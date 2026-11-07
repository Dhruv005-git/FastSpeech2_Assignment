"""
preprocess.py — English (Svarah) preprocessing pipeline for FastSpeech 2.

Steps
-----
1. Download Svarah subset (~5 h train, ~1 h val) from HuggingFace.
2. Extract phoneme sequences with phonemizer (espeak backend).
3. Compute mel spectrograms (librosa).
4. Extract pitch with PyWorld, interpolate unvoiced frames.
5. Compute energy (L2-norm per mel frame).
6. Run MFA forced alignment; parse TextGrid durations.
7. Compute & save normalisation statistics.
8. Write numpy arrays + train/val split lists.
"""

import os
import re
import sys
import json
import csv
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import yaml
from tqdm import tqdm
from num2words import num2words

# ── optional heavy imports — imported lazily or at top level ──────────────────
try:
    import pyworld as pw
except ImportError:
    pw = None  # will raise a clear error at runtime if needed

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
except ImportError:
    phonemize = None

try:
    import tgt
except ImportError:
    tgt = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config helper
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download Svarah subset
# ─────────────────────────────────────────────────────────────────────────────

def download_svarah_subset(
    output_dir: str,
    target_hours_train: float = 5.0,
    target_hours_val: float = 1.0,
) -> str:
    """
    Stream Svarah from HuggingFace and save a subset for training and
    validation.

    Returns
    -------
    Path to saved metadata CSV.
    """
    if load_dataset is None:
        raise ImportError("Install 'datasets': pip install datasets")

    output_dir = Path(output_dir)
    wav_dir = output_dir / "wavs"
    txt_dir = output_dir / "txts"
    wav_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading Svarah dataset from HuggingFace (streaming)…")
    # Svarah has a single split called 'train'
    dataset = load_dataset(
        "ai4bharat/Svarah",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    target_train_sec = target_hours_train * 3600.0
    target_val_sec = target_hours_val * 3600.0
    target_total_sec = target_train_sec + target_val_sec

    rows: List[dict] = []
    accumulated_sec = 0.0

    log.info(
        f"Collecting up to {target_hours_train + target_hours_val:.1f} h "
        f"({target_total_sec:.0f} s) of audio…"
    )

    for sample in tqdm(dataset, desc="Downloading Svarah", unit="sample"):
        # ── audio ──────────────────────────────────────────────────────────
        audio_arr = np.array(sample["audio"]["array"], dtype=np.float32)
        orig_sr = sample["audio"]["sampling_rate"]
        transcript = sample.get("text", sample.get("transcript", "")).strip()

        if not transcript:
            continue

        # Compute duration in seconds at original sample rate
        duration_sec = len(audio_arr) / orig_sr

        # Skip extremely short or long utterances
        if duration_sec < 0.5 or duration_sec > 20.0:
            continue

        accumulated_sec += duration_sec
        if accumulated_sec > target_total_sec:
            break

        # ── derive file id ──────────────────────────────────────────────
        file_id = sample.get("id", sample.get("utt_id", f"svarah_{len(rows):06d}"))
        # sanitise
        file_id = re.sub(r"[^A-Za-z0-9_\-]", "_", str(file_id))

        # ── resample to 22 050 Hz and save WAV ─────────────────────────
        if orig_sr != 22050:
            audio_arr = librosa.resample(
                audio_arr, orig_sr=orig_sr, target_sr=22050
            )
        wav_path = wav_dir / f"{file_id}.wav"
        sf.write(str(wav_path), audio_arr, 22050, subtype="PCM_16")

        # ── save transcript ─────────────────────────────────────────────
        txt_path = txt_dir / f"{file_id}.txt"
        txt_path.write_text(transcript, encoding="utf-8")

        rows.append(
            {
                "file_id": file_id,
                "text": transcript,
                "duration": round(duration_sec, 4),
                "split": "",  # assigned below
            }
        )

    if not rows:
        raise RuntimeError("No samples collected — check Svarah availability.")

    # ── assign train / val split ────────────────────────────────────────────
    total_dur = sum(r["duration"] for r in rows)
    val_budget = target_val_sec
    val_accumulated = 0.0

    for r in reversed(rows):
        if val_accumulated < val_budget:
            r["split"] = "val"
            val_accumulated += r["duration"]
        else:
            r["split"] = "train"

    # ── write metadata CSV ──────────────────────────────────────────────────
    meta_path = output_dir / "metadata.csv"
    with open(str(meta_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "text", "duration", "split"])
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    log.info(
        f"Saved {len(rows)} utterances: {n_train} train / {n_val} val — "
        f"total {total_dur / 3600:.2f} h"
    )
    log.info(f"Metadata saved → {meta_path}")
    return str(meta_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text / phoneme utilities
# ─────────────────────────────────────────────────────────────────────────────

def _expand_numbers(text: str) -> str:
    """Replace digit tokens with spoken-word equivalents."""

    def _replace(match):
        token = match.group(0)
        try:
            # Handle decimals
            if "." in token:
                return num2words(float(token))
            return num2words(int(token))
        except Exception:
            return token

    return re.sub(r"\b\d+(?:\.\d+)?\b", _replace, text)


def _basic_english_clean(text: str) -> str:
    """Lowercase, expand contractions, remove unusual characters."""
    text = text.lower()
    text = text.replace("'s", " is").replace("'re", " are")
    text = text.replace("'ll", " will").replace("'ve", " have")
    text = text.replace("'d", " would").replace("can't", "cannot")
    text = text.replace("n't", " not")
    text = _expand_numbers(text)
    # Keep only letters, spaces, and common punctuation that espeak handles
    text = re.sub(r"[^a-z\s\.,!\?;:\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_phonemes(text: str) -> List[str]:
    """
    Convert English text to a list of ARPAbet / IPA phonemes using
    phonemizer's espeak backend.

    Returns
    -------
    List of phoneme strings, e.g. ['DH', 'AH0', 'K', 'AE1', 'T']
    """
    if phonemize is None:
        raise ImportError("Install phonemizer: pip install phonemizer")

    text = _basic_english_clean(text)
    if not text:
        return []

    sep = Separator(phone=" ", word="  ", syllable="")
    backend = EspeakBackend(
        language="en-us",
        preserve_punctuation=False,
        with_stress=True,
        language_switch="remove-flags",
    )
    result = backend.phonemize([text], separator=sep)[0]

    # Split on spaces; filter empty strings
    phonemes = [p for p in result.strip().split(" ") if p]
    return phonemes


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mel spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def extract_mel_spectrogram(wav_path: str, config: dict) -> np.ndarray:
    """
    Load a WAV file and compute a log-mel spectrogram.

    Parameters
    ----------
    wav_path : str
        Path to the WAV file.
    config : dict
        Config dict with keys: sampling_rate, filter_length, hop_length,
        win_length, n_mel_channels, mel_fmin, mel_fmax, max_wav_value.

    Returns
    -------
    mel : np.ndarray, shape (n_mel_channels, T)
    """
    sr = config["sampling_rate"]
    audio, orig_sr = librosa.load(wav_path, sr=None, mono=True)

    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Normalise amplitude
    audio = audio / (np.abs(audio).max() + 1e-8)

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=config["filter_length"],
        n_mels=config["n_mel_channels"],
        fmin=config["mel_fmin"],
        fmax=config["mel_fmax"],
    )

    stft = librosa.stft(
        audio,
        n_fft=config["filter_length"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        window="hann",
        center=True,
    )
    magnitude = np.abs(stft) ** 2
    mel = np.dot(mel_basis, magnitude)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))  # log-mel

    return mel.astype(np.float32)  # (n_mel, T)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pitch (F0) extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pitch(wav_path: str, config: dict) -> np.ndarray:
    """
    Extract fundamental frequency (F0) using PyWorld, interpolate unvoiced
    regions, and downsample to mel-frame resolution.

    Returns
    -------
    pitch : np.ndarray, shape (T_mel,)  — values in Hz
    """
    if pw is None:
        raise ImportError("Install pyworld: pip install pyworld")

    sr = config["sampling_rate"]
    hop = config["hop_length"]

    audio, orig_sr = librosa.load(wav_path, sr=None, mono=True)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    audio = audio.astype(np.float64)

    # PyWorld expects float64 and 1-D
    _f0, t = pw.dio(audio, sr, frame_period=hop / sr * 1000)  # ms
    f0 = pw.stonemask(audio, _f0, t, sr)

    # Interpolate unvoiced (zero) regions
    voiced = f0 > 0
    if voiced.sum() < 2:
        # Fully unvoiced: return zeros
        return np.zeros(len(f0), dtype=np.float32)

    xp = np.where(voiced)[0]
    fp = f0[voiced]
    xi = np.arange(len(f0))
    f0_interp = np.interp(xi, xp, fp)

    # Restore original zero frames at boundaries (before first / after last voiced)
    f0_interp[: xp[0]] = 0.0
    f0_interp[xp[-1] + 1 :] = 0.0

    return f0_interp.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Energy
# ─────────────────────────────────────────────────────────────────────────────

def extract_energy(mel: np.ndarray) -> np.ndarray:
    """
    Compute per-frame energy as the L2-norm of the mel spectrogram.

    Parameters
    ----------
    mel : np.ndarray, shape (n_mel, T)

    Returns
    -------
    energy : np.ndarray, shape (T,)
    """
    energy = np.linalg.norm(mel, axis=0)  # shape (T,)
    return energy.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MFA forced alignment
# ─────────────────────────────────────────────────────────────────────────────

def run_mfa(
    data_dir: str,
    output_dir: str,
    language: str = "english",
    n_jobs: int = 4,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Run the Montreal Forced Aligner on a prepared corpus and parse the
    resulting TextGrid files into phoneme-duration sequences.

    The ``data_dir`` must contain:
        {speaker}/
            {utterance}.wav
            {utterance}.txt   (transcription)

    Parameters
    ----------
    data_dir : str
        Corpus root that MFA will read.
    output_dir : str
        Where MFA writes TextGrid files.
    language : str
        MFA acoustic-model / dictionary name.
    n_jobs : int
        Parallel jobs for MFA.

    Returns
    -------
    durations : dict
        {file_id: [(phoneme, n_frames), ...]}
    """
    if tgt is None:
        raise ImportError("Install tgt: pip install tgt")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── MFA validate (optional) ────────────────────────────────────────────
    validate_cmd = [
        "mfa",
        "validate",
        str(data_dir),
        "english_us_arpa",
        "english_us_arpa",
        "--clean",
    ]
    log.info(f"Running MFA validate: {' '.join(validate_cmd)}")
    try:
        subprocess.run(validate_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        log.warning(f"MFA validate returned non-zero exit ({exc.returncode}) — continuing.")
    except FileNotFoundError:
        raise RuntimeError(
            "MFA executable not found. Install with: "
            "conda install -c conda-forge montreal-forced-aligner"
        )

    # ── MFA align ─────────────────────────────────────────────────────────
    align_cmd = [
        "mfa",
        "align",
        str(data_dir),
        "english_us_arpa",
        "english_us_arpa",
        str(output_dir),
        "--clean",
        f"--num_jobs={n_jobs}",
    ]
    log.info(f"Running MFA align: {' '.join(align_cmd)}")
    result = subprocess.run(
        align_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"MFA stdout:\n{result.stdout}")
        log.error(f"MFA stderr:\n{result.stderr}")
        raise RuntimeError(f"MFA align failed (exit {result.returncode}).")

    log.info("MFA alignment complete — parsing TextGrids…")

    # ── Parse TextGrids ────────────────────────────────────────────────────
    durations: Dict[str, List[Tuple[str, int]]] = {}
    tg_files = list(out_path.rglob("*.TextGrid"))

    if not tg_files:
        raise RuntimeError(
            f"No TextGrid files found in {output_dir}. "
            "MFA may have failed silently — check logs."
        )

    # We need hop_length / sample_rate to convert seconds → frames.
    # These are stored in a small side-file written by preprocess_all.
    # Fallback to sensible defaults if not present.
    hop_length = 256
    sampling_rate = 22050
    sidecar = out_path.parent / "mfa_params.json"
    if sidecar.exists():
        with open(str(sidecar)) as fp:
            params = json.load(fp)
            hop_length = params.get("hop_length", hop_length)
            sampling_rate = params.get("sampling_rate", sampling_rate)

    for tg_path in tqdm(tg_files, desc="Parsing TextGrids", unit="file"):
        file_id = tg_path.stem
        try:
            tg = tgt.io.read_textgrid(str(tg_path))
        except Exception as exc:
            log.warning(f"Cannot read {tg_path}: {exc}")
            continue

        # Locate the phones tier
        tier = None
        for tier_name in ("phones", "phone", "Phone", "Phones"):
            try:
                tier = tg.get_tier_by_name(tier_name)
                break
            except Exception:
                pass

        if tier is None:
            log.warning(f"No phones tier in {tg_path}, skipping.")
            continue

        phone_durations: List[Tuple[str, int]] = []
        for interval in tier:
            phone = interval.text.strip()
            if not phone or phone in {"", "sp", "SIL", "sil", "<eps>"}:
                phone = "sp"  # silence marker
            dur_sec = interval.end_time - interval.start_time
            dur_frames = max(1, int(round(dur_sec * sampling_rate / hop_length)))
            phone_durations.append((phone, dur_frames))

        if phone_durations:
            durations[file_id] = phone_durations

    log.info(f"Parsed {len(durations)} TextGrids successfully.")
    return durations


# ─────────────────────────────────────────────────────────────────────────────
# 7. Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(
    pitch_list: List[np.ndarray],
    energy_list: List[np.ndarray],
    out_dir: str,
) -> dict:
    """
    Compute mean and standard deviation for pitch and energy across the
    training set and save them to stats.json.

    Parameters
    ----------
    pitch_list : list of 1-D arrays (voiced samples only used for pitch)
    energy_list : list of 1-D arrays
    out_dir : str

    Returns
    -------
    stats : dict with keys pitch_mean, pitch_std, energy_mean, energy_std
    """
    # For pitch, ignore zero (unvoiced) frames
    all_pitch = np.concatenate(pitch_list)
    voiced_pitch = all_pitch[all_pitch > 0]
    pitch_mean = float(voiced_pitch.mean()) if len(voiced_pitch) else 0.0
    pitch_std = float(voiced_pitch.std()) if len(voiced_pitch) else 1.0

    all_energy = np.concatenate(energy_list)
    energy_mean = float(all_energy.mean())
    energy_std = float(all_energy.std()) if all_energy.std() > 0 else 1.0

    stats = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }

    stats_path = Path(out_dir) / "stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(stats, f, indent=2)

    log.info(
        f"Stats saved → {stats_path}\n"
        f"  pitch  mean={pitch_mean:.2f} std={pitch_std:.2f}\n"
        f"  energy mean={energy_mean:.4f} std={energy_std:.4f}"
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MFA corpus preparation helper
# ─────────────────────────────────────────────────────────────────────────────

def prepare_mfa_corpus(raw_dir: str, mfa_input_dir: str, metadata: List[dict]) -> None:
    """
    Arrange files in the flat MFA corpus format expected by MFA:
        {mfa_input_dir}/{speaker}/{utt}.wav
        {mfa_input_dir}/{speaker}/{utt}.txt

    We use a single speaker directory called 'svarah'.
    """
    speaker_dir = Path(mfa_input_dir) / "svarah"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = Path(raw_dir) / "wavs"
    txt_dir = Path(raw_dir) / "txts"

    for row in tqdm(metadata, desc="Preparing MFA corpus", unit="file"):
        fid = row["file_id"]
        src_wav = wav_dir / f"{fid}.wav"
        src_txt = txt_dir / f"{fid}.txt"
        dst_wav = speaker_dir / f"{fid}.wav"
        dst_txt = speaker_dir / f"{fid}.txt"

        if src_wav.exists() and not dst_wav.exists():
            import shutil
            shutil.copy2(str(src_wav), str(dst_wav))
        if src_txt.exists() and not dst_txt.exists():
            import shutil
            shutil.copy2(str(src_txt), str(dst_txt))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_all(config_path: str) -> None:
    """
    Main preprocessing pipeline.

    1. Download Svarah subset.
    2. Run MFA alignment.
    3. For each utterance: compute mel, pitch, energy.
    4. Compute & save stats.
    5. Write numpy arrays and train/val lists.
    """
    cfg = load_config(config_path)

    raw_dir = Path(cfg["raw_path"])
    prep_dir = Path(cfg["preprocessed_path"])
    prep_dir.mkdir(parents=True, exist_ok=True)

    sampling_rate = cfg["sampling_rate"]
    hop_length = cfg["hop_length"]

    # ── Step 1: Download ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1 — Downloading Svarah subset")
    log.info("=" * 60)

    meta_path = raw_dir / "metadata.csv"
    if meta_path.exists():
        log.info(f"metadata.csv already exists at {meta_path}, skipping download.")
    else:
        download_svarah_subset(
            str(raw_dir),
            target_hours_train=5.0,
            target_hours_val=1.0,
        )

    # Read metadata
    metadata: List[dict] = []
    with open(str(meta_path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    # ── Step 2: MFA ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2 — Montreal Forced Alignment")
    log.info("=" * 60)

    mfa_input_dir = raw_dir / "mfa_input"
    mfa_output_dir = raw_dir / "mfa_output"

    prepare_mfa_corpus(str(raw_dir), str(mfa_input_dir), metadata)

    # Save params for TextGrid parser
    sidecar_path = mfa_output_dir.parent / "mfa_params.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(sidecar_path), "w") as fp:
        json.dump({"hop_length": hop_length, "sampling_rate": sampling_rate}, fp)

    durations_map = run_mfa(
        str(mfa_input_dir),
        str(mfa_output_dir),
        language="english",
    )

    # ── Step 3 & 4 & 5: Feature extraction ───────────────────────────────
    log.info("=" * 60)
    log.info("STEP 3–5 — Feature extraction")
    log.info("=" * 60)

    wav_dir = raw_dir / "wavs"

    train_ids: List[str] = []
    val_ids: List[str] = []

    train_pitches: List[np.ndarray] = []
    train_energies: List[np.ndarray] = []

    skipped = 0

    for row in tqdm(metadata, desc="Extracting features", unit="utt"):
        fid = row["file_id"]
        split = row["split"]
        wav_path = wav_dir / f"{fid}.wav"

        if not wav_path.exists():
            log.warning(f"WAV not found: {wav_path}, skipping.")
            skipped += 1
            continue

        if fid not in durations_map:
            log.warning(f"No MFA alignment for {fid}, skipping.")
            skipped += 1
            continue

        # Mel
        try:
            mel = extract_mel_spectrogram(str(wav_path), cfg)
        except Exception as exc:
            log.warning(f"Mel extraction failed for {fid}: {exc}")
            skipped += 1
            continue

        # Pitch
        try:
            pitch = extract_pitch(str(wav_path), cfg)
        except Exception as exc:
            log.warning(f"Pitch extraction failed for {fid}: {exc}")
            skipped += 1
            continue

        # Energy
        energy = extract_energy(mel)

        # Duration: sum frames per phoneme from MFA
        phone_dur_pairs = durations_map[fid]
        phones = [p for p, _ in phone_dur_pairs]
        dur_arr = np.array([d for _, d in phone_dur_pairs], dtype=np.int32)

        # Align lengths: trim / pad pitch & energy to mel length T
        T = mel.shape[1]
        if len(pitch) > T:
            pitch = pitch[:T]
        elif len(pitch) < T:
            pitch = np.pad(pitch, (0, T - len(pitch)))

        if len(energy) > T:
            energy = energy[:T]
        elif len(energy) < T:
            energy = np.pad(energy, (0, T - len(energy)))

        # Save arrays
        np.save(str(prep_dir / f"mel_{fid}.npy"), mel)
        np.save(str(prep_dir / f"pitch_{fid}.npy"), pitch)
        np.save(str(prep_dir / f"energy_{fid}.npy"), energy)
        np.save(str(prep_dir / f"duration_{fid}.npy"), dur_arr)

        # Save phoneme sequence
        (prep_dir / f"phoneme_{fid}.txt").write_text(
            " ".join(phones), encoding="utf-8"
        )

        if split == "train":
            train_ids.append(fid)
            train_pitches.append(pitch)
            train_energies.append(energy)
        else:
            val_ids.append(fid)

    log.info(
        f"Feature extraction complete. "
        f"train={len(train_ids)}, val={len(val_ids)}, skipped={skipped}"
    )

    # ── Step 6: Stats ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 6 — Computing normalisation statistics")
    log.info("=" * 60)
    compute_stats(train_pitches, train_energies, str(prep_dir))

    # ── Step 7: Write split lists ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 7 — Writing split lists")
    log.info("=" * 60)

    (prep_dir / "train.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (prep_dir / "val.txt").write_text("\n".join(val_ids), encoding="utf-8")

    log.info(f"train.txt  → {len(train_ids)} utterances")
    log.info(f"val.txt    → {len(val_ids)} utterances")
    log.info("Preprocessing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Svarah for FastSpeech 2")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    preprocess_all(args.config)
