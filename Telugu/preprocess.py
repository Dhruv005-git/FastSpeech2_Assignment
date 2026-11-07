"""
Telugu/preprocess.py
====================
Preprocessing pipeline for the IndicTTS Telugu dataset.

Steps:
  1. Download / load data from HuggingFace or local directory.
  2. Filter to ~5 hrs train / ~1 hr val.
  3. Normalize text and extract phoneme tokens.
  4. Extract mel spectrogram, pitch (F0), and energy features.
  5. Run MFA forced alignment (with aeneas / proportional fallback).
  6. Compute dataset-level statistics.
  7. Save .npy arrays + metadata CSV in the preprocessed directory.

Usage:
    python preprocess.py --config config.yaml
    python preprocess.py --config config.yaml --skip_download
    python preprocess.py --config config.yaml --skip_mfa
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Local imports — ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.audio import (
    compute_mel_spectrogram,
    get_duration_seconds,
    load_audio,
    normalize_audio,
    save_audio,
    trim_silence,
)
from utils.text_telugu import (
    aksharas_to_ipa,
    build_telugu_vocab,
    save_vocab,
    telugu_cleaners,
)

# Optional heavy dependencies
try:
    import pyworld as pw
    _HAS_PYWORLD = True
except ImportError:
    _HAS_PYWORLD = False
    warnings.warn("pyworld not found — pitch extraction will return zeros.")

try:
    import tgt
    _HAS_TGT = True
except ImportError:
    _HAS_TGT = False
    warnings.warn("tgt not found — MFA TextGrid parsing unavailable.")

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False
    warnings.warn("HuggingFace 'datasets' not found — only local loading available.")


# ===========================================================================
# 1.  DATA ACQUISITION
# ===========================================================================

def download_indic_tts_telugu(
    output_dir: str,
    target_hours_train: float = 5.0,
    target_hours_val: float = 1.0,
) -> str:
    """Download (or load) the IndicTTS Telugu dataset and produce metadata.csv.

    Data selection strategy
    -----------------------
    * We iterate through the HuggingFace split in order and accumulate
      audio duration until we have >= *target_hours_train* seconds of
      training data and >= *target_hours_val* seconds of validation data.
    * Clips shorter than 0.5 s or longer than 15 s are skipped (quality
      heuristic — very short clips often contain only silence; very long
      ones cause GPU OOM during training).
    * Files are written as 22050 Hz mono WAV to ``output_dir/wavs/``.
    * ``metadata.csv`` columns: file_id, text, duration, split

    HuggingFace dataset
    -------------------
    Dataset ID : ai4bharat/IndicTTS   |   config : te
    URL        : https://huggingface.co/datasets/ai4bharat/IndicTTS

    Local fallback
    --------------
    If HuggingFace is unavailable, download the corpus from:
        https://www.iitm.ac.in/donlab/tts/index.php
    Then call this function with ``hf_unavailable=True`` (see below).
    The expected local layout is:
        output_dir/
          wavs/        ← .wav files named {file_id}.wav
          metadata.csv ← file_id,text columns (no header)

    Args:
        output_dir:           Root directory for raw data.
        target_hours_train:   Target training-set duration in hours.
        target_hours_val:     Target validation-set duration in hours.

    Returns:
        Path to the generated metadata.csv file.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.csv")

    # -----------------------------------------------------------
    # Check if already downloaded
    # -----------------------------------------------------------
    if os.path.isfile(meta_path):
        print(f"[preprocess] metadata.csv already exists at {meta_path}. Skipping download.")
        return meta_path

    # -----------------------------------------------------------
    # Try HuggingFace
    # -----------------------------------------------------------
    if _HAS_DATASETS:
        print("[preprocess] Loading IndicTTS Telugu from HuggingFace …")
        try:
            ds = load_dataset("ai4bharat/IndicTTS", "te", trust_remote_code=True)
            # Merge all splits for custom duration-based selection
            all_items = list(ds.get("train", [])) + list(ds.get("test", []))
        except Exception as e:
            print(f"[preprocess] HuggingFace load failed: {e}")
            all_items = []
    else:
        all_items = []

    if not all_items:
        # -----------------------------------------------------------
        # Local fallback: read existing wavs + a simple text file
        # -----------------------------------------------------------
        print(
            "[preprocess] HuggingFace unavailable.\n"
            "  Please download the IndicTTS Telugu corpus from:\n"
            "    https://www.iitm.ac.in/donlab/tts/index.php\n"
            "  and unpack it into:\n"
            f"    {output_dir}/\n"
            "  with the following structure:\n"
            "    wavs/     ← .wav files\n"
            "    text.txt  ← lines of '<file_id>|<telugu_text>'\n"
            "  Then re-run preprocess.py.\n"
        )
        return _build_metadata_from_local(output_dir, target_hours_train, target_hours_val)

    # -----------------------------------------------------------
    # Build metadata from HuggingFace items
    # -----------------------------------------------------------
    target_train_s = target_hours_train * 3600.0
    target_val_s   = target_hours_val   * 3600.0
    min_dur = 0.5
    max_dur = 15.0

    train_rows: List[Dict] = []
    val_rows:   List[Dict] = []
    train_acc   = 0.0
    val_acc     = 0.0

    for item in all_items:
        # HF IndicTTS schema: {'audio': {'array': ..., 'sampling_rate': ...}, 'text': ...}
        audio_array = np.array(item["audio"]["array"], dtype=np.float32)
        sr_orig     = item["audio"]["sampling_rate"]
        raw_text    = item.get("text", item.get("sentence", ""))
        if not raw_text:
            continue

        # Resample to 22050 if needed
        if sr_orig != 22050:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr_orig, target_sr=22050)

        dur = len(audio_array) / 22050.0
        if dur < min_dur or dur > max_dur:
            continue

        file_id = f"te_{len(train_rows) + len(val_rows):06d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        save_audio(normalize_audio(audio_array), wav_path, sr=22050)

        row = {"file_id": file_id, "text": raw_text, "duration": round(dur, 4)}

        # Assign to train first, then val once train quota is filled
        if train_acc < target_train_s:
            row["split"] = "train"
            train_rows.append(row)
            train_acc += dur
        elif val_acc < target_val_s:
            row["split"] = "val"
            val_rows.append(row)
            val_acc += dur
        else:
            break  # Both quotas filled

    all_rows = train_rows + val_rows
    print(
        f"[preprocess] Selected {len(train_rows)} train clips ({train_acc/3600:.2f} hrs) "
        f"and {len(val_rows)} val clips ({val_acc/3600:.2f} hrs)."
    )

    # Write metadata CSV
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "text", "duration", "split"])
        writer.writeheader()
        writer.writerows(all_rows)

    return meta_path


def _build_metadata_from_local(
    output_dir: str,
    target_hours_train: float,
    target_hours_val: float,
) -> str:
    """Build metadata.csv from a locally extracted IndicTTS corpus."""
    wav_dir   = os.path.join(output_dir, "wavs")
    text_file = os.path.join(output_dir, "text.txt")
    meta_path = os.path.join(output_dir, "metadata.csv")

    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
    if not os.path.isfile(text_file):
        raise FileNotFoundError(
            f"Transcription file not found: {text_file}\n"
            "Expected format: '<file_id>|<telugu_text>' per line."
        )

    # Load transcription map
    text_map: Dict[str, str] = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                fid, txt = line.split("|", 1)
                text_map[fid.strip()] = txt.strip()

    target_train_s = target_hours_train * 3600.0
    target_val_s   = target_hours_val   * 3600.0
    min_dur, max_dur = 0.5, 15.0

    train_rows, val_rows = [], []
    train_acc = val_acc = 0.0

    wav_files = sorted(Path(wav_dir).glob("*.wav"))
    for wav_path in wav_files:
        fid = wav_path.stem
        if fid not in text_map:
            continue
        try:
            dur = get_duration_seconds(str(wav_path))
        except Exception:
            continue
        if dur < min_dur or dur > max_dur:
            continue
        row = {"file_id": fid, "text": text_map[fid], "duration": round(dur, 4)}
        if train_acc < target_train_s:
            row["split"] = "train"; train_rows.append(row); train_acc += dur
        elif val_acc < target_val_s:
            row["split"] = "val";   val_rows.append(row);   val_acc += dur
        else:
            break

    all_rows = train_rows + val_rows
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "text", "duration", "split"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"[preprocess] Local: {len(train_rows)} train ({train_acc/3600:.2f} h), "
        f"{len(val_rows)} val ({val_acc/3600:.2f} h)."
    )
    return meta_path


# ===========================================================================
# 2.  TEXT → PHONEMES
# ===========================================================================

def telugu_text_to_phonemes(text: str) -> List[str]:
    """Clean and phonemise a single Telugu text string.

    Args:
        text: Raw Telugu text.

    Returns:
        List of phoneme / grapheme token strings.
    """
    cleaned = telugu_cleaners(text)
    return aksharas_to_ipa(cleaned)


# ===========================================================================
# 3.  FEATURE EXTRACTION
# ===========================================================================

def extract_mel_spectrogram(wav_path: str, config: dict) -> np.ndarray:
    """Load audio and compute an 80-band log-mel spectrogram.

    Args:
        wav_path: Path to a 22050 Hz mono .wav file.
        config:   Config dict (needs audio section keys).

    Returns:
        (n_mels=80, T) float32 numpy array.
    """
    wav = load_audio(wav_path, sr=config["sampling_rate"])
    wav = normalize_audio(wav)
    mel = compute_mel_spectrogram(wav, config)
    return mel


def extract_pitch(wav_path: str, config: dict) -> np.ndarray:
    """Extract fundamental frequency (F0) using WORLD vocoder (pyworld).

    Unvoiced frames are set to 0.0.

    Args:
        wav_path: Path to a 22050 Hz mono .wav file.
        config:   Config dict (needs sampling_rate, hop_length).

    Returns:
        1-D float32 numpy array of F0 values, shape (T,).
    """
    if not _HAS_PYWORLD:
        # Graceful fallback: return zeros
        import librosa
        wav = load_audio(wav_path, sr=config["sampling_rate"])
        n_frames = 1 + len(wav) // config["hop_length"]
        return np.zeros(n_frames, dtype=np.float32)

    wav = load_audio(wav_path, sr=config["sampling_rate"])
    wav = normalize_audio(wav)

    sr         = config["sampling_rate"]
    hop_length = config["hop_length"]
    frame_period_ms = hop_length / sr * 1000.0  # milliseconds

    f0, timeaxis = pw.harvest(
        wav.astype(np.float64),
        sr,
        f0_floor=config.get("pitch_fmin", 50.0),
        f0_ceil=config.get("pitch_fmax", 600.0),
        frame_period=frame_period_ms,
    )
    return f0.astype(np.float32)


def extract_energy(mel: np.ndarray) -> np.ndarray:
    """Compute per-frame energy as the L2-norm of mel-spectrogram columns.

    Args:
        mel: (n_mels, T) log-mel spectrogram.

    Returns:
        1-D float32 numpy array of shape (T,).
    """
    # Convert from log to linear before computing norm
    mel_lin = np.exp(mel)
    energy  = np.linalg.norm(mel_lin, axis=0)  # (T,)
    return energy.astype(np.float32)


# ===========================================================================
# 4.  FORCED ALIGNMENT
# ===========================================================================

def run_mfa_telugu(data_dir: str, output_dir: str) -> bool:
    """Run Montreal Forced Aligner (MFA) for Telugu.

    Prerequisites:
        conda install -c conda-forge montreal-forced-aligner
        mfa model download acoustic telugu_mfa
        mfa model download dictionary telugu_mfa

    MFA expects the data directory to contain:
        wavs/<file_id>.wav
        txts/<file_id>.txt    ← transcription in Telugu script

    Args:
        data_dir:   Directory containing the wav + txt files.
        output_dir: Directory where TextGrid files are written.

    Returns:
        True if MFA completed successfully, False otherwise.
    """
    mfa_cmd = [
        "mfa", "align",
        data_dir,
        "telugu_mfa",   # dictionary
        "telugu_mfa",   # acoustic model
        output_dir,
        "--clean",
        "--overwrite",
    ]
    print(f"[preprocess] Running MFA: {' '.join(mfa_cmd)}")
    try:
        result = subprocess.run(
            mfa_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("[preprocess] WARNING: 'mfa' binary not found. "
              "Install MFA: https://montreal-forced-aligner.readthedocs.io/")
    except subprocess.CalledProcessError as e:
        print(f"[preprocess] WARNING: MFA failed.\n  stderr: {e.stderr}")
    return False


def parse_textgrid(textgrid_path: str, hop_length: int, sr: int) -> Optional[List[Tuple[str, int]]]:
    """Parse an MFA TextGrid file into (phoneme, duration_frames) pairs.

    Args:
        textgrid_path: Path to the .TextGrid file.
        hop_length:    STFT hop length in samples.
        sr:            Sampling rate.

    Returns:
        List of (phoneme, frames) tuples, or None on failure.
    """
    if not _HAS_TGT:
        print("[preprocess] tgt not installed — cannot parse TextGrid.")
        return None
    try:
        tg       = tgt.io.read_textgrid(textgrid_path)
        tier     = tg.get_tier_by_name("phones")
        entries  = []
        for interval in tier.intervals:
            phone  = interval.text.strip() or "<sil>"
            dur_s  = interval.end_time - interval.start_time
            frames = max(1, round(dur_s * sr / hop_length))
            entries.append((phone, frames))
        return entries
    except Exception as e:
        print(f"[preprocess] TextGrid parse error ({textgrid_path}): {e}")
        return None


def estimate_duration_proportional(
    phonemes: List[str],
    total_frames: int,
) -> List[int]:
    """Fallback: distribute total frames proportionally across phonemes.

    This is a rough heuristic — MFA should be preferred whenever available.
    Silence tokens (<sil>) are assigned a fixed 5-frame budget; the
    remainder is distributed equally among phonemes.

    Args:
        phonemes:     List of phoneme strings.
        total_frames: Total number of mel frames for the utterance.

    Returns:
        List of integer frame counts (same length as phonemes).
    """
    sil_frames_each = 5
    n_sil  = sum(1 for p in phonemes if p == "<sil>")
    n_real = len(phonemes) - n_sil
    sil_budget  = sil_frames_each * n_sil
    real_budget = max(total_frames - sil_budget, n_real)
    frames_per  = real_budget // max(n_real, 1)

    durations: List[int] = []
    allocated = 0
    for i, ph in enumerate(phonemes):
        if ph == "<sil>":
            durations.append(sil_frames_each)
            allocated += sil_frames_each
        elif i == len(phonemes) - 1:
            # Give the last phoneme any rounding remainder
            durations.append(total_frames - allocated)
        else:
            durations.append(frames_per)
            allocated += frames_per
    return durations


# ===========================================================================
# 5.  STATISTICS
# ===========================================================================

def compute_stats(data: List[Dict]) -> Dict:
    """Compute dataset-level pitch and energy statistics.

    Args:
        data: List of dicts, each with keys 'pitch' and 'energy'
              (1-D numpy arrays).

    Returns:
        Dict with keys: pitch_mean, pitch_std, energy_mean, energy_std.
    """
    all_pitch  = np.concatenate([d["pitch"][d["pitch"] > 0] for d in data if d["pitch"].any()])
    all_energy = np.concatenate([d["energy"] for d in data])
    return {
        "pitch_mean":  float(np.mean(all_pitch)),
        "pitch_std":   float(np.std(all_pitch)),
        "energy_mean": float(np.mean(all_energy)),
        "energy_std":  float(np.std(all_energy)),
    }


# ===========================================================================
# 6.  ORCHESTRATION
# ===========================================================================

def preprocess_all(config_path: str,
                   skip_download: bool = False,
                   skip_mfa: bool = False) -> None:
    """Run the full Telugu preprocessing pipeline.

    Args:
        config_path:    Path to config.yaml.
        skip_download:  If True, assume data already downloaded.
        skip_mfa:       If True, use proportional duration fallback.
    """
    # ----- Load config -----
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path   = config["raw_path"]
    prep_path  = config["preprocessed_path"]
    align_path = config["alignment_path"]
    vocab_path = config["vocab_path"]
    sr         = config["sampling_rate"]
    hop_length = config["hop_length"]

    for d in [raw_path, prep_path, align_path]:
        os.makedirs(d, exist_ok=True)

    # ----- Step 1: Download -----
    if not skip_download:
        meta_path = download_indic_tts_telugu(
            raw_path,
            target_hours_train=config.get("target_hours_train", 5.0),
            target_hours_val=config.get("target_hours_val", 1.0),
        )
    else:
        meta_path = os.path.join(raw_path, "metadata.csv")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"metadata.csv not found at {meta_path}.")

    # ----- Read metadata -----
    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # ----- Step 2: Phonemise + build vocab -----
    print("[preprocess] Building vocabulary …")
    all_texts = [r["text"] for r in rows]
    vocab = build_telugu_vocab(all_texts)
    save_vocab(vocab, vocab_path)
    print(f"[preprocess] Vocab size: {len(vocab)} tokens → {vocab_path}")

    # ----- Step 3: MFA -----
    # Prepare MFA input directory (wav + txt pairs)
    mfa_input = os.path.join(raw_path, "mfa_input")
    os.makedirs(mfa_input, exist_ok=True)

    if not skip_mfa:
        for row in rows:
            fid      = row["file_id"]
            src_wav  = os.path.join(raw_path, "wavs", f"{fid}.wav")
            dst_wav  = os.path.join(mfa_input, f"{fid}.wav")
            txt_path = os.path.join(mfa_input, f"{fid}.txt")
            if not os.path.isfile(dst_wav) and os.path.isfile(src_wav):
                shutil.copy(src_wav, dst_wav)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(telugu_cleaners(row["text"]))

        mfa_ok = run_mfa_telugu(mfa_input, align_path)
        if not mfa_ok:
            print("[preprocess] MFA unavailable — will use proportional duration fallback.")
            skip_mfa = True

    # ----- Step 4: Feature extraction -----
    processed_data: List[Dict] = []
    for split in ("train", "val"):
        split_dir = os.path.join(prep_path, split)
        for sub in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(split_dir, sub), exist_ok=True)

    sil_id = vocab.get("<sil>", 2)
    unk_id = vocab.get("<unk>", 1)

    for idx, row in enumerate(rows):
        fid   = row["file_id"]
        split = row["split"]
        text  = row["text"]
        wav_path = os.path.join(raw_path, "wavs", f"{fid}.wav")

        if not os.path.isfile(wav_path):
            print(f"[preprocess] WARNING: WAV not found — {wav_path}")
            continue

        # -- Mel
        mel = extract_mel_spectrogram(wav_path, config)
        # -- Pitch
        pitch = extract_pitch(wav_path, config)
        # Match frame count
        T = mel.shape[1]
        if len(pitch) > T:
            pitch = pitch[:T]
        elif len(pitch) < T:
            pitch = np.pad(pitch, (0, T - len(pitch)))
        # -- Energy
        energy = extract_energy(mel)

        # -- Phonemes
        phonemes = telugu_text_to_phonemes(text)
        phone_ids = [sil_id] + [vocab.get(p, unk_id) for p in phonemes] + [sil_id]
        phonemes_with_sil = ["<sil>"] + phonemes + ["<sil>"]

        # -- Durations
        tg_path = os.path.join(align_path, f"{fid}.TextGrid")
        if not skip_mfa and os.path.isfile(tg_path):
            dur_entries = parse_textgrid(tg_path, hop_length, sr)
        else:
            dur_entries = None

        if dur_entries is not None:
            durations = [d for _, d in dur_entries]
        else:
            durations = estimate_duration_proportional(phonemes_with_sil, T)

        # Clamp duration sum to mel length
        total_dur = sum(durations)
        if total_dur != T:
            # Adjust last phoneme
            durations[-1] = max(1, durations[-1] + (T - total_dur))

        # -- Save arrays
        split_dir = os.path.join(prep_path, split)
        np.save(os.path.join(split_dir, "mel",      f"{fid}.npy"), mel)
        np.save(os.path.join(split_dir, "pitch",    f"{fid}.npy"), pitch)
        np.save(os.path.join(split_dir, "energy",   f"{fid}.npy"), energy)
        np.save(os.path.join(split_dir, "duration", f"{fid}.npy"),
                np.array(durations, dtype=np.int32))

        processed_data.append({
            "file_id":  fid,
            "split":    split,
            "text":     text,
            "phonemes": phonemes_with_sil,
            "phone_ids": phone_ids,
            "pitch":    pitch,
            "energy":   energy,
        })

        if (idx + 1) % 100 == 0:
            print(f"[preprocess] Processed {idx+1}/{len(rows)} utterances …")

    # ----- Step 5: Stats -----
    if processed_data:
        stats = compute_stats(processed_data)
        stats_path = os.path.join(prep_path, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[preprocess] Stats: {stats}")
        print(f"[preprocess] Stats saved → {stats_path}")

    # ----- Step 6: Save phoneme ID lists -----
    for split in ("train", "val"):
        split_rows = [d for d in processed_data if d["split"] == split]
        phones_meta = [
            {"file_id": d["file_id"], "phone_ids": d["phone_ids"],
             "phonemes": d["phonemes"]}
            for d in split_rows
        ]
        phones_path = os.path.join(prep_path, split, "phones.json")
        with open(phones_path, "w", encoding="utf-8") as f:
            json.dump(phones_meta, f, ensure_ascii=False, indent=2)

    print(f"\n[preprocess] ✓ Done. Preprocessed data at: {prep_path}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IndicTTS Telugu data.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step (assume data already present).")
    parser.add_argument("--skip_mfa", action="store_true",
                        help="Skip MFA and use proportional duration fallback.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    preprocess_all(config_path,
                   skip_download=args.skip_download,
                   skip_mfa=args.skip_mfa)
