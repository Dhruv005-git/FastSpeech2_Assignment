"""
inference.py — FastSpeech 2 English inference script.

Usage
-----
python inference.py \\
    --checkpoint ./checkpoints/fs2_00300000.pt \\
    --config     ./config.yaml \\
    --output_dir ../Results/English

Outputs per sentence
--------------------
  wavs/{id}.wav
  duration/{id}_duration.png
  pitch/{id}_pitch.png
  energy/{id}_energy.png
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

# ── project imports ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

try:
    from model.fastspeech2 import FastSpeech2
except ImportError as exc:
    raise ImportError(
        "Cannot import model.fastspeech2. "
        "Make sure FastSpeech2_Assignment/ is on PYTHONPATH "
        f"and the model package exists.\n  Original: {exc}"
    )

try:
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
except ImportError:
    EspeakBackend = None  # type: ignore

try:
    from num2words import num2words
except ImportError:
    num2words = None  # type: ignore

import re

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Test sentences
# ─────────────────────────────────────────────────────────────────────────────

TEST_SENTENCES: Dict[str, str] = {
    "E1": (
        "The antidisestablishmentarianism movement gained considerable momentum "
        "during the parliamentary proceedings."
    ),
    "E2a": (
        "The orchestra's conduct during the ceremony was praised by the audience."
    ),
    "E2b": (
        "The researchers will conduct an experiment to evaluate the proposed method."
    ),
    "E3": "The rhythm of the orchestra changed unexpectedly.",
    "E4": (
        "The meeting will start at 10:45 AM on 12 November 2026, "
        "so andaroo should arrive on time."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def load_vocab(preprocessed_path: str) -> Dict[str, int]:
    vocab_path = Path(preprocessed_path) / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"vocab.txt not found at {vocab_path}. "
            "Run preprocessing and training first to generate it."
        )
    vocab: Dict[str, int] = {}
    for idx, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines()):
        token = line.strip()
        if token:
            vocab[token] = idx
    log.info(f"Loaded vocab: {len(vocab)} tokens from {vocab_path}")
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Text → phonemes
# ─────────────────────────────────────────────────────────────────────────────

def _expand_numbers(text: str) -> str:
    """Replace numeric tokens with spoken words."""

    def _replace(match):
        token = match.group(0)
        try:
            if num2words is None:
                return token
            if ":" in token:
                # e.g. "10:45" → "ten forty-five"
                parts = token.split(":")
                return num2words(int(parts[0])) + " " + num2words(int(parts[1]))
            if "." in token:
                return num2words(float(token))
            return num2words(int(token))
        except Exception:
            return token

    # Time pattern (e.g. 10:45)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", _replace, text)
    # Regular numbers
    text = re.sub(r"\b\d+(?:\.\d+)?\b", _replace, text)
    return text


def _clean_text(text: str) -> str:
    text = text.lower()
    # Basic contraction expansion
    contractions = {
        "'s": " is", "'re": " are", "'ll": " will",
        "'ve": " have", "'d": " would", "can't": "cannot",
        "won't": "will not", "n't": " not",
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = _expand_numbers(text)
    # Keep letters, spaces, and basic punctuation
    text = re.sub(r"[^a-z\s\.,!\?;:\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_phonemes(text: str, language: str = "en") -> List[str]:
    """
    Convert English text to a list of phoneme strings using espeak.

    Parameters
    ----------
    text : str
        Raw input text.
    language : str
        ISO language code (default 'en' → 'en-us' for espeak).

    Returns
    -------
    List of phoneme strings.
    """
    if EspeakBackend is None:
        raise ImportError("Install phonemizer: pip install phonemizer")

    lang_code = "en-us" if language in {"en", "en-us", "english"} else language
    text = _clean_text(text)
    if not text:
        return []

    sep = Separator(phone=" ", word="  ", syllable="")
    backend = EspeakBackend(
        language=lang_code,
        preserve_punctuation=False,
        with_stress=True,
        language_switch="remove-flags",
    )
    result = backend.phonemize([text], separator=sep)[0]
    phonemes = [p for p in result.strip().split(" ") if p]
    return phonemes


def phonemes_to_indices(
    phones: List[str],
    vocab: Dict[str, int],
) -> List[int]:
    unk = vocab.get("<unk>", 1)
    return [vocab.get(p, unk) for p in phones]


# ─────────────────────────────────────────────────────────────────────────────
# HiFi-GAN vocoder wrapper
# ─────────────────────────────────────────────────────────────────────────────

class HiFiGANVocoder:
    """
    Thin wrapper around a HiFi-GAN generator for mel → waveform conversion.

    The generator is loaded from a PyTorch checkpoint.  Its JSON config
    is read to determine upsample parameters.
    """

    def __init__(self, model_path: str, config_path: str, device: torch.device) -> None:
        self.device = device

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"HiFi-GAN model not found: {model_path}\n"
                "Download from https://github.com/jik876/hifi-gan"
            )
        if not Path(config_path).exists():
            raise FileNotFoundError(f"HiFi-GAN config not found: {config_path}")

        with open(config_path, "r") as f:
            self.hifi_config = json.load(f)

        # Dynamically import HiFi-GAN generator — assumed to be in PYTHONPATH
        try:
            from hifigan.models import Generator as HiFiGenerator  # type: ignore
        except ImportError:
            raise ImportError(
                "HiFi-GAN not found. Clone https://github.com/jik876/hifi-gan "
                "and add it to PYTHONPATH, or install with pip."
            )

        checkpoint = torch.load(model_path, map_location="cpu")
        generator_state = checkpoint.get("generator", checkpoint)

        self.generator = HiFiGenerator(self.hifi_config).to(device)
        self.generator.load_state_dict(generator_state)
        self.generator.eval()
        self.generator.remove_weight_norm()
        log.info(f"HiFi-GAN loaded from {model_path}")

    @torch.no_grad()
    def mel_to_wav(self, mel: torch.Tensor) -> np.ndarray:
        """
        Parameters
        ----------
        mel : torch.Tensor, shape (1, n_mel, T) or (n_mel, T)

        Returns
        -------
        wav : np.ndarray, shape (N,)
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)
        wav = self.generator(mel)
        wav = wav.squeeze().cpu().numpy()
        return wav


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    config: dict,
    vocab_size: int,
    device: torch.device,
) -> FastSpeech2:
    model = FastSpeech2(
        vocab_size=vocab_size,
        max_seq_len=config["max_seq_len"],
        encoder_hidden=config["encoder_hidden"],
        encoder_head=config["encoder_head"],
        encoder_layers=config["encoder_layers"],
        encoder_dropout=config["encoder_dropout"],
        encoder_kernel_size=config["encoder_kernel_size"],
        decoder_hidden=config["decoder_hidden"],
        decoder_head=config["decoder_head"],
        decoder_layers=config["decoder_layers"],
        decoder_dropout=config["decoder_dropout"],
        fft_conv1d_filter_size=config["fft_conv1d_filter_size"],
        fft_conv1d_kernel_size=config["fft_conv1d_kernel_size"],
        variance_predictor_filter_size=config["variance_predictor_filter_size"],
        variance_predictor_kernel_size=config["variance_predictor_kernel_size"],
        variance_predictor_dropout=config["variance_predictor_dropout"],
        n_mel_channels=config["n_mel_channels"],
        pitch_bins=config["pitch_bins"],
        energy_bins=config["energy_bins"],
        pitch_normalization=config["pitch_normalization"],
        energy_normalization=config["energy_normalization"],
    ).to(device)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    log.info(f"Model loaded from {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Variance plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_duration(
    phonemes: List[str],
    durations: np.ndarray,
    out_path: str,
    sentence_id: str,
) -> None:
    """Bar chart of predicted duration per phoneme."""
    fig, ax = plt.subplots(figsize=(max(8, len(phonemes) * 0.4), 4))
    x = np.arange(len(phonemes))
    ax.bar(x, durations, color="#4c72b0", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(phonemes, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Duration (frames)")
    ax.set_title(f"[{sentence_id}] Predicted Phoneme Durations")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  Duration plot → {out_path}")


def plot_pitch(
    pitch: np.ndarray,
    out_path: str,
    sentence_id: str,
) -> None:
    """Line plot of predicted F0 (pitch) over time."""
    fig, ax = plt.subplots(figsize=(10, 3))
    frames = np.arange(len(pitch))
    ax.plot(frames, pitch, linewidth=1.0, color="#dd8452")
    ax.set_xlabel("Frame")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(f"[{sentence_id}] Predicted Pitch (F0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  Pitch plot  → {out_path}")


def plot_energy(
    energy: np.ndarray,
    out_path: str,
    sentence_id: str,
) -> None:
    """Line plot of predicted energy over time."""
    fig, ax = plt.subplots(figsize=(10, 3))
    frames = np.arange(len(energy))
    ax.plot(frames, energy, linewidth=1.0, color="#55a868")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy")
    ax.set_title(f"[{sentence_id}] Predicted Energy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  Energy plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-sentence synthesis
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def synthesize(
    text: str,
    model: FastSpeech2,
    config: dict,
    vocoder: HiFiGANVocoder,
    output_dir: str,
    sentence_id: str,
    vocab: Dict[str, int],
) -> None:
    """
    Convert text to speech and save:
      - output_dir/wavs/{sentence_id}.wav
      - output_dir/duration/{sentence_id}_duration.png
      - output_dir/pitch/{sentence_id}_pitch.png
      - output_dir/energy/{sentence_id}_energy.png

    Parameters
    ----------
    text : str
        Input English text.
    model : FastSpeech2
        Trained model in eval() mode.
    config : dict
        Config dict.
    vocoder : HiFiGANVocoder
        HiFi-GAN vocoder wrapper.
    output_dir : str
        Root results directory.
    sentence_id : str
        Short identifier (e.g. 'E1', 'E2a').
    vocab : dict
        Phoneme → index mapping.
    """
    device = next(model.parameters()).device

    # ── 1. Text → phoneme indices ──────────────────────────────────────────
    log.info(f"[{sentence_id}] Text: {text!r}")
    phones = text_to_phonemes(text, language="en")
    log.info(f"[{sentence_id}] Phones ({len(phones)}): {' '.join(phones)}")

    if not phones:
        log.warning(f"[{sentence_id}] Empty phoneme sequence, skipping.")
        return

    phone_ids = phonemes_to_indices(phones, vocab)
    phoneme_tensor = torch.tensor(phone_ids, dtype=torch.long).unsqueeze(0).to(device)
    ph_mask = torch.zeros(1, len(phone_ids), dtype=torch.bool).to(device)

    # ── 2. Model forward (inference — no targets) ──────────────────────────
    output = model(
        phoneme_tensor,
        ph_mask,
        mel_target=None,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
    )

    # Expected keys from FastSpeech2 output dict:
    #   mel_pred      : (1, n_mel, T)
    #   duration_pred : (1, L)        — predicted durations in frames
    #   pitch_pred    : (1, T)        — frame-level predicted pitch
    #   energy_pred   : (1, T)        — frame-level predicted energy
    mel_pred = output["mel_pred"]          # (1, n_mel, T)
    dur_pred = output["duration_pred"]     # (1, L)
    pitch_pred = output["pitch_pred"]      # (1, T)
    energy_pred = output["energy_pred"]    # (1, T)

    # ── 3. Mel → waveform ─────────────────────────────────────────────────
    wav = vocoder.mel_to_wav(mel_pred)  # np.ndarray (N,)

    # ── 4. Save WAV ───────────────────────────────────────────────────────
    wav_dir = Path(output_dir) / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(wav_dir / f"{sentence_id}.wav")

    # Clip to avoid clipping artifacts
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(wav_path, wav, config["sampling_rate"], subtype="PCM_16")
    duration_secs = len(wav) / config["sampling_rate"]
    log.info(f"  WAV saved → {wav_path}  ({duration_secs:.2f}s)")

    # ── 5. Convert predictions to numpy ───────────────────────────────────
    dur_np = dur_pred.squeeze(0).cpu().numpy()        # (L,)
    pitch_np = pitch_pred.squeeze(0).cpu().numpy()    # (T,)
    energy_np = energy_pred.squeeze(0).cpu().numpy()  # (T,)

    # ── 6. Save variance adaptor plots ────────────────────────────────────
    dur_dir = Path(output_dir) / "duration"
    pitch_dir = Path(output_dir) / "pitch"
    energy_dir = Path(output_dir) / "energy"
    dur_dir.mkdir(parents=True, exist_ok=True)
    pitch_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    plot_duration(
        phones,
        dur_np,
        str(dur_dir / f"{sentence_id}_duration.png"),
        sentence_id,
    )
    plot_pitch(
        pitch_np,
        str(pitch_dir / f"{sentence_id}_pitch.png"),
        sentence_id,
    )
    plot_energy(
        energy_np,
        str(energy_dir / f"{sentence_id}_energy.png"),
        sentence_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastSpeech 2 English inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "Results" / "English"),
        help="Directory to save synthesised outputs",
    )
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Inference on: {device}")

    # ── Load vocabulary ────────────────────────────────────────────────────
    vocab = load_vocab(cfg["preprocessed_path"])
    vocab_size = len(vocab)

    # ── Load model ─────────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = load_model(args.checkpoint, cfg, vocab_size, device)

    # ── Load vocoder ───────────────────────────────────────────────────────
    vocoder = HiFiGANVocoder(
        model_path=cfg["vocoder_model_path"],
        config_path=cfg["vocoder_config_path"],
        device=device,
    )

    # ── Create output directory ────────────────────────────────────────────
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Saving results to: {out_dir}")

    # ── Run inference on all test sentences ────────────────────────────────
    for sid, text in TEST_SENTENCES.items():
        log.info("─" * 60)
        try:
            synthesize(
                text=text,
                model=model,
                config=cfg,
                vocoder=vocoder,
                output_dir=out_dir,
                sentence_id=sid,
                vocab=vocab,
            )
        except Exception as exc:
            log.error(f"[{sid}] Synthesis failed: {exc}", exc_info=True)

    log.info("=" * 60)
    log.info("Inference complete.")
    log.info(f"All outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
