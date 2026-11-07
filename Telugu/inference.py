"""
Telugu Inference Script — FastSpeech 2 TTS
Generates speech for Telugu test sentences and saves variance adaptor plots.

Usage:
    python Telugu/inference.py \
        --checkpoint Telugu/checkpoints/best_checkpoint.pth \
        --config     Telugu/config.yaml \
        --output_dir Results/Telugu \
        --vocoder_path /path/to/hifi-gan
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

# Allow imports from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import FastSpeech2
from utils.text_telugu import telugu_cleaners, build_telugu_vocab, load_vocab

# ---------------------------------------------------------------------------
# Telugu test sentences
# ---------------------------------------------------------------------------
TEST_SENTENCES = [
    ("T1",  "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."),
    ("T2a", "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."),
    ("T2b", "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."),
    ("T3",  "పులి అడవిలో నిశ్శబ్దంగా నడిచింది."),
    ("T4",  "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."),
]


# ---------------------------------------------------------------------------
# Text → phoneme indices
# ---------------------------------------------------------------------------

def telugu_text_to_indices(text: str, vocab: dict) -> torch.LongTensor:
    """
    Clean Telugu text, split into phoneme tokens, convert to index tensor.

    Args:
        text:  Raw Telugu unicode string.
        vocab: Dict mapping token → int index (built during preprocessing).

    Returns:
        1-D LongTensor of phoneme indices.
    """
    cleaned = telugu_cleaners(text)

    # Try epitran IPA, fall back to character-level
    try:
        import epitran
        epi = epitran.Epitran("tel-Telu")
        phonemes = list(epi.transliterate(cleaned))
    except Exception:
        # Character-level fallback: each Unicode code point is a token
        phonemes = list(cleaned)

    unk_id = vocab.get("<UNK>", 1)
    ids = [vocab.get(ph, unk_id) for ph in phonemes]

    if not ids:
        raise ValueError(f"Empty phoneme sequence for text: {text!r}")

    return torch.LongTensor(ids)


# ---------------------------------------------------------------------------
# HiFi-GAN vocoder wrapper
# ---------------------------------------------------------------------------

class HiFiGANVocoder:
    """Thin wrapper around a HiFi-GAN Generator checkpoint."""

    def __init__(self, vocoder_dir: str, device: torch.device):
        self.device = device

        # Locate config and checkpoint
        cfg_path = os.path.join(vocoder_dir, "config.json")
        ckpt_candidates = ["generator_v2", "generator_v1", "generator_v3", "generator"]
        ckpt_path = None
        for name in ckpt_candidates:
            p = os.path.join(vocoder_dir, name)
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None or not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"HiFi-GAN checkpoint or config not found in {vocoder_dir}. "
                "Clone https://github.com/jik876/hifi-gan and download a pretrained checkpoint."
            )

        # Load HiFi-GAN dynamically from its repo directory
        sys.path.insert(0, vocoder_dir)
        from models import Generator  # type: ignore  (from hifi-gan repo)

        with open(cfg_path) as f:
            h = json.load(f)

        class AttrDict(dict):
            def __getattr__(self, key):
                return self[key]

        h = AttrDict(h)
        self.generator = Generator(h).to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.generator.load_state_dict(state["generator"])
        self.generator.eval()
        self.generator.remove_weight_norm()
        print(f"[HiFiGAN] Loaded checkpoint from {ckpt_path}")

    @torch.no_grad()
    def mel_to_wav(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convert a mel spectrogram (shape: 1 × n_mel × T) to a waveform.

        Returns numpy array (float32).
        """
        mel = mel.to(self.device)
        wav = self.generator(mel)
        return wav.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Variance-adaptor plots
# ---------------------------------------------------------------------------

def _save_duration_plot(durations: np.ndarray, phonemes, title: str, path: str):
    """Bar chart of predicted phoneme durations."""
    fig, ax = plt.subplots(figsize=(max(8, len(durations) * 0.4), 4))
    x = np.arange(len(durations))
    ax.bar(x, durations, color="steelblue", edgecolor="navy", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(phonemes if phonemes else [""] * len(durations),
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Duration (frames)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_pitch_plot(pitch: np.ndarray, title: str, path: str):
    """Line plot of predicted F0 contour."""
    fig, ax = plt.subplots(figsize=(10, 4))
    frames = np.arange(len(pitch))
    ax.plot(frames, pitch, color="crimson", linewidth=1.2)
    ax.fill_between(frames, pitch, alpha=0.15, color="crimson")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pitch (normalized)")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_energy_plot(energy: np.ndarray, title: str, path: str):
    """Line plot of predicted energy."""
    fig, ax = plt.subplots(figsize=(10, 4))
    frames = np.arange(len(energy))
    ax.plot(frames, energy, color="darkorange", linewidth=1.2)
    ax.fill_between(frames, energy, alpha=0.15, color="darkorange")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy (normalized)")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-sentence synthesis
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize(
    text: str,
    sentence_id: str,
    model: FastSpeech2,
    vocab: dict,
    config: dict,
    vocoder: HiFiGANVocoder,
    output_dir: str,
    device: torch.device,
):
    """
    Synthesise one Telugu sentence and save audio + variance adaptor plots.

    Outputs:
        <output_dir>/test_generated_audios/<sentence_id>.wav
        <output_dir>/variance_adaptor_plots/duration/<sentence_id>_duration.png
        <output_dir>/variance_adaptor_plots/pitch/<sentence_id>_pitch.png
        <output_dir>/variance_adaptor_plots/energy/<sentence_id>_energy.png
    """
    # Directories
    audio_dir = os.path.join(output_dir, "test_generated_audios")
    dur_dir   = os.path.join(output_dir, "variance_adaptor_plots", "duration")
    pit_dir   = os.path.join(output_dir, "variance_adaptor_plots", "pitch")
    eng_dir   = os.path.join(output_dir, "variance_adaptor_plots", "energy")
    for d in (audio_dir, dur_dir, pit_dir, eng_dir):
        os.makedirs(d, exist_ok=True)

    # Text → indices
    src_seq = telugu_text_to_indices(text, vocab).unsqueeze(0).to(device)   # (1, L)
    src_len = torch.LongTensor([src_seq.shape[1]]).to(device)               # (1,)

    # Build reverse vocab for display labels
    id2tok = {v: k for k, v in vocab.items()}

    model.eval()
    (
        mel_output,
        mel_postnet,
        p_pred,
        e_pred,
        log_d_pred,
        duration_rounded,
        src_mask,
        mel_mask,
        mel_len,
    ) = model(src_seq, src_len)

    # ---- Audio ----
    # mel_postnet: (1, T_mel, n_mel) → (1, n_mel, T_mel)
    mel_for_vocoder = mel_postnet.transpose(1, 2)
    wav = vocoder.mel_to_wav(mel_for_vocoder)

    sr = config.get("sampling_rate", 22050)
    wav_path = os.path.join(audio_dir, f"{sentence_id}.wav")
    sf.write(wav_path, wav.astype(np.float32), sr)
    print(f"  [Audio] Saved → {wav_path}")

    # ---- Variance adaptor data ----
    phoneme_ids = src_seq.squeeze(0).cpu().numpy()
    phoneme_tokens = [id2tok.get(int(i), "?") for i in phoneme_ids]

    durations = duration_rounded.squeeze(0).cpu().numpy()          # (L,)
    pitch     = p_pred.squeeze(0).cpu().numpy()                    # (L,)
    energy    = e_pred.squeeze(0).cpu().numpy()                    # (L,)

    # Duration plot
    dur_path = os.path.join(dur_dir, f"{sentence_id}_duration.png")
    _save_duration_plot(durations, phoneme_tokens,
                        f"Duration — {sentence_id}", dur_path)
    print(f"  [Plot]  Duration → {dur_path}")

    # Pitch plot  (expand to mel-frame level for readability)
    pitch_expanded = np.repeat(pitch, np.maximum(durations.astype(int), 1))
    pit_path = os.path.join(pit_dir, f"{sentence_id}_pitch.png")
    _save_pitch_plot(pitch_expanded, f"Pitch (F0) — {sentence_id}", pit_path)
    print(f"  [Plot]  Pitch   → {pit_path}")

    # Energy plot
    energy_expanded = np.repeat(energy, np.maximum(durations.astype(int), 1))
    eng_path = os.path.join(eng_dir, f"{sentence_id}_energy.png")
    _save_energy_plot(energy_expanded, f"Energy — {sentence_id}", eng_path)
    print(f"  [Plot]  Energy  → {eng_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FastSpeech 2 Telugu Inference")
    parser.add_argument("--checkpoint",    required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--config",        required=True, help="Path to Telugu config.yaml")
    parser.add_argument("--output_dir",    required=True, help="Directory to save audio and plots")
    parser.add_argument("--vocoder_path",  required=True, help="Directory containing HiFi-GAN generator + config.json")
    parser.add_argument("--vocab_path",    default=None,  help="Path to vocab.json (default: same dir as checkpoint)")
    parser.add_argument("--p_control",     type=float, default=1.0, help="Pitch control multiplier")
    parser.add_argument("--e_control",     type=float, default=1.0, help="Energy control multiplier")
    parser.add_argument("--d_control",     type=float, default=1.0, help="Duration control multiplier")
    args = parser.parse_args()

    # ---- Config ----
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---- Vocabulary ----
    vocab_path = args.vocab_path
    if vocab_path is None:
        ckpt_dir   = os.path.dirname(os.path.abspath(args.checkpoint))
        vocab_path = os.path.join(ckpt_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        # Try preprocessed_path from config
        pre_path   = config.get("preprocessed_path", "data/preprocessed/telugu")
        vocab_path = os.path.join(pre_path, "vocab.json")

    if os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
        print(f"[Vocab]  Loaded {len(vocab)} tokens from {vocab_path}")
    else:
        raise FileNotFoundError(
            f"vocab.json not found. Expected at {vocab_path}. "
            "Run Telugu/preprocess.py first to generate it."
        )

    # ---- Model ----
    model = FastSpeech2(config).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model]  Loaded checkpoint — {total_params:.1f}M parameters")

    # ---- Vocoder ----
    vocoder = HiFiGANVocoder(args.vocoder_path, device)

    # ---- Inference loop ----
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[Inference] Output → {args.output_dir}\n")

    for sentence_id, text in TEST_SENTENCES:
        print(f"{'='*60}")
        print(f"[{sentence_id}] {text}")
        try:
            synthesize(
                text=text,
                sentence_id=sentence_id,
                model=model,
                vocab=vocab,
                config=config,
                vocoder=vocoder,
                output_dir=args.output_dir,
                device=device,
            )
        except Exception as exc:
            print(f"  [ERROR] {sentence_id} failed: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("[Done] Telugu inference complete.")
    print(f"  Audio  → {os.path.join(args.output_dir, 'test_generated_audios')}")
    print(f"  Plots  → {os.path.join(args.output_dir, 'variance_adaptor_plots')}")


if __name__ == "__main__":
    main()

