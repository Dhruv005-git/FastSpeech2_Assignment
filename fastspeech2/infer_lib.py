import os
import sys

import numpy as np
import soundfile as sf
import torch

from .model import FastSpeech2
from .text import get_frontend, units_to_ids
from .utils import read_json, setup_font


# ------------------------------------------------------------------ vocoder
class Vocoder:
    """NVIDIA BigVGAN-v2 (22.05 kHz, 80 mel, hop 256): universal pretrained vocoder, whose mel front-end is
    the one used in fastspeech2/audio.py. Falls back to Griffin-Lim if BigVGAN cannot be loaded."""

    def __init__(self, cfg, dev):
        self.cfg, self.dev, self.net = cfg, dev, None
        name = cfg["inference"].get("vocoder", "nvidia/bigvgan_v2_22khz_80band_256x")
        if name != "griffin-lim":
            try:
                from huggingface_hub import snapshot_download
                path = snapshot_download(name)
                sys.path.insert(0, path)
                import bigvgan
                self.net = bigvgan.BigVGAN.from_pretrained(path, use_cuda_kernel=False)
                self.net.remove_weight_norm()
                self.net = self.net.eval().to(dev)
                print("[vocoder] BigVGAN loaded")
            except Exception as e:                                           # noqa
                print(f"[warn] BigVGAN unavailable ({e!r}); using Griffin-Lim (lower quality)")

    @torch.no_grad()
    def __call__(self, mel):                                                 # mel (T,80) log-mel
        if self.net is not None:
            return self.net(mel.T[None].to(self.dev))[0, 0].float().cpu().numpy()
        import librosa
        a = self.cfg["audio"]
        m = np.exp(mel.numpy().T)
        return librosa.feature.inverse.mel_to_audio(m, sr=a["sampling_rate"], n_fft=a["n_fft"], hop_length=a["hop_length"],
                                                    win_length=a["win_length"], fmin=a["fmin"], fmax=a["fmax"], power=1.0, n_iter=60)


# ------------------------------------------------------------------ model loading
def load_model(ckpt_path, dev):
    s = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = s["cfg"]
    model = FastSpeech2(cfg, len(s["symbols"]), len(s["speakers"]), s["stats"]).to(dev)
    model.load_state_dict(s["model"])
    return model.eval(), cfg, s


@torch.no_grad()
def synthesize(model, ckpt, text, cfg, dev, speaker=None, p_ctrl=1.0, e_ctrl=1.0, d_ctrl=1.0):
    symbols, to_units = get_frontend(cfg["language"])
    units = to_units(text)
    ids = torch.tensor([units_to_ids(units, symbols)], device=dev)
    kept_units = [symbols[i] for i in ids[0].tolist()]
    spk = ckpt["speakers"]
    spk_id = torch.tensor([spk.index(speaker) if speaker in spk else 0], device=dev)
    out = model(ids, torch.tensor([ids.size(1)], device=dev), spk_id, p_ctrl=p_ctrl, e_ctrl=e_ctrl, d_ctrl=d_ctrl)
    st = ckpt["stats"]
    return dict(units=kept_units, mel=out["post"][0].cpu(), dur=out["dur"][0].cpu().numpy(),
                pitch=(out["pitch"][0].cpu().numpy() * st["pitch"]["std"] + st["pitch"]["mean"]),
                energy=(out["energy"][0].cpu().numpy() * st["energy"]["std"] + st["energy"]["mean"]))


# ------------------------------------------------------------------ plots
def _label(u):
    return " " if u == "<sp>" else u


def plot_variance(r, sid, text, out_dir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hop, sr = cfg["audio"]["hop_length"], cfg["audio"]["sampling_rate"]
    units, dur = r["units"], r["dur"]
    n = len(units)
    x = np.arange(n)
    w = max(10, min(32, n * 0.28))
    ms = dur * hop / sr * 1000
    bounds = np.concatenate([[0], np.cumsum(dur)])
    centers = (bounds[:-1] + bounds[1:]) / 2
    title = f"{sid}: {text}"

    def style(ax, labels=True):
        ax.set_xticks(x); ax.set_xticklabels([_label(u) for u in units], rotation=90, fontsize=7)
        ax.grid(alpha=.3)

    # --- duration: predicted frames per token
    fig, ax = plt.subplots(figsize=(w, 4))
    ax.bar(x, dur, color=["#bbbbbb" if u == "<sp>" else "#3b78c4" for u in units])
    for i, v in enumerate(dur):
        ax.text(i, v + 0.1, str(int(v)), ha="center", fontsize=6)
    ax.set(ylabel=f"predicted duration (frames, 1 frame = {hop / sr * 1000:.1f} ms)", title="Duration - " + title)
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "duration", sid + ".png"), dpi=140); plt.close(fig)

    # --- pitch / energy: phoneme-level prediction (step curve over time) with token labels
    for key, ylabel, folder, col in (("pitch", "predicted F0 (Hz)", "pitch", "#d1495b"), ("energy", "predicted energy (L2 of STFT magnitude)", "energy", "#2e8b57")):
        v = r[key]
        fig, ax = plt.subplots(2, 1, figsize=(w, 6.5), gridspec_kw=dict(height_ratios=[1, 1.3]))
        ax[0].bar(x, v, color=col); ax[0].set(ylabel=ylabel, title=f"{key.title()} - " + title); style(ax[0])
        ax[1].imshow(r["mel"].T.numpy(), origin="lower", aspect="auto", cmap="magma", extent=[0, bounds[-1], 0, 80])
        ax[1].step(np.concatenate([bounds[:-1], [bounds[-1]]]), np.concatenate([v, [v[-1]]]) / max(v.max(), 1e-6) * 75, where="post", color="w", lw=1.4)
        ax[1].set_xticks(centers); ax[1].set_xticklabels([_label(u) for u in units], rotation=90, fontsize=7)
        ax[1].set(ylabel="mel bin (white = scaled prediction)", xlabel="token (placed at its predicted time span)")
        for b in bounds:
            ax[1].axvline(b, color="w", lw=.25, alpha=.5)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, folder, sid + ".png"), dpi=140); plt.close(fig)


def run_inference(cfg, ckpt_path, tests, out_dir=None, speaker=None, p=1.0, e=1.0, d=1.0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    setup_font(cfg["language"])
    model, cfg, ck = load_model(ckpt_path, dev)
    setup_font(cfg["language"])
    voc = Vocoder(cfg, dev)
    out_dir = out_dir or cfg["path"]["results_path"]
    speaker = speaker or cfg["inference"].get("speaker")
    wav_dir = os.path.join(out_dir, "test_generated_audios")
    va = os.path.join(out_dir, "variance_adaptor_plots")
    for dname in (wav_dir, *[os.path.join(va, k) for k in ("duration", "pitch", "energy")]):
        os.makedirs(dname, exist_ok=True)
    for sid, text in tests:
        r = synthesize(model, ck, text, cfg, dev, speaker, p, e, d)
        wav = voc(r["mel"])
        sf.write(os.path.join(wav_dir, sid + ".wav"), wav, cfg["audio"]["sampling_rate"])
        plot_variance(r, sid, text, va, cfg)
        np.savez(os.path.join(va, f"{sid}_predictions.npz"), units=np.array(r["units"]), duration=r["dur"], pitch=r["pitch"], energy=r["energy"])
        print(f"[infer] {sid}: {len(r['units'])} tokens, {len(r['mel'])} frames = {len(wav) / cfg['audio']['sampling_rate']:.2f}s")
