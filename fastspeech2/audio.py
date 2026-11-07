"""Audio front-end. The mel-spectrogram is computed *exactly* as in NVIDIA BigVGAN
(`bigvgan_v2_22khz_80band_256x`) so that the pretrained universal vocoder can be used
directly on the FastSpeech 2 output (no vocoder training needed)."""
import numpy as np
import torch
import librosa

_mel_basis, _hann = {}, {}


def mel_spectrogram(wav, cfg):
    """wav: 1-D float tensor in [-1, 1]. Returns (n_mels, T) log-mel and (T,) frame energy."""
    a = cfg["audio"]
    n_fft, hop, win = a["n_fft"], a["hop_length"], a["win_length"]
    fmax = a.get("fmax")
    key = (a["sampling_rate"], n_fft, a["n_mels"], a["fmin"], fmax, str(wav.device))
    if key not in _mel_basis:
        m = librosa.filters.mel(sr=a["sampling_rate"], n_fft=n_fft, n_mels=a["n_mels"],
                                fmin=a["fmin"], fmax=fmax)
        _mel_basis[key] = torch.from_numpy(m).float().to(wav.device)
        _hann[key] = torch.hann_window(win).to(wav.device)
    y = torch.nn.functional.pad(wav[None, None], ((n_fft - hop) // 2, (n_fft - hop) // 2), mode="reflect")[:, 0]
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_hann[key], center=False,
                      pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)          # (1, F, T)
    mel = torch.log(torch.clamp(torch.matmul(_mel_basis[key], mag[0]), min=1e-5))
    energy = torch.norm(mag[0], dim=0)                                      # L2 norm per frame (as in FS2)
    return mel, energy


def load_wav(path, sr):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def trim_silence(wav, sr, top_db=40):
    _, idx = librosa.effects.trim(wav, top_db=top_db, frame_length=1024, hop_length=256)
    pad = int(0.05 * sr)
    return wav[max(idx[0] - pad, 0): idx[1] + pad]


def extract_f0(wav, cfg):
    """Frame-level F0 (0 = unvoiced). Uses WORLD (pyworld) like FS2; falls back to librosa.pyin."""
    a = cfg["audio"]
    sr, hop = a["sampling_rate"], a["hop_length"]
    try:
        import pyworld as pw
        x = wav.astype(np.float64)
        f0, t = pw.dio(x, sr, frame_period=hop / sr * 1000, f0_floor=a.get("f0_min", 60), f0_ceil=a.get("f0_max", 600))
        f0 = pw.stonemask(x, f0, t, sr)
    except ImportError:
        f0, _, _ = librosa.pyin(wav, fmin=a.get("f0_min", 60), fmax=a.get("f0_max", 600), sr=sr,
                                frame_length=1024, hop_length=hop, center=False)
        f0 = np.nan_to_num(f0)
    return f0.astype(np.float32)
