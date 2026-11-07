"""
utils/audio.py
==============
Audio utility functions shared across all language pipelines.

Dependencies:
    pip install librosa soundfile numpy scipy
"""

import os
import numpy as np
import librosa
import librosa.filters
import soundfile as sf
from scipy.signal import get_window


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 22050) -> np.ndarray:
    """Load an audio file and resample to *sr* Hz.

    Args:
        path: Absolute or relative path to .wav / .flac / .mp3 file.
        sr:   Target sampling rate.  Defaults to 22050 Hz.

    Returns:
        1-D float32 numpy array in the range [-1, 1].
    """
    wav, orig_sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav


def save_audio(wav: np.ndarray, path: str, sr: int = 22050) -> None:
    """Write a float32 waveform to disk as a 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wav_int16 = np.clip(wav, -1.0, 1.0)
    wav_int16 = (wav_int16 * 32767).astype(np.int16)
    sf.write(path, wav_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_audio(wav: np.ndarray) -> np.ndarray:
    """Peak-normalise a waveform to [-1, 1].

    Returns the input unchanged if the peak is effectively zero (silence).
    """
    peak = np.abs(wav).max()
    if peak < 1e-8:
        return wav
    return wav / peak


# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------

def _build_hann_window(win_length: int, device_type: str = "numpy") -> np.ndarray:
    return get_window("hann", win_length, fftbins=True).astype(np.float32)


def _stft(wav: np.ndarray,
          filter_length: int,
          hop_length: int,
          win_length: int) -> np.ndarray:
    """Compute complex STFT using librosa."""
    return librosa.stft(
        wav,
        n_fft=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

def compute_mel_spectrogram(wav: np.ndarray, config: dict) -> np.ndarray:
    """Compute an 80-band log-mel spectrogram from a waveform.

    Args:
        wav:    1-D float32 waveform (already at config["sampling_rate"]).
        config: Dictionary with keys: sampling_rate, filter_length,
                hop_length, win_length, n_mel_channels, mel_fmin, mel_fmax.

    Returns:
        (n_mels, T) float32 numpy array — log-scale mel spectrogram.
    """
    sr           = config["sampling_rate"]
    n_fft        = config["filter_length"]
    hop_length   = config["hop_length"]
    win_length   = config["win_length"]
    n_mels       = config["n_mel_channels"]
    fmin         = config.get("mel_fmin", 0.0)
    fmax         = config.get("mel_fmax", 8000.0)

    stft_mag = np.abs(_stft(wav, n_fft, hop_length, win_length))

    mel_fb = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )  # shape: (n_mels, 1 + n_fft // 2)

    mel = np.dot(mel_fb, stft_mag)                    # (n_mels, T)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))  # log compression
    return mel.astype(np.float32)


# ---------------------------------------------------------------------------
# Mel → audio (Griffin-Lim fallback)
# ---------------------------------------------------------------------------

def mel_to_audio(mel: np.ndarray, config: dict, n_iter: int = 60) -> np.ndarray:
    """Reconstruct a waveform from a log-mel spectrogram via Griffin-Lim.

    This is a CPU fallback; for high-quality synthesis use HiFi-GAN.

    Args:
        mel:    (n_mels, T) log-mel spectrogram (output of compute_mel_spectrogram).
        config: Same config dict as compute_mel_spectrogram.
        n_iter: Number of Griffin-Lim iterations (default 60).

    Returns:
        1-D float32 waveform.
    """
    sr         = config["sampling_rate"]
    n_fft      = config["filter_length"]
    hop_length = config["hop_length"]
    win_length = config["win_length"]
    n_mels     = config["n_mel_channels"]
    fmin       = config.get("mel_fmin", 0.0)
    fmax       = config.get("mel_fmax", 8000.0)

    # Inverse log
    mel_lin = np.exp(mel)                              # (n_mels, T)

    mel_fb = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    # Pseudo-inverse of mel filterbank
    mel_fb_inv = np.linalg.pinv(mel_fb)               # (n_fft//2+1, n_mels)
    stft_mag = np.maximum(np.dot(mel_fb_inv, mel_lin), 0.0)

    wav = librosa.griffinlim(
        stft_mag,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
    )
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def get_duration_seconds(path: str) -> float:
    """Return the duration of an audio file in seconds without full decoding."""
    info = sf.info(path)
    return info.duration


def trim_silence(wav: np.ndarray,
                 top_db: float = 35.0,
                 frame_length: int = 1024,
                 hop_length: int = 256) -> np.ndarray:
    """Trim leading / trailing silence using librosa energy criterion."""
    wav_trimmed, _ = librosa.effects.trim(
        wav,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return wav_trimmed


def mu_law_encode(wav: np.ndarray, mu: int = 255) -> np.ndarray:
    """μ-law compand a waveform (useful for WaveNet-style vocoders)."""
    wav = np.clip(wav, -1.0, 1.0)
    return np.sign(wav) * np.log1p(mu * np.abs(wav)) / np.log1p(mu)


def mu_law_decode(wav_encoded: np.ndarray, mu: int = 255) -> np.ndarray:
    """Inverse μ-law companding."""
    return np.sign(wav_encoded) * (np.expm1(np.abs(wav_encoded) * np.log1p(mu)) / mu)
