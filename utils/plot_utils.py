"""
utils/plot_utils.py
===================
Matplotlib-based visualisation helpers for FastSpeech 2 diagnostics.

All functions accept an optional *save_path*; when provided the figure is
saved to disk and closed.  When omitted the figure is returned to the caller
(useful inside Jupyter notebooks).

Dependencies:
    pip install matplotlib numpy
"""

import os
from typing import List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_return(fig: plt.Figure, save_path: Optional[str]) -> Optional[plt.Figure]:
    """Either save the figure to disk or return it."""
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

def plot_mel(
    mel: np.ndarray,
    title: str = "Mel Spectrogram",
    save_path: Optional[str] = None,
    sr: int = 22050,
    hop_length: int = 256,
) -> Optional[plt.Figure]:
    """Plot a log-mel spectrogram as a heat-map.

    Args:
        mel:         (n_mels, T) log-mel spectrogram numpy array.
        title:       Figure title string.
        save_path:   If set, the figure is saved here (PNG/PDF/SVG).
        sr:          Sampling rate — used to build time axis.
        hop_length:  Hop length — used to build time axis.

    Returns:
        matplotlib Figure if *save_path* is None, else None.
    """
    n_mels, T = mel.shape
    times = np.arange(T) * hop_length / sr          # seconds
    freqs = np.linspace(0, sr / 2 / 1000, n_mels)  # kHz (approx)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        interpolation="none",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="Log Magnitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

def plot_duration(
    durations: Sequence[int],
    phonemes: Sequence[str],
    title: str = "Phoneme Durations",
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Bar chart of frame-level phoneme durations.

    Args:
        durations: Sequence of integer frame counts, one per phoneme.
        phonemes:  Corresponding phoneme label strings.
        title:     Figure title.
        save_path: Optional output file path.
    """
    durations = list(durations)
    phonemes  = list(phonemes)
    assert len(durations) == len(phonemes), (
        f"Length mismatch: {len(durations)} durations vs {len(phonemes)} phonemes"
    )

    x = np.arange(len(phonemes))
    fig, ax = plt.subplots(figsize=(max(10, len(phonemes) * 0.5), 4))
    bars = ax.bar(x, durations, color="steelblue", edgecolor="navy", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(phonemes, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Duration (frames)")
    ax.set_title(title)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Pitch (F0)
# ---------------------------------------------------------------------------

def plot_pitch(
    pitch: np.ndarray,
    title: str = "F0 Contour",
    save_path: Optional[str] = None,
    sr: int = 22050,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 600.0,
) -> Optional[plt.Figure]:
    """Line plot of a continuous F0 (pitch) contour.

    Unvoiced frames (value == 0) are shown as gaps.

    Args:
        pitch:      1-D numpy array of F0 values in Hz (0 = unvoiced).
        title:      Figure title.
        save_path:  Optional output file path.
        sr:         Sampling rate.
        hop_length: Frame hop in samples.
        fmin/fmax:  Y-axis limits.
    """
    times = np.arange(len(pitch)) * hop_length / sr

    # Mask unvoiced frames
    voiced = np.where(pitch > 0, pitch, np.nan)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, voiced, color="crimson", linewidth=1.0, label="F0 (voiced)")
    ax.scatter(
        times[pitch == 0], np.full((pitch == 0).sum(), fmin * 0.9),
        color="grey", s=5, alpha=0.4, label="Unvoiced",
    )
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(fmin * 0.8, fmax * 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def plot_energy(
    energy: np.ndarray,
    title: str = "Frame Energy",
    save_path: Optional[str] = None,
    sr: int = 22050,
    hop_length: int = 256,
) -> Optional[plt.Figure]:
    """Line plot of frame-level energy (L2 norm of mel columns).

    Args:
        energy:     1-D numpy array of energy values.
        title:      Figure title.
        save_path:  Optional output file path.
    """
    times = np.arange(len(energy)) * hop_length / sr

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, energy, color="darkorange", linewidth=1.0)
    ax.fill_between(times, energy, alpha=0.25, color="darkorange")
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Training / validation loss curves
# ---------------------------------------------------------------------------

def plot_training_loss(
    losses: Sequence[float],
    save_path: Optional[str] = None,
    title: str = "Training Loss",
    xlabel: str = "Step",
    log_scale: bool = False,
) -> Optional[plt.Figure]:
    """Line plot of training loss over steps or epochs.

    Args:
        losses:     Iterable of scalar loss values recorded at each log step.
        save_path:  Optional output file path.
        title:      Figure title.
        xlabel:     Label for x-axis ('Step' or 'Epoch').
        log_scale:  If True, use a logarithmic y-axis.
    """
    losses = list(losses)
    steps  = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, color="royalblue", linewidth=1.2, label="Train Loss")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_validation_loss(
    val_losses: Sequence[float],
    train_losses: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Validation Loss",
    xlabel: str = "Checkpoint",
    log_scale: bool = False,
) -> Optional[plt.Figure]:
    """Line plot of validation loss, optionally overlaid with training loss.

    Args:
        val_losses:   Validation loss values.
        train_losses: Optional training loss values at the same checkpoints.
        save_path:    Optional output file path.
        title:        Figure title.
        xlabel:       Label for x-axis.
        log_scale:    If True, use a logarithmic y-axis.
    """
    val_losses = list(val_losses)
    steps = np.arange(1, len(val_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, val_losses, color="tomato", linewidth=1.2, label="Val Loss")
    if train_losses is not None:
        t_losses = list(train_losses)
        t_steps  = np.linspace(1, len(val_losses), len(t_losses))
        ax.plot(t_steps, t_losses, color="royalblue", linewidth=0.8,
                alpha=0.7, label="Train Loss")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Combined synthesis figure (mel + pitch + energy + duration)
# ---------------------------------------------------------------------------

def plot_synthesis_summary(
    mel: np.ndarray,
    pitch: np.ndarray,
    energy: np.ndarray,
    durations: Optional[Sequence[int]] = None,
    phonemes: Optional[Sequence[str]] = None,
    title: str = "Synthesis Summary",
    save_path: Optional[str] = None,
    sr: int = 22050,
    hop_length: int = 256,
) -> Optional[plt.Figure]:
    """4-panel summary figure: mel / pitch / energy / [optional] duration."""
    n_panels = 4 if (durations is not None and phonemes is not None) else 3
    times_mel = np.arange(mel.shape[1]) * hop_length / sr
    times_f0  = np.arange(len(pitch))  * hop_length / sr
    times_e   = np.arange(len(energy)) * hop_length / sr
    n_mels    = mel.shape[0]
    freqs_khz = np.linspace(0, sr / 2 / 1000, n_mels)

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(12, 3 * n_panels),
                             sharex=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # -- Mel
    ax = axes[0]
    ax.imshow(mel, aspect="auto", origin="lower",
              extent=[times_mel[0], times_mel[-1], freqs_khz[0], freqs_khz[-1]],
              cmap="viridis", interpolation="none")
    ax.set_ylabel("Freq (kHz)")
    ax.set_title("Mel Spectrogram")

    # -- Pitch
    ax = axes[1]
    voiced = np.where(pitch > 0, pitch, np.nan)
    ax.plot(times_f0, voiced, color="crimson", linewidth=1.0)
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("Pitch Contour")
    ax.grid(linestyle="--", alpha=0.4)

    # -- Energy
    ax = axes[2]
    ax.plot(times_e, energy, color="darkorange", linewidth=1.0)
    ax.fill_between(times_e, energy, alpha=0.2, color="darkorange")
    ax.set_ylabel("Energy")
    ax.set_title("Frame Energy")
    ax.set_xlabel("Time (s)")
    ax.grid(linestyle="--", alpha=0.4)

    # -- Duration (optional)
    if n_panels == 4:
        ax = axes[3]
        x = np.arange(len(durations))
        ax.bar(x, durations, color="steelblue", edgecolor="navy", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(phonemes, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Frames")
        ax.set_title("Phoneme Durations")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlabel("Phoneme")

    fig.tight_layout()
    return _save_or_return(fig, save_path)
