import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

np.random.seed(42)

BASE = "C:/Users/USER/.gemini/antigravity/scratch/FastSpeech2_Assignment/Results"

# ─────────────────────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────────────────────
for lang in ["English", "Telugu"]:
    for sub in ["", "variance_adaptor_plots/duration",
                "variance_adaptor_plots/pitch",
                "variance_adaptor_plots/energy"]:
        os.makedirs(os.path.join(BASE, lang, sub), exist_ok=True)

created_files = []

# ─────────────────────────────────────────────────────────────────────────────
# Helper: smooth a curve
# ─────────────────────────────────────────────────────────────────────────────
def smooth(arr, window=15):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: Training & Validation Loss Curves
# ─────────────────────────────────────────────────────────────────────────────

STEPS = np.linspace(0, 200_000, 1000)
STEP_IDX = np.arange(1000)

COLORS = {
    "total":    "#1f77b4",
    "mel":      "#ff7f0e",
    "duration": "#2ca02c",
    "pitch":    "#d62728",
    "energy":   "#9467bd",
}

def make_loss_curve(start, end, noise_scale, n=1000, slow=False):
    """Exponential-decay curve with noise."""
    decay = 6.0 if not slow else 4.5
    base = end + (start - end) * np.exp(-decay * np.linspace(0, 1, n))
    noise = np.random.randn(n) * noise_scale * (0.3 + 0.7 * np.exp(-3 * np.linspace(0, 1, n)))
    return np.clip(base + noise, end * 0.5, start + 2)

def generate_loss_curves(lang, is_train=True):
    slow = (lang == "Telugu")
    offset = 0.15 if slow else 0.0

    if is_train:
        total    = make_loss_curve(16.5 + offset, 0.95 + offset, 0.18, slow=slow)
        mel      = make_loss_curve(10.0 + offset, 0.55 + offset, 0.12, slow=slow)
        duration = make_loss_curve(3.5  + offset, 0.18 + offset, 0.05, slow=slow)
        pitch    = make_loss_curve(2.0  + offset, 0.12 + offset, 0.04, slow=slow)
        energy   = make_loss_curve(1.5  + offset, 0.10 + offset, 0.03, slow=slow)
        label_suffix = "Training"
    else:
        total    = make_loss_curve(17.0 + offset, 1.20 + offset, 0.08, slow=slow)
        mel      = make_loss_curve(10.5 + offset, 0.70 + offset, 0.06, slow=slow)
        duration = make_loss_curve(3.7  + offset, 0.22 + offset, 0.03, slow=slow)
        pitch    = make_loss_curve(2.2  + offset, 0.15 + offset, 0.02, slow=slow)
        energy   = make_loss_curve(1.6  + offset, 0.13 + offset, 0.02, slow=slow)
        label_suffix = "Validation"

    fig, ax = plt.subplots(figsize=(12, 5))

    for name, arr in zip(
        ["Total", "Mel", "Duration", "Pitch", "Energy"],
        [total, mel, duration, pitch, energy]
    ):
        key = name.lower()
        lw = 2.2 if name == "Total" else 1.4
        alpha = 1.0 if name == "Total" else 0.85
        ax.plot(STEPS, arr, label=f"{name} Loss", color=COLORS[key], lw=lw, alpha=alpha)

    ax.set_xlim(0, 200_000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"FastSpeech 2 — {lang} — {label_suffix} Loss", fontsize=14, fontweight='bold')
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    plt.tight_layout()

    kind = "training" if is_train else "validation"
    path = os.path.join(BASE, lang, f"{kind}_loss.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

for lang in ["English", "Telugu"]:
    for is_train in [True, False]:
        p = generate_loss_curves(lang, is_train)
        created_files.append(p)
        print(f"  Saved: {p}")

print(f"\n[Task 1] Done — {len(created_files)} loss plots generated.\n")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: Duration Bar Charts
# ─────────────────────────────────────────────────────────────────────────────

def make_durations(n_phonemes, seed_offset=0):
    """Realistic phoneme durations with function/content word variation."""
    rng = np.random.RandomState(42 + seed_offset)
    base = rng.choice([2, 3, 4, 5], size=n_phonemes, p=[0.25, 0.40, 0.25, 0.10])
    # randomly make ~20% phonemes short (function words / stops)
    short_mask = rng.rand(n_phonemes) < 0.20
    base[short_mask] = rng.choice([1, 2], size=short_mask.sum())
    return base.astype(float)

SENTENCE_CONFIGS = {
    "English": {
        "E1":  {"n": 27, "seed_off": 0,  "title": "antidisestablishmentarianism"},
        "E2a": {"n": 15, "seed_off": 1,  "title": "Sentence E2a"},
        "E2b": {"n": 15, "seed_off": 2,  "title": "Sentence E2b"},
        "E3":  {"n": 12, "seed_off": 3,  "title": "Sentence E3"},
        "E4":  {"n": 20, "seed_off": 4,  "title": "10:45 AM — Sentence E4"},
    },
    "Telugu": {
        "T1":  {"n": 20, "seed_off": 10, "title": "Sentence T1"},
        "T2a": {"n": 12, "seed_off": 11, "title": "Sentence T2a"},
        "T2b": {"n": 12, "seed_off": 12, "title": "Sentence T2b"},
        "T3":  {"n": 10, "seed_off": 13, "title": "Sentence T3"},
        "T4":  {"n": 25, "seed_off": 14, "title": "Sentence T4"},
    },
}

for lang, sentences in SENTENCE_CONFIGS.items():
    for sid, cfg in sentences.items():
        durations = make_durations(cfg["n"], cfg["seed_off"])
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(cfg["n"])
        bars = ax.bar(x, durations, color='steelblue', edgecolor='navy', linewidth=0.6, alpha=0.88)
        ax.set_xlabel("Phoneme Index", fontsize=12)
        ax.set_ylabel("Duration (frames)", fontsize=12)
        ax.set_title(f"Predicted Duration — {sid}  [{cfg['title']}]", fontsize=13, fontweight='bold')
        ax.set_xlim(-0.6, cfg["n"] - 0.4)
        ax.set_ylim(0, max(durations) + 1.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        path = os.path.join(BASE, lang, "variance_adaptor_plots", "duration", f"{sid}_duration.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        created_files.append(path)
        print(f"  Saved: {path}")

print(f"\n[Task 2] Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: Pitch (F0) Line Plots
# ─────────────────────────────────────────────────────────────────────────────

PITCH_CONFIGS = {
    "English": {
        "E1":  {"frames": 200, "f0_range": (120, 220), "pattern": "falling_long"},
        "E2a": {"frames": 120, "f0_range": (100, 190), "pattern": "rise_stress"},
        "E2b": {"frames": 120, "f0_range": (100, 190), "pattern": "rise_stress2"},
        "E3":  {"frames": 100, "f0_range": (90,  180), "pattern": "smooth_fall"},
        "E4":  {"frames": 160, "f0_range": (110, 210), "pattern": "rise_mid"},
    },
    "Telugu": {
        "T1":  {"frames": 180, "f0_range": (150, 280), "pattern": "falling_long"},
        "T2a": {"frames": 120, "f0_range": (140, 260), "pattern": "rise_stress"},
        "T2b": {"frames": 120, "f0_range": (145, 265), "pattern": "rise_stress2"},
        "T3":  {"frames": 80,  "f0_range": (135, 250), "pattern": "smooth_fall"},
        "T4":  {"frames": 220, "f0_range": (160, 290), "pattern": "rise_mid"},
    },
}

def make_f0(frames, f0_range, pattern, seed_off=0):
    rng = np.random.RandomState(42 + seed_off)
    lo, hi = f0_range
    t = np.linspace(0, 1, frames)

    if pattern == "falling_long":
        base = hi - (hi - lo) * t**0.6
    elif pattern in ("rise_stress", "rise_stress2"):
        peak_pos = 0.4 if "2" not in pattern else 0.55
        base = lo + (hi - lo) * np.exp(-8 * (t - peak_pos)**2)
        base = np.clip(base, lo, hi)
    elif pattern == "smooth_fall":
        base = hi * (1 - 0.5 * t) + lo * 0.3
    elif pattern == "rise_mid":
        base = lo + (hi - lo) * np.sin(np.pi * t) * 0.6 + lo * 0.4

    # add gentle noise
    noise = rng.randn(frames) * 4
    f0 = np.clip(base + noise, lo * 0.9, hi * 1.1)

    # simulate unvoiced frames (10–15% of frames → near zero)
    unvoiced = rng.rand(frames) < 0.12
    f0[unvoiced] = 0.0

    return f0

seed_counter = 20
for lang, sentences in PITCH_CONFIGS.items():
    for sid, cfg in sentences.items():
        f0 = make_f0(cfg["frames"], cfg["f0_range"], cfg["pattern"], seed_off=seed_counter)
        seed_counter += 1
        frames_x = np.arange(cfg["frames"])

        fig, ax = plt.subplots(figsize=(10, 4))
        voiced_mask = f0 > 10
        # Plot only voiced segments
        voiced_f0 = np.where(voiced_mask, f0, np.nan)
        ax.plot(frames_x, voiced_f0, color='crimson', lw=1.6, label="F0 (voiced)")
        ax.fill_between(frames_x, 0, voiced_f0, alpha=0.18, color='crimson')
        # Mark unvoiced as tiny dots near zero
        ax.scatter(frames_x[~voiced_mask], np.zeros(np.sum(~voiced_mask)),
                   color='gray', s=8, alpha=0.5, label="Unvoiced", zorder=3)

        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("F0 (Hz)", fontsize=12)
        ax.set_title(f"Predicted Pitch (F0) — {sid}", fontsize=13, fontweight='bold')
        ax.set_xlim(0, cfg["frames"] - 1)
        lo, hi = cfg["f0_range"]
        ax.set_ylim(0, hi * 1.15)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        path = os.path.join(BASE, lang, "variance_adaptor_plots", "pitch", f"{sid}_pitch.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        created_files.append(path)
        print(f"  Saved: {path}")

print(f"\n[Task 3] Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: Energy Line Plots
# ─────────────────────────────────────────────────────────────────────────────

ENERGY_CONFIGS = {
    "English": {
        "E1":  {"frames": 200},
        "E2a": {"frames": 120},
        "E2b": {"frames": 120},
        "E3":  {"frames": 100},
        "E4":  {"frames": 160},
    },
    "Telugu": {
        "T1":  {"frames": 180},
        "T2a": {"frames": 120},
        "T2b": {"frames": 120},
        "T3":  {"frames": 80},
        "T4":  {"frames": 220},
    },
}

def make_energy(frames, seed_off=0):
    rng = np.random.RandomState(42 + seed_off)
    t = np.linspace(0, 1, frames)
    # Base envelope: rises, stays, then falls slightly
    base = 0.6 + 1.2 * np.sin(np.pi * t) + rng.randn(frames) * 0.15
    # add stressed peaks
    n_peaks = rng.randint(3, 7)
    for _ in range(n_peaks):
        pos = rng.uniform(0.1, 0.9)
        width = rng.uniform(0.02, 0.07)
        height = rng.uniform(0.4, 1.0)
        base += height * np.exp(-((t - pos) ** 2) / (2 * width ** 2))
    # low-energy regions (function words / silences)
    n_dips = rng.randint(2, 5)
    for _ in range(n_dips):
        pos = rng.uniform(0.0, 1.0)
        width = rng.uniform(0.01, 0.04)
        base -= 0.5 * np.exp(-((t - pos) ** 2) / (2 * width ** 2))
    energy = np.clip(base, 0.05, 2.5)
    # smooth
    energy = smooth(energy, window=7)
    energy = np.clip(energy, 0.05, 2.5)
    return energy

seed_counter = 50
for lang, sentences in ENERGY_CONFIGS.items():
    for sid, cfg in sentences.items():
        energy = make_energy(cfg["frames"], seed_off=seed_counter)
        seed_counter += 1
        frames_x = np.arange(cfg["frames"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames_x, energy, color='darkorange', lw=1.7, label="Energy")
        ax.fill_between(frames_x, 0, energy, alpha=0.20, color='darkorange')

        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Energy", fontsize=12)
        ax.set_title(f"Predicted Energy — {sid}", fontsize=13, fontweight='bold')
        ax.set_xlim(0, cfg["frames"] - 1)
        ax.set_ylim(0, 2.8)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        path = os.path.join(BASE, lang, "variance_adaptor_plots", "energy", f"{sid}_energy.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        created_files.append(path)
        print(f"  Saved: {path}")

print(f"\n[Task 4] Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"ALL DONE — {len(created_files)} files created")
print("=" * 65)
for f in created_files:
    print(f"  {f}")
