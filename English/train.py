"""
train.py — FastSpeech 2 training script (English / Svarah).

Usage
-----
python train.py --config config.yaml
python train.py --config config.yaml --checkpoint ./checkpoints/fs2_300000.pt
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt

# ── project imports ───────────────────────────────────────────────────────────
# Assumes the repo root (parent of English/) is on PYTHONPATH, or we add it.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

try:
    from model.fastspeech2 import FastSpeech2
    from model.loss import FastSpeech2Loss
except ImportError as exc:
    raise ImportError(
        "Cannot import model.fastspeech2 or model.loss. "
        "Make sure FastSpeech2_Assignment/ is on PYTHONPATH "
        f"and the model package exists.\n  Original: {exc}"
    )

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FastSpeech2Dataset(Dataset):
    """
    Loads preprocessed mel / pitch / energy / duration / phoneme arrays for
    a given split (train or val).
    """

    def __init__(
        self,
        preprocessed_path: str,
        split: str,
        config: dict,
    ) -> None:
        super().__init__()
        self.prep_dir = Path(preprocessed_path)
        self.config = config
        self.split = split

        split_file = self.prep_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split list not found: {split_file}")

        self.ids: List[str] = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        log.info(f"[{split}] {len(self.ids)} utterances loaded from {split_file}")

        # Build phoneme vocab from training data (or load if exists)
        self.vocab = self._build_vocab()

    # ── Vocabulary ──────────────────────────────────────────────────────────

    def _build_vocab(self) -> Dict[str, int]:
        vocab_path = self.prep_dir / "vocab.txt"
        if vocab_path.exists():
            vocab: Dict[str, int] = {}
            for idx, line in enumerate(
                vocab_path.read_text(encoding="utf-8").splitlines()
            ):
                token = line.strip()
                if token:
                    vocab[token] = idx
            return vocab

        # Build from scratch (only on training set to avoid leakage)
        if self.split != "train":
            raise FileNotFoundError(
                f"vocab.txt not found at {vocab_path}. "
                "Run training split first to build vocabulary."
            )

        log.info("Building phoneme vocabulary…")
        tokens: set = {"<pad>", "<unk>", "sp"}
        for fid in self.ids:
            ph_file = self.prep_dir / f"phoneme_{fid}.txt"
            if ph_file.exists():
                phones = ph_file.read_text(encoding="utf-8").strip().split()
                tokens.update(phones)

        sorted_tokens = sorted(tokens)
        vocab = {tok: idx for idx, tok in enumerate(sorted_tokens)}
        vocab_path.write_text(
            "\n".join(sorted_tokens), encoding="utf-8"
        )
        log.info(f"Vocabulary size: {len(vocab)} — saved to {vocab_path}")
        return vocab

    def phonemes_to_indices(self, phones: List[str]) -> List[int]:
        unk = self.vocab.get("<unk>", 1)
        return [self.vocab.get(p, unk) for p in phones]

    # ── __len__ / __getitem__ ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        fid = self.ids[idx]
        prep = self.prep_dir

        mel = np.load(str(prep / f"mel_{fid}.npy"))       # (n_mel, T)
        pitch = np.load(str(prep / f"pitch_{fid}.npy"))   # (T,)
        energy = np.load(str(prep / f"energy_{fid}.npy")) # (T,)
        duration = np.load(str(prep / f"duration_{fid}.npy"))  # (L,)

        phones_str = (prep / f"phoneme_{fid}.txt").read_text(encoding="utf-8").strip().split()
        phone_ids = self.phonemes_to_indices(phones_str)

        return {
            "id": fid,
            "phonemes": np.array(phone_ids, dtype=np.int64),  # (L,)
            "mel": mel,          # (n_mel, T)
            "pitch": pitch,      # (T,)
            "energy": energy,    # (T,)
            "duration": duration,  # (L,)
            "mel_len": mel.shape[1],
            "phoneme_len": len(phone_ids),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> dict:
    """
    Pad variable-length sequences to the maximum length in the batch and
    build boolean masks.
    """
    ids = [item["id"] for item in batch]

    max_ph_len = max(item["phoneme_len"] for item in batch)
    max_mel_len = max(item["mel_len"] for item in batch)
    n_mel = batch[0]["mel"].shape[0]

    # Allocate padded tensors
    phonemes = torch.zeros(len(batch), max_ph_len, dtype=torch.long)
    ph_mask = torch.ones(len(batch), max_ph_len, dtype=torch.bool)  # True = padding

    mels = torch.zeros(len(batch), n_mel, max_mel_len)
    mel_mask = torch.ones(len(batch), max_mel_len, dtype=torch.bool)

    pitches = torch.zeros(len(batch), max_mel_len)
    energies = torch.zeros(len(batch), max_mel_len)
    durations = torch.zeros(len(batch), max_ph_len, dtype=torch.long)

    for i, item in enumerate(batch):
        pl = item["phoneme_len"]
        tl = item["mel_len"]

        phonemes[i, :pl] = torch.tensor(item["phonemes"], dtype=torch.long)
        ph_mask[i, :pl] = False

        mels[i, :, :tl] = torch.tensor(item["mel"], dtype=torch.float32)
        mel_mask[i, :tl] = False

        pitches[i, :tl] = torch.tensor(item["pitch"], dtype=torch.float32)
        energies[i, :tl] = torch.tensor(item["energy"], dtype=torch.float32)
        durations[i, :pl] = torch.tensor(item["duration"], dtype=torch.long)

    return {
        "ids": ids,
        "phonemes": phonemes,          # (B, L)
        "ph_mask": ph_mask,            # (B, L)  True = pad
        "mels": mels,                  # (B, n_mel, T)
        "mel_mask": mel_mask,          # (B, T)   True = pad
        "pitches": pitches,            # (B, T)
        "energies": energies,          # (B, T)
        "durations": durations,        # (B, L)
        "mel_lens": torch.tensor([item["mel_len"] for item in batch]),
        "ph_lens": torch.tensor([item["phoneme_len"] for item in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    preprocessed_path: str,
    split: str,
    config: dict,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = FastSpeech2Dataset(preprocessed_path, split, config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

class NoamScheduler:
    """
    Noam (transformer) warmup schedule:
        lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

    Optionally applies step-wise annealing after certain global steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        anneal_steps: Optional[List[int]] = None,
        anneal_rate: float = 0.3,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.anneal_steps = sorted(anneal_steps or [])
        self.anneal_rate = anneal_rate
        self._step = 0
        self._base_lr: Optional[float] = None

    def step(self) -> float:
        self._step += 1
        s = self._step
        lr = (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup ** -1.5)

        # Apply annealing multiplier
        n_anneals = sum(1 for a in self.anneal_steps if s >= a)
        lr *= self.anneal_rate ** n_anneals

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loss: float,
    ckpt_dir: str,
) -> str:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(ckpt_dir) / f"fs2_{step:08d}.pt")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    log.info(f"  ✓ Checkpoint saved → {path}")
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, int, float]:
    log.info(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return state.get("step", 0), state.get("epoch", 0), state.get("loss", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Loss logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def append_loss_csv(csv_path: str, step: int, loss_dict: dict) -> None:
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step"] + list(loss_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({"step": step, **loss_dict})


def plot_losses(csv_path: str, out_png: str, title: str = "Loss") -> None:
    if not Path(csv_path).exists():
        return
    steps, losses = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                losses.append(float(row.get("total_loss", row.get("loss", 0))))
            except ValueError:
                pass

    if not steps:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info(f"Loss plot saved → {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    step: int,
    val_csv: str,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        phonemes = batch["phonemes"].to(device)
        ph_mask = batch["ph_mask"].to(device)
        mels = batch["mels"].to(device)
        mel_mask = batch["mel_mask"].to(device)
        pitches = batch["pitches"].to(device)
        energies = batch["energies"].to(device)
        durations = batch["durations"].to(device)

        output = model(
            phonemes,
            ph_mask,
            mel_target=mels,
            duration_target=durations,
            pitch_target=pitches,
            energy_target=energies,
        )
        loss_dict = criterion(output, mels, durations, pitches, energies, mel_mask)
        total_loss += loss_dict["total_loss"].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    append_loss_csv(val_csv, step, {"total_loss": avg_loss})
    log.info(f"  [Val step {step}] loss = {avg_loss:.4f}")
    model.train()
    return avg_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str, checkpoint_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    # ── Directories ───────────────────────────────────────────────────────
    ckpt_dir = cfg["checkpoint_path"]
    result_dir = cfg["result_path"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    train_csv = str(Path(result_dir) / "training_loss.csv")
    val_csv = str(Path(result_dir) / "validation_loss.csv")

    # ── DataLoaders ───────────────────────────────────────────────────────
    prep_path = cfg["preprocessed_path"]
    batch_size = cfg["batch_size"]

    train_loader = get_dataloader(prep_path, "train", cfg, batch_size, shuffle=True)
    val_loader = get_dataloader(prep_path, "val", cfg, batch_size, shuffle=False)

    # Derive vocab size from saved vocab.txt
    vocab_path = Path(prep_path) / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"vocab.txt not found at {vocab_path}. Run preprocess.py first."
        )
    vocab_size = sum(1 for l in vocab_path.read_text(encoding="utf-8").splitlines() if l.strip())
    log.info(f"Vocabulary size: {vocab_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FastSpeech2(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        encoder_hidden=cfg["encoder_hidden"],
        encoder_head=cfg["encoder_head"],
        encoder_layers=cfg["encoder_layers"],
        encoder_dropout=cfg["encoder_dropout"],
        encoder_kernel_size=cfg["encoder_kernel_size"],
        decoder_hidden=cfg["decoder_hidden"],
        decoder_head=cfg["decoder_head"],
        decoder_layers=cfg["decoder_layers"],
        decoder_dropout=cfg["decoder_dropout"],
        fft_conv1d_filter_size=cfg["fft_conv1d_filter_size"],
        fft_conv1d_kernel_size=cfg["fft_conv1d_kernel_size"],
        variance_predictor_filter_size=cfg["variance_predictor_filter_size"],
        variance_predictor_kernel_size=cfg["variance_predictor_kernel_size"],
        variance_predictor_dropout=cfg["variance_predictor_dropout"],
        n_mel_channels=cfg["n_mel_channels"],
        pitch_bins=cfg["pitch_bins"],
        energy_bins=cfg["energy_bins"],
        pitch_normalization=cfg["pitch_normalization"],
        energy_normalization=cfg["energy_normalization"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"FastSpeech 2 — {n_params:,} trainable parameters")

    criterion = FastSpeech2Loss().to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=cfg["encoder_hidden"],
        warmup_steps=cfg["warm_up_step"],
        anneal_steps=cfg["anneal_steps"],
        anneal_rate=cfg["anneal_rate"],
    )

    # ── Mixed precision scaler ────────────────────────────────────────────
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_step = 0
    start_epoch = 0

    if checkpoint_path and Path(checkpoint_path).exists():
        start_step, start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
        # Fast-forward scheduler
        for _ in range(start_step):
            scheduler.step()
        log.info(f"Resuming from step {start_step}, epoch {start_epoch}")
    elif not checkpoint_path:
        # Auto-detect latest checkpoint
        ckpts = sorted(Path(ckpt_dir).glob("fs2_*.pt"))
        if ckpts:
            start_step, start_epoch, _ = load_checkpoint(str(ckpts[-1]), model, optimizer)
            for _ in range(start_step):
                scheduler.step()
            log.info(f"Auto-resumed from {ckpts[-1]} (step {start_step})")

    # ── Training loop ─────────────────────────────────────────────────────
    total_steps = len(train_loader) * cfg["epochs"]
    global_step = start_step
    accumulation_steps = cfg.get("accumulation_steps", 1)

    log.info(f"Starting training — epochs={cfg['epochs']}, steps/epoch={len(train_loader)}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            phonemes = batch["phonemes"].to(device)
            ph_mask = batch["ph_mask"].to(device)
            mels = batch["mels"].to(device)
            mel_mask = batch["mel_mask"].to(device)
            pitches = batch["pitches"].to(device)
            energies = batch["energies"].to(device)
            durations = batch["durations"].to(device)

            # ── Forward ───────────────────────────────────────────────────
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(
                    phonemes,
                    ph_mask,
                    mel_target=mels,
                    duration_target=durations,
                    pitch_target=pitches,
                    energy_target=energies,
                )
                loss_dict = criterion(output, mels, durations, pitches, energies, mel_mask)
                loss = loss_dict["total_loss"] / accumulation_steps

            # ── Backward ──────────────────────────────────────────────────
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["grad_clip_thresh"]
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                lr = scheduler.step()
                global_step += 1

                epoch_loss += loss_dict["total_loss"].item()

                # ── Logging ───────────────────────────────────────────────
                if global_step % cfg["log_step"] == 0:
                    mel_loss = loss_dict.get("mel_loss", torch.tensor(0.0)).item()
                    dur_loss = loss_dict.get("dur_loss", torch.tensor(0.0)).item()
                    pitch_loss = loss_dict.get("pitch_loss", torch.tensor(0.0)).item()
                    energy_loss = loss_dict.get("energy_loss", torch.tensor(0.0)).item()
                    total = loss_dict["total_loss"].item()

                    log.info(
                        f"[Epoch {epoch+1} | Step {global_step}] "
                        f"total={total:.4f}  mel={mel_loss:.4f}  "
                        f"dur={dur_loss:.4f}  pitch={pitch_loss:.4f}  "
                        f"energy={energy_loss:.4f}  lr={lr:.2e}"
                    )
                    append_loss_csv(
                        train_csv,
                        global_step,
                        {
                            "total_loss": total,
                            "mel_loss": mel_loss,
                            "dur_loss": dur_loss,
                            "pitch_loss": pitch_loss,
                            "energy_loss": energy_loss,
                            "lr": lr,
                        },
                    )

                # ── Validation ────────────────────────────────────────────
                if global_step % cfg["val_step"] == 0:
                    validate(model, criterion, val_loader, device, global_step, val_csv)

                # ── Checkpoint ────────────────────────────────────────────
                if global_step % cfg["save_step"] == 0:
                    save_checkpoint(
                        model, optimizer, global_step, epoch,
                        loss_dict["total_loss"].item(), ckpt_dir
                    )

        elapsed = time.time() - t0
        log.info(
            f"[Epoch {epoch+1}/{cfg['epochs']}] "
            f"avg_loss={epoch_loss / max(len(train_loader), 1):.4f}  "
            f"elapsed={elapsed/60:.1f} min"
        )

    # ── Final checkpoint ──────────────────────────────────────────────────
    save_checkpoint(model, optimizer, global_step, cfg["epochs"], 0.0, ckpt_dir)

    # ── Plot loss curves ──────────────────────────────────────────────────
    plot_losses(
        train_csv,
        str(Path(result_dir) / "training_loss.png"),
        title="FastSpeech 2 — Training Loss (English)",
    )
    plot_losses(
        val_csv,
        str(Path(result_dir) / "validation_loss.png"),
        title="FastSpeech 2 — Validation Loss (English)",
    )

    log.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastSpeech 2 — English")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (optional)",
    )
    args = parser.parse_args()
    train(args.config, checkpoint_path=args.checkpoint)
