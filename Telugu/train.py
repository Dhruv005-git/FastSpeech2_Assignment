"""
Telugu/train.py
===============
Training script for the Telugu FastSpeech 2 model.

Architecture: Shared FastSpeech 2 model (imported from ../model/).
Data:         IndicTTS Telugu preprocessed by preprocess.py.
Config:       Telugu/config.yaml

Usage:
    # From the FastSpeech2_Assignment/ root:
    python Telugu/train.py --config Telugu/config.yaml

    # Resume from checkpoint:
    python Telugu/train.py --config Telugu/config.yaml \\
        --restore_step 50000

    # Multi-GPU (DataParallel):
    python Telugu/train.py --config Telugu/config.yaml --gpus 0,1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml

# ---------------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Local imports
# NOTE: Adjust the import path to match your actual model module structure.
# ---------------------------------------------------------------------------
try:
    from model.fastspeech2 import FastSpeech2
    from model.loss import FastSpeech2Loss
except ImportError:
    raise ImportError(
        "Cannot import model.fastspeech2 or model.loss.\n"
        "Make sure the model/ directory exists at the project root and contains:\n"
        "  model/__init__.py\n"
        "  model/fastspeech2.py   ← FastSpeech2 nn.Module\n"
        "  model/loss.py          ← FastSpeech2Loss\n"
    )

from utils.text_telugu import load_vocab
from utils.plot_utils import plot_training_loss, plot_validation_loss


# ===========================================================================
# Dataset
# ===========================================================================

class TeluguTTSDataset(Dataset):
    """PyTorch Dataset for preprocessed IndicTTS Telugu features.

    Loads mel spectrograms, phoneme ID sequences, pitch, energy, and
    duration arrays from the .npy files produced by preprocess.py.

    Args:
        prep_path: Root of the preprocessed directory (e.g. data/preprocessed/telugu).
        split:     'train' or 'val'.
        phones_json: Path to the phones.json file for this split.
    """

    def __init__(self, prep_path: str, split: str, phones_json: str) -> None:
        super().__init__()
        self.split_dir = os.path.join(prep_path, split)
        with open(phones_json, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        self._check_dirs()

    def _check_dirs(self) -> None:
        for sub in ("mel", "pitch", "energy", "duration"):
            d = os.path.join(self.split_dir, sub)
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Missing directory: {d}\n"
                    "Run Telugu/preprocess.py first."
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry  = self.entries[idx]
        fid    = entry["file_id"]

        mel      = np.load(os.path.join(self.split_dir, "mel",      f"{fid}.npy"))
        pitch    = np.load(os.path.join(self.split_dir, "pitch",    f"{fid}.npy"))
        energy   = np.load(os.path.join(self.split_dir, "energy",   f"{fid}.npy"))
        duration = np.load(os.path.join(self.split_dir, "duration", f"{fid}.npy"))
        phone_ids = np.array(entry["phone_ids"], dtype=np.int64)

        return {
            "file_id":  fid,
            "phone_ids": torch.from_numpy(phone_ids).long(),
            "mel":       torch.from_numpy(mel).float(),        # (n_mels, T)
            "pitch":     torch.from_numpy(pitch).float(),      # (T,)
            "energy":    torch.from_numpy(energy).float(),     # (T,)
            "duration":  torch.from_numpy(duration).long(),    # (L,)
        }


def _pad_sequence(seqs: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Right-pad a list of 1-D or 2-D tensors to the same length."""
    max_len = max(s.shape[-1] for s in seqs)
    out = []
    for s in seqs:
        pad = max_len - s.shape[-1]
        if s.dim() == 1:
            out.append(torch.nn.functional.pad(s, (0, pad), value=pad_value))
        else:  # 2-D: (n_mels, T)
            out.append(torch.nn.functional.pad(s, (0, pad), value=pad_value))
    return torch.stack(out, dim=0)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Dynamic padding collate function for variable-length sequences."""
    file_ids   = [b["file_id"] for b in batch]
    phone_ids  = _pad_sequence([b["phone_ids"]  for b in batch], pad_value=0)
    mels       = _pad_sequence([b["mel"]        for b in batch], pad_value=0.0)
    pitches    = _pad_sequence([b["pitch"]      for b in batch], pad_value=0.0)
    energies   = _pad_sequence([b["energy"]     for b in batch], pad_value=0.0)
    durations  = _pad_sequence([b["duration"]   for b in batch], pad_value=0)

    # Lengths (for masking)
    phone_lengths = torch.tensor([b["phone_ids"].shape[0] for b in batch], dtype=torch.long)
    mel_lengths   = torch.tensor([b["mel"].shape[1]       for b in batch], dtype=torch.long)

    return {
        "file_ids":      file_ids,
        "phone_ids":     phone_ids,
        "phone_lengths": phone_lengths,
        "mels":          mels,
        "mel_lengths":   mel_lengths,
        "pitches":       pitches,
        "energies":      energies,
        "durations":     durations,
    }


# ===========================================================================
# Scheduler (Noam / Transformer LR schedule)
# ===========================================================================

class NoamScheduler:
    """Noam learning-rate schedule from 'Attention Is All You Need'."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 d_model: int, warmup_steps: int) -> None:
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0

    def step(self) -> float:
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        s = self._step
        w = self.warmup_steps
        return (self.d_model ** -0.5) * min(s ** -0.5, s * w ** -1.5)

    @property
    def current_lr(self) -> float:
        return self._get_lr()


# ===========================================================================
# Training loop
# ===========================================================================

def train(config_path: str,
          restore_step: int = 0,
          gpus: str = "0") -> None:
    """Main training function.

    Args:
        config_path:  Path to Telugu/config.yaml.
        restore_step: Global step to restore from (0 = train from scratch).
        gpus:         Comma-separated GPU indices string (e.g. '0' or '0,1').
    """
    # --- Config ---
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prep_path  = config["preprocessed_path"]
    ckpt_path  = config["checkpoint_path"]
    result_path = config["result_path"]
    vocab_path  = config["vocab_path"]
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    # --- Device ---
    gpu_list = [int(g) for g in gpus.split(",") if g.strip()]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}  |  GPUs: {gpu_list}")

    # --- Vocabulary ---
    vocab = load_vocab(vocab_path)
    n_symbols = len(vocab)
    print(f"[train] Vocabulary size: {n_symbols}")

    # --- Datasets & Loaders ---
    train_phones = os.path.join(prep_path, "train", "phones.json")
    val_phones   = os.path.join(prep_path, "val",   "phones.json")

    train_dataset = TeluguTTSDataset(prep_path, "train", train_phones)
    val_dataset   = TeluguTTSDataset(prep_path, "val",   val_phones)

    train_cfg  = config["train"]
    batch_size = train_cfg["batch_size"]
    n_workers  = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers, pin_memory=True,
    )

    # --- Model ---
    model_cfg = config["model"]
    model = FastSpeech2(
        n_symbols=n_symbols,
        **model_cfg,
    ).to(device)

    if len(gpu_list) > 1:
        model = nn.DataParallel(model, device_ids=gpu_list)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {total_params:,}")

    # --- Loss ---
    criterion = FastSpeech2Loss(config)

    # --- Optimiser & Scheduler ---
    optimizer = Adam(
        model.parameters(),
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=train_cfg.get("d_model", model_cfg["encoder_hidden"]),
        warmup_steps=train_cfg.get("lr_warmup_steps", 4000),
    )

    # --- Restore checkpoint ---
    global_step = 0
    if restore_step > 0:
        ckpt_file = os.path.join(ckpt_path, f"checkpoint_{restore_step}.pt")
        if os.path.isfile(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=device)
            (model.module if hasattr(model, "module") else model).load_state_dict(
                ckpt["model"]
            )
            optimizer.load_state_dict(ckpt["optimizer"])
            global_step = ckpt.get("global_step", restore_step)
            print(f"[train] Restored checkpoint from step {global_step}.")
        else:
            print(f"[train] WARNING: Checkpoint not found at {ckpt_file}. Training from scratch.")

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(result_path, "tensorboard"))

    # --- Training ---
    total_steps   = train_cfg.get("total_step", 900_000)
    log_step      = train_cfg.get("log_step",   100)
    val_step      = train_cfg.get("val_step",   1000)
    save_step     = train_cfg.get("save_step",  10_000)
    grad_clip     = train_cfg.get("grad_clip_thresh", 1.0)
    acc_steps     = train_cfg.get("acc_steps", 1)

    train_losses: List[float] = []
    val_losses:   List[float] = []

    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    t_start      = time.time()

    print(f"[train] Starting training — total steps: {total_steps}")

    epoch = 0
    while global_step < total_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= total_steps:
                break

            # Move to device
            phone_ids     = batch["phone_ids"].to(device)
            phone_lengths = batch["phone_lengths"].to(device)
            mels          = batch["mels"].to(device)
            mel_lengths   = batch["mel_lengths"].to(device)
            pitches       = batch["pitches"].to(device)
            energies      = batch["energies"].to(device)
            durations     = batch["durations"].to(device)

            # Forward pass
            output = model(
                src_seq=phone_ids,
                src_len=phone_lengths,
                mel_len=mel_lengths,
                d_target=durations,
                p_target=pitches,
                e_target=energies,
            )

            # Loss
            loss, loss_dict = criterion(output, batch)
            loss = loss / acc_steps
            loss.backward()

            if (global_step + 1) % acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                lr = scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss_dict["total"].item()
            global_step  += 1

            # Logging
            if global_step % log_step == 0:
                avg_loss = running_loss / log_step
                elapsed  = time.time() - t_start
                print(
                    f"[train] Step {global_step:>7d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {scheduler.current_lr:.2e} | "
                    f"Elapsed: {elapsed:.1f}s"
                )
                writer.add_scalar("Loss/train_total", avg_loss, global_step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f"Loss/train_{k}", v.item(), global_step)
                writer.add_scalar("LR", scheduler.current_lr, global_step)
                train_losses.append(avg_loss)
                running_loss = 0.0
                t_start = time.time()

            # Validation
            if global_step % val_step == 0:
                val_loss = _validate(model, val_loader, criterion, device, config)
                val_losses.append(val_loss)
                writer.add_scalar("Loss/val_total", val_loss, global_step)
                print(f"[train] *** Validation Step {global_step} | Val Loss: {val_loss:.4f} ***")
                model.train()

            # Checkpoint
            if global_step % save_step == 0:
                _save_checkpoint(model, optimizer, global_step, ckpt_path)

    # --- Final save ---
    _save_checkpoint(model, optimizer, global_step, ckpt_path)
    writer.close()

    # --- Plot loss curves ---
    plot_training_loss(
        train_losses,
        save_path=os.path.join(result_path, "train_loss.png"),
        title="Telugu FastSpeech 2 — Training Loss",
    )
    plot_validation_loss(
        val_losses,
        save_path=os.path.join(result_path, "val_loss.png"),
        title="Telugu FastSpeech 2 — Validation Loss",
    )
    print(f"\n[train] ✓ Training complete.  Checkpoints at: {ckpt_path}")


# ===========================================================================
# Validation helper
# ===========================================================================

@torch.no_grad()
def _validate(model: nn.Module,
              loader: DataLoader,
              criterion: nn.Module,
              device: torch.device,
              config: dict) -> float:
    """Run one full validation pass and return mean total loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        phone_ids     = batch["phone_ids"].to(device)
        phone_lengths = batch["phone_lengths"].to(device)
        mels          = batch["mels"].to(device)
        mel_lengths   = batch["mel_lengths"].to(device)
        pitches       = batch["pitches"].to(device)
        energies      = batch["energies"].to(device)
        durations     = batch["durations"].to(device)

        output = model(
            src_seq=phone_ids,
            src_len=phone_lengths,
            mel_len=mel_lengths,
            d_target=durations,
            p_target=pitches,
            e_target=energies,
        )
        loss, loss_dict = criterion(output, batch)
        total_loss += loss_dict["total"].item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _save_checkpoint(model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     step: int,
                     ckpt_dir: str) -> None:
    ckpt = {
        "global_step": step,
        "model":       (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer":   optimizer.state_dict(),
    }
    path = os.path.join(ckpt_dir, f"checkpoint_{step}.pt")
    torch.save(ckpt, path)
    print(f"[train] Checkpoint saved → {path}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastSpeech 2 — Telugu")
    parser.add_argument("--config", default="Telugu/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--restore_step", type=int, default=0,
                        help="Checkpoint step to restore (0 = fresh start)")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU indices, e.g. '0' or '0,1'")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            Path(__file__).resolve().parent.parent, config_path
        )

    train(config_path, restore_step=args.restore_step, gpus=args.gpus)
