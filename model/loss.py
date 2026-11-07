"""
loss.py

FastSpeech 2 loss function.

Computes masked MSE losses for mel spectrogram (pre- and post-PostNet),
pitch and energy, plus a masked MAE loss for log-duration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class FastSpeech2Loss(nn.Module):
    """
    Composite loss for FastSpeech 2 training.

    Loss terms
    ----------
    mel_loss         : MSE between mel_output (pre-PostNet) and mel_target,
                       averaged over non-padded mel frames.
    mel_postnet_loss : MSE between mel_postnet_output (post-PostNet) and mel_target,
                       averaged over non-padded mel frames.
    pitch_loss       : MSE between pitch_pred and pitch_target,
                       averaged over non-padded mel frames.
    energy_loss      : MSE between energy_pred and energy_target,
                       averaged over non-padded mel frames.
    duration_loss    : MAE between log_duration_pred and log_duration_target,
                       averaged over non-padded *source* frames.

    Total loss = sum of all five terms.
    """

    def __init__(self) -> None:
        super().__init__()

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss averaged over unmasked (valid) positions.

        Args:
            pred   (Tensor):     Any shape matching *target*.
            target (Tensor):     Ground-truth values, same shape as *pred*.
            mask (BoolTensor):   Same shape as *pred* (True → padded / ignore).

        Returns:
            Scalar tensor — mean squared error on valid positions.
        """
        pred = pred.masked_fill(mask, 0.0)
        target = target.masked_fill(mask, 0.0)
        n_valid = (~mask).float().sum().clamp(min=1.0)
        return F.mse_loss(pred, target, reduction="sum") / n_valid

    @staticmethod
    def _masked_mae(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MAE (L1) loss averaged over unmasked (valid) positions.

        Args:
            pred   (Tensor):     Any shape matching *target*.
            target (Tensor):     Ground-truth values, same shape as *pred*.
            mask (BoolTensor):   Same shape as *pred* (True → padded / ignore).

        Returns:
            Scalar tensor — mean absolute error on valid positions.
        """
        pred = pred.masked_fill(mask, 0.0)
        target = target.masked_fill(mask, 0.0)
        n_valid = (~mask).float().sum().clamp(min=1.0)
        return F.l1_loss(pred, target, reduction="sum") / n_valid

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        inputs: Tuple,
        predictions: Tuple,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the FastSpeech 2 composite loss.

        Args:
            inputs (tuple):
                src_len              (LongTensor)  (batch,)
                mel_len              (LongTensor)  (batch,)
                mel_target           (Tensor)      (batch, mel_len, n_mel)
                pitch_target         (Tensor)      (batch, mel_len)
                energy_target        (Tensor)      (batch, mel_len)
                log_duration_target  (Tensor)      (batch, src_len)

            predictions (tuple):
                mel_output           (Tensor)      (batch, mel_len, n_mel)
                mel_postnet_output   (Tensor)      (batch, mel_len, n_mel)
                pitch_pred           (Tensor)      (batch, mel_len)
                energy_pred          (Tensor)      (batch, mel_len)
                log_duration_pred    (Tensor)      (batch, src_len)
                src_mask             (BoolTensor)  (batch, src_len)
                mel_mask             (BoolTensor)  (batch, mel_len)

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                total_loss (Tensor): Scalar — sum of all five loss terms.
                loss_dict  (dict):   {
                    "mel_loss":          scalar Tensor,
                    "mel_postnet_loss":  scalar Tensor,
                    "pitch_loss":        scalar Tensor,
                    "energy_loss":       scalar Tensor,
                    "duration_loss":     scalar Tensor,
                }
        """
        # ---- Unpack ---------------------------------------------------------
        (
            src_len,
            mel_len,
            mel_target,
            pitch_target,
            energy_target,
            log_duration_target,
        ) = inputs

        (
            mel_output,
            mel_postnet_output,
            pitch_pred,
            energy_pred,
            log_duration_pred,
            src_mask,
            mel_mask,
        ) = predictions

        # ---- Build per-element masks ----------------------------------------
        # mel_mask: (batch, mel_len) True at padding
        # We need (batch, mel_len, 1) for broadcasting with mel (batch, mel_len, n_mel)
        mel_mask_3d = mel_mask.unsqueeze(-1)   # (batch, mel_len, 1) → will broadcast

        # src_mask: (batch, src_len) True at padding
        # For duration (batch, src_len) → same shape, no expansion needed

        # ---- Individual losses ----------------------------------------------
        mel_loss = self._masked_mse(
            mel_output,
            mel_target,
            mel_mask_3d.expand_as(mel_output),
        )

        mel_postnet_loss = self._masked_mse(
            mel_postnet_output,
            mel_target,
            mel_mask_3d.expand_as(mel_postnet_output),
        )

        pitch_loss = self._masked_mse(pitch_pred, pitch_target, mel_mask)

        energy_loss = self._masked_mse(energy_pred, energy_target, mel_mask)

        duration_loss = self._masked_mae(
            log_duration_pred,
            log_duration_target,
            src_mask,
        )

        # ---- Total ----------------------------------------------------------
        total_loss = (
            mel_loss
            + mel_postnet_loss
            + pitch_loss
            + energy_loss
            + duration_loss
        )

        loss_dict: Dict[str, torch.Tensor] = {
            "mel_loss": mel_loss,
            "mel_postnet_loss": mel_postnet_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "duration_loss": duration_loss,
        }

        return total_loss, loss_dict
