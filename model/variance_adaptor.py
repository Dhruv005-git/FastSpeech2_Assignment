"""
variance_adaptor.py

Variance Adaptor module for FastSpeech 2:
  - VariancePredictor  — shared architecture for duration / pitch / energy prediction
  - LengthRegulator    — expands encoder output according to phoneme durations
  - VarianceAdaptor    — orchestrates the above with quantised pitch & energy embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LayerNorm


# ---------------------------------------------------------------------------
# VariancePredictor
# ---------------------------------------------------------------------------

class VariancePredictor(nn.Module):
    """
    Two-layer 1-D convolutional variance predictor shared by duration, pitch,
    and energy predictors.

    Architecture (per layer):
        Conv1d → ReLU → LayerNorm → Dropout
    Followed by a linear projection to scalar output.

    Args:
        input_size (int):   Input feature dimension.
        filter_size (int):  Number of conv filters.
        kernel_size (int):  Convolution kernel size.
        dropout (float):    Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        filter_size: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Layer 1
        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(
                input_size,
                filter_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.norm_1 = LayerNorm(filter_size)
        self.dropout_1 = nn.Dropout(dropout)

        # Layer 2
        self.conv_layer_2 = nn.Sequential(
            nn.Conv1d(
                filter_size,
                filter_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.norm_2 = LayerNorm(filter_size)
        self.dropout_2 = nn.Dropout(dropout)

        # Linear projection to scalar
        self.linear_layer = nn.Linear(filter_size, 1)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            encoder_output (Tensor): (batch, seq_len, input_size)
            mask (BoolTensor | None):(batch, seq_len) — True for padding; masked positions
                                     are zeroed in the output.

        Returns:
            Tensor: (batch, seq_len) — predicted scalar per frame.
        """
        # Conv1d expects (batch, channels, length)
        out = encoder_output.transpose(1, 2)

        # Layer 1
        out = self.conv_layer_1(out)
        out = self.dropout_1(self.norm_1(out.transpose(1, 2))).transpose(1, 2)

        # Layer 2
        out = self.conv_layer_2(out)
        out = self.dropout_2(self.norm_2(out.transpose(1, 2))).transpose(1, 2)

        # Linear: (batch, filter_size, len) → (batch, len, filter_size) → (batch, len)
        out = self.linear_layer(out.transpose(1, 2)).squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


# ---------------------------------------------------------------------------
# LengthRegulator
# ---------------------------------------------------------------------------

class LengthRegulator(nn.Module):
    """
    Repeats each encoder frame according to its integer phoneme duration to
    produce a mel-length output.

    Args:
        None — stateless module.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _regulate(encoder_output: torch.Tensor, duration: torch.Tensor, max_len: int = None):
        """
        Expand encoder_output along the time axis using duration counts.

        Args:
            encoder_output (Tensor): (batch, src_len, hidden)
            duration (LongTensor):   (batch, src_len) — integer repeat counts.
            max_len (int | None):    If provided, pad / truncate output to this length.

        Returns:
            Tuple[Tensor, Tensor]:
                output   (batch, mel_len, hidden) — length-regulated output
                mel_len  (LongTensor) (batch,)    — actual mel lengths per sample
        """
        output_list = []
        mel_len_list = []

        for enc_out, dur in zip(encoder_output, duration):
            # enc_out: (src_len, hidden), dur: (src_len,)
            expanded = torch.repeat_interleave(enc_out, dur, dim=0)  # (mel_len, hidden)
            output_list.append(expanded)
            mel_len_list.append(expanded.size(0))

        mel_len = torch.tensor(mel_len_list, dtype=torch.long, device=encoder_output.device)

        # Pad to same length
        target_len = max_len if max_len is not None else int(mel_len.max().item())
        padded = torch.zeros(
            encoder_output.size(0), target_len, encoder_output.size(-1),
            device=encoder_output.device,
            dtype=encoder_output.dtype,
        )
        for i, out in enumerate(output_list):
            L = min(out.size(0), target_len)
            padded[i, :L] = out[:L]

        return padded, mel_len

    def forward(
        self,
        encoder_output: torch.Tensor,
        duration: torch.Tensor,
        max_len: int = None,
    ):
        """
        Args:
            encoder_output (Tensor):  (batch, src_len, hidden)
            duration (LongTensor):    (batch, src_len) — integer repeat counts (≥ 0).
            max_len (int | None):     Desired output length for padding.

        Returns:
            Tuple[Tensor, Tensor]:
                output   (batch, mel_len, hidden)
                mel_len  (LongTensor) (batch,)
        """
        output, mel_len = self._regulate(encoder_output, duration, max_len)
        return output, mel_len


# ---------------------------------------------------------------------------
# VarianceAdaptor
# ---------------------------------------------------------------------------

class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor as described in FastSpeech 2 (Ren et al., 2021).

    Applies three variance predictors (duration, pitch, energy) to the encoder
    output, then uses the LengthRegulator to convert phoneme-level features to
    frame-level features.

    Pitch and energy are quantised into discrete bins and embedded via learned
    nn.Embedding tables, which are added to the regulated output.

    Args:
        config (dict): Model configuration containing at minimum:
            - encoder_hidden (int)
            - variance_predictor_filter_size (int)
            - variance_predictor_kernel_size (int)
            - variance_predictor_dropout (float)
            - pitch_min, pitch_max (float)
            - energy_min, energy_max (float)
            - n_bins (int)              number of quantisation bins (e.g. 256)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.duration_predictor = VariancePredictor(
            input_size=config["encoder_hidden"],
            filter_size=config["variance_predictor_filter_size"],
            kernel_size=config["variance_predictor_kernel_size"],
            dropout=config["variance_predictor_dropout"],
        )
        self.pitch_predictor = VariancePredictor(
            input_size=config["encoder_hidden"],
            filter_size=config["variance_predictor_filter_size"],
            kernel_size=config["variance_predictor_kernel_size"],
            dropout=config["variance_predictor_dropout"],
        )
        self.energy_predictor = VariancePredictor(
            input_size=config["encoder_hidden"],
            filter_size=config["variance_predictor_filter_size"],
            kernel_size=config["variance_predictor_kernel_size"],
            dropout=config["variance_predictor_dropout"],
        )

        self.length_regulator = LengthRegulator()

        n_bins: int = config.get("n_bins", 256)

        # Pitch quantisation boundaries (log-scale or linear, stored as buffer)
        pitch_min: float = config["pitch_min"]
        pitch_max: float = config["pitch_max"]
        self.register_buffer(
            "pitch_bins",
            torch.linspace(pitch_min, pitch_max, n_bins - 1),
        )

        # Energy quantisation boundaries
        energy_min: float = config["energy_min"]
        energy_max: float = config["energy_max"]
        self.register_buffer(
            "energy_bins",
            torch.linspace(energy_min, energy_max, n_bins - 1),
        )

        # Embedding tables
        self.pitch_embedding = nn.Embedding(n_bins, config["encoder_hidden"])
        self.energy_embedding = nn.Embedding(n_bins, config["encoder_hidden"])

    # ------------------------------------------------------------------
    # Helper: quantise a continuous value tensor to bin indices
    # ------------------------------------------------------------------

    def _quantise(self, values: torch.Tensor, bins: torch.Tensor) -> torch.LongTensor:
        """
        Quantise *values* into integer bin indices using ``torch.bucketize``.

        Args:
            values (Tensor): Any shape, float.
            bins   (Tensor): 1-D boundary tensor of shape (n_bins - 1,).

        Returns:
            LongTensor: Same shape as *values* with values in [0, n_bins - 1].
        """
        return torch.bucketize(values.contiguous(), bins)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        mel_mask: torch.Tensor = None,
        max_len: int = None,
        pitch_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
        duration_target: torch.Tensor = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ):
        """
        Args:
            x (Tensor):                   (batch, src_len, hidden) — encoder output.
            src_mask (BoolTensor):        (batch, src_len) — True for padding.
            mel_mask (BoolTensor | None): (batch, mel_len) — True for padding (used at
                                          training time when mel_len is known).
            max_len (int | None):         Maximum target mel length for padding.
            pitch_target (Tensor | None): (batch, mel_len) — ground-truth pitch values.
            energy_target (Tensor| None): (batch, mel_len) — ground-truth energy values.
            duration_target (LongTensor | None): (batch, src_len) — GT durations (training).
            p_control (float):            Pitch scaling factor (default: 1.0).
            e_control (float):            Energy scaling factor (default: 1.0).
            d_control (float):            Duration scaling factor (default: 1.0).

        Returns:
            Tuple:
                output           (batch, mel_len, hidden)
                p_pred           (batch, mel_len)
                e_pred           (batch, mel_len)
                log_d_pred       (batch, src_len)
                duration_rounded (LongTensor) (batch, src_len)
                mel_len          (LongTensor) (batch,)
                mel_mask         (BoolTensor) (batch, mel_len)
        """
        # ---- Duration -------------------------------------------------------
        log_d_pred = self.duration_predictor(x, src_mask)

        if duration_target is not None:
            # Training: use ground-truth integer durations
            duration_rounded = duration_target
        else:
            # Inference: convert predicted log-duration to rounded integers
            duration_rounded = torch.clamp(
                torch.round((torch.exp(log_d_pred) - 1) * d_control).long(),
                min=0,
            )

        # Length regulation: phoneme → frame
        x, mel_len = self.length_regulator(x, duration_rounded, max_len)

        # Build mel mask if not supplied
        if mel_mask is None:
            mel_mask = self._get_mask_from_lengths(mel_len, x.size(1))

        # ---- Pitch ----------------------------------------------------------
        p_pred = self.pitch_predictor(x, mel_mask)

        if pitch_target is not None:
            pitch_emb = self.pitch_embedding(self._quantise(pitch_target, self.pitch_bins))
        else:
            p_pred_controlled = p_pred * p_control
            pitch_emb = self.pitch_embedding(self._quantise(p_pred_controlled, self.pitch_bins))

        x = x + pitch_emb

        # ---- Energy ---------------------------------------------------------
        e_pred = self.energy_predictor(x, mel_mask)

        if energy_target is not None:
            energy_emb = self.energy_embedding(self._quantise(energy_target, self.energy_bins))
        else:
            e_pred_controlled = e_pred * e_control
            energy_emb = self.energy_embedding(self._quantise(e_pred_controlled, self.energy_bins))

        x = x + energy_emb

        return x, p_pred, e_pred, log_d_pred, duration_rounded, mel_len, mel_mask

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mask_from_lengths(lengths: torch.Tensor, max_len: int = None) -> torch.BoolTensor:
        """
        Build a boolean padding mask.

        Args:
            lengths (LongTensor): (batch,) — valid length per sample.
            max_len (int | None): If None, uses ``lengths.max()``.

        Returns:
            BoolTensor: (batch, max_len) — True at padded positions.
        """
        batch_size = lengths.size(0)
        if max_len is None:
            max_len = int(lengths.max().item())
        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        return ids >= lengths.unsqueeze(1)
