"""
modules.py

Core building blocks for FastSpeech 2:
  - LayerNorm
  - MultiHeadAttention
  - PositionwiseFeedForward
  - FFTBlock
  - ConvNorm
  - PostNet
  - get_sinusoid_encoding_table
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> np.ndarray:
    """
    Return a sinusoidal positional-encoding table.

    Args:
        n_position (int): Maximum number of positions (sequence length + 1).
        d_hid (int):      Dimensionality of the model / hidden size.

    Returns:
        np.ndarray: Shape (n_position, d_hid), dtype float32.
    """

    def _angle(position: int, hid_j: int) -> float:
        return position / (10000 ** (2 * (hid_j // 2) / d_hid))

    table = np.array(
        [
            [_angle(pos, hid_j) for hid_j in range(d_hid)]
            for pos in range(n_position)
        ],
        dtype=np.float32,
    )
    # Apply sin to even indices, cos to odd indices
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return table


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """
    Thin wrapper around ``torch.nn.LayerNorm``.

    Args:
        d_hid (int):          Hidden dimension.
        eps (float):          Epsilon for numerical stability (default: 1e-6).
    """

    def __init__(self, d_hid: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_hid, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape (..., d_hid).

        Returns:
            Tensor: Same shape as *x*, normalised along last dimension.
        """
        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.

    Args:
        n_head (int):    Number of attention heads.
        d_model (int):   Model dimensionality (must be divisible by n_head).
        d_k (int):       Dimensionality of each head's key/query vectors.
        d_v (int):       Dimensionality of each head's value vectors.
        dropout (float): Dropout probability on attention weights (default: 0.1).
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

        nn.init.normal_(self.w_qs.weight, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, std=math.sqrt(2.0 / (d_model + d_v)))
        nn.init.xavier_normal_(self.fc.weight)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            q (Tensor): (batch, n_head, len_q, d_k)
            k (Tensor): (batch, n_head, len_k, d_k)
            v (Tensor): (batch, n_head, len_v, d_v)  len_v == len_k
            mask (BoolTensor | None): (batch, 1, 1, len_k) — True where padded.

        Returns:
            Tuple[Tensor, Tensor]:
                output   (batch, n_head, len_q, d_v)
                attn_w   (batch, n_head, len_q, len_k)
        """
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (..., len_q, len_k)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn_w = self.attn_dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn_w, v)
        return output, attn_w

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            q (Tensor):            (batch, len_q, d_model)
            k (Tensor):            (batch, len_k, d_model)
            v (Tensor):            (batch, len_v, d_model)
            mask (BoolTensor|None):(batch, len_q, len_k) — True where padded.

        Returns:
            Tuple[Tensor, Tensor]:
                output   (batch, len_q, d_model)
                attn_w   (batch, n_head, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch, len_q = q.size(0), q.size(1)
        len_k = k.size(1)

        residual = q

        # Linear projections then split into heads
        q = self.w_qs(q).view(batch, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch, len_k, n_head, d_v).transpose(1, 2)

        if mask is not None:
            # mask: (batch, len_q, len_k) → (batch, 1, 1, len_k) for src_mask
            # or (batch, 1, len_q, len_k) depending on caller — accept both
            if mask.dim() == 2:
                # (batch, len_k) key-padding mask
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        output, attn_w = self._scaled_dot_product_attention(q, k, v, mask)

        # Merge heads
        output = output.transpose(1, 2).contiguous().view(batch, len_q, n_head * d_v)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn_w


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network with ReLU activation and dropout.

    Architecture: Linear → ReLU → Dropout → Linear → Dropout → LayerNorm (residual)

    Args:
        d_in (int):      Input (and output) dimension.
        d_hid (int):     Inner hidden dimension.
        kernel_size (list[int]):  Kernel sizes [kernel_1, kernel_2] for the two conv layers.
        dropout (float): Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        kernel_size: list = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size is None:
            kernel_size = [9, 1]

        # Convolution-based position-wise FFN (matches ming024 style)
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (batch, seq_len, d_in)

        Returns:
            Tensor: (batch, seq_len, d_in)
        """
        residual = x

        # Conv1d expects (batch, channels, length)
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output).transpose(1, 2)

        return self.layer_norm(output + residual)


# ---------------------------------------------------------------------------
# Feed-Forward Transformer Block
# ---------------------------------------------------------------------------

class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer (FFT) block as used in FastSpeech 2.

    Consists of:
      1. Multi-Head Self-Attention (with residual + LayerNorm — handled inside MHA)
      2. Position-wise Feed-Forward Network

    Args:
        d_model (int):         Model hidden dimension.
        n_head (int):          Number of attention heads.
        d_k (int):             Key/query dimension per head.
        d_v (int):             Value dimension per head.
        d_inner (int):         Inner dimension of the FFN.
        kernel_size (list):    Kernel sizes for the FFN conv layers.
        dropout (float):       Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
        d_inner: int,
        kernel_size: list = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size is None:
            kernel_size = [9, 1]

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size=kernel_size, dropout=dropout
        )

    def forward(
        self,
        enc_input: torch.Tensor,
        mask: torch.Tensor = None,
        slf_attn_mask: torch.Tensor = None,
    ):
        """
        Args:
            enc_input (Tensor):         (batch, seq_len, d_model)
            mask (BoolTensor | None):   (batch, seq_len) — True for padding positions.
                                        Used to zero out padded positions in FFN output.
            slf_attn_mask (BoolTensor | None): attention mask fed to MHA.

        Returns:
            Tuple[Tensor, Tensor]:
                enc_output  (batch, seq_len, d_model)
                attn_w      (batch, n_head, seq_len, seq_len)
        """
        enc_output, attn_w = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )

        # Zero out padded positions
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0.0)

        enc_output = self.pos_ffn(enc_output)

        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0.0)

        return enc_output, attn_w


# ---------------------------------------------------------------------------
# ConvNorm
# ---------------------------------------------------------------------------

class ConvNorm(nn.Module):
    """
    1-D convolution with optional weight normalisation.

    Args:
        in_channels (int):   Input channel count.
        out_channels (int):  Output channel count.
        kernel_size (int):   Convolution kernel size (default: 1).
        stride (int):        Stride (default: 1).
        padding (int|None):  Explicit padding; if None, computed for 'same' output.
        dilation (int):      Dilation (default: 1).
        bias (bool):         Whether to use bias (default: True).
        w_init_gain (str):   Gain string for ``nn.init.calculate_gain`` (default: 'linear').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1, "kernel_size must be odd when padding is None"
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (batch, in_channels, length)

        Returns:
            Tensor: (batch, out_channels, length)
        """
        return self.conv(x)


# ---------------------------------------------------------------------------
# PostNet
# ---------------------------------------------------------------------------

class PostNet(nn.Module):
    """
    Post-processing network that refines the mel-spectrogram output.

    Five 1-D convolutional layers:
      - Layers 1–4: ConvNorm → BatchNorm1d → Tanh → Dropout
      - Layer  5:   ConvNorm → BatchNorm1d → Dropout  (no activation)

    The PostNet output is added as a residual to the pre-PostNet mel output.

    Args:
        n_mel_channels (int):   Number of mel filterbank channels (default: 80).
        postnet_embedding_dim (int): Internal channel width (default: 512).
        postnet_kernel_size (int):   Convolution kernel size (default: 5).
        postnet_n_convolutions (int):Number of conv layers (default: 5).
        dropout (float):             Dropout probability (default: 0.5).
    """

    def __init__(
        self,
        n_mel_channels: int = 80,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.convolutions = nn.ModuleList()

        # First layer: mel → embedding_dim
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding=int((postnet_kernel_size - 1) / 2),
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        # Middle layers: embedding_dim → embedding_dim
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        padding=int((postnet_kernel_size - 1) / 2),
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        # Last layer: embedding_dim → mel  (linear, no tanh)
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    padding=int((postnet_kernel_size - 1) / 2),
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

        self.n_convolutions = postnet_n_convolutions
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Mel spectrogram  (batch, n_mel_channels, time).

        Returns:
            Tensor: Residual correction  (batch, n_mel_channels, time).
        """
        for i, conv in enumerate(self.convolutions):
            if i < self.n_convolutions - 1:
                x = self.dropout(torch.tanh(conv(x)))
            else:
                x = self.dropout(conv(x))
        return x
