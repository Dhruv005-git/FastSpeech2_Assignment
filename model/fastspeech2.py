"""
fastspeech2.py

Top-level FastSpeech 2 model (Ren et al., 2021).

References:
    - https://arxiv.org/abs/2006.04558
    - https://github.com/ming024/FastSpeech2
"""

import torch
import torch.nn as nn
import numpy as np

from .modules import (
    FFTBlock,
    PostNet,
    get_sinusoid_encoding_table,
)
from .variance_adaptor import VarianceAdaptor


class FastSpeech2(nn.Module):
    """
    FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.

    The model consists of:
      1. **Encoder** — phoneme embedding + positional encoding + N FFT blocks.
      2. **Variance Adaptor** — duration / pitch / energy conditioning.
      3. **Decoder** — positional encoding + N FFT blocks.
      4. **mel_linear** — linear projection from hidden to mel channels.
      5. **PostNet** — 5-layer convolutional refinement network (residual).

    Args:
        config (dict): Full model configuration.  Expected keys (with examples):

            Vocabulary & embedding
            ----------------------
            vocab_size               (int)  e.g. 200
            encoder_hidden           (int)  e.g. 256

            Encoder / Decoder shared
            ------------------------
            encoder_layer            (int)  e.g. 4
            encoder_head             (int)  e.g. 2
            encoder_hidden           (int)  e.g. 256
            encoder_conv1d_filter_size (int) e.g. 1024
            encoder_conv1d_kernel_size (list[int]) e.g. [9, 1]
            encoder_dropout          (float) e.g. 0.2

            decoder_layer            (int)  e.g. 4
            decoder_head             (int)  e.g. 2
            decoder_hidden           (int)  e.g. 256
            decoder_conv1d_filter_size (int) e.g. 1024
            decoder_conv1d_kernel_size (list[int]) e.g. [9, 1]
            decoder_dropout          (float) e.g. 0.2

            Variance Adaptor
            ----------------
            variance_predictor_filter_size (int) e.g. 256
            variance_predictor_kernel_size (int) e.g. 3
            variance_predictor_dropout (float)   e.g. 0.5
            pitch_min, pitch_max     (float)
            energy_min, energy_max   (float)
            n_bins                   (int)  e.g. 256

            Mel / PostNet
            -------------
            n_mel_channels           (int)  e.g. 80
            postnet_embedding_dim    (int)  e.g. 512
            postnet_kernel_size      (int)  e.g. 5
            postnet_n_convolutions   (int)  e.g. 5

            Misc
            ----
            max_seq_len              (int)  e.g. 1000
            pad_idx                  (int)  e.g. 0
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

        # ------------------------------------------------------------------ #
        # Encoder                                                              #
        # ------------------------------------------------------------------ #
        vocab_size: int = config["vocab_size"]
        encoder_hidden: int = config["encoder_hidden"]
        max_seq_len: int = config.get("max_seq_len", 1000)
        pad_idx: int = config.get("pad_idx", 0)

        self.src_word_emb = nn.Embedding(vocab_size, encoder_hidden, padding_idx=pad_idx)

        # Sinusoidal positional encoding — not learnt, registered as buffer
        self.register_buffer(
            "enc_pos_table",
            torch.FloatTensor(
                get_sinusoid_encoding_table(max_seq_len + 1, encoder_hidden)
            ).unsqueeze(0),  # (1, max_seq_len+1, encoder_hidden)
        )

        self.encoder = nn.ModuleList(
            [
                FFTBlock(
                    d_model=encoder_hidden,
                    n_head=config["encoder_head"],
                    d_k=encoder_hidden // config["encoder_head"],
                    d_v=encoder_hidden // config["encoder_head"],
                    d_inner=config["encoder_conv1d_filter_size"],
                    kernel_size=config.get("encoder_conv1d_kernel_size", [9, 1]),
                    dropout=config["encoder_dropout"],
                )
                for _ in range(config["encoder_layer"])
            ]
        )

        # ------------------------------------------------------------------ #
        # Variance Adaptor                                                     #
        # ------------------------------------------------------------------ #
        self.variance_adaptor = VarianceAdaptor(config)

        # ------------------------------------------------------------------ #
        # Decoder                                                              #
        # ------------------------------------------------------------------ #
        decoder_hidden: int = config["decoder_hidden"]

        self.register_buffer(
            "dec_pos_table",
            torch.FloatTensor(
                get_sinusoid_encoding_table(max_seq_len + 1, decoder_hidden)
            ).unsqueeze(0),  # (1, max_seq_len+1, decoder_hidden)
        )

        self.decoder = nn.ModuleList(
            [
                FFTBlock(
                    d_model=decoder_hidden,
                    n_head=config["decoder_head"],
                    d_k=decoder_hidden // config["decoder_head"],
                    d_v=decoder_hidden // config["decoder_head"],
                    d_inner=config["decoder_conv1d_filter_size"],
                    kernel_size=config.get("decoder_conv1d_kernel_size", [9, 1]),
                    dropout=config["decoder_dropout"],
                )
                for _ in range(config["decoder_layer"])
            ]
        )

        # ------------------------------------------------------------------ #
        # Output projections                                                   #
        # ------------------------------------------------------------------ #
        n_mel_channels: int = config["n_mel_channels"]

        self.mel_linear = nn.Linear(decoder_hidden, n_mel_channels)

        self.postnet = PostNet(
            n_mel_channels=n_mel_channels,
            postnet_embedding_dim=config.get("postnet_embedding_dim", 512),
            postnet_kernel_size=config.get("postnet_kernel_size", 5),
            postnet_n_convolutions=config.get("postnet_n_convolutions", 5),
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def get_mask_from_lengths(lengths: torch.Tensor, max_len: int = None) -> torch.BoolTensor:
        """
        Create a boolean padding mask from a lengths tensor.

        Args:
            lengths (LongTensor): (batch,) — valid length per sample.
            max_len (int | None): If None, uses ``lengths.max()``.

        Returns:
            BoolTensor: (batch, max_len) — True at padded (invalid) positions.
        """
        batch_size = lengths.size(0)
        if max_len is None:
            max_len = int(lengths.max().item())
        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        return ids >= lengths.unsqueeze(1)

    def _add_positional_encoding(
        self,
        x: torch.Tensor,
        pos_table: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Add sinusoidal positional encoding and zero-out padded positions.

        Args:
            x         (Tensor):     (batch, seq_len, hidden)
            pos_table (Tensor):     (1, max_seq_len+1, hidden)
            mask (BoolTensor|None): (batch, seq_len) — True for padding.

        Returns:
            Tensor: (batch, seq_len, hidden)
        """
        seq_len = x.size(1)
        x = x + pos_table[:, :seq_len, :].to(x.device)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x

    # ---------------------------------------------------------------------- #
    # Encoder sub-forward                                                      #
    # ---------------------------------------------------------------------- #

    def _encode(self, src_seq: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder stack.

        Args:
            src_seq  (LongTensor): (batch, src_len) — phoneme indices.
            src_mask (BoolTensor): (batch, src_len) — True for padding.

        Returns:
            Tensor: (batch, src_len, encoder_hidden)
        """
        # Key-padding mask for self-attention: (batch, 1, 1, src_len)
        slf_attn_mask = src_mask.unsqueeze(1).unsqueeze(1)

        output = self.src_word_emb(src_seq)
        output = self._add_positional_encoding(output, self.enc_pos_table, src_mask)

        for layer in self.encoder:
            output, _ = layer(output, mask=src_mask, slf_attn_mask=slf_attn_mask)

        return output

    # ---------------------------------------------------------------------- #
    # Decoder sub-forward                                                      #
    # ---------------------------------------------------------------------- #

    def _decode(self, x: torch.Tensor, mel_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the decoder stack.

        Args:
            x        (Tensor):     (batch, mel_len, decoder_hidden)
            mel_mask (BoolTensor): (batch, mel_len) — True for padding.

        Returns:
            Tensor: (batch, mel_len, decoder_hidden)
        """
        slf_attn_mask = mel_mask.unsqueeze(1).unsqueeze(1)

        output = self._add_positional_encoding(x, self.dec_pos_table, mel_mask)

        for layer in self.decoder:
            output, _ = layer(output, mask=mel_mask, slf_attn_mask=slf_attn_mask)

        return output

    # ---------------------------------------------------------------------- #
    # Main forward                                                             #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        src_seq: torch.Tensor,
        src_len: torch.Tensor,
        mel_len: torch.Tensor = None,
        pitch_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
        duration_target: torch.Tensor = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ):
        """
        Full FastSpeech 2 forward pass.

        Args:
            src_seq (LongTensor):         (batch, src_len) — phoneme index sequence.
            src_len (LongTensor):         (batch,) — valid phoneme lengths.
            mel_len (LongTensor | None):  (batch,) — GT mel lengths (training).
            pitch_target (Tensor | None): (batch, mel_len) — GT pitch values.
            energy_target (Tensor| None): (batch, mel_len) — GT energy values.
            duration_target (LongTensor | None): (batch, src_len) — GT durations.
            p_control (float):            Pitch scaling factor.
            e_control (float):            Energy scaling factor.
            d_control (float):            Duration scaling factor.

        Returns:
            Tuple:
                mel_output          (batch, mel_len, n_mel_channels) — pre-PostNet mel.
                mel_postnet_output  (batch, mel_len, n_mel_channels) — post-PostNet mel.
                p_pred              (batch, mel_len) — pitch predictions.
                e_pred              (batch, mel_len) — energy predictions.
                log_d_pred          (batch, src_len) — log-duration predictions.
                duration_rounded    (LongTensor) (batch, src_len) — integer durations used.
                src_mask            (BoolTensor) (batch, src_len).
                mel_mask            (BoolTensor) (batch, mel_len).
                mel_len             (LongTensor) (batch,).
        """
        # ---- Masks ----------------------------------------------------------
        src_mask = self.get_mask_from_lengths(src_len)
        mel_mask = (
            self.get_mask_from_lengths(mel_len)
            if mel_len is not None
            else None
        )
        max_mel_len = int(mel_len.max().item()) if mel_len is not None else None

        # ---- Encoder --------------------------------------------------------
        encoder_output = self._encode(src_seq, src_mask)

        # ---- Variance Adaptor -----------------------------------------------
        (
            va_output,
            p_pred,
            e_pred,
            log_d_pred,
            duration_rounded,
            mel_len_out,
            mel_mask,
        ) = self.variance_adaptor(
            encoder_output,
            src_mask,
            mel_mask=mel_mask,
            max_len=max_mel_len,
            pitch_target=pitch_target,
            energy_target=energy_target,
            duration_target=duration_target,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )

        # ---- Decoder --------------------------------------------------------
        decoder_output = self._decode(va_output, mel_mask)

        # ---- Mel linear + PostNet -------------------------------------------
        mel_output = self.mel_linear(decoder_output)        # (batch, mel_len, n_mel)

        # PostNet takes (batch, n_mel, mel_len)
        postnet_input = mel_output.transpose(1, 2)
        postnet_residual = self.postnet(postnet_input)
        mel_postnet_output = (postnet_input + postnet_residual).transpose(1, 2)

        # Zero out padded positions
        if mel_mask is not None:
            mel_output = mel_output.masked_fill(mel_mask.unsqueeze(-1), 0.0)
            mel_postnet_output = mel_postnet_output.masked_fill(mel_mask.unsqueeze(-1), 0.0)

        return (
            mel_output,
            mel_postnet_output,
            p_pred,
            e_pred,
            log_d_pred,
            duration_rounded,
            src_mask,
            mel_mask,
            mel_len_out,
        )
