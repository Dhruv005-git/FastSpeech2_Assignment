"""Unsupervised text-mel aligner used to obtain phoneme durations (replaces Montreal Forced Aligner).

Why: the reference repo needs MFA + a pretrained acoustic model/lexicon per language. There is no
pretrained MFA model for Telugu and MFA is painful to install on Colab, so we learn the alignment
from the data itself (RAD-TTS / FastPitch "alignment learning framework", Badlani et al. 2021):
  * a small text encoder and mel encoder produce a soft-alignment matrix
  * it is trained with the CTC forward-sum loss + a beta-binomial diagonal prior
  * Monotonic Alignment Search (Viterbi, as in Glow-TTS) turns it into integer durations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from scipy.stats import betabinom


class Aligner(nn.Module):
    def __init__(self, n_symbols, n_mels=80, d_text=256, d_att=80, temperature=0.0005):
        super().__init__()
        self.emb = nn.Embedding(n_symbols, d_text, padding_idx=0)
        self.key = nn.Sequential(nn.Conv1d(d_text, 2 * d_text, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(2 * d_text, d_att, 1))
        self.query = nn.Sequential(nn.Conv1d(n_mels, 2 * n_mels, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(2 * n_mels, n_mels, 1), nn.ReLU(),
                                   nn.Conv1d(n_mels, d_att, 1))
        self.temperature = temperature

    def forward(self, ids, mel, prior=None):
        """ids (B,N); mel (B,T,80) (roughly standardised). Returns log-attention (B,T,N)."""
        k = self.key(self.emb(ids).transpose(1, 2))                  # (B,D,N)
        q = self.query(mel.transpose(1, 2))                           # (B,D,T)
        dist = (q[:, :, :, None] - k[:, :, None, :]).pow(2).sum(1)   # (B,T,N)
        logp = F.log_softmax(-self.temperature * dist, dim=2)
        if prior is not None:
            logp = logp + torch.log(prior + 1e-8)
        return logp


def norm_mel(mel):
    return (mel + 5.0) / 3.0


def beta_binomial_prior(n_text, n_mel, scale=1.0):
    x = np.arange(n_text)
    rows = [betabinom(n_text - 1, scale * i, scale * (n_mel + 1 - i)).pmf(x) for i in range(1, n_mel + 1)]
    return np.array(rows, dtype=np.float32)                          # (T, N)


def forward_sum_loss(logp, n_len, t_len, blank_logprob=-1.0):
    padded = F.pad(logp, (1, 0), value=blank_logprob)                # (B,T,N+1), blank = index 0
    total = 0.0
    for b in range(logp.size(0)):
        lp = padded[b, : t_len[b], : n_len[b] + 1]
        lp = F.log_softmax(lp, dim=-1)[:, None]                       # (T,1,N+1)
        tgt = torch.arange(1, n_len[b] + 1, device=logp.device)[None]
        total = total + F.ctc_loss(lp, tgt, t_len[b:b + 1], n_len[b:b + 1], blank=0, reduction="sum", zero_infinity=True)
    return total / logp.size(0) / float(t_len.float().mean())


@njit(cache=True)
def _mas(logp):
    T, N = logp.shape
    Q = np.full((T, N), -1e18)
    back = np.zeros((T, N), dtype=np.int8)
    Q[0, 0] = logp[0, 0]
    for t in range(1, T):
        for n in range(min(t + 1, N)):
            stay = Q[t - 1, n]
            adv = Q[t - 1, n - 1] if n > 0 else -1e18
            if adv > stay:
                Q[t, n], back[t, n] = adv + logp[t, n], 1
            else:
                Q[t, n] = stay + logp[t, n]
    dur = np.zeros(N, dtype=np.int64)
    n = N - 1
    for t in range(T - 1, -1, -1):
        dur[n] += 1
        if back[t, n] == 1:
            n -= 1
    return dur


def monotonic_align(logp):
    """logp (T,N) numpy -> integer durations (N,) that sum to T (every symbol >= 1 frame; needs T >= N)."""
    return _mas(np.ascontiguousarray(logp, dtype=np.float64))
