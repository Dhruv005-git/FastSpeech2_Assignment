"""FastSpeech 2 (Ren et al., 2020): FFT encoder -> variance adaptor (duration / pitch / energy)
-> FFT mel decoder -> PostNet.  Architecture/hyper-parameters follow ming024/FastSpeech2."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(lengths, max_len=None):
    max_len = int(max_len or lengths.max())
    return torch.arange(max_len, device=lengths.device)[None] >= lengths[:, None]       # True = pad


def sinusoid(n_pos, d):
    pos = torch.arange(n_pos).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n_pos, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class FFTBlock(nn.Module):
    def __init__(self, d, heads, d_ff, kernels, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.conv1 = nn.Conv1d(d, d_ff, kernels[0], padding=(kernels[0] - 1) // 2)
        self.conv2 = nn.Conv1d(d_ff, d, kernels[1], padding=(kernels[1] - 1) // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.ln1(x + self.drop(a)).masked_fill(pad_mask[..., None], 0)
        h = self.conv2(F.relu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        return self.ln2(x + self.drop(h)).masked_fill(pad_mask[..., None], 0)


class VariancePredictor(nn.Module):
    def __init__(self, d, filt, k, dropout):
        super().__init__()
        self.c1 = nn.Conv1d(d, filt, k, padding=(k - 1) // 2)
        self.c2 = nn.Conv1d(filt, filt, k, padding=(k - 1) // 2)
        self.n1, self.n2 = nn.LayerNorm(filt), nn.LayerNorm(filt)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filt, 1)

    def forward(self, x, pad_mask):
        h = self.drop(self.n1(F.relu(self.c1(x.transpose(1, 2))).transpose(1, 2)))
        h = self.drop(self.n2(F.relu(self.c2(h.transpose(1, 2))).transpose(1, 2)))
        return self.out(h).squeeze(-1).masked_fill(pad_mask, 0)


def length_regulate(x, dur, max_len=None):
    """x: (B,N,D), dur: (B,N) int -> (B,T,D), mel_len (B,)"""
    outs = [torch.repeat_interleave(x[b], dur[b].clamp(min=0), dim=0) for b in range(x.size(0))]
    lens = torch.tensor([o.size(0) for o in outs], device=x.device)
    T = max_len or int(lens.max().clamp(min=1))
    outs = [F.pad(o[:T], (0, 0, 0, T - min(o.size(0), T))) for o in outs]
    return torch.stack(outs), lens.clamp(max=T)


class VarianceAdaptor(nn.Module):
    def __init__(self, c, stats):
        super().__init__()
        d, v = c["transformer"]["encoder_hidden"], c["variance_predictor"]
        mk = lambda: VariancePredictor(d, v["filter_size"], v["kernel_size"], v["dropout"])
        self.duration_predictor, self.pitch_predictor, self.energy_predictor = mk(), mk(), mk()
        nb = v["n_bins"]
        # bins live on the *normalised* (z-score) scale, range taken from training-set statistics
        self.register_buffer("pitch_bins", torch.linspace(stats["pitch"]["min"], stats["pitch"]["max"], nb - 1))
        self.register_buffer("energy_bins", torch.linspace(stats["energy"]["min"], stats["energy"]["max"], nb - 1))
        self.pitch_emb, self.energy_emb = nn.Embedding(nb, d), nn.Embedding(nb, d)

    def forward(self, x, src_mask, mel_len=None, max_mel_len=None, p_t=None, e_t=None, d_t=None,
                p_ctrl=1.0, e_ctrl=1.0, d_ctrl=1.0):
        log_d = self.duration_predictor(x, src_mask)
        p = self.pitch_predictor(x, src_mask)
        x = x + self.pitch_emb(torch.bucketize(p_t if p_t is not None else p * p_ctrl, self.pitch_bins))
        e = self.energy_predictor(x, src_mask)
        x = x + self.energy_emb(torch.bucketize(e_t if e_t is not None else e * e_ctrl, self.energy_bins))
        if d_t is not None:
            dur = d_t
        else:
            dur = torch.clamp(torch.round((torch.exp(log_d) - 1) * d_ctrl), min=0).long().masked_fill(src_mask, 0)
        x, mel_len = length_regulate(x, dur, max_mel_len)
        return x, p, e, log_d, dur, mel_len


class PostNet(nn.Module):
    def __init__(self, n_mel=80, d=512, k=5, n=5):
        super().__init__()
        ch = [n_mel] + [d] * (n - 1) + [n_mel]
        self.convs = nn.ModuleList()
        for i in range(n):
            layers = [nn.Conv1d(ch[i], ch[i + 1], k, padding=(k - 1) // 2), nn.BatchNorm1d(ch[i + 1])]
            self.convs.append(nn.Sequential(*layers))

    def forward(self, x):
        x = x.transpose(1, 2)
        for i, c in enumerate(self.convs):
            x = c(x)
            if i < len(self.convs) - 1:
                x = torch.tanh(x)
            x = F.dropout(x, 0.5, self.training)
        return x.transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(self, cfg, n_symbols, n_speakers, stats):
        super().__init__()
        c = cfg["model"]
        t = c["transformer"]
        d = t["encoder_hidden"]
        self.emb = nn.Embedding(n_symbols, d, padding_idx=0)
        self.register_buffer("pe", sinusoid(4096, d), persistent=False)
        mk = lambda n: nn.ModuleList([FFTBlock(d, t["n_head"], t["conv_filter_size"], t["conv_kernel_size"], t["dropout"]) for _ in range(n)])
        self.encoder, self.decoder = mk(t["encoder_layer"]), mk(t["decoder_layer"])
        self.variance_adaptor = VarianceAdaptor(c, stats)
        self.n_mel = cfg["audio"]["n_mels"]
        self.mel_linear = nn.Linear(t["decoder_hidden"], self.n_mel)
        self.postnet = PostNet(self.n_mel)
        self.speaker_emb = nn.Embedding(n_speakers, d) if n_speakers > 1 else None
        self.scale = d ** 0.5

    def forward(self, ids, src_len, spk=None, mel_len=None, max_mel_len=None, p_t=None, e_t=None, d_t=None,
                p_ctrl=1.0, e_ctrl=1.0, d_ctrl=1.0):
        src_mask = get_mask(src_len, ids.size(1))
        x = self.emb(ids) * self.scale + self.pe[: ids.size(1)]
        for l in self.encoder:
            x = l(x, src_mask)
        if self.speaker_emb is not None:
            x = x + self.speaker_emb(spk)[:, None]
        x, p, e, log_d, dur, mel_len = self.variance_adaptor(x, src_mask, mel_len, max_mel_len, p_t, e_t, d_t,
                                                              p_ctrl, e_ctrl, d_ctrl)
        mel_mask = get_mask(mel_len, x.size(1))
        x = (x + self.pe[: x.size(1)]).masked_fill(mel_mask[..., None], 0)
        for l in self.decoder:
            x = l(x, mel_mask)
        mel = self.mel_linear(x)
        post = (mel + self.postnet(mel)).masked_fill(mel_mask[..., None], 0)
        return dict(mel=mel.masked_fill(mel_mask[..., None], 0), post=post, pitch=p, energy=e, log_dur=log_d,
                    dur=dur, src_mask=src_mask, mel_mask=mel_mask, mel_len=mel_len)


def fs2_loss(out, batch):
    sm, mm = ~out["src_mask"], ~out["mel_mask"]
    T = min(out["mel"].size(1), batch["mel"].size(1))
    mm = mm[:, :T]
    m = mm[..., None]
    tgt = batch["mel"][:, :T]
    mel_l = F.l1_loss(out["mel"][:, :T].masked_select(m), tgt.masked_select(m))
    post_l = F.l1_loss(out["post"][:, :T].masked_select(m), tgt.masked_select(m))
    pitch_l = F.mse_loss(out["pitch"].masked_select(sm), batch["pitch"].masked_select(sm))
    energy_l = F.mse_loss(out["energy"].masked_select(sm), batch["energy"].masked_select(sm))
    dur_l = F.mse_loss(out["log_dur"].masked_select(sm), torch.log(batch["dur"].float() + 1).masked_select(sm))
    total = mel_l + post_l + pitch_l + energy_l + dur_l
    return dict(total=total, mel=mel_l, postnet=post_l, pitch=pitch_l, energy=energy_l, duration=dur_l)
