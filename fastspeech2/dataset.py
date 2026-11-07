import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .aligner import beta_binomial_prior, norm_mel


def _pad(arrs, value=0.0):
    n = max(a.shape[0] for a in arrs)
    out = np.full((len(arrs), n) + arrs[0].shape[1:], value, dtype=arrs[0].dtype)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = a
    return torch.from_numpy(out)


class AlignDataset(Dataset):
    """(units, mel, prior) for training the aligner."""

    def __init__(self, root, split):
        self.root = root
        self.items = [l.split("|")[0] for l in open(os.path.join(root, f"{split}.txt"), encoding="utf-8").read().splitlines() if l]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        z = np.load(os.path.join(self.root, "feats", self.items[i] + ".npz"))
        ids, mel = z["ids"].astype(np.int64), z["mel"]
        return dict(id=self.items[i], ids=ids, mel=norm_mel(mel).astype(np.float32),
                    prior=beta_binomial_prior(len(ids), len(mel)))

    @staticmethod
    def collate(b):
        T, N = max(len(x["mel"]) for x in b), max(len(x["ids"]) for x in b)
        prior = np.full((len(b), T, N), 1e-8, dtype=np.float32)
        for i, x in enumerate(b):
            prior[i, : x["prior"].shape[0], : x["prior"].shape[1]] = x["prior"]
        return dict(id=[x["id"] for x in b], ids=_pad([x["ids"] for x in b]), mel=_pad([x["mel"] for x in b]),
                    prior=torch.from_numpy(prior), n_len=torch.tensor([len(x["ids"]) for x in b]),
                    t_len=torch.tensor([len(x["mel"]) for x in b]))


class TTSDataset(Dataset):
    def __init__(self, root, split, speakers):
        self.root = root
        self.rows = [l.split("|") for l in open(os.path.join(root, f"{split}.txt"), encoding="utf-8").read().splitlines() if l]
        self.spk = {s: i for i, s in enumerate(speakers)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        uid, spk = self.rows[i][0], self.rows[i][1]
        z = np.load(os.path.join(self.root, "feats", uid + ".npz"))
        ld = lambda d: np.load(os.path.join(self.root, d, uid + ".npy"))
        return dict(id=uid, spk=self.spk[spk], ids=z["ids"].astype(np.int64), mel=z["mel"],
                    dur=ld("duration").astype(np.int64), pitch=ld("pitch"), energy=ld("energy"))

    @staticmethod
    def collate(b):
        return dict(id=[x["id"] for x in b], spk=torch.tensor([x["spk"] for x in b]),
                    ids=_pad([x["ids"] for x in b]), src_len=torch.tensor([len(x["ids"]) for x in b]),
                    mel=_pad([x["mel"] for x in b]), mel_len=torch.tensor([len(x["mel"]) for x in b]),
                    dur=_pad([x["dur"] for x in b]), pitch=_pad([x["pitch"] for x in b]),
                    energy=_pad([x["energy"] for x in b]))
