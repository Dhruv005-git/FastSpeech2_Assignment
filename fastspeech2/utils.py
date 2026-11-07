import json
import os
import random
import urllib.request

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for k in ("preprocessed_path", "checkpoint_path", "results_path"):
        cfg["path"][k] = os.path.normpath(os.path.join(base, cfg["path"][k]))
    return cfg


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def setup_font(language):
    """Make matplotlib able to draw Telugu glyphs (DejaVu has none). Returns the font name or None."""
    import matplotlib
    from matplotlib import font_manager as fm
    if language != "telugu":
        return None
    wanted = ["Noto Sans Telugu", "Nirmala UI", "Gautami", "Lohit Telugu", "Pothana2000", "Vani", "Noto Serif Telugu"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for w in wanted:
        if w in installed:
            matplotlib.rcParams["font.family"] = w
            return w
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    p = os.path.join(d, "NotoSansTelugu-Regular.ttf")
    if not os.path.exists(p):
        os.makedirs(d, exist_ok=True)
        try:
            urllib.request.urlretrieve("https://github.com/notofonts/telugu/raw/main/fonts/NotoSansTelugu/full/ttf/NotoSansTelugu-Regular.ttf", p)
        except Exception as e:                                              # noqa
            print("[warn] could not download a Telugu font; plot labels may show boxes:", e)
            return None
    fm.fontManager.addfont(p)
    name = fm.FontProperties(fname=p).get_name()
    matplotlib.rcParams["font.family"] = name
    return name


def plot_losses(csv_path, out_png, title, keys=("total", "mel", "postnet", "pitch", "energy", "duration")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    steps = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(steps, [float(r["total"]) for r in rows], "k", lw=1.6)
    ax[0].set(title=f"{title} - total loss", xlabel="step", ylabel="loss")
    for k in keys[1:]:
        ax[1].plot(steps, [float(r[k]) for r in rows], label=k, lw=1.2)
    ax[1].set(title=f"{title} - components", xlabel="step", yscale="log")
    ax[1].legend(ncol=2, fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
