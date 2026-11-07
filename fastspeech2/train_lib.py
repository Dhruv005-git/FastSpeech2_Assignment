import csv
import glob
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TTSDataset
from .model import FastSpeech2, fs2_loss
from .utils import plot_losses, read_json, seed_everything

KEYS = ["total", "mel", "postnet", "pitch", "energy", "duration"]


def build_model(cfg, root):
    meta, stats = read_json(os.path.join(root, "meta.json")), read_json(os.path.join(root, "stats.json"))
    return FastSpeech2(cfg, len(meta["symbols"]), len(meta["speakers"]), stats), meta, stats


def noam(step, d, warm, anneal_steps, anneal_rate):
    lr = d ** -0.5 * min(max(step, 1) ** -0.5, max(step, 1) * warm ** -1.5)
    for s in anneal_steps:
        if step > s:
            lr *= anneal_rate
    return lr


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def validate(model, dl, dev, amp):
    model.eval()
    acc, n = {k: 0.0 for k in KEYS}, 0
    for b in dl:
        b = to_dev(b, dev)
        with torch.autocast("cuda", enabled=amp):
            out = model(b["ids"], b["src_len"], b["spk"], b["mel_len"], b["mel"].size(1), b["pitch"], b["energy"], b["dur"])
        l = fs2_loss({k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}, b)
        bs = b["ids"].size(0)
        for k in KEYS:
            acc[k] += l[k].item() * bs
        n += bs
    model.train()
    return {k: v / n for k, v in acc.items()}


def train(cfg):
    tc, oc = cfg["train"], cfg["optimizer"]
    root, ck_dir = cfg["path"]["preprocessed_path"], cfg["path"]["checkpoint_path"]
    res = cfg["path"]["results_path"]
    log_dir = os.path.join(res, "logs")
    for d in (ck_dir, res, log_dir):
        os.makedirs(d, exist_ok=True)
    seed_everything(tc["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = tc["amp"] and dev == "cuda"
    model, meta, stats = build_model(cfg, root)
    model.to(dev).train()
    print(f"[train] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | device {dev} | amp {amp}")
    tr = TTSDataset(root, "train", meta["speakers"]); va = TTSDataset(root, "val", meta["speakers"])
    tl = DataLoader(tr, tc["batch_size"], shuffle=True, collate_fn=TTSDataset.collate, num_workers=tc["num_workers"], drop_last=True,
                    persistent_workers=tc["num_workers"] > 0)
    vl = DataLoader(va, tc["batch_size"], shuffle=False, collate_fn=TTSDataset.collate, num_workers=tc["num_workers"])
    opt = torch.optim.Adam(model.parameters(), betas=tuple(oc["betas"]), eps=oc["eps"], weight_decay=oc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    d_model = cfg["model"]["transformer"]["encoder_hidden"]

    step, best = 0, float("inf")
    last = os.path.join(ck_dir, "last.pt")
    if os.path.exists(last):                                   # resume (Colab sessions get disconnected!)
        s = torch.load(last, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        step, best = s["step"], s["best"]
        print(f"[train] resumed from step {step}")
    tcsv, vcsv = os.path.join(log_dir, "train_log.csv"), os.path.join(log_dir, "val_log.csv")
    for p in (tcsv, vcsv):
        if not os.path.exists(p):
            csv.writer(open(p, "w", newline="")).writerow(["step"] + KEYS)

    def save(path):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, best=best,
                        cfg=cfg, symbols=meta["symbols"], speakers=meta["speakers"], stats=stats), path)

    run, nrun, t0 = {k: 0.0 for k in KEYS}, 0, time.time()
    bar = tqdm(total=tc["total_step"], initial=step, desc="train")
    while step < tc["total_step"]:
        for b in tl:
            b = to_dev(b, dev)
            for g in opt.param_groups:
                g["lr"] = noam(step + 1, d_model, oc["warm_up_step"], oc["anneal_steps"], oc["anneal_rate"]) * oc["lr_scale"]
            with torch.autocast("cuda", enabled=amp):
                out = model(b["ids"], b["src_len"], b["spk"], b["mel_len"], b["mel"].size(1), b["pitch"], b["energy"], b["dur"])
            l = fs2_loss({k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip_thresh"])
            scaler.step(opt); scaler.update()
            step += 1; bar.update(1)
            for k in KEYS:
                run[k] += l[k].item()
            nrun += 1
            if step % tc["log_step"] == 0:
                row = [step] + [run[k] / nrun for k in KEYS]
                csv.writer(open(tcsv, "a", newline="")).writerow(row)
                bar.set_postfix(loss=f"{row[1]:.3f}", mel=f"{row[2]:.3f}")
                run, nrun = {k: 0.0 for k in KEYS}, 0
            if step % tc["val_step"] == 0 or step == tc["total_step"]:
                v = validate(model, vl, dev, amp)
                csv.writer(open(vcsv, "a", newline="")).writerow([step] + [v[k] for k in KEYS])
                print(f"\n[val] step {step}: " + " ".join(f"{k}={v[k]:.4f}" for k in KEYS))
                if v["total"] < best:
                    best = v["total"]; save(os.path.join(ck_dir, "best.pt"))
                save(last)
                plot_losses(tcsv, os.path.join(res, "training_loss.png"), cfg["language"].title() + " training")
                plot_losses(vcsv, os.path.join(res, "validation_loss.png"), cfg["language"].title() + " validation")
            if step % tc["save_step"] == 0:
                save(os.path.join(ck_dir, f"step_{step}.pt"))
                for old in sorted(glob.glob(os.path.join(ck_dir, "step_*.pt")), key=os.path.getmtime)[:-tc["keep_checkpoints"]]:
                    os.remove(old)
            if step >= tc["total_step"]:
                break
    save(last)
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min | best val loss {best:.4f}")
