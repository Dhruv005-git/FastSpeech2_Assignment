"""Three-stage preprocessing shared by English/ and Telugu/:
  extract  - select & subsample the HF dataset (~train_hours + val_hours), resample, trim, mel / F0 / energy
  align    - train the unsupervised aligner on the train split, derive phoneme durations (MAS)
  variance - phoneme-level pitch & energy, z-score statistics
"""
import os
import re
import shutil

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm

from . import audio as A
from .aligner import Aligner, beta_binomial_prior, forward_sum_loss, monotonic_align, norm_mel
from .dataset import AlignDataset
from .text import get_frontend, units_to_ids
from .utils import write_json, read_json, seed_everything

TEXT_COLS = ["text", "transcript", "transcription", "sentence", "normalized_text", "verbatim", "raw_text"]
SPK_COLS = ["speaker_id", "speaker", "spk_id", "spk", "client_id", "user_id", "speaker_name"]


def clean_transcript(t, language):
    t = re.sub(r"[\[\(<{][^\]\)>}]*[\]\)>}]", " ", t)           # drop annotation tags like [noise], <laugh>
    t = t.replace("_", " ")
    if language == "english":
        t = re.sub(r"[^A-Za-z0-9',.?!;:\- ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _detect_columns(ds, dcfg):
    from datasets import Audio
    audio_col = next(k for k, v in ds.features.items() if isinstance(v, Audio))
    text_col = dcfg.get("text_column") or next((c for c in TEXT_COLS if c in ds.features), None)
    if text_col is None:
        text_col = next(k for k, v in ds.features.items() if getattr(v, "dtype", "") == "string")
    spk_col = dcfg.get("speaker_column") or next((c for c in SPK_COLS if c in ds.features), None) \
        or ("gender" if "gender" in ds.features else None)
    return audio_col, text_col, spk_col


def snr_proxy(wav, sr):
    """Dynamic-range estimate (p90 - p10 of frame RMS in dB): high = clean, low = noisy / flat."""
    rms = librosa.feature.rms(y=wav, frame_length=int(.025 * sr), hop_length=int(.010 * sr))[0]
    db = 20 * np.log10(rms + 1e-6)
    return float(np.percentile(db, 90) - np.percentile(db, 10))


# --------------------------------------------------------------------------------- stage 1
def stage_extract(cfg):
    from datasets import Audio, concatenate_datasets, load_dataset
    root, a, d = cfg["path"]["preprocessed_path"], cfg["audio"], cfg["dataset"]
    language = cfg["language"]
    os.makedirs(os.path.join(root, "feats"), exist_ok=True)
    symbols, to_units = get_frontend(language)
    token = os.environ.get(d.get("hf_token_env", "HF_TOKEN"))
    print(f"[extract] loading {d['name']} ...")
    dd = load_dataset(d["name"], data_files=d.get("data_files"), token=token)
    ds = concatenate_datasets(list(dd.values())) if hasattr(dd, "values") else dd
    audio_col, text_col, spk_col = _detect_columns(ds, d)
    print(f"[extract] columns: audio={audio_col} text={text_col} speaker={spk_col} | rows={len(ds)}")
    print("[extract] all columns:", list(ds.features))
    ds = ds.cast_column(audio_col, Audio(sampling_rate=a["sampling_rate"]))
    order = np.random.RandomState(d["seed"]).permutation(len(ds))

    target = (d["train_hours"] + d["val_hours"]) * 3600
    got, seen, kept, rejected = 0.0, set(), [], dict(dur=0, snr=0, text=0, dup=0, align=0)
    for i in tqdm(order, desc="select+extract"):
        if got >= target:
            break
        row = ds[int(i)]
        text = clean_transcript(str(row[text_col]), language)
        wav = row[audio_col]["array"].astype(np.float32)
        dur = len(wav) / a["sampling_rate"]
        if not d["min_dur"] <= dur <= d["max_dur"]:
            rejected["dur"] += 1; continue
        units = to_units(text)
        ids = units_to_ids(units, symbols)
        if len(ids) < 3:
            rejected["text"] += 1; continue
        if text.lower() in seen:
            rejected["dup"] += 1; continue
        wav = A.trim_silence(wav, a["sampling_rate"], a["trim_top_db"])
        if snr_proxy(wav, a["sampling_rate"]) < d["min_snr_db"]:
            rejected["snr"] += 1; continue
        wav = wav / max(np.abs(wav).max(), 1e-6) * 0.95
        mel, energy = A.mel_spectrogram(torch.from_numpy(wav), cfg)
        mel, energy = mel.T.numpy(), energy.numpy()
        T = mel.shape[0]
        if T < len(ids) + 2:
            rejected["align"] += 1; continue
        f0 = A.extract_f0(wav, cfg)
        f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T]
        spk = str(row[spk_col]) if spk_col else "spk0"
        uid = f"{language[:2]}_{int(i):06d}"
        np.savez(os.path.join(root, "feats", uid + ".npz"), mel=mel.astype(np.float32), f0=f0, energy=energy.astype(np.float32),
                 ids=np.array(ids, dtype=np.int32))
        seen.add(text.lower())
        kept.append((uid, spk, text, T * a["hop_length"] / a["sampling_rate"]))
        got += kept[-1][3]
    print(f"[extract] kept {len(kept)} utts = {got / 3600:.2f} h | rejected: {rejected}")
    if got < target:
        print(f"[warn] only {got / 3600:.2f} h available after filtering (wanted {target / 3600:.1f} h); "
              f"lower min_snr_db / widen the duration range in config.yaml if you need more.")

    # speaker-aware random split into val (val_hours) and train (rest); texts are unique -> no overlap
    rng = np.random.RandomState(d["seed"])
    idx = rng.permutation(len(kept))
    val, train, vs = [], [], 0.0
    for j in idx:
        if vs < d["val_hours"] * 3600:
            val.append(kept[j]); vs += kept[j][3]
        else:
            train.append(kept[j])
    # every validation speaker must be present in train
    tr_spk = {k[1] for k in train}
    for k in [k for k in val if k[1] not in tr_spk]:
        val.remove(k); train.append(k)
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(root, name + ".txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{u}|{s}|{t}\n" for u, s, t, _ in rows)
    speakers = sorted({k[1] for k in kept})
    write_json(dict(symbols=symbols, speakers=speakers, language=language,
                    train_hours=sum(k[3] for k in train) / 3600, val_hours=sum(k[3] for k in val) / 3600,
                    n_train=len(train), n_val=len(val), rejected=rejected, columns=dict(audio=audio_col, text=text_col, speaker=spk_col)),
               os.path.join(root, "meta.json"))
    print(f"[extract] train {len(train)} utts / {sum(k[3] for k in train) / 3600:.2f} h | "
          f"val {len(val)} utts / {sum(k[3] for k in val) / 3600:.2f} h | speakers: {len(speakers)}")


# --------------------------------------------------------------------------------- stage 2
def stage_align(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, ac = cfg["path"]["preprocessed_path"], cfg["aligner"]
    meta = read_json(os.path.join(root, "meta.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg["dataset"]["seed"])
    model = Aligner(len(meta["symbols"]), cfg["audio"]["n_mels"]).to(dev)
    ckpt = os.path.join(root, "aligner.pt")
    if os.path.exists(ckpt) and not ac.get("retrain", False):
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        print("[align] loaded existing aligner.pt")
    else:
        ds = AlignDataset(root, "train")
        dl = torch.utils.data.DataLoader(ds, batch_size=ac["batch_size"], shuffle=True, num_workers=ac["num_workers"],
                                         collate_fn=AlignDataset.collate, drop_last=True, persistent_workers=ac["num_workers"] > 0)
        opt = torch.optim.Adam(model.parameters(), lr=ac["lr"], weight_decay=1e-6)
        log = []
        for ep in range(1, ac["epochs"] + 1):
            model.train(); tot = 0.0
            for b in tqdm(dl, desc=f"aligner epoch {ep}/{ac['epochs']}", leave=False):
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                logp = model(b["ids"], b["mel"], b["prior"])
                loss = forward_sum_loss(logp, b["n_len"], b["t_len"])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                tot += loss.item()
            log.append(tot / len(dl))
            print(f"[align] epoch {ep} forward-sum loss {log[-1]:.4f}")
            if ep % 10 == 0 or ep == ac["epochs"]:
                torch.save(model.state_dict(), ckpt)
        write_json(log, os.path.join(root, "aligner_loss.json"))
        torch.save(model.state_dict(), ckpt)

    # ------- derive durations for train+val
    model.eval()
    os.makedirs(os.path.join(root, "duration"), exist_ok=True)
    bad, shown = 0, 0
    for split in ("train", "val"):
        for uid in tqdm(AlignDataset(root, split).items, desc=f"durations[{split}]"):
            z = np.load(os.path.join(root, "feats", uid + ".npz"))
            ids, mel = z["ids"].astype(np.int64), z["mel"]
            prior = torch.from_numpy(beta_binomial_prior(len(ids), len(mel)))[None].to(dev)
            with torch.no_grad():
                logp = model(torch.from_numpy(ids)[None].to(dev), torch.from_numpy(norm_mel(mel))[None].to(dev), prior)[0]
            dur = monotonic_align(logp.cpu().numpy())
            if dur.sum() != len(mel) or (dur < 1).any():
                bad += 1; continue
            np.save(os.path.join(root, "duration", uid + ".npy"), dur)
            if split == "train" and shown < 3:
                shown += 1
                fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
                ax[0].imshow(logp.exp().cpu().numpy().T, aspect="auto", origin="lower"); ax[0].set(title=f"soft alignment {uid}", xlabel="mel frame", ylabel="symbol")
                hard = np.zeros((len(ids), len(mel))); s = 0
                for n, dd in enumerate(dur):
                    hard[n, s:s + dd] = 1; s += dd
                ax[1].imshow(hard, aspect="auto", origin="lower"); ax[1].set(title="MAS hard alignment", xlabel="mel frame")
                fig.tight_layout(); fig.savefig(os.path.join(root, f"alignment_{shown}.png"), dpi=110); plt.close(fig)
    print(f"[align] durations written; {bad} utterances dropped (invalid alignment)")
    _drop_missing(root, "duration")


def _drop_missing(root, folder):
    for split in ("train", "val"):
        p = os.path.join(root, split + ".txt")
        rows = [l for l in open(p, encoding="utf-8").read().splitlines() if l]
        keep = [l for l in rows if os.path.exists(os.path.join(root, folder, l.split("|")[0] + ".npy"))]
        open(p, "w", encoding="utf-8").write("\n".join(keep) + "\n")
        print(f"[{split}] {len(keep)}/{len(rows)} utterances have {folder}")


# --------------------------------------------------------------------------------- stage 3
def _interp_f0(f0):
    nz = f0 > 0
    if nz.sum() == 0:
        return f0
    x = np.arange(len(f0))
    return interp1d(x[nz], f0[nz], bounds_error=False, fill_value=(f0[nz][0], f0[nz][-1]))(x)


def _phoneme_avg(v, dur):
    out, s = np.zeros(len(dur), dtype=np.float32), 0
    for i, d in enumerate(dur):
        out[i] = v[s:s + d].mean() if d > 0 else 0.0
        s += d
    return out


def stage_variance(cfg):
    root = cfg["path"]["preprocessed_path"]
    for d in ("pitch", "energy"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    raw = {}
    for split in ("train", "val"):
        for uid in [l.split("|")[0] for l in open(os.path.join(root, split + ".txt"), encoding="utf-8").read().splitlines() if l]:
            z = np.load(os.path.join(root, "feats", uid + ".npz"))
            dur = np.load(os.path.join(root, "duration", uid + ".npy"))
            raw[uid] = (split, _phoneme_avg(_interp_f0(z["f0"]), dur), _phoneme_avg(z["energy"], dur))
    stats = {}
    for k, j in (("pitch", 1), ("energy", 2)):
        tr = np.concatenate([v[j] for v in raw.values() if v[0] == "train"])
        if k == "pitch":
            tr = tr[tr > 0]
        mu, sd = float(tr.mean()), float(tr.std())
        stats[k] = dict(mean=mu, std=sd)
        norm = {u: (v[j] - mu) / sd for u, v in raw.items()}
        allv = np.concatenate(list(norm.values()))
        stats[k]["min"], stats[k]["max"] = float(allv.min()), float(allv.max())
        for u, arr in norm.items():
            np.save(os.path.join(root, k, u + ".npy"), arr.astype(np.float32))
    write_json(stats, os.path.join(root, "stats.json"))
    print("[variance] stats:", stats)


def run_all(cfg, stage="all"):
    if stage in ("all", "extract"):
        stage_extract(cfg)
    if stage in ("all", "align"):
        stage_align(cfg)
    if stage in ("all", "variance"):
        stage_variance(cfg)
