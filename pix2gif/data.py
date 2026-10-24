"""TGIF motion-pair dataset: frame pairs labelled with their mean Lucas-Kanade optical flow."""
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageSequence
from torch.utils.data import Dataset

from .model import motion_prompt

TSV_URL = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv"
SPLIT_FILE = "pairs.json"


def flow_magnitude(f0, f1, eps=0.5):
    """Grayscale -> Shi-Tomasi corners -> Lucas-Kanade tracking -> mean displacement, ignoring tiny moves."""
    g0, g1 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY), cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None or len(p0) < 5:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(15, 15), maxLevel=2,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if p1 is None:
        return None
    ok = st.reshape(-1) == 1
    if ok.sum() < 5:
        return None
    mag = np.linalg.norm((p1 - p0).reshape(-1, 2)[ok], axis=1)
    mag = mag[mag > eps]
    return float(mag.mean()) if len(mag) else 0.0


def _gif_frames(data, res, max_frames=96):
    frames = []
    for i, f in enumerate(ImageSequence.Iterator(Image.open(io.BytesIO(data)))):
        if i >= max_frames:
            break
        frames.append(np.asarray(f.convert("RGB").resize((res, res), Image.Resampling.LANCZOS)))
    return frames


def _process_gif(cfg, gid, url, caption):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        frames = _gif_frames(r.content, cfg.res)
    except Exception:
        return []
    n = len(frames)
    if n < 2:
        return []
    cands = [(i, j) for i in range(n) for j in range(i + 1, n)]  # forward pairs only
    random.Random(gid).shuffle(cands)
    kept = []
    for i, j in cands[:80]:
        m = flow_magnitude(frames[i], frames[j], cfg.flow_eps)
        if m is not None and cfg.motion_range[0] <= m <= cfg.motion_range[1]:
            kept.append((i, j, m))
        if len(kept) >= cfg.pairs_per_gif:
            break
    rows = []
    for i, j, m in kept:
        paths = []
        for k in (i, j):
            p = f"{cfg.data_dir}/frames/{gid:06d}_{k:03d}.jpg"
            if not os.path.exists(p):
                Image.fromarray(frames[k]).save(p, quality=95)
            paths.append(p)
        rows.append([paths[0], paths[1], caption.strip(), round(m, 3), gid])
    return rows


def build_dataset(cfg, workers=32):
    """Download cfg.n_gifs TGIF GIFs, mine motion pairs, split train/val by GIF. Returns {"train": [...], "val": [...]},
    rows are [source_path, target_path, caption, flow]."""
    os.makedirs(f"{cfg.data_dir}/frames", exist_ok=True)
    rows = [l.split("\t") for l in requests.get(TSV_URL, timeout=60).text.strip().split("\n")]
    random.Random(cfg.seed).shuffle(rows)
    jobs = [(i, u, c) for i, (u, c) in enumerate(rows[:cfg.n_gifs])]
    pairs, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        for k, res in enumerate(ex.map(lambda j: _process_gif(cfg, *j), jobs), 1):
            pairs += res
            if k % 250 == 0:
                print(f"  {k}/{len(jobs)} GIFs · {len(pairs)} pairs · {time.time() - t0:.0f}s")
    gids = sorted({p[4] for p in pairs})
    val = set(random.Random(cfg.seed).sample(gids, max(1, int(len(gids) * cfg.val_fraction))))
    split = {"train": [p[:4] for p in pairs if p[4] not in val], "val": [p[:4] for p in pairs if p[4] in val]}
    with open(f"{cfg.data_dir}/{SPLIT_FILE}", "w") as f:
        json.dump(split, f)
    return split


def load_or_build(cfg):
    path = f"{cfg.data_dir}/{SPLIT_FILE}"
    if os.path.exists(path):
        return json.load(open(path))
    print(f"downloading {cfg.n_gifs} TGIF GIFs and mining motion pairs...")
    return build_dataset(cfg)


class MotionPairs(Dataset):
    """(source, target, caption, flow) -> 256² tensors in [-1, 1], prompt with motion text, int motion."""
    def __init__(self, rows, res=256, flip_prob=0.5):
        self.rows, self.res, self.flip_prob = rows, res, flip_prob

    def __len__(self):
        return len(self.rows)

    def _load(self, p):
        im = Image.open(p).convert("RGB").resize((self.res, self.res), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1

    def __getitem__(self, i):
        s, t, cap, m = self.rows[i]
        s, t = self._load(s), self._load(t)
        if random.random() < self.flip_prob:
            s, t = s.flip(-1), t.flip(-1)
        return {"src": s, "tgt": t, "prompt": motion_prompt(cap, m), "motion": int(m)}
