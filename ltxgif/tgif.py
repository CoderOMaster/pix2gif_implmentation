"""TGIF -> paired image-to-video training clips.

Each kept GIF becomes one fixed-size MP4 (cfg.num_frames frames at its real playback rate). Its first frame is the
condition image; LTX's first-frame conditioning extracts it from the clip, so the manifest needs only {video, caption}.
TGIF is released for non-commercial research use only."""
import io
import json
import random
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageSequence

TSV_URL = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv"
SPLIT_URL = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/{}.txt"
SPLITS = ("train", "val", "test")


def _fetch_text(url, cache: Path) -> str:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        cache.write_text(r.text)
    return cache.read_text()


def load_split(cfg, split):
    """[(gif_id, url, caption)] for an official TGIF split, in a fixed shuffled order (first caption per GIF)."""
    meta = cfg.data_dir / "meta"
    captions = {}
    for line in _fetch_text(TSV_URL, meta / "tgif-v1.0.tsv").splitlines():
        url, _, cap = line.partition("\t")
        captions.setdefault(url.strip(), cap.strip())
    urls = [u.strip() for u in _fetch_text(SPLIT_URL.format(split), meta / f"{split}.txt").splitlines() if u.strip()]
    rows = [(f"{split}_{i:06d}", u, captions[u]) for i, u in enumerate(urls) if captions.get(u)]
    random.Random(cfg.seed).shuffle(rows)
    return rows


def decode_gif(data):
    """Fully composited RGB frames and their display durations (ms). Pillow applies each frame's disposal method;
    durations below 20 ms are played as 100 ms by browsers, so they are treated the same way here."""
    im = Image.open(io.BytesIO(data))
    frames, durations = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(np.asarray(f.convert("RGB")))
        d = f.info.get("duration", 100) or 100
        durations.append(100 if d < 20 else d)
    return frames, durations


def resample(frames, durations, n, fps, max_fps):
    """Pick n frames at evenly spaced timestamps (no padding with repeated frames). Returns (indices, fps) or None.
    A GIF shorter than (n - 1) / fps seconds is sampled at a higher rate, up to max_fps."""
    starts = np.cumsum([0] + durations[:-1]) / 1000.0
    total = (starts[-1] + durations[-1] / 1000.0)
    rate = fps
    if (n - 1) / rate >= total:
        rate = (n - 1) / (total * 0.999)
        if rate > max_fps:
            return None
    t = np.arange(n) / rate
    idx = np.searchsorted(starts, t, side="right") - 1
    return idx.tolist(), float(rate)


def center_square(frame, size):
    h, w = frame.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return np.asarray(Image.fromarray(frame[y:y + s, x:x + s]).resize((size, size), Image.Resampling.LANCZOS))


def clip_stats(frames):
    """Temporal-quality statistics on a list of RGB uint8 frames (computed on 128 px greyscale)."""
    g = np.stack([np.asarray(Image.fromarray(f).convert("L").resize((128, 128)), dtype=np.float32) / 255
                  for f in frames])
    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    med = float(np.median(d))
    luma = g.mean(axis=(1, 2))
    return {
        "motion": float(d.mean()),
        "static_fraction": float((d < 0.002).mean()),
        "cut_ratio": float(d.max() / (med + 1e-4)),
        "luma_flicker": float(np.abs(np.diff(luma)).mean()),
        "loop_gap": float(np.abs(g[-1] - g[0]).mean() / (med + 1e-4)),
        "min_frame_std": float(g.std(axis=(1, 2)).min()),
    }


def reject_reason(cfg, stats, caption):
    if len(caption.split()) < cfg.min_caption_words:
        return "caption_too_short"
    if stats["min_frame_std"] < 0.02:
        return "blank_frame"
    if stats["motion"] < cfg.min_motion:
        return "too_static"
    if stats["static_fraction"] > cfg.max_static_fraction:
        return "duplicated_frames"
    if stats["cut_ratio"] > cfg.max_cut_ratio:
        return "scene_cut"
    if stats["luma_flicker"] > cfg.max_luma_flicker:
        return "flicker"
    if cfg.max_loop_gap > 0 and stats["loop_gap"] > cfg.max_loop_gap:
        return "loop_gap"
    return None


def normalize_caption(c):
    c = re.sub(r"\s+", " ", c).strip().rstrip(".")
    return c[:1].upper() + c[1:] + "."


def write_mp4(frames, path, fps):
    import imageio.v2 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    with iio.get_writer(tmp, fps=fps, codec="libx264", quality=None, macro_block_size=16, pixelformat="yuv420p",
                        ffmpeg_params=["-crf", "16"]) as w:
        for f in frames:
            w.append_data(f)
    tmp.replace(path)


def read_video(path):
    import imageio.v3 as iio
    return [np.asarray(f) for f in iio.imiter(path)]


def process_gif(cfg, gid, url, caption, split):
    """Download, decode, filter and convert one GIF. Returns a record dict (with 'reject' set if dropped)."""
    rec = {"id": gid, "url": url, "caption_orig": caption}
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        frames, durations = decode_gif(r.content)
    except Exception as e:  # dead links, truncated files
        return {**rec, "reject": f"download_or_decode: {type(e).__name__}"}
    if len(frames) < cfg.min_unique_frames:
        return {**rec, "reject": "too_few_frames"}
    picked = resample(frames, durations, cfg.num_frames, cfg.fps, cfg.max_fps)
    if picked is None:
        return {**rec, "reject": "too_short"}
    idx, rate = picked
    if len(set(idx)) < cfg.min_unique_frames:
        return {**rec, "reject": "too_few_unique_frames"}
    clip = [center_square(frames[i], cfg.size) for i in idx]
    stats = clip_stats(clip)
    rec.update(stats=stats, fps=rate, source_frames=len(frames))
    why = reject_reason(cfg, stats, caption)
    if why:
        return {**rec, "reject": why}
    rel = Path("clips") / split / f"{gid}.mp4"
    write_mp4(clip, cfg.data_dir / rel, rate)
    return {**rec, "video": rel.as_posix(), "caption": normalize_caption(caption)}


def build_split(cfg, split, n=None):
    """Process GIFs from an official split until n clips pass the filters (n defaults to the split's target size
    x headroom). Resumable: every processed GIF is appended to the split's records file and skipped next time."""
    target = n or int(round(getattr(cfg, f"n_{split}") * cfg.headroom))
    rec_path = cfg.records(split)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    kept = sum("video" in r for r in done.values())
    todo = iter([row for row in load_split(cfg, split) if row[0] not in done])
    print(f"{split}: {kept} clips already built, target {target}")
    with ThreadPoolExecutor(cfg.download_workers) as ex, open(rec_path, "a") as out:
        running = set()
        while kept < target:
            while len(running) < cfg.download_workers * 2 and (row := next(todo, None)) is not None:
                running.add(ex.submit(process_gif, cfg, *row, split))
            if not running:
                print(f"{split}: ran out of GIFs at {kept} clips")
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                r = f.result()
                done[r["id"]] = r
                out.write(json.dumps(r) + "\n")
                kept += "video" in r
                if len(done) % 1000 == 0:
                    out.flush()
                    print(f"  {split}: {kept}/{target} kept of {len(done)} processed")
        for f in running:
            f.cancel()
    summarize(cfg, split)
    return write_manifest(cfg, split)


def load_records(cfg, split):
    """Latest record per GIF id (later lines, e.g. from recaptioning, override earlier ones)."""
    recs = {}
    for line in cfg.records(split).read_text().splitlines():
        r = json.loads(line)
        recs[r["id"]] = {**recs.get(r["id"], {}), **r}
    return list(recs.values())


def summarize(cfg, split):
    from collections import Counter
    recs = load_records(cfg, split)
    reasons = Counter(r.get("reject", "kept").split(":")[0] for r in recs)
    print(f"{split}: {len(recs)} GIFs processed -> " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))


def write_manifest(cfg, split, n=None):
    """Write the LTX manifest: the first n kept clips (default: the split's target size)."""
    n = n or getattr(cfg, f"n_{split}")
    keep = [r for r in load_records(cfg, split) if "video" in r and not r.get("reject")]
    keep.sort(key=lambda r: r["id"])
    rows = [{"video": r["video"], "caption": r["caption"]} for r in keep[:n]]
    cfg.manifest(split).write_text(json.dumps(rows, indent=1))
    print(f"{split}: manifest with {len(rows)} clips -> {cfg.manifest(split)}")
    return rows


def first_frame(cfg, video_rel, out_dir):
    """Save a clip's first frame as PNG (the condition image for validation / evaluation)."""
    out = Path(out_dir) / (Path(video_rel).stem + ".png")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio
        Image.fromarray(next(iter(iio.imiter(cfg.data_dir / video_rel)))).save(out)
    return out
