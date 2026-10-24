"""Clip -> looping GIF: loop-boundary handling, global palette, frame rate and file-size budget.

Loop modes:
  "trim"     cut the clip where a frame best matches the first one, so the wrap looks like an ordinary step;
  "pingpong" play forward then backward (always seamless; reverses the motion);
  "auto"     trim if the best cut point is close to an ordinary step, else pingpong;
  "crossfade" blend the last frames into the first ones (can ghost / double-expose; opt-in only);
  "none"     export as generated."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .tgif import clip_stats, read_video


def _small(f):
    return np.asarray(Image.fromarray(f).convert("L").resize((128, 128)), dtype=np.float32) / 255


def loop_trim(frames, min_len=None):
    """Return (frames[:k], gap) where the wrap frames[k-1] -> frames[0] is closest to a typical step;
    gap = that wrap distance / median consecutive-frame distance (1 = indistinguishable from a normal step)."""
    g = [_small(f) for f in frames]
    steps = [np.abs(a - b).mean() for a, b in zip(g, g[1:])]
    med = float(np.median(steps)) + 1e-4
    min_len = min_len or max(4, (2 * len(frames)) // 3)
    gaps = {k: float(np.abs(g[k - 1] - g[0]).mean()) / med for k in range(min_len, len(frames) + 1)}
    k = min(gaps, key=gaps.get)
    return frames[:k], gaps[k]


def make_loop(frames, mode="auto", max_gap=1.5, crossfade=3):
    info = {"mode": mode, "gap_before": clip_stats(frames)["loop_gap"]}
    if mode == "none":
        return frames, info
    if mode in ("trim", "auto"):
        trimmed, gap = loop_trim(frames)
        if mode == "trim" or gap <= max_gap:
            return trimmed, {**info, "mode": "trim", "gap_after": gap, "frames": len(trimmed)}
    if mode in ("pingpong", "auto"):
        return frames + frames[-2:0:-1], {**info, "mode": "pingpong", "gap_after": 1.0}
    if mode == "crossfade":
        n = min(crossfade, len(frames) // 3)
        body = [f.astype(np.float32) for f in frames]
        for i in range(n):
            a = (i + 1) / (n + 1)
            body[-n + i] = (1 - a) * body[-n + i] + a * body[i]
        return [f.round().astype(np.uint8) for f in body], {**info, "mode": "crossfade"}
    raise ValueError(f"unknown loop mode {mode}")


def _pil_gif(frames, path, fps, colors):
    sample = np.concatenate([f[::4, ::4] for f in frames], axis=0)
    palette = Image.fromarray(sample).quantize(colors, method=Image.Quantize.MEDIANCUT)
    q = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=round(1000 / fps), loop=0, disposal=1)


def _gifski(frames, path, fps, quality):
    with tempfile.TemporaryDirectory() as d:
        names = []
        for i, f in enumerate(frames):
            names.append(f"{d}/{i:04d}.png")
            Image.fromarray(f).save(names[-1])
        subprocess.run(["gifski", "--fps", str(fps), "--quality", str(quality), "-o", str(path), *names],
                       check=True, capture_output=True)


def save_gif(frames, path, fps, max_kb=2048, width=None):
    """Write a GIF (gifski if installed, otherwise Pillow with one global palette, which avoids palette flicker),
    shrinking width and colour count until it fits max_kb."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = width or frames[0].shape[1]
    for colors in (256, 192, 128, 96, 64):
        size = w
        while True:
            fr = frames if size == frames[0].shape[1] else [
                np.asarray(Image.fromarray(f).resize((size, round(size * f.shape[0] / f.shape[1])),
                                                     Image.Resampling.LANCZOS)) for f in frames]
            if shutil.which("gifski"):
                _gifski(fr, path, fps, quality=round(100 * colors / 256))
            else:
                _pil_gif(fr, path, fps, colors)
            kb = path.stat().st_size / 1024
            if not max_kb or kb <= max_kb or size <= 192:
                break
            size = int(size * 0.85) // 2 * 2
        if not max_kb or kb <= max_kb:
            break
    return {"path": str(path), "kb": round(kb), "width": size, "colors": colors, "frames": len(frames), "fps": fps}


def video_to_gif(video, gif, fps, loop="auto", max_kb=2048, width=None):
    frames = read_video(video)
    looped, info = make_loop(frames, loop)
    return {**save_gif(looped, gif, fps, max_kb, width), "loop": info}
