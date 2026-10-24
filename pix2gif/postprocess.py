"""Frame post-processing and jitter measurement."""
import math

import cv2
import numpy as np
from PIL import Image


def _global_motion(a, b, orb):
    """Similarity transform mapping b onto a. RANSAC rejects the moving subject, so this tracks the background."""
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)
    if len(ms) < 12:
        return None
    M, inl = cv2.estimateAffinePartial2D(np.float32([kb[m.queryIdx].pt for m in ms]),
                                         np.float32([ka[m.trainIdx].pt for m in ms]),
                                         method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return M if M is not None and inl.sum() >= 10 else None


def stabilize(frames, max_frac=0.08):
    """Register every frame's background to frame 0, then crop the shared valid area and resize back."""
    arrs = [np.array(f.convert("RGB")) for f in frames]
    h, w = arrs[0].shape[:2]
    orb = cv2.ORB_create(2000)
    ref = cv2.cvtColor(arrs[0], cv2.COLOR_RGB2GRAY)
    Ms = [np.float32([[1, 0, 0], [0, 1, 0]])]
    for a in arrs[1:]:
        M = _global_motion(ref, cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), orb)
        ok = M is not None and abs(M[0, 2]) < w * max_frac and abs(M[1, 2]) < h * max_frac \
            and abs(math.hypot(M[0, 0], M[1, 0]) - 1) < 0.05
        Ms.append(np.float32(M) if ok else Ms[-1])
    shift = max(math.hypot(M[0, 2], M[1, 2]) for M in Ms)
    if shift < 0.5:
        return frames
    pad = min(int(math.ceil(shift)) + 2, int(min(h, w) * max_frac))
    out = [cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REFLECT)[pad:h - pad, pad:w - pad]
           for a, M in zip(arrs, Ms)]
    out = [Image.fromarray(o).resize((w, h), Image.Resampling.LANCZOS) for o in out]
    before, after = jitter_metrics(frames)["camera_shake_px"], jitter_metrics(out)["camera_shake_px"]
    if before is not None and after is not None and after >= before:
        return frames  # registration didn't help (e.g. no stable background): keep the original frames
    return out


def jitter_metrics(frames):
    """camera_shake_px: mean background shift between consecutive frames (lower = steadier).
    flicker: mean absolute pixel change between consecutive frames, 0-100 (includes real motion; compare settings
    on the same image). jumpiness: std/mean of the per-step change (lower = evenly paced, no sudden jumps)."""
    g = [cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2GRAY) for f in frames]
    orb = cv2.ORB_create(1500)
    shifts = [math.hypot(M[0, 2], M[1, 2]) for M in (_global_motion(a, b, orb) for a, b in zip(g, g[1:]))
              if M is not None]
    d = np.array([np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 2.55 for a, b in zip(g, g[1:])])
    return {"camera_shake_px": round(float(np.mean(shifts)), 2) if shifts else None,
            "flicker": round(float(d.mean()), 2),
            "jumpiness": round(float(d.std() / (d.mean() + 1e-6)), 2)}
