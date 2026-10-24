"""MotiF motion heatmaps, cached on LTX's latent token grid.

For each clip: RAFT optical flow between consecutive frames -> per-pixel flow magnitude -> normalised to [0, 1] with
MotiF's sigmoid-like map -> pooled to the latent grid. Files mirror the latent layout
(precomputed/motion_heatmaps/<clip path>.pt, {"motion": [F', H', W']}) so LTX's dataset loader pairs them by path.

Normalisation: MotiF applies sigma(x) = 1 / (1 + exp(100 * (0.05 - x))) to the flow intensity without stating its
units. Here x = |flow| / max(peak |flow| in the clip, floor), so regions moving at more than ~5% of the clip's peak
motion approach 1 and the static background approaches 0; the floor stops nearly static clips from amplifying flow
noise."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import TEMPORAL_FACTOR
from .tgif import read_video


def motif_heatmap(mag, steepness=100.0, midpoint=0.05, floor=2.0):
    """mag: [T, H, W] flow magnitudes in pixels -> heat in [0, 1]."""
    x = mag / max(float(mag.max()), floor)
    return torch.sigmoid(steepness * (x - midpoint))


def to_latent_grid(heat, latent_frames, latent_h, latent_w):
    """heat: [F - 1, H, W], where heat[j] is the motion from pixel frame j to j + 1.

    The video VAE is causal: latent frame 0 encodes pixel frame 0 alone, and latent frame k >= 1 encodes pixel
    frames 8(k-1)+1 .. 8k. Pixel frame j (j >= 1) is assigned the motion that arrives at it, heat[j - 1]; frame 0 gets
    zero (it is the clean condition frame and is excluded from the loss anyway). Spatial pooling is an average over each
    token's 32x32 pixel patch."""
    pooled = F.adaptive_avg_pool2d(heat[:, None].float(), (latent_h, latent_w))[:, 0]   # [F - 1, h, w]
    out = torch.zeros(latent_frames, latent_h, latent_w)
    for k in range(1, latent_frames):
        out[k] = pooled[TEMPORAL_FACTOR * (k - 1):TEMPORAL_FACTOR * k].mean(0)
    return out


def load_raft(device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    weights = Raft_Large_Weights.DEFAULT
    return raft_large(weights=weights).eval().to(device), weights.transforms()


@torch.no_grad()
def clip_flow_magnitude(cfg, frames, raft, transforms, device):
    """frames: list of RGB uint8 arrays -> [F - 1, flow_res, flow_res] flow magnitude (pixels at flow_res)."""
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    x = F.interpolate(x.float(), size=(cfg.flow_res, cfg.flow_res), mode="bilinear", antialias=True)
    x = x.round().clamp(0, 255).to(torch.uint8)
    a, b = transforms(x[:-1], x[1:])
    flow = raft(a.to(device), b.to(device), num_flow_updates=cfg.raft_updates)[-1]
    return flow.norm(dim=1).cpu()


def heatmap_path(cfg, video_rel, split="train"):
    return cfg.precomputed_dir(split) / "motion_heatmaps" / Path(video_rel).with_suffix(".pt")


def compute_heatmaps(cfg, split="train", device="cuda", overwrite=False):
    """Cache a motion heatmap for every clip in the split's manifest."""
    rows = json.loads(cfg.manifest(split).read_text())
    raft, transforms = load_raft(device)
    lf, lh, lw = cfg.latent_grid
    todo = [r for r in rows if overwrite or not heatmap_path(cfg, r["video"], split).exists()]
    print(f"{split}: {len(rows) - len(todo)} heatmaps cached, computing {len(todo)}")
    coverage = []
    for i, r in enumerate(todo):
        frames = read_video(cfg.data_dir / r["video"])[:cfg.num_frames]
        if len(frames) != cfg.num_frames:
            print(f"  skipping {r['video']}: {len(frames)} frames")
            continue
        mag = clip_flow_magnitude(cfg, frames, raft, transforms, device)
        heat = to_latent_grid(motif_heatmap(mag, cfg.motif_steepness, cfg.motif_midpoint, cfg.flow_floor_px),
                              lf, lh, lw)
        out = heatmap_path(cfg, r["video"], split)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"motion": heat.half()}, out.with_suffix(".tmp"))
        out.with_suffix(".tmp").replace(out)
        coverage.append(float(heat[1:].mean()))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(todo)} · mean heat {np.mean(coverage[-1000:]):.3f}")
    if coverage:
        print(f"done · mean heat over generated tokens {np.mean(coverage):.3f} "
              f"(fraction of the loss MotiF up-weights; ~0 or ~1 everywhere means the normalisation is off)")


def check_alignment(cfg, split="train", n=20):
    """Confirm cached heatmaps have exactly the shape of LTX's cached latents (frames, height, width)."""
    root = cfg.precomputed_dir(split)
    lat_files = sorted((root / "latents").rglob("*.pt"))
    if not lat_files:
        raise FileNotFoundError(f"no latents under {root / 'latents'}: run LTX preprocessing first")
    missing = 0
    for p in lat_files[:n]:
        lat = torch.load(p, map_location="cpu", weights_only=True)
        hp = root / "motion_heatmaps" / p.relative_to(root / "latents")
        if not hp.exists():
            missing += 1
            continue
        m = torch.load(hp, map_location="cpu", weights_only=True)["motion"]
        want = (int(lat["num_frames"]), int(lat["height"]), int(lat["width"]))
        assert tuple(m.shape) == want, f"{hp}: heatmap {tuple(m.shape)} vs latent grid {want}"
    n_lat = len(lat_files)
    n_heat = len(list((root / "motion_heatmaps").rglob("*.pt")))
    print(f"alignment ok on {min(n, n_lat) - missing} samples · {n_lat} latents, {n_heat} heatmaps")
    if n_heat < n_lat:
        print("  some clips have no heatmap: LTX's loader will skip those samples")
