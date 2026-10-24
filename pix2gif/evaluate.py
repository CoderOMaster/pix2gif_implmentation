import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import MotionPairs, flow_magnitude
from .sample import generate

EVAL_MOTIONS = (2, 4, 6, 8, 11, 14, 17, 19)


class ClipScorer:
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        self.proc, self.device = CLIPProcessor.from_pretrained(model_id), device

    @torch.no_grad()
    def __call__(self, frames, caption):
        inp = self.proc(text=[caption], images=frames, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**{k: v.half() if v.dtype == torch.float32 else v for k, v in inp.items()})
        i, t = F.normalize(out.image_embeds.float(), dim=-1), F.normalize(out.text_embeds.float(), dim=-1)
        return float((i @ t.T).mean())


def evaluate(cfg, c, split, n=30, motions=EVAL_MOTIONS, steps=50, clip=True, **gen_kwargs):
    """Motion control (RMSE and Pearson r between requested motion and measured LK flow source->frame) and
    CLIPSim (frames vs caption) on held-out GIFs. Returns (metrics dict, sample GIF frames)."""
    scorer = ClipScorer(cfg.device) if clip else None
    rows = random.Random(0).sample(split["val"], min(n, len(split["val"])))
    asked, measured, sims, samples = [], [], [], []
    for s, _, cap, _ in rows:
        img = Image.open(s).convert("RGB")
        frames = generate(cfg, c, img, cap, motions, steps=steps, **gen_kwargs)
        src = np.asarray(img.resize((cfg.res, cfg.res)))
        for m, f in zip(motions, frames):
            mm = flow_magnitude(src, np.asarray(f.resize((cfg.res, cfg.res))), cfg.flow_eps)
            if mm is not None:
                asked.append(m)
                measured.append(mm)
        if scorer:
            sims.append(scorer(frames, cap))
        if len(samples) < 4:
            samples.append(frames)
    a, b = np.array(asked, float), np.array(measured, float)
    metrics = {"motion_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
               "motion_pearson": float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan"),
               "clipsim": float(np.mean(sims)) if sims else None,
               "n_frames": len(a), "asked": a.tolist(), "measured": b.tolist()}
    return metrics, samples


@torch.no_grad()
def validation_loss(cfg, c, split, n=64, timesteps=(100, 300, 500, 700, 900), seed=0, batch_size=8):
    """Diffusion loss on held-out pairs with FIXED noise and FIXED timesteps (no CFG dropout, no flip), so the
    number is comparable across checkpoints, unlike the noisy training loss. Lower = better denoiser."""
    dev, model = cfg.device, c.model.eval()
    ds = MotionPairs(split["val"][:n], cfg.res, flip_prob=0.0)
    g = torch.Generator(dev).manual_seed(seed)
    per_t = {t: [] for t in timesteps}
    for i in range(0, len(ds), batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        src = torch.stack([b["src"] for b in items]).to(dev)
        tgt = torch.stack([b["tgt"] for b in items]).to(dev)
        motion = torch.tensor([b["motion"] for b in items], device=dev)
        z = c.vae.encode(tgt).latent_dist.mode() * c.scale
        src_lat = c.vae.encode(src).latent_dist.mode()
        text = c.encode_text([b["prompt"] for b in items])
        for t in timesteps:
            noise = torch.randn(z.shape, generator=g, device=dev)
            tt = torch.full((len(items),), t, device=dev, dtype=torch.long)
            with torch.autocast(dev, dtype=cfg.dtype):
                pred, _ = model(c.noise_scheduler.add_noise(z, noise, tt), tt, text, motion, src_lat)
            per_t[t] += ((pred.float() - noise) ** 2).mean(dim=(1, 2, 3)).tolist()
    out = {f"t={t}": float(np.mean(v)) for t, v in per_t.items()}
    out["mean"] = float(np.mean(list(out.values())))
    return out


def _psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-8)))


@torch.no_grad()
def ground_truth_metrics(cfg, c, split, n=50, steps=50, use_lpips=True, **gen_kwargs):
    """For held-out pairs, generate the target frame from the source with the pair's TRUE motion value and compare it
    to the REAL target frame. 'copy' is the trivial baseline that returns the source frame unchanged; a useful model
    must beat it (lower LPIPS, higher PSNR) while also producing the requested amount of motion."""
    lp = None
    if use_lpips:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(cfg.device)
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(cfg.device)
    rows = random.Random(1).sample(split["val"], min(n, len(split["val"])))
    res = {"lpips_model": [], "lpips_copy": [], "psnr_model": [], "psnr_copy": [], "flow_model": [], "flow_true": []}
    for s, t, cap, m in rows:
        src = np.array(Image.open(s).convert("RGB").resize((cfg.res, cfg.res)))
        tgt = np.array(Image.open(t).convert("RGB").resize((cfg.res, cfg.res)))
        pred = np.array(generate(cfg, c, Image.fromarray(src), cap, [int(m)], steps=steps, **gen_kwargs)[0])
        res["psnr_model"].append(_psnr(pred, tgt))
        res["psnr_copy"].append(_psnr(src, tgt))
        if lp is not None:
            res["lpips_model"].append(float(lp(to_t(pred), to_t(tgt))))
            res["lpips_copy"].append(float(lp(to_t(src), to_t(tgt))))
        f = flow_magnitude(src, pred, cfg.flow_eps)
        if f is not None:
            res["flow_model"].append(f)
            res["flow_true"].append(m)
    return {k: float(np.mean(v)) if v else None for k, v in res.items()} | {"n": len(rows)}
