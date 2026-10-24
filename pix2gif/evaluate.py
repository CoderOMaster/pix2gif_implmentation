import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import flow_magnitude
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
