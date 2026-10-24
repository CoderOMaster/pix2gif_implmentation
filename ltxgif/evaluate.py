"""Compare base LTX, LoRA (lambda = 0) and LoRA + MotiF on identical images, prompts and seeds.

Automatic metrics are a screen, not a verdict; the blinded side-by-side page is for people to rate whether the
requested action happens, the source stays recognisable, the motion looks natural, and the loop is smooth."""
import csv
import html
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from . import ltx
from .gif import video_to_gif
from .tgif import center_square, clip_stats, first_frame, read_video

RATING_COLUMNS = ["action_happens", "source_preserved", "natural_motion", "loop_smooth"]


def _square_png(src, out, size):
    out = Path(out)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(center_square(np.asarray(Image.open(src).convert("RGB")), size)).save(out)
    return str(out)


def test_items(cfg, n=50):
    """First frames + captions of held-out TGIF test clips."""
    rows = json.loads(cfg.manifest("test").read_text())[:n]
    return [{"id": Path(r["video"]).stem, "image": str(first_frame(cfg, r["video"], cfg.data_dir / "test_frames")),
             "prompt": r["caption"]} for r in rows]


def ti2v_bench_items(cfg, n=None):
    """TI2V-Bench (MotiF's evaluation set): images paired with several different action prompts each."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("wang-sj16/TI2V-Bench", repo_type="dataset",
                                  local_dir=Path(cfg.work_dir) / "ti2v_bench"))
    with open(root / "prompt.csv") as f:
        rows = list(csv.DictReader(f))[:n]
    return [{"id": f"{Path(r['file_name']).stem}_{i:03d}", "prompt": r["prompt"],
             "image": _square_png(root / r["file_name"], root / "square" / (Path(r["file_name"]).stem + ".png"),
                                  cfg.size)} for i, r in enumerate(rows)]


def run_variants(cfg, items, variants, tag, seed=1234):
    """variants: {name: LoRA path or None}. Writes <work>/eval/<tag>/<name>/<id>.mp4, one model load per variant."""
    out = {}
    for name, lora in variants.items():
        d = Path(cfg.work_dir) / "eval" / tag / name
        jobs = [{"image": it["image"], "prompt": it["prompt"], "seed": seed, "output": str(d / f"{it['id']}.mp4")}
                for it in items]
        ltx.generate(cfg, jobs, lora=lora, jobs_file=d / "jobs.json")
        out[name] = d
    return out


def _clip_model(device):
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-large-patch14"
    return CLIPModel.from_pretrained(name).to(device).eval(), CLIPProcessor.from_pretrained(name)


def metrics(cfg, items, dirs, device="cuda"):
    """Per-variant means of:
    clip_text   CLIP similarity between frames and the prompt (x100): does it look like the described scene/action;
    source_sim  CLIP image similarity between the condition image and the last frame: is the source preserved;
    motion      mean |frame difference|: how much moves (0 = frozen);
    flicker     mean |change in brightness| between frames;
    loop_gap    last -> first frame jump relative to a normal step (1 = seamless)."""
    import pandas as pd
    import torch
    model, proc = _clip_model(device)

    @torch.no_grad()
    def img_emb(images):
        e = model.get_image_features(**proc(images=images, return_tensors="pt").to(device))
        return torch.nn.functional.normalize(e, dim=-1)

    @torch.no_grad()
    def txt_emb(text):
        e = model.get_text_features(**proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(device))
        return torch.nn.functional.normalize(e, dim=-1)

    rows = []
    for it in items:
        src = img_emb([Image.open(it["image"]).convert("RGB")])
        t = txt_emb(it["prompt"])
        for name, d in dirs.items():
            p = Path(d) / f"{it['id']}.mp4"
            if not p.exists():
                continue
            frames = read_video(p)
            e = img_emb([Image.fromarray(f) for f in frames])
            s = clip_stats(frames)
            rows.append({"variant": name, "id": it["id"], "clip_text": float((e @ t.T).mean() * 100),
                         "source_sim": float((e[-1:] @ src.T).item()), "motion": s["motion"],
                         "flicker": s["luma_flicker"], "loop_gap": s["loop_gap"]})
    df = pd.DataFrame(rows)
    return df, df.groupby("variant")[["clip_text", "source_sim", "motion", "flicker", "loop_gap"]].mean().round(4)


def blind_review(cfg, items, dirs, tag, loop="auto", seed=0):
    """Build GIFs and a blinded review page: per item the source image, the prompt and the variants in random order
    labelled A, B, C... Writes review.html, ratings.csv (to fill in, 1-5 per column) and key.json (label -> variant)."""
    root = Path(cfg.work_dir) / "eval" / tag / "review"
    rng = random.Random(seed)
    key, cards, ratings = {}, [], []
    names = list(dirs)
    for it in items:
        order = rng.sample(names, len(names))
        labels = {}
        src = _square_png(it["image"], root / "sources" / f"{it['id']}.png", cfg.size)
        cells = [f'<figure><img src="sources/{Path(src).name}"><figcaption>source</figcaption></figure>']
        for i, name in enumerate(order):
            lab = chr(65 + i)
            mp4 = Path(dirs[name]) / f"{it['id']}.mp4"
            if not mp4.exists():
                continue
            gif = root / "gifs" / f"{it['id']}_{lab}.gif"
            if not gif.exists():
                video_to_gif(mp4, gif, cfg.fps, loop=loop, max_kb=0)
            labels[lab] = name
            cells.append(f'<figure><img src="gifs/{gif.name}"><figcaption>{lab}</figcaption></figure>')
            ratings.append({"id": it["id"], "label": lab, **{c: "" for c in RATING_COLUMNS}})
        key[it["id"]] = labels
        cards.append(f'<section><p><b>{html.escape(it["id"])}</b> — {html.escape(it["prompt"])}</p>'
                     f'<div class="row">{"".join(cells)}</div></section>')
    root.mkdir(parents=True, exist_ok=True)
    (root / "review.html").write_text(
        "<!doctype html><meta charset=utf-8><title>GIF review</title><style>body{font:14px sans-serif;margin:16px}"
        ".row{display:flex;gap:8px;flex-wrap:wrap}figure{margin:0}img{width:200px;height:200px;object-fit:cover}"
        "figcaption{text-align:center}section{margin-bottom:24px}</style>"
        f"<h1>{html.escape(tag)}</h1><p>Rate each lettered GIF 1-5 on: {', '.join(RATING_COLUMNS)} "
        "(ratings.csv).</p>" + "".join(cards))
    (root / "key.json").write_text(json.dumps(key, indent=1))
    with open(root / "ratings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", *RATING_COLUMNS])
        w.writeheader()
        w.writerows(ratings)
    print(f"review page: {root / 'review.html'} · fill in {root / 'ratings.csv'} · key: {root / 'key.json'}")
    return root


def score_ratings(cfg, tag):
    """Unblind a filled-in ratings.csv: mean score per variant and criterion."""
    import pandas as pd
    root = Path(cfg.work_dir) / "eval" / tag / "review"
    key = json.loads((root / "key.json").read_text())
    df = pd.read_csv(root / "ratings.csv").dropna(subset=RATING_COLUMNS, how="all")
    df["variant"] = [key[str(i)][lab] for i, lab in zip(df["id"], df["label"])]
    return df.groupby("variant")[RATING_COLUMNS].mean().round(2)
