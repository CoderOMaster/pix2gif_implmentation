"""CPU smoke test for ltxgif: real TGIF clips -> filters -> MP4 -> RAFT heatmaps -> MotiF loss through LTX's own
FlexibleStrategy and dataset loader (with fake latents) -> trainer config validation -> loop + GIF -> review page.

    LTX_REPO=/path/to/LTX-2 python tests/ltx_smoke_test.py

LTX_REPO is a clone of github.com/Lightricks/LTX-2; its trainer source is put on the path (no install needed)."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ltx_repo = Path(os.environ.get("LTX_REPO", "LTX-2")).resolve()
sys.path[:0] = [str(ltx_repo / "packages/ltx-core/src"), str(ltx_repo / "packages/ltx-trainer/src")]

from ltxgif import LtxGifConfig, gif, heatmaps, ltx, tgif  # noqa: E402
from ltxgif.evaluate import blind_review, score_ratings  # noqa: E402

work = Path(tempfile.mkdtemp())
cfg = LtxGifConfig(work_dir=str(work), ltx_repo=str(ltx_repo), models_dir=str(work / "models"), size=64,
                   num_frames=9, n_train=3, n_val=2, n_test=2, headroom=1.0, download_workers=4, flow_res=128,
                   raft_updates=4, n_val_prompts=2, vram_mode="standard")
assert cfg.latent_grid == (2, 2, 2)

# --- unit checks: resampling and filters -------------------------------------------------------------------------
idx, rate = tgif.resample([0] * 20, [100] * 20, 9, 8.0, 16.0)          # 2 s GIF at 8 fps: every 1.25th frame
assert rate == 8.0 and idx == [0, 1, 2, 3, 5, 6, 7, 8, 10], idx
assert tgif.resample([0] * 6, [100] * 6, 9, 8.0, 16.0)[1] > 8.0           # 0.6 s GIF: sampled faster
assert tgif.resample([0] * 4, [100] * 4, 9, 8.0, 16.0) is None             # 0.4 s GIF: too short even at 16 fps
still = [np.full((64, 64, 3), 128, np.uint8)] * 9
assert tgif.reject_reason(cfg, tgif.clip_stats(still), "a man is waving his hand") == "blank_frame"

# --- real TGIF clips ---------------------------------------------------------------------------------------------
for split in ("train", "val", "test"):
    rows = tgif.build_split(cfg, split)
    assert len(rows) == getattr(cfg, f"n_{split}"), (split, len(rows))
rows = json.loads(cfg.manifest("train").read_text())
frames = tgif.read_video(cfg.data_dir / rows[0]["video"])
assert len(frames) == cfg.num_frames and frames[0].shape == (64, 64, 3), (len(frames), frames[0].shape)
tgif.build_split(cfg, "train")                                       # resume: nothing new to do
print("captions:", [r["caption"] for r in rows])

# --- heatmaps ----------------------------------------------------------------------------------------------------
mag = torch.zeros(8, 32, 32)
mag[:, 8:16, 8:16] = 10.0
heat = heatmaps.motif_heatmap(mag)
assert heat[0, 10, 10] > 0.99 and heat[0, 0, 0] < 0.01
grid = heatmaps.to_latent_grid(heat, 2, 2, 2)
assert grid.shape == (2, 2, 2) and grid[0].abs().max() == 0 and grid[1, 0, 0] > 0.2 and grid[1, 1, 1] < 0.01
heatmaps.compute_heatmaps(cfg, "train", device="cpu")

# fake LTX preprocessing output (what process_dataset.py writes) so the real loader can pair files by path
root = cfg.precomputed_dir("train")
for r in rows:
    rel = Path(r["video"]).with_suffix(".pt")
    for sub, data in (("latents", {"latents": torch.randn(128, 2, 2, 2), "num_frames": 2, "height": 2, "width": 2,
                                   "fps": 8.0}),
                      ("conditions", {"video_prompt_embeds": torch.randn(8, 32),
                                      "prompt_attention_mask": torch.ones(8, dtype=torch.bool)})):
        (root / sub / rel).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, root / sub / rel)
heatmaps.check_alignment(cfg, "train")

# --- MotiF strategy through LTX's FlexibleStrategy + PrecomputedDataset ------------------------------------------
from ltx_trainer.datasets import PrecomputedDataset  # noqa: E402
from ltx_trainer.timestep_samplers import SAMPLERS  # noqa: E402
from ltx_trainer.training_strategies.flexible import FlexibleStrategy, FlexibleStrategyConfig  # noqa: E402
from ltxgif.motif import MotifStrategy, MotifStrategyConfig  # noqa: E402

flex_cfg = FlexibleStrategyConfig(name="flexible", video={"is_generated": True, "latents_dir": "latents",
                                                          "conditions": [{"type": "first_frame", "probability": 1.0}]})
ds = PrecomputedDataset(str(root), MotifStrategyConfig(**flex_cfg.model_dump()).get_data_sources())
assert len(ds) == len(rows)
batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
sampler = SAMPLERS["uniform"]()


def run(strategy, seed=0):
    torch.manual_seed(seed)
    inputs = strategy.prepare_training_inputs({k: v for k, v in batch.items()}, sampler)
    torch.manual_seed(1)
    pred = torch.randn_like(inputs.video_targets)
    return inputs, pred, strategy.compute_loss(pred, None, inputs)


_, _, ref = run(FlexibleStrategy(flex_cfg))
inp0, pred, l0 = run(MotifStrategy(MotifStrategyConfig(**flex_cfg.model_dump(), lambda_motion=0.0)))
assert torch.allclose(l0, ref), "lambda = 0 must reproduce LTX's own loss"
inp1, pred1, l1 = run(MotifStrategy(MotifStrategyConfig(**flex_cfg.model_dump(), lambda_motion=1.0)))
mask = inp1.video_loss_mask.float()
assert mask[:, :4].sum() == 0 and mask[:, 4:].all(), "first latent frame (4 tokens) must be excluded from the loss"
err2 = (pred1 - inp1.video_targets).square().mean(-1)
manual = ((err2 * mask).sum(1) + (err2 * inp1.video_motion.square() * mask).sum(1)) / mask.sum(1)
assert torch.allclose(l1, manual, atol=1e-5), (l1, manual)
assert inp1.video_motion.shape == (len(rows), 8)
print("MotiF loss ok:", l0.tolist(), "->", l1.tolist())

# --- trainer config passes LTX's own validation ------------------------------------------------------------------
for key in ("transformer", "text_encoder", "video_vae", "audio_vae"):
    cfg.model_path(key).parent.mkdir(parents=True, exist_ok=True)
    cfg.model_path(key).write_bytes(b"")
from ltx_trainer.config import LtxTrainerConfig  # noqa: E402

conf = yaml.safe_load(ltx.train_config(cfg, "motif", 1.0).read_text())
motif = conf.pop("motif")
tc = LtxTrainerConfig(**conf)
assert tc.training_strategy.video.conditions[0].probability == 1.0 and tc.training_strategy.audio is None
assert not tc.validation.generate_audio and tc.lora.target_modules == ltx.VIDEO_LORA_TARGETS
strategy = MotifStrategyConfig(**tc.training_strategy.model_dump(), **motif)
tc = tc.model_copy(update={"training_strategy": strategy})
assert tc.training_strategy.get_data_sources()["motion_heatmaps"] == "motion"
print("trainer config ok:", ltx.vram_preset(cfg))

# --- loop + GIF ----------------------------------------------------------------------------------------------------
looped, info = gif.make_loop(frames, "auto")
assert info["mode"] in ("trim", "pingpong") and len(looped) >= 4
assert len(gif.make_loop(frames, "pingpong")[0]) == 2 * len(frames) - 2
out = gif.save_gif(looped, work / "out.gif", cfg.fps, max_kb=20)
assert out["kb"] <= 20 or out["width"] <= 192, out
print("gif:", out, info)

# --- blinded review page -------------------------------------------------------------------------------------------
items = [{"id": Path(r["video"]).stem, "image": str(tgif.first_frame(cfg, r["video"], work / "frames")),
          "prompt": r["caption"]} for r in rows]
dirs = {}
for name in ("base", "lora", "motif"):
    dirs[name] = work / "fake" / name
    dirs[name].mkdir(parents=True)
    for r in rows:
        shutil.copy(cfg.data_dir / r["video"], dirs[name] / (Path(r["video"]).stem + ".mp4"))
review = blind_review(cfg, items, dirs, "smoke")
csv_path = review / "ratings.csv"
lines = csv_path.read_text().splitlines()
csv_path.write_text("\n".join([lines[0]] + [ln.replace(",,,,", ",3,4,5,2") for ln in lines[1:]]) + "\n")
scores = score_ratings(cfg, "smoke")
assert set(scores.index) == {"base", "lora", "motif"} and (scores["action_happens"] == 3).all()

shutil.rmtree(work)
print("LTX SMOKE TEST PASSED")
