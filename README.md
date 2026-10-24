# Pix2Gif

Implementation of [Pix2Gif: Motion-Guided Diffusion for GIF Generation (arXiv:2403.04634)](https://arxiv.org/abs/2403.04634):
image + caption + motion magnitude → animated GIF.

[Open `pix2gif_colab.ipynb` in Colab](https://colab.research.google.com/github/CoderOMaster/pix2gif_implmentation/blob/main/pix2gif_colab.ipynb).
Use a GPU runtime (A100 recommended) and run the cells in order.

## How it works

* **Conditioning:** caption + `"The optical flow is <n>."` → CLIP embedding. The motion value goes through a learned embedding that is added to it.
* **Warping:** a small FlowNet predicts a flow field from the source latent and the condition, and the source latent is warped with it.
* **Denoiser:** a Stable Diffusion UNet with a 12-channel input `[noisy latent, source latent, warped latent]`, initialised from
  InstructPix2Pix (or SD-1.5). The extra input channels start at zero.
* **Loss:** diffusion loss + 0.01 × VGG perceptual loss on the warped latent. CFG dropout is 5% text / 5% image / 5% both.
* **Sampling:** Euler ancestral with 3-way classifier-free guidance (text 7.5, image 1.8), one frame per motion value from the same source image.
* **Data:** TGIF GIFs, frame pairs labelled with their mean Lucas–Kanade optical flow, keeping 2–20 px and up to 10 pairs per GIF.

## Layout

```
pix2gif/
  config.py     Config dataclass (data, model, training settings)
  data.py       TGIF download, optical-flow pair mining, Dataset
  model.py      motion embedding, FlowNet, warp, perceptual loss, Pix2Gif, model loading
  train.py      training loop (bf16, 8-bit AdamW, EMA, checkpoint + resume)
  sample.py     GIF generation, checkpoint loading, text -> first frame
  evaluate.py   motion RMSE / Pearson r, CLIP similarity
pix2gif_colab.ipynb   Colab notebook
tests/smoke_test.py   CPU smoke test with tiny models
tools/build_notebook.py  regenerates the notebook
```

## Usage

```python
from pix2gif import Config, load_or_build, load_components, train, generate, save_gif

cfg = Config(n_gifs=3000, max_steps=5000)
split = load_or_build(cfg)                 # data/tgif_pairs
c = load_components(cfg)
train(cfg, c, split)                       # outputs/pix2gif_*.pt
frames = generate(cfg, c, image, "a dog is running.")
save_gif(frames, "outputs/dog.gif")
```

## Test

```bash
pip install -r requirements.txt
python tests/smoke_test.py   # downloads 8 GIFs, trains tiny models for a few steps on CPU
```

---

# LTX-2.5 LoRA + MotiF (image → GIF)

A second approach, in `ltxgif/`: fine-tune the LTX-2.5 image-to-video model with LoRA on curated TGIF clips, adding the
[MotiF](https://arxiv.org/abs/2412.16153) motion focal loss, then export looping GIFs.
[Open `ltx_gif_colab.ipynb`](https://colab.research.google.com/github/CoderOMaster/pix2gif_implmentation/blob/ltx-motif/ltx_gif_colab.ipynb)
on a GPU with at least 32 GB.

* **Data:** TGIF official train/val/test splits → 17-frame 512×512 clips resampled by timestamp (first frame = condition
  image). Static, duplicated, cut and flickering clips are dropped. Optional Gemini recaptioning into
  "subject + action + direction + camera" form. TGIF is for non-commercial research only.
* **Training:** the official [LTX-2 trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer), with
  LoRA rank 32 on the video attention layers, first-frame conditioning probability 1.0, video only, lr 5e-5,
  batch 1 × 4 accumulation, 5,000 steps. Low-VRAM preset (int8) under 70 GB.
* **MotiF:** RAFT flow → heatmap m in [0, 1] (σ(100·(x − 0.05)), x = |flow| / clip peak) pooled onto the latent token grid
  (causal temporal grouping). Loss = LTX loss + λ · mean[m² · (v̂ − v)²], implemented as a trainer strategy
  (`ltxgif/motif.py`) without modifying the LTX-2 checkout.
* **Evaluation:** base vs LoRA (λ = 0) vs LoRA + MotiF on held-out TGIF and
  [TI2V-Bench](https://huggingface.co/datasets/wang-sj16/TI2V-Bench): CLIP text / source similarity, motion,
  flicker, loop gap, plus a blinded side-by-side page for human ratings.

```
ltxgif/
  config.py          LtxGifConfig
  tgif.py            TGIF download, decoding, filters, resampling, MP4 clips, manifests
  recaption.py       optional Gemini captions
  heatmaps.py        RAFT → MotiF heatmaps on the latent grid, alignment check
  ltx.py             model download, VRAM presets, trainer YAML, preprocess / train / generate wrappers
  motif.py           MotiF strategy for the LTX trainer      (runs in the LTX-2 environment)
  train_ltx.py       training launcher                      (runs in the LTX-2 environment)
  batch_generate.py  one-model-load batch inference          (runs in the LTX-2 environment)
  gif.py             loop handling, GIF export
  evaluate.py        metrics, blinded review
tests/ltx_smoke_test.py   CPU test: real TGIF clips, heatmaps, MotiF loss through LTX's own strategy + loader
```

```bash
LTX_REPO=/path/to/LTX-2 python tests/ltx_smoke_test.py
```
