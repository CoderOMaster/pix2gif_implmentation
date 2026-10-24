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
