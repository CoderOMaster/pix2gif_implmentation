"""Regenerate the Colab notebook:  python tools/build_notebook.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})


def code(s):
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None,
                  "outputs": [], "source": s.strip()})


md("""
# Pix2Gif: motion-guided GIF generation

Implementation of [Pix2Gif (arXiv:2403.04634)](https://arxiv.org/abs/2403.04634). Given an image, a caption and a motion
magnitude, the model generates the next frame. Generating with increasing motion values gives a GIF.

**Model:** an image-conditioned Stable Diffusion UNet. A learned motion embedding is added to the CLIP text embedding. A small
FlowNet predicts a flow field that warps the source latent, and the UNet sees `[noisy latent, source latent, warped latent]`.
Training loss: diffusion loss + 0.01 × perceptual loss on the warped latent.

**Data:** TGIF frame pairs whose Lucas–Kanade optical flow is between 2 and 20 px.

**Runtime:** GPU, A100 recommended (L4 works). Run the cells in order.
""")

code('''
#@title 1 · Setup
import os
REPO = "https://github.com/CoderOMaster/pix2gif_implmentation.git"
if not os.path.exists("pix2gif"):
    if not os.path.exists("pix2gif_implmentation"):
        !git clone -q {REPO}
    os.chdir("pix2gif_implmentation")
!pip -q install -r requirements.txt

import torch
assert torch.cuda.is_available(), "Select a GPU runtime (Runtime -> Change runtime type)"
print(torch.cuda.get_device_name(0))
''')

code('''
#@title 2 · Config
from pix2gif import Config

N_GIFS = 3000  #@param {type:"integer"}
INIT = "instruct-pix2pix"  #@param ["instruct-pix2pix", "sd-1.5"]
USE_WARP = True  #@param {type:"boolean"}
MAX_STEPS = 5000  #@param {type:"integer"}
BATCH_SIZE = 16  #@param {type:"integer"}
GRAD_ACCUM = 2  #@param {type:"integer"}
USE_8BIT_ADAM = True  #@param {type:"boolean"}
SAVE_EVERY = 1000  #@param {type:"integer"}

cfg = Config(n_gifs=N_GIFS, init=INIT, use_warp=USE_WARP, max_steps=MAX_STEPS, batch_size=BATCH_SIZE,
             grad_accum=GRAD_ACCUM, use_8bit_adam=USE_8BIT_ADAM, save_every=SAVE_EVERY)
cfg
''')

code('''
#@title 3 · Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pix2gif import load_or_build

split = load_or_build(cfg)
print(f"train pairs: {len(split['train'])} · val pairs: {len(split['val'])}")

fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
ax[0].hist([r[3] for r in split["train"]], bins=36)
ax[0].set_title("optical flow of training pairs")
for a, (s, t, cap, m) in zip(ax[1:], random.Random(1).sample(split["train"], 3)):
    a.imshow(np.hstack([np.asarray(Image.open(s)), np.asarray(Image.open(t))]))
    a.set_title(f"{cap[:38]}… | flow {m:.1f}", fontsize=8)
    a.axis("off")
plt.show()
''')

code('''
#@title 4 · Load model
from pix2gif import load_components

c = load_components(cfg)
n = sum(p.numel() for p in c.model.parameters()) / 1e6
print(f"trainable parameters: {n:.0f}M")
''')

code('''
#@title 5 · Train (resumes from outputs/ if a checkpoint exists)
from pix2gif import train

log = train(cfg, c, split)
if log:
    l = np.array(log)
    plt.figure(figsize=(8, 3))
    plt.plot(np.convolve(l[:, 0], np.ones(50) / 50, mode="valid"))
    plt.title("diffusion loss (moving avg)")
    plt.show()
''')

code('''
#@title 6 · Generate a GIF
import base64, io, time, requests
from IPython.display import HTML, display
from pix2gif import generate, load_checkpoint, save_gif, text_to_image

SOURCE = "val"  #@param ["val", "upload", "url", "text"]
URL = ""  #@param {type:"string"}
CAPTION = "a cat is dancing."  #@param {type:"string"}
MOTIONS = "2,4,6,8,11,13,16,19"  #@param {type:"string"}
STEPS = 50  #@param {type:"slider", min:20, max:100, step:5}
TEXT_CFG = 7.5  #@param {type:"number"}
IMAGE_CFG = 1.8  #@param {type:"number"}
FPS = 4  #@param {type:"integer"}
BOOMERANG = False  #@param {type:"boolean"}


def show(path, width=256):
    b = base64.b64encode(open(path, "rb").read()).decode()
    display(HTML(f"<img src='data:image/gif;base64,{b}' width={width}>"))


load_checkpoint(cfg, c)
if SOURCE == "val":
    s, _, CAPTION, _ = random.choice(split["val"])
    image = Image.open(s).convert("RGB")
elif SOURCE == "upload":
    from google.colab import files
    image = Image.open(io.BytesIO(next(iter(files.upload().values())))).convert("RGB")
elif SOURCE == "url":
    image = Image.open(io.BytesIO(requests.get(URL, timeout=30).content)).convert("RGB")
else:
    image = text_to_image(CAPTION)

frames = generate(cfg, c, image, CAPTION, [int(x) for x in MOTIONS.split(",")], steps=STEPS,
                  text_cfg=TEXT_CFG, image_cfg=IMAGE_CFG)
path = save_gif(frames, f"outputs/gif_{int(time.time())}.gif", fps=FPS, boomerang=BOOMERANG)
print(CAPTION, "->", path)
display(image.resize((256, int(256 * image.height / image.width))))
show(path)
''')

code('''
#@title 7 · Evaluate (motion control + CLIP similarity on held-out GIFs)
from pix2gif import evaluate

N_EVAL = 30  #@param {type:"integer"}
metrics, samples = evaluate(cfg, c, split, n=N_EVAL, steps=STEPS, text_cfg=TEXT_CFG, image_cfg=IMAGE_CFG)
print(f"motion RMSE {metrics['motion_rmse']:.2f} · motion Pearson r {metrics['motion_pearson']:.3f} · "
      f"CLIPSim {metrics['clipsim']:.4f} · {metrics['n_frames']} frames")
for i, fr in enumerate(samples):
    show(save_gif(fr, f"outputs/eval_{i}.gif"), 192)
plt.figure(figsize=(4, 4))
plt.scatter(metrics["asked"], metrics["measured"], s=8, alpha=0.5)
plt.plot([0, 20], [0, 20], "k--", lw=1)
plt.xlabel("requested motion")
plt.ylabel("measured flow")
plt.show()
''')

code('''
#@title 8 · Quality checks: validation loss + comparison with the real next frame
#@markdown Run with `CHECKPOINT = "base"` once (initial weights, no training) to get the baseline, then with
#@markdown `"trained"`. Training helps only if the trained numbers beat both the base model and the copy-source baseline.
from pix2gif import ground_truth_metrics, load_components, validation_loss

CHECKPOINT = "trained"  #@param ["trained", "base"]
N_PAIRS = 50  #@param {type:"integer"}

if CHECKPOINT == "base":
    c = load_components(cfg)          # fresh initial weights
else:
    load_checkpoint(cfg, c)

vl = validation_loss(cfg, c, split)
print("validation diffusion loss (fixed noise/timesteps):", {k: round(v, 4) for k, v in vl.items()})

gt = ground_truth_metrics(cfg, c, split, n=N_PAIRS, steps=STEPS, text_cfg=TEXT_CFG, image_cfg=IMAGE_CFG)
print(f"LPIPS vs real frame  model {gt['lpips_model']:.4f} | copy-source {gt['lpips_copy']:.4f}   (lower is better)")
print(f"PSNR  vs real frame  model {gt['psnr_model']:.2f} | copy-source {gt['psnr_copy']:.2f}   (higher is better)")
print(f"mean flow            model {gt['flow_model']:.2f} | real pairs  {gt['flow_true']:.2f}   (should match)")
''')

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(ROOT, "pix2gif_colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
