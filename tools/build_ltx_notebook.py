"""Regenerate the LTX + MotiF notebook:  python tools/build_ltx_notebook.py"""
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
# Image → GIF: LTX-2.5 LoRA + MotiF on curated TGIF clips

Fine-tune the LTX-2.5 image-to-video model so it animates a still image from a motion instruction, then export
looping GIFs.

* **Data:** TGIF animated GIFs from the official train / val / test splits. Each GIF becomes a 17-frame, 512×512 clip
  resampled by timestamp; its first frame is the condition image. Clips that are static, duplicated, cut or
  flickering are dropped. Captions can optionally be rewritten as explicit animation instructions with Gemini.
* **Model:** LTX-2.5 dev transformer with the base weights frozen and LoRA (rank 32) on the video attention layers.
  The first frame is always given (first-frame conditioning probability 1.0), video only (no audio branch).
* **MotiF loss:** RAFT optical flow → motion heatmap on the latent token grid; the loss adds
  λ · mean[m² · (v̂ − v)²] to LTX's flow-matching loss, so moving regions count more than the static background.
* **Experiment:** base LTX vs LoRA (λ = 0) vs LoRA + MotiF (λ = 1) on held-out TGIF and TI2V-Bench, with the same
  images, prompts and seeds.

**Runtime:** GPU with ≥ 32 GB (A100 40/80 GB, H100). The LTX-2 code needs a recent NVIDIA driver (CUDA 13); cell 1
checks it. TGIF is licensed for non-commercial research only.
""")

code('''
#@title 1 · Setup (repo, LTX-2 trainer, Python environments)
import os, subprocess
REPO = "https://github.com/CoderOMaster/pix2gif_implmentation.git"
BRANCH = "ltx-motif"
if not os.path.exists("ltxgif"):
    if not os.path.exists("pix2gif_implmentation"):
        !git clone -q -b {BRANCH} {REPO}
    os.chdir("pix2gif_implmentation")
!git fetch -q origin {BRANCH} && git checkout -q {BRANCH} && git pull -q origin {BRANCH}

driver = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True).stdout.strip()
print("GPU:", driver)
if driver and int(driver.split(",")[-1].strip().split(".")[0]) < 580:
    print("WARNING: LTX-2 pins torch built for CUDA 13, which needs NVIDIA driver >= 580. "
          "This runtime's driver is older, so the LTX steps may fail; use a runtime/VM with a newer driver.")

# LTX-2 runs in its own uv environment (it pins its own torch); this notebook's Python only drives it.
if not os.path.exists("LTX-2"):
    !git clone -q https://github.com/Lightricks/LTX-2
!pip -q install uv
!cd LTX-2 && uv sync -q
!pip -q install -r requirements-ltx.txt
''')

code('''
#@title 2 · Config
from ltxgif import LtxGifConfig

N_TRAIN = 18000  #@param {type:"integer"}
N_VAL = 1000  #@param {type:"integer"}
N_TEST = 1000  #@param {type:"integer"}
STEPS = 5000  #@param {type:"integer"}
LORA_RANK = 32  #@param {type:"integer"}
LEARNING_RATE = 5e-5  #@param {type:"number"}
GRAD_ACCUM = 4  #@param {type:"integer"}
CHECKPOINT_EVERY = 500  #@param {type:"integer"}
VRAM_MODE = "auto"  #@param ["auto", "standard", "low"]
WORK_DIR = "ltx_work"  #@param {type:"string"}

cfg = LtxGifConfig(work_dir=WORK_DIR, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, steps=STEPS,
                   lora_rank=LORA_RANK, lora_alpha=LORA_RANK, learning_rate=LEARNING_RATE, grad_accum=GRAD_ACCUM,
                   checkpoint_every=CHECKPOINT_EVERY, validate_every=CHECKPOINT_EVERY, vram_mode=VRAM_MODE)
from ltxgif import ltx
print(cfg)
print("latent grid (frames, h, w):", cfg.latent_grid, "· VRAM preset:", ltx.vram_preset(cfg))
''')

code('''
#@title 3 · Download LTX-2.5 (dev transformer, text encoder, VAEs: ~70 GB)
# Accept the model terms at huggingface.co/Lightricks/LTX-2.5 and add a read token as the Colab secret HF_TOKEN.
from google.colab import userdata
ltx.download_models(cfg, token=userdata.get("HF_TOKEN"))
''')

code('''
#@title 4 · Build TGIF clips (resumable; rerun to continue)
from ltxgif import tgif
for split in ("val", "test", "train"):
    tgif.build_split(cfg, split)
''')

code('''
#@title 4b · Look at a few clips
from IPython.display import HTML, display
import base64, json
rows = json.loads(cfg.manifest("train").read_text())[:6]
cells = []
for r in rows:
    data = base64.b64encode((cfg.data_dir / r["video"]).read_bytes()).decode()
    cells.append(f'<figure style="margin:0;width:200px"><video src="data:video/mp4;base64,{data}" autoplay loop muted '
                 f'width=200></video><figcaption style="font-size:12px">{r["caption"]}</figcaption></figure>')
display(HTML('<div style="display:flex;gap:8px;flex-wrap:wrap">' + "".join(cells) + "</div>"))
''')

code('''
#@title 5 · (Optional) Rewrite captions as animation instructions with Gemini
# Needs the Colab secret GEMINI_API_KEY. Clips where no action is visible are dropped; review the examples printed below.
RECAPTION = False  #@param {type:"boolean"}
if RECAPTION:
    from ltxgif import recaption
    key = userdata.get("GEMINI_API_KEY")
    for split in ("val", "test", "train"):
        recaption.recaption(cfg, split, key)
    recaption.show_examples(cfg, "train")
''')

code('''
#@title 6 · Encode the training set with LTX (video latents + text embeddings)
ltx.preprocess(cfg, "train")
''')

code('''
#@title 7 · MotiF motion heatmaps (RAFT) + alignment check
from ltxgif import heatmaps
heatmaps.compute_heatmaps(cfg, "train")
heatmaps.check_alignment(cfg, "train")
''')

code('''
#@title 7b · Show a clip next to its heatmap
import matplotlib.pyplot as plt, torch
r = json.loads(cfg.manifest("train").read_text())[0]
frames = tgif.read_video(cfg.data_dir / r["video"])
heat = torch.load(heatmaps.heatmap_path(cfg, r["video"]), weights_only=True)["motion"].float()
fig, ax = plt.subplots(1, 1 + heat.shape[0], figsize=(3 * (1 + heat.shape[0]), 3))
ax[0].imshow(frames[-1]); ax[0].set_title("last frame")
for k in range(heat.shape[0]):
    ax[k + 1].imshow(heat[k], vmin=0, vmax=1, cmap="magma"); ax[k + 1].set_title(f"latent frame {k}")
for a in ax: a.axis("off")
plt.suptitle(r["caption"]); plt.show()
''')

code('''
#@title 8 · Train the baseline LoRA (λ = 0)
# Validation videos: ltx_work/runs/lora/samples. Rerunning resumes from the latest checkpoint.
ltx.train(cfg, "lora", lambda_motion=0.0)
''')

code('''
#@title 9 · Train LoRA + MotiF (λ = 1)
LAMBDA_MOTION = 1.0  #@param {type:"number"}
ltx.train(cfg, "motif", lambda_motion=LAMBDA_MOTION)
''')

code('''
#@title 10 · Compare base vs LoRA vs LoRA + MotiF
from ltxgif import evaluate
N_TEST_ITEMS = 50  #@param {type:"integer"}
N_BENCH_ITEMS = 80  #@param {type:"integer"}
variants = {"base": None, "lora": ltx.latest_lora(cfg, "lora"), "motif": ltx.latest_lora(cfg, "motif")}
print(variants)
for tag, items in (("tgif_test", evaluate.test_items(cfg, N_TEST_ITEMS)),
                   ("ti2v_bench", evaluate.ti2v_bench_items(cfg, N_BENCH_ITEMS))):
    dirs = evaluate.run_variants(cfg, items, variants, tag)
    per_item, summary = evaluate.metrics(cfg, items, dirs)
    print(tag); display(summary)
    evaluate.blind_review(cfg, items, dirs, tag)
''')

code('''
#@title 10b · After rating review/ratings.csv (1-5 per column), unblind the scores
TAG = "ti2v_bench"  #@param ["ti2v_bench", "tgif_test"]
display(evaluate.score_ratings(cfg, TAG))
''')

code('''
#@title 11 · Animate your own image
from ltxgif import gif
from PIL import Image
IMAGE = "my_image.jpg"  #@param {type:"string"}
PROMPT = "A cat sits on a windowsill and slowly turns its head to the left. The camera remains static."  #@param {type:"string"}
VARIANT = "motif"  #@param ["motif", "lora", "base"]
LOOP = "auto"  #@param ["auto", "trim", "pingpong", "none"]
SEED = 42  #@param {type:"integer"}
src = evaluate._square_png(IMAGE, f"{cfg.work_dir}/my/source.png", cfg.size)
out = ltx.generate(cfg, [{"image": src, "prompt": PROMPT, "seed": SEED, "output": f"{cfg.work_dir}/my/{VARIANT}_{SEED}.mp4"}],
                   lora=None if VARIANT == "base" else ltx.latest_lora(cfg, VARIANT), jobs_file=f"{cfg.work_dir}/my/jobs.json")[0]
info = gif.video_to_gif(out, out.replace(".mp4", ".gif"), cfg.fps, loop=LOOP)
print(info)
from IPython.display import Image as IPImage
display(IPImage(filename=info["path"]))
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(ROOT, "ltx_gif_colab.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote ltx_gif_colab.ipynb")
