import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
read = lambda f: open(os.path.join(HERE, f)).read().strip()
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
code = lambda s: cells.append({"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None,
                               "outputs": [], "source": s})

md("""
# Pix2Gif replication: motion-guided diffusion for GIF generation

**Paper:** Kandala, Gao, Yang. *Pix2Gif: Motion-Guided Diffusion for GIF Generation*, [arXiv 2403.04634](https://arxiv.org/abs/2403.04634) (Microsoft Research)

**Idea.** Treat GIF generation as **image-to-image translation** (InstructPix2Pix style). Given a source image, a caption and a
**motion magnitude** *m*, the model predicts the frame that is *m* pixels of optical flow away. A GIF is the same source image
rendered at increasing *m* (e.g. 2, 4, 6, 8, 11, 13, 16, 19).

**Architecture** (final "Warp-concat" model):
1. Caption + `" The optical flow is <m in words>."` → CLIP → **c_T**. Motion *m* → learned embedding → **c_M**. **c_L = c_T + c_M**.
2. **FlowNet** (a small UNet; c_L is injected at 16×16) predicts a flow field over the source latent *E(c_I)*. **WarpNet**
   (`grid_sample`) warps it into **z_W**.
3. The SD-1.5 UNet denoises with input **[z_t, E(c_I), z_W]** (12 channels) and cross-attention context c_L.
4. Loss: **L_LDM + 0.01 · L_perceptual(z_W, E(c_I))**, where the perceptual term uses a VGG19 whose first conv is widened to 4 channels so it runs on latents.
5. Data: TGIF frame pairs whose Lucas–Kanade mean flow is in **[2, 20]**, with ≥10 pairs per GIF (783k pairs in the paper).

**Provenance.** The authors deleted the code from [hiteshK03/Pix2Gif](https://github.com/hiteshK03/Pix2Gif) in Sept 2024. This notebook
ports it from commit **`781ca78`**, the last commit that still has it (cloned in cell 1 for reference). **No weights were ever released**, and the TGIF
pair-mining script was never published. So this notebook rebuilds the dataset from the paper's Algorithm 1 and **trains the model**.

| | Paper / official code | This notebook |
|---|---|---|
| FlowNet, WarpNet, InjectionModule, motion embedding, prompt template | ✓ | ported line by line |
| 12-ch UNet input, `hybrid` conditioning (context = c_T + c_M) | ✓ | ✓ |
| Perceptual loss (VGG19-4ch, layers 0/5/10/19/28, λ = 0.01) | ✓ | ✓ |
| CFG dropout 5% text / 5% image / 5% both; AdamW lr 1e-4; 256²; LitEma | ✓ | ✓ |
| Sampler: Euler ancestral, text CFG 7.5, image CFG 1.8, per-frame re-seed | ✓ (`demo.py`) | ✓ |
| Init | SD-1.5 + ft-MSE VAE | **InstructPix2Pix** by default (a big head start on Colab compute); SD-1.5 is selectable |
| Data scale | ~100k GIFs → 783k pairs | `N_GIFS` (default 3k GIFs → ~25k pairs) |
| Compute | 16×V100, eff. batch 256, ~24k steps | 1 GPU, eff. batch 32, `MAX_STEPS` (default 5k), bf16, 8-bit AdamW, gradient checkpointing |
| LK threshold / feature parameters | not stated | `FLOW_EPS = 0.5`, OpenCV defaults (flow measured at 256²) |

**Runtime:** Colab **A100** (recommended) or **L4**. A T4 doesn't have enough memory for full UNet fine-tuning. With the defaults, building the
dataset takes about 10 min and training about 1–2 h on an A100 (an estimate, not measured). Everything persists on Google Drive, so a dropped
session resumes where it stopped. Expect lower quality than the paper at this scale. Raise `N_GIFS` and `MAX_STEPS` to close the gap.

**Run order:** 1 → 2 → 3 → 4 → 5 (train) → 6 (make GIFs) → 7 (evaluate).
""")
code(read("c1_setup.py"))
code(read("c2_config.py"))
md("### Dataset\nTGIF GIF URLs (tumblr) still resolve as of Oct 2026. Pairs are saved in the official JSON format "
   "(`[src, tgt, caption, flow]`, split by GIF), so the official `edit_dataset.py` can read them too.")
code(read("c3_data.py"))
code(read("c4_model.py"))
md("### Training\nRe-running this cell resumes from the last checkpoint on Drive. To train longer, raise `MAX_STEPS` in cell 2 "
   "and re-run cells 2 and 5. The optimizer state isn't saved (it's about 7 GB), so it restarts on resume.")
code(read("c5_train.py"))
md("### Generate\nPix2Gif is **image → GIF**. For text-only input, pick `text (sd-turbo)` to create the first frame. Larger or more "
   "spread-out motion values give bigger movement. `IMAGE_CFG` in 1.6–2.2 is the paper's best range: higher sticks closer to the source.")
code(read("c6_infer.py"))
md("### Evaluate\nThis covers the paper's **motion coherency** (L2 and Pearson correlation between the requested motion and the LK flow measured between the "
   "source and each generated frame) and **CLIPSim**, on held-out TGIF GIFs. FVD and PIC need UCF-101 / MSR-VTT. The official scripts "
   "(`calculate_fvd.py`, `metrics/`) are in `/content/Pix2Gif-official`.")
code(read("c7_eval.py"))

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
