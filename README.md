# Pix2Gif implementation (Colab)

A Colab reimplementation of **Pix2Gif: Motion-Guided Diffusion for GIF Generation** (Kandala, Gao, Yang, Microsoft Research,
[arXiv 2403.04634](https://arxiv.org/abs/2403.04634)): image + caption + motion magnitude → animated GIF.

[Open `Pix2Gif_Replication.ipynb` in Colab](https://colab.research.google.com/github/CoderOMaster/pix2gif_implmentation/blob/main/Pix2Gif_Replication.ipynb), select an **A100** (or L4) runtime, and run cells 1 → 7.

## Why a reimplementation

* The authors removed the code from [hiteshK03/Pix2Gif](https://github.com/hiteshK03/Pix2Gif) in Sept 2024 (commit `0be59b6`).
  The last commit that still has it is **`781ca78`**, and the model code here is ported from it.
* **No trained weights were ever released.** The only checkpoint references in the repo history point to the authors' private machines.
* The TGIF pair-mining script was never published, so it is rebuilt here from the paper's Algorithm 1.

So the notebook **rebuilds the dataset and trains the model** at Colab scale.

## Pipeline

| Cell | What it does |
|---|---|
| 1 Setup | GPU check, installs, Google Drive mount, clones the official code at `781ca78` for reference |
| 2 Config | Data scale, init (InstructPix2Pix or SD-1.5), warp on/off (ablation), training hyperparameters |
| 3 Dataset | Downloads TGIF GIFs, finds Lucas–Kanade motion pairs with flow in [2, 20], ≤10 per GIF, saves in the official JSON format, tars to Drive |
| 4 Model | FlowNet / WarpNet / InjectionModule / IntervalEmbedding / 4-channel VGG perceptual loss (ported line by line) on a diffusers UNet with a 12-channel input |
| 5 Train | L_LDM + 0.01·L_perceptual, official CFG dropout, AdamW 1e-4 (8-bit optional), bf16, LitEma, atomic checkpoints, auto-resume |
| 6 Generate | Official `demo.py` sampling: Euler ancestral, text CFG 7.5 / image CFG 1.8, one frame per motion value from the same source; optional text→first frame (sd-turbo) |
| 7 Evaluate | Motion coherency (L2 / PCC between requested and measured flow) and CLIPSim on held-out GIFs |

## Fidelity vs. the paper

Faithful: the architecture, prompt template (`"<caption> The optical flow is <words>."`), 12-channel "Warp-concat" input, hybrid conditioning,
perceptual loss (layers 0/5/10/19/28, λ = 0.01), CFG dropout 5/5/5%, lr 1e-4, 256² resolution, motion range [2, 20], and the sampler settings.

Scaled or changed for Colab: initialised from InstructPix2Pix by default (SD-1.5 is selectable); ~3k GIFs and 5k steps at effective batch 32
(the paper used 783k pairs, ~24k steps at batch 256 on 16×V100); 8-bit AdamW and gradient checkpointing; the LK threshold and
feature parameters aren't stated in the paper, so they are chosen here.

## Repo layout

```
Pix2Gif_Replication.ipynb   # the Colab notebook (built from src/)
src/c1_setup.py … c7_eval.py # notebook cells as plain Python
src/build.py                # rebuilds the notebook: python src/build.py Pix2Gif_Replication.ipynb
src/test_cpu.py             # runs cells 2–7 on CPU with tiny random models (smoke test, needs network)
```

## Status

Every cell passes a CPU smoke test: real TGIF download, train, resume, generate and eval, using tiny random-weight models.
It has not yet been trained on a GPU at full scale, so there are no quality numbers yet.

## Citation

```bibtex
@misc{kandala2024pix2gif,
  title={Pix2Gif: Motion-Guided Diffusion for GIF Generation},
  author={Hitesh Kandala and Jianfeng Gao and Jianwei Yang},
  year={2024}, eprint={2403.04634}, archivePrefix={arXiv}, primaryClass={cs.CV}
}
```
