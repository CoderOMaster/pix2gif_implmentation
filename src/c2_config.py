#@title 2 · Config
# ---- data (paper: 783,184 train pairs from TGIF) ----
N_GIFS = 3000  #@param {type:"integer"}
PAIRS_PER_GIF = 10         # paper: ">= 10 pairs per GIF (or all, if fewer)"
MOTION_RANGE = (2, 20)     # paper Sec. 3.3: keep pairs whose mean flow magnitude is in [2, 20]
FLOW_EPS = 0.5             # Algorithm 1 drops "small" displacements; paper doesn't give the value
RES = 256                  # paper / official config
VAL_FRACTION = 0.05

# ---- model ----
INIT = "instruct-pix2pix"  #@param ["instruct-pix2pix", "sd-1.5 (paper)"]
USE_WARP = True  #@param {type:"boolean"}
# USE_WARP=False is the paper's "Pix2Gif-Motion-embed" ablation (no FlowNet/WarpNet, 8-ch UNet input)
PERCEPTUAL_WEIGHT = 0.01   # paper λ_P (ddpm_edit.py: per_lambda = 0.01)
UNCOND_P = 0.05            # official CFG dropout: 5% text-only, 5% image-only, 5% both

# ---- training (paper: 16×V100, eff. batch 2×8 accum×16 GPUs = 256, ~24k steps, AdamW 1e-4) ----
MAX_STEPS = 5000  #@param {type:"integer"}
BATCH_SIZE = 16  #@param {type:"integer"}
GRAD_ACCUM = 2  #@param {type:"integer"}
LR = 1e-4
WARMUP_STEPS = 500         # official: 10k warm-up at eff. batch 256; scaled down
USE_8BIT_ADAM = True  #@param {type:"boolean"}
USE_EMA = True  #@param {type:"boolean"}
SAVE_EVERY = 1000  #@param {type:"integer"}
SEED = 0

MODEL_IDS = {"instruct-pix2pix": "timbrooks/instruct-pix2pix",
             "sd-1.5 (paper)": "stable-diffusion-v1-5/stable-diffusion-v1-5"}
VAE_ID = "stabilityai/sd-vae-ft-mse"  # paper: "ft-MSE autoencoder"
DATA_DIR = "/content/tgif_pairs"
CKPT_PATH = f"{ROOT}/pix2gif_{'warp' if USE_WARP else 'noflow'}_{INIT.split()[0]}.pt"
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("checkpoint ->", CKPT_PATH, "| autocast", DTYPE)
