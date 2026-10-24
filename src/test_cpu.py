"""Run notebook cells 2-7 on CPU with tiny random models (validates code paths, shapes, I/O; not quality)."""
import os, sys, types, shutil
import matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_test")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)


import torch
ns = {"os": os, "torch": torch, "ROOT": WORK}
src = lambda f: open(os.path.join(HERE, f)).read()

c2 = src("c2_config.py") + f"""
N_GIFS, MAX_STEPS, BATCH_SIZE, GRAD_ACCUM, SAVE_EVERY, WARMUP_STEPS = 8, 3, 2, 2, 2, 2
USE_8BIT_ADAM = False
DEVICE, DTYPE = "cpu", torch.bfloat16
DATA_DIR = "{WORK}/tgif_pairs"
CKPT_PATH = f"{{ROOT}}/test.pt"
"""
TINY = '''
def load_components():
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    te = CLIPTextModel(CLIPTextConfig(hidden_size=768, intermediate_size=256, num_hidden_layers=1,
                                      num_attention_heads=12, max_position_embeddings=77)).eval()
    vae = AutoencoderKL(block_out_channels=(16, 16, 16, 16), down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4, norm_num_groups=8).eval()
    unet = UNet2DConditionModel(sample_size=32, in_channels=8, out_channels=4, block_out_channels=(32, 64),
                                layers_per_block=1, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=768,
                                attention_head_dim=8, norm_num_groups=8)
    sched = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    unet = widen_conv_in(unet, 12 if USE_WARP else 8)
    return tok, te, vae, sched, Pix2Gif(unet, USE_WARP)

tok, text_encoder, vae, noise_sched, model = load_components()'''
c4 = src("c4_model.py").replace("weights=models.VGG19_Weights.IMAGENET1K_V1", "weights=None") \
    .replace("\n\ntok, text_encoder, vae, noise_sched, model = load_components()", "\n" + TINY)
c5 = src("c5_train.py").replace("num_workers=4", "num_workers=0").replace(", persistent_workers=True", "") \
    .replace("pin_memory=True", "pin_memory=False")
c6 = src("c6_infer.py").replace('STEPS = 50  #@param', 'STEPS = 3  #@param').replace('"/content/_src.gif"', f'"{WORK}/_src.gif"')
c7 = src("c7_eval.py").replace("N_EVAL = 30", "N_EVAL = 2") \
    .replace('clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(DEVICE).eval()', "") \
    .replace('clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")', "clip_sim = lambda f, c: 0.0") \
    .replace('f"/content/eval_{k}.gif"', f'f"{WORK}/eval_{{k}}.gif"')
c7 = c7.replace("@torch.no_grad()\ndef clip_sim", "@torch.no_grad()\ndef _unused_clip_sim")

for name, code in [("config", c2), ("data", src("c3_data.py")), ("model", c4), ("train", c5),
                   ("infer", c6), ("eval", c7)]:
    print(f"\n######## {name}")
    exec(compile(code, name, "exec"), ns)

# resume path: run training again with more steps from the saved checkpoint
print("\n######## resume")
ns["MAX_STEPS"] = 4
exec(compile(c5, "train2", "exec"), ns)
print("\nALL CELLS OK · files:", sorted(os.listdir(WORK)))
