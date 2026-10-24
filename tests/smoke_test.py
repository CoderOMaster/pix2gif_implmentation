"""CPU smoke test: real TGIF download + tiny random models through dataset -> train -> resume -> generate -> evaluate.
Run from the repo root:  python tests/smoke_test.py"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from PIL import Image

from pix2gif import Config, evaluate, generate, load_checkpoint, load_or_build, save_gif, train
from pix2gif.model import assemble

work = tempfile.mkdtemp()
cfg = Config(n_gifs=8, max_steps=3, batch_size=2, grad_accum=2, save_every=2, warmup_steps=2, num_workers=0,
             use_8bit_adam=False, device="cpu", dtype=torch.bfloat16,
             data_dir=f"{work}/data", out_dir=f"{work}/out")


def tiny_components():
    unet = UNet2DConditionModel(sample_size=32, in_channels=8, out_channels=4, block_out_channels=(32, 64),
                                layers_per_block=1, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=768,
                                attention_head_dim=8, norm_num_groups=8)
    vae = AutoencoderKL(block_out_channels=(16,) * 4, down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4, norm_num_groups=8)
    te = CLIPTextModel(CLIPTextConfig(hidden_size=768, intermediate_size=256, num_hidden_layers=1,
                                      num_attention_heads=12, max_position_embeddings=77))
    sched = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    return assemble(cfg, CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"), te, vae, sched, unet)


split = load_or_build(cfg)
print("pairs:", len(split["train"]), len(split["val"]))
assert split["train"] and split["val"]

c = tiny_components()
train(cfg, c, split, perceptual_pretrained=False)
assert os.path.exists(cfg.ckpt_path)
cfg.max_steps = 4
train(cfg, c, split, perceptual_pretrained=False)  # resume path

assert load_checkpoint(cfg, c) == 4
frames = generate(cfg, c, Image.open(split["val"][0][0]), split["val"][0][2], steps=3)
assert len(frames) == 8
print("gif:", save_gif(frames, f"{cfg.out_dir}/sample.gif"))

metrics, samples = evaluate(cfg, c, split, n=2, steps=2, clip=False)
print({k: v for k, v in metrics.items() if k not in ("asked", "measured")})
shutil.rmtree(work)
print("SMOKE TEST PASSED")
