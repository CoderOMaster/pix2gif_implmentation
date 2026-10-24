import os

import numpy as np
import torch
from PIL import Image

from .model import motion_prompt

DEFAULT_MOTIONS = (2, 4, 6, 8, 11, 13, 16, 19)


def load_checkpoint(cfg, c, use_ema=True):
    if not os.path.exists(cfg.ckpt_path):
        print("no checkpoint found: using current weights")
        return None
    ck = torch.load(cfg.ckpt_path, map_location="cpu")
    sd = ck["ema"] if use_ema and ck.get("ema") else ck["model"]
    full = c.model.state_dict()
    full.update({k: v.to(full[k].dtype) for k, v in sd.items() if k in full})
    c.model.load_state_dict(full)
    print(f"loaded step {ck['step']} ({'EMA' if use_ema and ck.get('ema') else 'raw'} weights)")
    return ck["step"]


def motion_sweep(n_frames=8, max_motion=19.0, min_motion=2.0):
    """Evenly spaced (fractional) motion values: more frames = smaller steps between frames."""
    return [float(m) for m in np.linspace(min_motion, max_motion, n_frames)]


@torch.no_grad()
def generate(cfg, c, image, caption, motions=DEFAULT_MOTIONS, steps=50, seed=1371,
             text_cfg=7.5, image_cfg=1.8, sampler="euler_a", strength=1.0, shared_noise=True):
    """One frame per motion value, each generated from the same source image, with 3-way CFG
    [text+image, image only, unconditional].

    sampler="euler_a" (paper): ancestral sampling injects fresh noise every step, so frames flicker.
    sampler="euler": deterministic; with shared noise, neighbouring motion values give neighbouring frames.
    strength < 1: start every frame from the source latent noised to that level (img2img), anchoring all frames to
    the source image; lower = steadier but less motion.
    Fractional motion values are supported (embedding is interpolated; the prompt uses the rounded value)."""
    from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
    model, dev = c.model.eval(), cfg.device
    w, h = image.size
    src = torch.from_numpy(np.array(image.convert("RGB").resize((cfg.res, cfg.res), Image.Resampling.LANCZOS)))
    src = (src.permute(2, 0, 1).float() / 127.5 - 1)[None].to(dev)
    src_lat = c.vae.encode(src).latent_dist.mode()
    null = c.encode_text([""])
    img = torch.cat([src_lat, src_lat, torch.zeros_like(src_lat)])
    Sched = EulerAncestralDiscreteScheduler if sampler == "euler_a" else EulerDiscreteScheduler
    sch = Sched.from_config(c.noise_scheduler.config, timestep_spacing="linspace")
    frames = []
    for k, m in enumerate(motions):
        g = torch.Generator(dev).manual_seed(seed if shared_noise else seed + k)
        txt = torch.cat([c.encode_text([motion_prompt(caption, round(m))]), null, null])
        mot = torch.full((3,), float(m), device=dev)
        sch.set_timesteps(steps, device=dev)
        noise = torch.randn(src_lat.shape, generator=g, device=dev)
        start = min(int(round(steps * (1 - strength))), steps - 1)
        timesteps = sch.timesteps[start:]
        if start > 0:
            sch.set_begin_index(start)
            lat = sch.add_noise(src_lat * c.scale, noise, timesteps[:1])
        else:
            lat = noise * sch.init_noise_sigma
        for t in timesteps:
            inp = sch.scale_model_input(lat, t).repeat(3, 1, 1, 1)
            with torch.autocast(dev, dtype=cfg.dtype):
                eps, _ = model(inp, t.expand(3), txt, mot, img)
            e_c, e_i, e_u = eps.float().chunk(3)
            lat = sch.step(e_u + text_cfg * (e_c - e_i) + image_cfg * (e_i - e_u), t, lat, generator=g).prev_sample
        x = c.vae.decode(lat / c.scale).sample.clamp(-1, 1)
        x = ((x[0].permute(1, 2, 0) + 1) * 127.5).round().byte().cpu().numpy()
        frames.append(Image.fromarray(x).resize((w, h), Image.Resampling.LANCZOS))
    return frames


def save_gif(frames, path, fps=4, boomerang=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    seq = frames + frames[-2:0:-1] if boomerang else frames
    seq[0].save(path, save_all=True, append_images=seq[1:], duration=int(1000 / fps), loop=0)
    return path


def text_to_image(prompt, device="cuda"):
    """First frame from text (Pix2Gif itself is image -> GIF)."""
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16).to(device)
    image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    del pipe
    torch.cuda.empty_cache()
    return image
