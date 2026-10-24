import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import MotionPairs
from .model import LatentPerceptual


class EMA:
    """Exponential moving average with warm-up: decay = min(0.9999, (1 + n) / (10 + n))."""
    def __init__(self, module):
        self.shadow = {k: v.detach().clone().float() for k, v in module.state_dict().items()
                       if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, module):
        self.n += 1
        d = min(0.9999, (1 + self.n) / (10 + self.n))
        for k, v in module.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.float(), alpha=1 - d)


def _enable_checkpointing(unet, mode):
    """Activation checkpointing for the UNet. diffusers defaults to non-reentrant checkpointing, which checks that the
    recomputed forward saves exactly the same tensors as the original; under autocast that check can fail with
    'A different number of tensors was saved'. Reentrant checkpointing recomputes without that check."""
    if mode == "off":
        return
    if mode == "non-reentrant":
        unet.enable_gradient_checkpointing()
        return
    from torch.utils.checkpoint import checkpoint
    unet.enable_gradient_checkpointing(
        gradient_checkpointing_func=lambda module, *args: checkpoint(module.__call__, *args, use_reentrant=True))


def save_checkpoint(cfg, model, step, ema=None):
    os.makedirs(cfg.out_dir, exist_ok=True)
    half = lambda sd: {k: v.half() if v.dtype == torch.float32 else v for k, v in sd.items()}
    state = {"step": step, "model": half(model.state_dict()), "ema": half(ema.shadow) if ema else None,
             "config": {"init": cfg.init, "use_warp": cfg.use_warp, "flow_residual": cfg.flow_residual,
                        "res": cfg.res}}
    torch.save(state, cfg.ckpt_path + ".tmp")
    os.replace(cfg.ckpt_path + ".tmp", cfg.ckpt_path)


def train(cfg, c, split, perceptual_pretrained=True):
    """Fine-tune Pix2Gif on motion pairs. Resumes from cfg.ckpt_path if it exists."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model, dev = c.model, cfg.device
    loader = DataLoader(MotionPairs(split["train"], cfg.res), batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=dev == "cuda",
                        persistent_workers=cfg.num_workers > 0)
    perceptual = LatentPerceptual(pretrained=perceptual_pretrained).to(dev)
    _enable_checkpointing(model.unet, cfg.grad_checkpointing)
    params = [p for p in model.parameters() if p.requires_grad]
    # Clip the UNet and the flow branch separately: the perceptual loss is orders of magnitude larger than the
    # diffusion loss, so one joint clip would shrink the UNet's updates whenever FlowNet's gradients are big.
    unet_params = [p for p in model.unet.parameters() if p.requires_grad]
    flow_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("unet.")]
    if cfg.use_8bit_adam:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=cfg.lr)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps))
    scaler = torch.amp.GradScaler(enabled=cfg.dtype == torch.float16 and dev == "cuda")
    ema = EMA(model) if cfg.use_ema else None
    null_emb = c.encode_text([""])

    step = 0
    if os.path.exists(cfg.ckpt_path):
        ck = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict({k: v.float() if v.dtype == torch.float16 else v for k, v in ck["model"].items()})
        if ema and ck.get("ema"):
            ema.shadow, ema.n = {k: v.float().to(dev) for k, v in ck["ema"].items()}, ck["step"]
        step = ck["step"]
        for _ in range(step):
            lr_sched.step()
        print(f"resumed from step {step}")

    model.train()
    log, t0, start = [], time.time(), step
    while step < cfg.max_steps:
        for i, b in enumerate(loader):
            src, tgt = b["src"].to(dev, non_blocking=True), b["tgt"].to(dev, non_blocking=True)
            bs = src.shape[0]
            with torch.no_grad():
                z = c.vae.encode(tgt).latent_dist.sample() * c.scale    # target latent (scaled)
                src_lat = c.vae.encode(src).latent_dist.mode()          # condition latent (unscaled)
                text = c.encode_text(b["prompt"])
                r = torch.rand(bs, device=dev)                          # classifier-free guidance dropout
                text = torch.where((r < 2 * cfg.uncond_p)[:, None, None], null_emb, text)
                keep_img = 1 - ((r >= cfg.uncond_p) & (r < 3 * cfg.uncond_p)).float()
                src_lat = src_lat * keep_img[:, None, None, None]
                noise = torch.randn_like(z)
                t = torch.randint(0, c.noise_scheduler.config.num_train_timesteps, (bs,), device=dev)
                noisy = c.noise_scheduler.add_noise(z, noise, t)
            with torch.autocast(dev, dtype=cfg.dtype):
                pred, warped = model(noisy, t, text, b["motion"].to(dev), src_lat)
            loss_ldm = F.mse_loss(pred.float(), noise)
            loss_p = perceptual(warped.float(), src_lat) if cfg.use_warp else torch.zeros((), device=dev)
            scaler.scale((loss_ldm + cfg.perceptual_weight * loss_p) / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum:
                continue
            scaler.unscale_(opt)
            gn_unet = float(torch.nn.utils.clip_grad_norm_(unet_params, 1.0))
            gn_flow = float(torch.nn.utils.clip_grad_norm_(flow_params, 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            lr_sched.step()
            if ema:
                ema.update(model)
            step += 1
            log.append((loss_ldm.item(), loss_p.item(), gn_unet, gn_flow))
            if step % 50 == 0:
                l = np.mean(log[-50:], axis=0)
                print(f"step {step:>6}/{cfg.max_steps} · ldm {l[0]:.4f} · perceptual {l[1]:.1f} · "
                      f"grad-norm unet {l[2]:.2f} flow {l[3]:.1f} · {(time.time() - t0) / (step - start):.2f}s/step")
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                save_checkpoint(cfg, model, step, ema)
                print("  saved", cfg.ckpt_path)
            if step >= cfg.max_steps:
                break
    model.eval()
    return log
