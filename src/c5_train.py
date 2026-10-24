#@title 5 · Train (resumes automatically from the Drive checkpoint)
import copy, math
from torch.utils.data import DataLoader, Dataset


class MotionPairs(Dataset):
    """edit_dataset.EditDatasetMotionInterval: (source, target, caption, flow) -> 256² tensors, shared random h-flip."""
    def __init__(self, rows, flip_prob=0.5):
        self.rows, self.flip_prob = rows, flip_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        s, t, cap, m = self.rows[i]
        load = lambda p: torch.from_numpy(np.asarray(
            Image.open(p).convert("RGB").resize((RES, RES), Image.Resampling.LANCZOS))).permute(2, 0, 1).float() / 127.5 - 1
        s, t = load(s), load(t)
        if random.random() < self.flip_prob:
            s, t = s.flip(-1), t.flip(-1)
        return {"src": s, "tgt": t, "prompt": motion_prompt(cap, m), "motion": int(m)}


class EMA:
    """LitEma from the official code: decay = min(0.9999, (1 + n) / (10 + n))."""
    def __init__(self, module):
        self.shadow = {k: v.detach().clone().float() for k, v in module.state_dict().items() if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, module):
        self.n += 1
        d = min(0.9999, (1 + self.n) / (10 + self.n))
        for k, v in module.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.float(), alpha=1 - d)


def save_ckpt(step, ema, opt=None):
    state = {"step": step, "model": {k: v.half() if v.dtype == torch.float32 else v for k, v in model.state_dict().items()},
             "ema": {k: v.half() for k, v in ema.shadow.items()} if ema else None,
             "config": {"INIT": INIT, "USE_WARP": USE_WARP, "RES": RES}}
    torch.save(state, CKPT_PATH + ".tmp")
    os.replace(CKPT_PATH + ".tmp", CKPT_PATH)  # atomic: a Colab disconnect can't corrupt it


torch.manual_seed(SEED)
random.seed(SEED)
loader = DataLoader(MotionPairs(split["train"]), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True, persistent_workers=True)
perceptual = LatentPerceptual().to(DEVICE)
model.unet.enable_gradient_checkpointing()
params = [p for p in model.parameters() if p.requires_grad]
if USE_8BIT_ADAM:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=LR)
else:
    opt = torch.optim.AdamW(params, lr=LR)
lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP_STEPS))
scaler = torch.amp.GradScaler(enabled=DTYPE == torch.float16)
ema = EMA(model) if USE_EMA else None
null_emb = encode_text([""])
step = 0
if os.path.exists(CKPT_PATH):
    ck = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict({k: v.float() if v.dtype == torch.float16 else v for k, v in ck["model"].items()})
    if ema and ck.get("ema"):
        ema.shadow = {k: v.float().to(DEVICE) for k, v in ck["ema"].items()}
        ema.n = ck["step"]
    step = ck["step"]
    for _ in range(step):
        lr_sched.step()
    print(f"resumed from step {step} (optimizer state restarts)")

model.train()
log, t0, start_step = [], time.time(), step
while step < MAX_STEPS:
    for i, b in enumerate(loader):
        src, tgt = b["src"].to(DEVICE, non_blocking=True), b["tgt"].to(DEVICE, non_blocking=True)
        bs = src.shape[0]
        with torch.no_grad():
            z = vae.encode(tgt).latent_dist.sample() * SCALE          # target: scaled sample (get_first_stage_encoding)
            src_lat = vae.encode(src).latent_dist.mode()             # condition: unscaled mode (encode_first_stage().mode())
            text = encode_text(b["prompt"])
            r = torch.rand(bs, device=DEVICE)                        # official CFG dropout
            text = torch.where((r < 2 * UNCOND_P)[:, None, None], null_emb, text)
            src_lat = src_lat * (1 - ((r >= UNCOND_P) & (r < 3 * UNCOND_P)).float())[:, None, None, None]
            noise = torch.randn_like(z)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (bs,), device=DEVICE)
            noisy = noise_sched.add_noise(z, noise, t)
        with torch.autocast(DEVICE, dtype=DTYPE):
            pred, warped = model(noisy, t, text, b["motion"].to(DEVICE), src_lat)
        loss_ldm = F.mse_loss(pred.float(), noise)
        loss_p = perceptual(warped.float(), src_lat) if USE_WARP else torch.zeros((), device=DEVICE)
        loss = (loss_ldm + PERCEPTUAL_WEIGHT * loss_p) / GRAD_ACCUM   # L_T = L'_LDM + λ_P · L_P
        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        lr_sched.step()
        ema and ema.update(model)
        step += 1
        log.append((loss_ldm.item(), loss_p.item()))
        if step % 50 == 0:
            l = np.mean(log[-50:], axis=0)
            print(f"step {step:>6}/{MAX_STEPS} · ldm {l[0]:.4f} · perceptual {l[1]:.4f} · "
                  f"{(time.time() - t0) / (step - start_step):.2f}s/step")
        if step % SAVE_EVERY == 0 or step == MAX_STEPS:
            save_ckpt(step, ema)
            print("  saved", CKPT_PATH)
        if step >= MAX_STEPS:
            break
model.eval()
