#@title 7 · Evaluate (paper Sec. 4: motion coherency L2 / PCC, CLIPSim)
from transformers import CLIPModel, CLIPProcessor

N_EVAL = 30  #@param {type:"integer"}
EVAL_MOTIONS = [2, 4, 6, 8, 11, 14, 17, 19]  # the paper's ablation motion list

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(DEVICE).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


@torch.no_grad()
def clip_sim(frames, caption):
    inp = clip_proc(text=[caption], images=frames, return_tensors="pt", padding=True).to(DEVICE)
    out = clip(**{k: v.half() if v.dtype == torch.float32 else v for k, v in inp.items()})
    i = F.normalize(out.image_embeds.float(), dim=-1)
    t = F.normalize(out.text_embeds.float(), dim=-1)
    return float((i @ t.T).mean())


rows_eval = random.Random(0).sample(split["val"], min(N_EVAL, len(split["val"])))
asked, measured, sims = [], [], []
for k, (s, _, cap, _) in enumerate(rows_eval):
    img = Image.open(s).convert("RGB")
    frames = pix2gif(img, cap, EVAL_MOTIONS, steps=STEPS, text_cfg=TEXT_CFG, image_cfg=IMAGE_CFG)
    src = np.asarray(img.resize((RES, RES)))
    for m, f in zip(EVAL_MOTIONS, frames):
        mm = flow_magnitude(src, np.asarray(f.resize((RES, RES))))  # same Algorithm 1 used to label the data
        if mm is not None:
            asked.append(m)
            measured.append(mm)
    sims.append(clip_sim(frames, cap))
    if k < 4:
        show(save_gif(frames, f"/content/eval_{k}.gif"), 192)

asked, measured = np.array(asked, float), np.array(measured, float)
print(f"motion coherency  L2 = {np.sqrt(np.mean((asked - measured) ** 2)):.2f}  "
      f"PCC = {np.corrcoef(asked, measured)[0, 1]:.3f}   (n={len(asked)} frames)")
print(f"CLIPSim (frames vs caption) = {np.mean(sims):.4f}")
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(asked, measured, s=8, alpha=0.5)
ax.plot([0, 20], [0, 20], "k--", lw=1)
ax.set_xlabel("input motion")
ax.set_ylabel("measured flow (LK)")
ax.set_title("motion control")
plt.show()
