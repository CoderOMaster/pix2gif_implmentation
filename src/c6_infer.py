#@title 6 · Generate a GIF (demo.py sampling: Euler ancestral, 3-way CFG, one frame per motion value)
from diffusers import EulerAncestralDiscreteScheduler
from IPython.display import HTML, display
import base64

USE_EMA_WEIGHTS = True  #@param {type:"boolean"}


def load_for_inference(path=CKPT_PATH, use_ema=USE_EMA_WEIGHTS):
    if not os.path.exists(path):
        print("no checkpoint yet: using the current in-memory weights")
        return
    ck = torch.load(path, map_location="cpu")
    sd = ck["ema"] if use_ema and ck.get("ema") else ck["model"]
    full = model.state_dict()
    full.update({k: v.to(full[k].dtype) for k, v in sd.items() if k in full})
    model.load_state_dict(full)
    print(f"loaded step {ck['step']} ({'EMA' if use_ema and ck.get('ema') else 'raw'} weights)")


@torch.no_grad()
def pix2gif(image, caption, motions=(2, 4, 6, 8, 11, 13, 16, 19), steps=100, seed=1371,
            text_cfg=7.5, image_cfg=1.8, shared_noise=True):
    """Every frame is generated from the SAME source image, with a different motion value (as in demo.py).
    shared_noise=True re-seeds per frame like demo.py (more consistent frames); False = demo_parallel.py."""
    model.eval()
    w, h = image.size
    src = torch.from_numpy(np.asarray(image.convert("RGB").resize((RES, RES), Image.Resampling.LANCZOS)))
    src = (src.permute(2, 0, 1).float() / 127.5 - 1)[None].to(DEVICE)
    src_lat = vae.encode(src).latent_dist.mode()
    null = encode_text([""])
    sch = EulerAncestralDiscreteScheduler.from_config(noise_sched.config, timestep_spacing="linspace")
    frames = []
    for k, m in enumerate(motions):
        g = torch.Generator(DEVICE).manual_seed(seed if shared_noise else seed + k)
        cond = encode_text([motion_prompt(caption, m)])
        txt = torch.cat([cond, null, null])                                    # [cond, img-only, uncond]
        img = torch.cat([src_lat, src_lat, torch.zeros_like(src_lat)])
        iv = torch.full((3,), int(m), device=DEVICE)                           # motion embedding on all 3 branches
        sch.set_timesteps(steps, device=DEVICE)
        lat = torch.randn(src_lat.shape, generator=g, device=DEVICE) * sch.init_noise_sigma
        for t in sch.timesteps:
            inp = sch.scale_model_input(lat, t).repeat(3, 1, 1, 1)
            with torch.autocast(DEVICE, dtype=DTYPE):
                eps, _ = model(inp, t.expand(3), txt, iv, img)
            e_c, e_i, e_u = eps.float().chunk(3)
            eps = e_u + text_cfg * (e_c - e_i) + image_cfg * (e_i - e_u)
            lat = sch.step(eps, t, lat, generator=g).prev_sample
        x = vae.decode(lat / SCALE).sample.clamp(-1, 1)
        x = ((x[0].permute(1, 2, 0) + 1) * 127.5).round().byte().cpu().numpy()
        frames.append(Image.fromarray(x).resize((w, h), Image.Resampling.LANCZOS))
    return frames


def save_gif(frames, path, fps=4, boomerang=False):
    seq = frames + frames[-2:0:-1] if boomerang else frames
    seq[0].save(path, save_all=True, append_images=seq[1:], duration=int(1000 / fps), loop=0)
    return path


def show(path, width=320):
    b = base64.b64encode(open(path, "rb").read()).decode()
    display(HTML(f"<img src='data:image/gif;base64,{b}' width={width}>"))


load_for_inference()

SOURCE = "val"  #@param ["val", "upload", "url", "text (sd-turbo)"]
URL = ""  #@param {type:"string"}
CAPTION = "a cat is dancing."  #@param {type:"string"}
MOTIONS = "2,4,6,8,11,13,16,19"  #@param {type:"string"}
STEPS = 50  #@param {type:"slider", min:20, max:100, step:5}
TEXT_CFG = 7.5  #@param {type:"number"}
IMAGE_CFG = 1.8  #@param {type:"number"}
FPS = 4  #@param {type:"integer"}
BOOMERANG = False  #@param {type:"boolean"}

if SOURCE == "val":
    s, _, cap, _ = random.choice(split["val"])
    image, CAPTION = Image.open(s).convert("RGB"), cap
elif SOURCE == "upload":
    from google.colab import files
    image = Image.open(io.BytesIO(next(iter(files.upload().values())))).convert("RGB")
elif SOURCE == "url":
    image = Image.open(io.BytesIO(requests.get(URL, timeout=30).content)).convert("RGB")
else:  # Pix2Gif is image-to-GIF; for text-only input, make the first frame with a fast T2I model
    from diffusers import AutoPipelineForText2Image
    t2i = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16).to(DEVICE)
    image = t2i(CAPTION, num_inference_steps=1, guidance_scale=0.0).images[0]
    del t2i
    torch.cuda.empty_cache()

frames = pix2gif(image, CAPTION, [int(x) for x in MOTIONS.split(",")], steps=STEPS,
                 text_cfg=TEXT_CFG, image_cfg=IMAGE_CFG)
out = save_gif(frames, f"{ROOT}/sample_{int(time.time())}.gif", fps=FPS, boomerang=BOOMERANG)
print(CAPTION, "->", out)
image.resize((256, int(256 * image.height / image.width))).save("/content/_src.gif")
show("/content/_src.gif", 256)
show(out, 256)
