"""Text -> GIF: a free text-to-image model makes the first frame from the prompt, the LTX LoRA animates it with the
same prompt, then loop handling + GIF export.

The LoRA was trained only for image-to-video (first frame always given), so text-to-GIF goes through an image first.
Images come from FLUX.1-schnell (Apache-2.0) through Hugging Face's inference API by default (runs on HF's servers,
so it does not compete with LTX for GPU memory), or locally with diffusers (backend="local")."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import gif, ltx
from .tgif import center_square
from .worker import get_worker

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
_local = {}


def hf_image(prompt, token, model=IMAGE_MODEL, seed=42, size=1024, steps=4):
    """Generate an image with Hugging Face Inference Providers (uses the HF token's free monthly credits)."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="auto", api_key=token)
    return client.text_to_image(prompt, model=model, width=size, height=size, num_inference_steps=steps,
                                seed=seed).convert("RGB")


def local_image(prompt, model=IMAGE_MODEL, seed=42, size=1024, steps=4, token=None):
    """Generate an image on this GPU with diffusers. If the LTX worker holds most of the GPU, layers are streamed
    from CPU one at a time (slower, a few GB of GPU memory); otherwise whole components are offloaded."""
    import torch
    from diffusers import AutoPipelineForText2Image
    if model not in _local:
        _local[model] = {"pipe": AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.bfloat16,
                                                                           token=token), "mode": None}
    entry = _local[model]
    mode = "model" if torch.cuda.mem_get_info()[0] / 2**30 > 36 else "sequential"
    if mode != entry["mode"]:
        if mode == "model":
            entry["pipe"].enable_model_cpu_offload()
        else:
            entry["pipe"].enable_sequential_cpu_offload()
        entry["mode"] = mode
    g = torch.Generator("cpu").manual_seed(seed)
    return entry["pipe"](prompt, width=size, height=size, num_inference_steps=steps, guidance_scale=0.0,
                         generator=g).images[0].convert("RGB")


def text_to_gif(cfg, prompt, lora=None, seed=42, loop="auto", backend="hf", token=None, image_model=IMAGE_MODEL,
                image=None, max_kb=2048, persistent=True):
    """prompt -> first frame (text-to-image) -> LTX (+ LoRA) clip with the same prompt -> looping GIF.

    backend: "hf" (Hugging Face inference API, needs `token`) or "local" (diffusers on this GPU).
    image: a PIL image or path to skip text-to-image and animate your own first frame.
    lora=None runs the base model. persistent=True keeps LTX loaded between calls (ltxgif.worker.stop_worker()).
    Returns {"image", "mp4", "gif", "prompt", ...}."""
    out_dir = Path(cfg.work_dir) / "text2gif"
    out_dir.mkdir(parents=True, exist_ok=True)
    if image is not None:
        image = Image.open(image) if isinstance(image, (str, Path)) else image
        frame_key = hashlib.md5(np.asarray(image.convert("RGB")).tobytes()).hexdigest()[:10]
    else:
        frame_key = hashlib.md5(f"{prompt}|{seed}|{image_model}".encode()).hexdigest()[:10]
    src = out_dir / f"{frame_key}_frame.png"
    if image is None and not src.exists():
        if backend == "hf":
            image = hf_image(prompt, token, image_model, seed)
        elif backend == "local":
            image = local_image(prompt, image_model, seed, token=token)
        else:
            raise ValueError(f"backend must be 'hf' or 'local', got {backend!r}")
    if image is not None:
        Image.fromarray(center_square(np.asarray(image.convert("RGB")), cfg.size)).save(src)

    tag = "base" if lora is None else Path(lora).stem.split("_")[-1]
    prompt_key = hashlib.md5(prompt.encode()).hexdigest()[:6]
    output = out_dir / f"{frame_key}_{prompt_key}_{tag}_{seed}.mp4"
    if output.exists():
        mp4 = str(output)
    elif persistent:
        mp4 = get_worker(cfg, lora).generate(src, prompt, seed, output)
    else:
        mp4 = ltx.generate(cfg, [{"image": str(src), "prompt": prompt, "seed": seed, "output": str(output)}],
                           lora=lora, jobs_file=out_dir / f"{frame_key}_jobs.json")[0]
    info = gif.video_to_gif(mp4, mp4.replace(".mp4", ".gif"), cfg.fps, loop=loop, max_kb=max_kb)
    return {"image": str(src), "mp4": mp4, "gif": info["path"], "prompt": prompt,
            **{k: v for k, v in info.items() if k != "path"}}
