"""Text -> GIF: Gemini makes the first frame, the LTX LoRA animates it, then loop handling + GIF export.

The LoRA was trained only for image-to-video (first frame always given), so text-to-GIF goes through an image first.
One idea can be split by Gemini into a scene prompt (what the first frame shows) and a motion prompt written like the
training captions (subject + action + direction + camera)."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import gif, ltx
from .tgif import center_square
from .worker import get_worker

PLAN_PROMPT = """Turn this idea for a short looping animated GIF into two prompts.
Idea: "{idea}"

scene_prompt: a detailed description of the single still first frame for an image generator (subject, appearance,
setting, lighting, style, composition with the subject clearly visible). No motion words.
motion_prompt: one to three sentences describing only the motion that starts from that frame, in this form: subject
and scene, explicit action, direction or type of movement, camera behaviour (e.g. "The camera remains static.").
Keep the motion simple and short enough to happen in about two seconds."""


def plan_prompts(idea, api_key, model="gemini-flash-latest"):
    """Split one idea into {"scene_prompt", "motion_prompt"} with a Gemini text model."""
    from google import genai
    from google.genai import types
    schema = types.Schema(type="OBJECT", required=["scene_prompt", "motion_prompt"], properties={
        "scene_prompt": types.Schema(type="STRING"), "motion_prompt": types.Schema(type="STRING")})
    client = genai.Client(api_key=api_key)   # keep a reference: the client closes its connection when collected
    resp = client.models.generate_content(
        model=model, contents=PLAN_PROMPT.format(idea=idea),
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema))
    return json.loads(resp.text)


def gemini_image(prompt, api_key, model="gemini-2.5-flash-image"):
    """Generate one square image with a Gemini image model; returns a PIL image."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    try:
        conf = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                           image_config=types.ImageConfig(aspect_ratio="1:1"))
    except (AttributeError, TypeError):   # older google-genai without ImageConfig: ask for square in the prompt
        conf, prompt = types.GenerateContentConfig(response_modalities=["IMAGE"]), prompt + " Square 1:1 image."
    resp = client.models.generate_content(model=model, contents=[prompt], config=conf)
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError(f"no image returned by {model}: {resp.text if hasattr(resp, 'text') else resp}")


def text_to_gif(cfg, api_key, idea=None, scene_prompt=None, motion_prompt=None, lora=None, seed=42,
                loop="auto", image_model="gemini-2.5-flash-image", text_model="gemini-flash-latest",
                max_kb=2048, image=None, persistent=True):
    """idea -> (scene_prompt, motion_prompt) -> Gemini first frame -> LTX (+ LoRA) clip -> looping GIF.

    Give either `idea` (Gemini writes both prompts) or `scene_prompt` + `motion_prompt` yourself. Pass `image` (a PIL
    image or path) to reuse a first frame you liked and only change the motion. lora=None runs the base model.
    persistent=True keeps LTX loaded in a background worker between calls (stop it with ltxgif.worker.stop_worker()).
    Returns {"image", "mp4", "gif", "scene_prompt", "motion_prompt", ...}."""
    if not (scene_prompt and motion_prompt):
        if not idea:
            raise ValueError("give an idea, or both scene_prompt and motion_prompt")
        plan = plan_prompts(idea, api_key, text_model)
        scene_prompt, motion_prompt = scene_prompt or plan["scene_prompt"], motion_prompt or plan["motion_prompt"]
    print("scene: ", scene_prompt, "\nmotion:", motion_prompt)

    out_dir = Path(cfg.work_dir) / "text2gif"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_key = hashlib.md5(f"{scene_prompt}|{seed}".encode()).hexdigest()[:10]   # same frame for every checkpoint
    key = hashlib.md5(f"{frame_key}|{motion_prompt}|{image is not None}".encode()).hexdigest()[:10]
    src = out_dir / (f"{key}_frame.png" if image is not None else f"{frame_key}_frame.png")
    if image is not None:
        image = Image.open(image) if isinstance(image, (str, Path)) else image
    elif not src.exists():
        image = gemini_image(scene_prompt, api_key, image_model)
    if image is not None:
        Image.fromarray(center_square(np.asarray(image.convert("RGB")), cfg.size)).save(src)

    tag = "base" if lora is None else Path(lora).stem.split("_")[-1]
    output = out_dir / f"{key}_{tag}_{seed}.mp4"
    if output.exists():
        mp4 = str(output)
    elif persistent:
        mp4 = get_worker(cfg, lora).generate(src, motion_prompt, seed, output)
    else:
        mp4 = ltx.generate(cfg, [{"image": str(src), "prompt": motion_prompt, "seed": seed, "output": str(output)}],
                           lora=lora, jobs_file=out_dir / f"{key}_jobs.json")[0]
    info = gif.video_to_gif(mp4, mp4.replace(".mp4", ".gif"), cfg.fps, loop=loop, max_kb=max_kb)
    return {"image": str(src), "mp4": mp4, "gif": info["path"], "scene_prompt": scene_prompt,
            "motion_prompt": motion_prompt, **{k: v for k, v in info.items() if k != "path"}}
