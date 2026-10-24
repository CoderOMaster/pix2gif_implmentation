"""Model download, GPU presets, trainer configs and subprocess wrappers around the LTX-2 tools.

The LTX-2 code runs in its own uv environment (it pins its own torch build); these helpers run in the notebook's
Python and call into that environment with `uv run`."""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .config import HF_REPO, MODEL_FILES
from .tgif import first_frame

REPO_ROOT = Path(__file__).resolve().parents[1]

# LoRA on the video branch only: PEFT matches a target when the module name ends with "." + target, so
# "attn1.to_q" matches the video self-attention but not "audio_attn1.to_q".
VIDEO_LORA_TARGETS = [f"{a}.{p}" for a in ("attn1", "attn2") for p in ("to_q", "to_k", "to_v", "to_out.0")]


def run(cmd, cwd=None, env=None):
    """Run a command, streaming its output; raise if it fails."""
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env={**os.environ, **(env or {})},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
    if p.wait():
        raise RuntimeError(f"command failed with exit code {p.returncode}")


def uv_python(cfg, *args, cwd=None):
    run(["uv", "run", "--project", Path(cfg.ltx_repo).resolve(), "python", *args], cwd=cwd)


def download_models(cfg, token=None):
    from huggingface_hub import hf_hub_download
    for key, name in MODEL_FILES.items():
        path = hf_hub_download(HF_REPO, name, local_dir=cfg.models_dir, token=token)
        print(f"{key:13s} {path} ({Path(path).stat().st_size / 2**30:.1f} GiB)")


def gpu_memory_gb():
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0


def vram_preset(cfg):
    """'standard' (bf16 weights, ~80 GB) or 'low' (int8-quantised transformer, 8-bit text encoder, 8-bit AdamW;
    the trainer documents it for 32 GB cards)."""
    mode = cfg.vram_mode
    if mode == "auto":
        gb = gpu_memory_gb()
        if gb < 30:
            raise RuntimeError(f"{gb:.0f} GB GPU: LTX-2.5 LoRA training needs >= 32 GB (A100 40/80 GB or H100)")
        mode = "standard" if gb >= 70 else "low"
    low = mode == "low"
    return {"mode": mode, "quantization": "int8-quanto" if low else None, "text_encoder_8bit": low,
            "optimizer": "adamw8bit" if low else "adamw", "offload_optimizer": low}


def model_section(cfg):
    return {"model_path": str(cfg.model_path("transformer").resolve()),
            "text_encoder_path": str(cfg.model_path("text_encoder").resolve()),
            "video_vae_path": str(cfg.model_path("video_vae").resolve()),
            "audio_vae_path": str(cfg.model_path("audio_vae").resolve())}


def preprocess(cfg, split="train"):
    """Encode clips to video latents and captions to text embeddings with LTX's process_dataset.py (video only)."""
    preset = vram_preset(cfg)
    m = model_section(cfg)
    args = ["scripts/process_dataset.py", cfg.manifest(split).resolve(),
            "--resolution-buckets", f"{cfg.size}x{cfg.size}x{cfg.num_frames}",
            "--model-path", m["model_path"], "--text-encoder-path", m["text_encoder_path"],
            "--video-vae-path", m["video_vae_path"], "--audio-vae-path", m["audio_vae_path"],
            "--output-dir", cfg.precomputed_dir(split).resolve(), "--skip-audio"]
    if preset["text_encoder_8bit"]:
        args.append("--load-text-encoder-in-8bit")
    uv_python(cfg, *args, cwd=Path(cfg.ltx_repo) / "packages" / "ltx-trainer")


def validation_samples(cfg, n=None):
    """First frames + captions of held-out val clips, rendered by the trainer at each validation step."""
    rows = json.loads(cfg.manifest("val").read_text())[:n or cfg.n_val_prompts]
    out = cfg.data_dir / "val_frames"
    return [{"prompt": r["caption"], "conditions": [{"type": "first_frame",
                                                      "image_or_video": str(first_frame(cfg, r["video"], out).resolve())}]}
            for r in rows]


def train_config(cfg, name, lambda_motion, steps=None):
    """Write the trainer YAML for one run: image-to-video LoRA (first frame always given), video only."""
    preset = vram_preset(cfg)
    run_dir = cfg.run_dir(name)
    conf = {
        "model": {**model_section(cfg), "training_mode": "lora", "load_checkpoint": None},
        "lora": {"rank": cfg.lora_rank, "alpha": cfg.lora_alpha, "dropout": 0.0, "target_modules": VIDEO_LORA_TARGETS},
        "training_strategy": {"name": "flexible", "video": {
            "is_generated": True, "latents_dir": "latents",
            "conditions": [{"type": "first_frame", "probability": 1.0}]}},
        "optimization": {"learning_rate": cfg.learning_rate, "steps": steps or cfg.steps,
                         "batch_size": cfg.batch_size, "gradient_accumulation_steps": cfg.grad_accum,
                         "max_grad_norm": cfg.max_grad_norm, "optimizer_type": preset["optimizer"],
                         "scheduler_type": "linear", "scheduler_params": {}, "enable_gradient_checkpointing": True},
        "acceleration": {"mixed_precision_mode": "bf16", "quantization": preset["quantization"],
                         "load_text_encoder_in_8bit": preset["text_encoder_8bit"],
                         "offload_optimizer_during_validation": preset["offload_optimizer"]},
        "data": {"preprocessed_data_root": str(cfg.precomputed_dir("train").resolve()), "num_dataloader_workers": 2},
        "validation": {"samples": validation_samples(cfg), "video_dims": [cfg.size, cfg.size, cfg.num_frames],
                       "frame_rate": cfg.fps, "seed": cfg.seed, "inference_steps": 30,
                       "interval": cfg.validate_every, "generate_audio": False, "skip_initial_validation": False},
        "checkpoints": {"interval": cfg.checkpoint_every, "keep_last_n": -1, "precision": "bfloat16"},
        "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal", "timestep_sampling_params": {}},
        "wandb": {"enabled": False},
        "seed": cfg.seed,
        "output_dir": str(run_dir.resolve()),
        "motif": {"lambda_motion": lambda_motion, "motion_dir": "motion_heatmaps"},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    print(f"{name}: {preset['mode']} VRAM preset, lambda_motion={lambda_motion} -> {path}")
    return path


def train(cfg, name, lambda_motion, steps=None):
    """Train (or resume) one LoRA run. Checkpoints: <run>/checkpoints/lora_weights_step_XXXXX.safetensors;
    validation videos: <run>/samples."""
    path = train_config(cfg, name, lambda_motion, steps)
    uv_python(cfg, REPO_ROOT / "ltxgif" / "train_ltx.py", path, cwd=Path(cfg.ltx_repo) / "packages" / "ltx-trainer")


def latest_lora(cfg, name):
    ckpts = sorted(cfg.run_dir(name).glob("checkpoints/*step_*.safetensors"))
    return ckpts[-1] if ckpts else None


def generate(cfg, jobs, lora=None, lora_strength=1.0, steps=30, quantization=None, jobs_file=None):
    """Image-to-video with the dev transformer (+ optional LoRA) through LTX's one-stage pipeline, loading the model
    once for all jobs. jobs: [{"image", "prompt", "seed", "output"}]; outputs are MP4s."""
    m = model_section(cfg)
    common = ["--transformer-path", m["model_path"], "--text-encoder-path", m["text_encoder_path"],
              "--video-vae-path", m["video_vae_path"], "--audio-vae-path", m["audio_vae_path"],
              "--height", cfg.size, "--width", cfg.size, "--num-frames", cfg.num_frames,
              "--frame-rate", cfg.fps, "--num-inference-steps", steps]
    if lora:
        common += ["--lora", Path(lora).resolve(), lora_strength]
    if quantization or vram_preset(cfg)["mode"] == "low":
        common += ["--quantization", quantization or "fp8-cast"]
    jobs = [{**j, "image": str(Path(j["image"]).resolve()), "output": str(Path(j["output"]).resolve())} for j in jobs]
    jobs_file = Path(jobs_file or Path(cfg.work_dir) / "jobs.json")
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps({"common": list(map(str, common)), "jobs": jobs}, indent=1))
    uv_python(cfg, REPO_ROOT / "ltxgif" / "batch_generate.py", jobs_file.resolve(), cwd=cfg.ltx_repo)
    return [j["output"] for j in jobs]
