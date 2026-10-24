from dataclasses import dataclass
from pathlib import Path

HF_REPO = "Lightricks/LTX-2.5"
MODEL_FILES = {
    "transformer": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}

# LTX video VAE compression: 32x32 pixels per latent token, 8 frames per latent frame (plus the causal first frame).
SPATIAL_FACTOR = 32
TEMPORAL_FACTOR = 8


@dataclass
class LtxGifConfig:
    work_dir: str = "ltx_work"          # clips, manifests, cached latents, runs
    ltx_repo: str = "LTX-2"             # clone of github.com/Lightricks/LTX-2
    models_dir: str = "models/ltx-2.5"

    # data: TGIF subset drawn from the official train / val / test splits
    n_train: int = 18000
    n_val: int = 1000
    n_test: int = 1000
    headroom: float = 1.1               # build this many extra clips so later caption rejections still leave n_*
    size: int = 512                     # square clips, divisible by 32
    num_frames: int = 17                # 1 + 8k
    fps: float = 8.0                    # target timing; shorter GIFs get a proportionally higher fps
    max_fps: float = 16.0               # reject GIFs too short to fill num_frames at this rate
    min_unique_frames: int = 9          # distinct source frames among the num_frames samples
    min_motion: float = 0.004           # mean |frame difference| (0..1) between consecutive frames
    max_static_fraction: float = 0.4    # fraction of near-identical consecutive frames
    max_cut_ratio: float = 6.0          # largest frame difference / median: above = scene cut
    max_luma_flicker: float = 0.03      # mean |change in mean brightness| between frames
    max_loop_gap: float = 0.0           # >0: reject clips whose last->first jump exceeds this x the median step
    min_caption_words: int = 4
    download_workers: int = 32

    # MotiF heatmaps
    flow_res: int = 256                 # RAFT input size (the heatmap is pooled to the latent grid anyway)
    raft_updates: int = 12
    motif_steepness: float = 100.0      # sigma(x) = 1 / (1 + exp(steepness * (midpoint - x)))
    motif_midpoint: float = 0.05
    flow_floor_px: float = 2.0          # x = |flow| / max(peak |flow| of the clip, floor)

    # LoRA training
    lora_rank: int = 32
    lora_alpha: int = 32
    learning_rate: float = 5e-5
    batch_size: int = 1
    grad_accum: int = 4
    steps: int = 5000
    max_grad_norm: float = 1.0
    checkpoint_every: int = 500
    validate_every: int = 500
    n_val_prompts: int = 4
    seed: int = 42
    vram_mode: str = "auto"             # "auto", "standard" (~80 GB) or "low" (int8, ~32-40 GB)

    def __post_init__(self):
        assert self.size % SPATIAL_FACTOR == 0, "size must be divisible by 32"
        assert self.num_frames % TEMPORAL_FACTOR == 1, "num_frames must be 1 + 8k"

    @property
    def data_dir(self) -> Path:
        return Path(self.work_dir) / "data"

    def manifest(self, split: str) -> Path:
        """LTX dataset manifest (JSON list of {video, caption}); video paths are relative to data_dir."""
        return self.data_dir / f"{split}.json"

    def records(self, split: str) -> Path:
        """Every processed GIF of a split (kept and rejected, with stats); makes clip building resumable."""
        return self.data_dir / f"{split}_records.jsonl"

    def precomputed_dir(self, split: str = "train") -> Path:
        return Path(self.work_dir) / f"precomputed_{split}"

    def run_dir(self, name: str) -> Path:
        return Path(self.work_dir) / "runs" / name

    @property
    def latent_grid(self) -> tuple:
        """(frames, height, width) of the video latent: e.g. 17x512x512 -> 3x16x16 = 768 tokens."""
        s = self.size // SPATIAL_FACTOR
        return (self.num_frames - 1) // TEMPORAL_FACTOR + 1, s, s

    def model_path(self, key: str) -> Path:
        return Path(self.models_dir) / MODEL_FILES[key]
