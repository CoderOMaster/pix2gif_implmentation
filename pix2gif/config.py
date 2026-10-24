from dataclasses import dataclass
from typing import Optional

import torch

MODEL_IDS = {
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",
    "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


@dataclass
class Config:
    # data
    n_gifs: int = 3000                 # TGIF GIFs to download
    pairs_per_gif: int = 10            # max frame pairs kept per GIF
    motion_range: tuple = (2, 20)      # keep pairs whose mean optical flow is in this range
    flow_eps: float = 0.5              # ignore tracked-point displacements below this (px)
    res: int = 256
    val_fraction: float = 0.05
    data_dir: str = "data/tgif_pairs"

    # model
    init: str = "instruct-pix2pix"     # or "sd-1.5"
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    use_warp: bool = True              # False = motion embedding only (no FlowNet / WarpNet)
    flow_residual: bool = True         # warp = identity + learned offset (False = raw grid, as in the paper)
    perceptual_weight: float = 0.01
    uncond_p: float = 0.05             # CFG dropout: p text-only, p image-only, p both

    # training
    max_steps: int = 5000
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 1e-4
    warmup_steps: int = 500
    use_8bit_adam: bool = True
    use_ema: bool = True
    save_every: int = 1000
    num_workers: int = 4
    seed: int = 0
    out_dir: str = "outputs"

    device: str = "cuda"
    dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        if self.dtype is None:
            bf16 = self.device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if bf16 else torch.float16

    @property
    def model_id(self) -> str:
        return MODEL_IDS[self.init]

    @property
    def ckpt_path(self) -> str:
        tag = "nowarp" if not self.use_warp else ("warpres" if self.flow_residual else "warp")
        return f"{self.out_dir}/pix2gif_{tag}_{self.init}.pt"
