"""MotiF motion focal loss as an LTX trainer strategy (runs inside the LTX-2 environment).

Everything except the loss comes from LTX's FlexibleStrategy: first-frame conditioning, flow-matching noise, the
velocity target (noise - clean latent) and the loss mask that excludes the clean condition frame. The loss becomes

    L = mean[mask * (v_hat - v)^2] + lambda * mean[mask * m^2 * (v_hat - v)^2]

with m the cached motion heatmap on the same latent tokens. LTX applies no per-timestep loss weight, so there is no
w_t to carry over. With lambda = 0 the motion term is still logged, which makes it comparable across the two runs."""
from dataclasses import dataclass, fields

import torch
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.flexible import FlexibleStrategy, FlexibleStrategyConfig


class MotifStrategyConfig(FlexibleStrategyConfig):
    motion_dir: str = Field(default="motion_heatmaps", description="Directory of cached motion heatmaps")
    lambda_motion: float = Field(default=1.0, ge=0.0, description="Weight of the motion focal loss")
    log_every: int = Field(default=100, ge=1, description="Log the loss terms every N micro-batches")

    def get_data_sources(self) -> dict[str, str]:
        return {**super().get_data_sources(), self.motion_dir: "motion"}


@dataclass
class MotifModelInputs(ModelInputs):
    video_motion: Tensor | None = None   # [B, seq_len] heat per target token


class MotifStrategy(FlexibleStrategy):
    config: MotifStrategyConfig

    def __init__(self, config: MotifStrategyConfig):
        super().__init__(config)
        self._log = []

    def prepare_training_inputs(self, batch, timestep_sampler) -> MotifModelInputs:
        inputs = super().prepare_training_inputs(batch, timestep_sampler)
        motion = batch["motion"]["motion"]                       # [B, F', H', W'], token order (f h w) = patchify
        motion = motion.reshape(motion.shape[0], -1).to(inputs.video_targets.device, torch.float32)
        if motion.shape[1] != inputs.video_targets.shape[1]:
            raise ValueError(f"motion heatmap has {motion.shape[1]} tokens but the video latent has "
                             f"{inputs.video_targets.shape[1]}: recompute heatmaps for this resolution / frame count")
        return MotifModelInputs(**{f.name: getattr(inputs, f.name) for f in fields(ModelInputs)}, video_motion=motion)

    def compute_loss(self, video_pred: Tensor, audio_pred, inputs: MotifModelInputs) -> Tensor:
        base = super().compute_loss(video_pred, audio_pred, inputs)
        n = inputs.video_targets.shape[1]
        err2 = (video_pred[:, -n:].float() - inputs.video_targets.float()).square()
        mask = inputs.video_loss_mask[:, -n:].unsqueeze(-1).float()
        weight = inputs.video_motion.unsqueeze(-1).square()
        motion = (err2 * weight * mask).mean(dim=[-2, -1]) / mask.mean(dim=[-2, -1]).clamp(min=1e-8)
        self._record(base, motion, inputs.video_motion, mask)
        return base + self.config.lambda_motion * motion

    def get_checkpoint_metadata(self):
        return {**super().get_checkpoint_metadata(), "motif_lambda": self.config.lambda_motion}

    @torch.no_grad()
    def _record(self, base, motion, heat, mask):
        tok = mask[..., 0]
        cover = float((heat * tok).sum() / tok.sum().clamp(min=1))
        self._log.append((float(base.mean()), float(motion.mean()), cover))
        if len(self._log) >= self.config.log_every:
            b, m, c = (sum(x) / len(self._log) for x in zip(*self._log))
            logger.info(f"MotiF terms · base {b:.4f} · motion {m:.4f} (x lambda {self.config.lambda_motion}) "
                        f"· mean heat {c:.3f}")
            self._log.clear()
