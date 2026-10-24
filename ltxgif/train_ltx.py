"""Launch LTX LoRA training with the MotiF strategy, without modifying the LTX-2 checkout.

    uv run --project LTX-2 python ltxgif/train_ltx.py run.yaml

The YAML is a normal LTX trainer config (flexible strategy) plus a top-level `motif:` block
({lambda_motion, motion_dir, log_every}). Training resumes from the run's latest checkpoint automatically."""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(config_path):
    data = yaml.safe_load(Path(config_path).read_text())
    motif = data.pop("motif", {})

    from ltx_trainer import trainer as trainer_mod
    from ltx_trainer.config import LtxTrainerConfig
    from ltxgif.motif import MotifStrategy, MotifStrategyConfig

    ckpt_dir = Path(data["output_dir"]) / "checkpoints"
    if not data["model"].get("load_checkpoint") and list(ckpt_dir.glob("*step_*.safetensors")):
        data["model"]["load_checkpoint"] = str(ckpt_dir)
        print(f"resuming from the latest checkpoint in {ckpt_dir}")

    cfg = LtxTrainerConfig(**data)
    strategy = MotifStrategyConfig(**cfg.training_strategy.model_dump(), **motif)
    motion_dir = Path(cfg.data.preprocessed_data_root) / strategy.motion_dir
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"motion heatmaps not found at {motion_dir}")
    cfg = cfg.model_copy(update={"training_strategy": strategy})

    factory = trainer_mod.get_training_strategy
    trainer_mod.get_training_strategy = lambda c: MotifStrategy(c) if isinstance(c, MotifStrategyConfig) else factory(c)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "motif.json").write_text(json.dumps(motif))
    trainer_mod.LtxvTrainer(cfg).train()


if __name__ == "__main__":
    main(sys.argv[1])
