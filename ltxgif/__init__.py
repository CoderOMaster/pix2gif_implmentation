"""Image-to-GIF fine-tuning of LTX-2.5: LoRA on curated TGIF clips with the MotiF motion focal loss.

Modules (import the ones you need; motif.py, train_ltx.py and batch_generate.py run inside the LTX-2 environment):
  config      LtxGifConfig: paths, data filters, heatmap and LoRA settings
  tgif        TGIF download, GIF decoding, filtering, timestamp resampling, MP4 clips + manifests
  recaption   optional Gemini rewrite of captions as animation instructions
  heatmaps    RAFT flow -> MotiF heatmaps on the latent token grid
  ltx         model download, VRAM presets, trainer YAML, preprocessing / training / generation wrappers
  motif       MotiF training strategy for the LTX trainer
  gif         loop handling and GIF export
  evaluate    base vs LoRA vs LoRA + MotiF: metrics and blinded review
  text2gif    text -> Gemini first frame -> LoRA animation -> GIF
  worker      keeps LTX loaded between cell runs (serve_ltx.py runs in the LTX-2 environment)
"""
from .config import LtxGifConfig

__all__ = ["LtxGifConfig"]
