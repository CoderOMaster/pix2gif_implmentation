#@title 1 · Setup: GPU check, installs, Drive, official code
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip -q install -U diffusers transformers accelerate bitsandbytes inflect opencv-python-headless safetensors
import os, subprocess, torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> GPU (A100 recommended, L4 works)"

USE_DRIVE = True  #@param {type:"boolean"}
if USE_DRIVE:  # Colab sessions die; keep the dataset tarball + checkpoints on Drive
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/pix2gif"
else:
    ROOT = "/content/pix2gif"
os.makedirs(ROOT, exist_ok=True)

# Official code, for reference. The authors deleted it from the repo in Sept 2024 (commit 0be59b6);
# 781ca78 is the last commit that still has it. The model code below is ported from these files:
#   stable_diffusion/ldm/modules/diffusionmodules/openaimodel.py  (FlowNet, WarpNet, InjectionModule, IntervalEmbedding)
#   stable_diffusion/ldm/modules/diffusionmodules/util.py         (conv / deconv / i_conv / predict_flow)
#   stable_diffusion/ldm/models/diffusion/ddpm_edit.py           (conditioning, CFG dropout, losses)
#   stable_diffusion/ldm/modules/losses/perceptual.py            (4-channel VGG19 perceptual loss)
#   edit_dataset.py, demo.py, demo_parallel.py, configs/*.yaml
if not os.path.exists("/content/Pix2Gif-official"):
    subprocess.run("git clone -q https://github.com/hiteshK03/Pix2Gif.git /content/Pix2Gif-official"
                   " && cd /content/Pix2Gif-official && git checkout -q 781ca7859e", shell=True, check=True)
print("official code at /content/Pix2Gif-official ·", torch.cuda.get_device_name(0))
