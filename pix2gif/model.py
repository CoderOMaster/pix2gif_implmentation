"""Pix2Gif: motion embedding + FlowNet/WarpNet latent warping on top of an image-conditioned SD UNet."""
from dataclasses import dataclass

import inflect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_inflect = inflect.engine()


def motion_prompt(caption, motion):
    """'<caption>. The optical flow is <motion in words>.' (CLIP handles number words better than digits)."""
    caption = caption.strip()
    caption = caption if caption.endswith(".") else caption + "."
    return f"{caption} The optical flow is {_inflect.number_to_words(int(motion))}."


def _conv(i, o, norm=nn.BatchNorm2d, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, k, stride, (k - 1) // 2, bias=True), norm(o), nn.LeakyReLU(0.2, inplace=True))


def _deconv(i, o, norm):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True), norm(o), nn.LeakyReLU(0.2, inplace=True))


def _predict_flow(i):
    return nn.Sequential(nn.Conv2d(i, 2, 3, 1, 1, bias=True), nn.Tanh())


class IntervalEmbedding(nn.Module):
    """Motion magnitude -> learned 768-d vector, broadcast-added to every CLIP token.
    Trained on integers; fractional values linearly interpolate the two neighbouring embeddings (smoother sweeps)."""
    def __init__(self, num_embeddings=1024, dim=768):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, dim)

    def forward(self, motion):
        motion = motion.float().clamp(0, self.embedding.num_embeddings - 1)
        lo = motion.floor().long()
        hi = (lo + 1).clamp(max=self.embedding.num_embeddings - 1)
        w = (motion - lo.float())[:, None]
        return (self.embedding(lo) * (1 - w) + self.embedding(hi) * w).unsqueeze(1)  # (B, 1, 768)


class InjectionModule(nn.Module):
    """(B, 77, 768) text+motion condition -> (B, size, 16, 16) feature map."""
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.linear = nn.Conv1d(77, size, kernel_size=3, stride=3, padding=1)
        self.norm = nn.LayerNorm([size ** 2, size])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm(self.linear(x).transpose(1, 2))).transpose(1, 2)
        return x.reshape(x.shape[0], -1, self.size, self.size)


class FlowNet(nn.Module):
    """Predicts a sampling grid (tanh, in [-1, 1]) over the 32×32 source latent, conditioned on text + motion."""
    def __init__(self, ngf=64, norm=nn.BatchNorm2d, in_ch=4):
        super().__init__()
        self.conv0 = _conv(in_ch, ngf, norm)
        self.conv1 = _conv(ngf, ngf, norm, stride=2)
        self.conv1_1 = _conv(ngf, ngf * 2, norm)
        self.conv2 = _conv(ngf * 2, ngf * 2, norm, stride=2)
        self.conv2_1 = _conv(ngf * 2, ngf * 2, norm)
        self.deconv1 = _deconv(ngf * 2, ngf // 2, norm)
        self.deconv0 = _deconv(ngf // 2 + ngf // 4 + 2, ngf // 4, norm)
        self.inter_conv1 = _conv(ngf // 2 + ngf // 4 + 2, ngf // 2, norm)
        self.inter_conv0 = _conv(ngf // 4 + 2, ngf // 4, norm)
        self.predict_flow2 = _predict_flow(ngf * 2)
        self.predict_flow1 = _predict_flow(ngf // 2)
        self.predict_flow0 = _predict_flow(ngf // 4)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.injection1 = InjectionModule(ngf // 4)

    def forward(self, x, cond):
        out_conv2 = self.conv2_1(self.conv2(self.conv1_1(self.conv1(self.conv0(x)))))  # B×128×8×8
        flow2 = self.predict_flow2(out_conv2)
        concat1 = torch.cat((self.deconv1(out_conv2), self.upsampled_flow2_to_1(flow2), self.injection1(cond)), 1)
        flow1 = self.predict_flow1(self.inter_conv1(concat1))                           # 16×16
        concat0 = torch.cat((self.deconv0(concat1), self.upsampled_flow1_to_0(flow1)), 1)
        flow0 = self.predict_flow0(self.inter_conv0(concat0))                           # 32×32
        return flow0, flow1, flow2


def warp(latent, flow):
    """WarpNet: the predicted flow is used directly as the grid_sample grid."""
    return F.grid_sample(latent, flow.permute(0, 2, 3, 1), mode="bilinear", align_corners=False)


class LatentPerceptual(nn.Module):
    """VGG19 perceptual loss on 4-channel latents (first conv widened; 4th channel = mean of RGB filters)."""
    def __init__(self, layers=(0, 5, 10, 19, 28), pretrained=True):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights).features[: max(layers) + 1]
        first = vgg[0]
        new = nn.Conv2d(4, first.out_channels, first.kernel_size, first.stride, first.padding)
        with torch.no_grad():
            new.weight[:, :3] = first.weight
            new.weight[:, 3] = first.weight.mean(dim=1)
            new.bias.copy_(first.bias)
        vgg[0] = new
        for m in vgg:
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.vgg, self.layers = vgg.eval().requires_grad_(False), set(layers)

    def forward(self, a, b):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            a, b = layer(a), layer(b)
            if i in self.layers:
                loss = loss + torch.mean((a - b) ** 2)
        return loss


class Pix2Gif(nn.Module):
    def __init__(self, unet, use_warp=True):
        super().__init__()
        self.unet, self.use_warp = unet, use_warp
        self.interval_embedding = IntervalEmbedding()
        self.flownet = FlowNet(64, in_ch=4)

    def forward(self, noisy, t, text_emb, motion, src_latent):
        ctx = text_emb + self.interval_embedding(motion)   # text + motion condition, also the cross-attn context
        x, warped = [noisy, src_latent], None
        if self.use_warp:
            warped = warp(src_latent, self.flownet(src_latent, ctx)[0])
            x.append(warped)                                # UNet input: [z_t, source latent, warped latent]
        return self.unet(torch.cat(x, 1), t, encoder_hidden_states=ctx).sample, warped


def widen_conv_in(unet, in_ch):
    """New input channels start at zero, so the initial model behaves exactly like the base model."""
    old = unet.conv_in
    new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.zero_()
        n = min(in_ch, old.in_channels)
        new.weight[:, :n] = old.weight[:, :n]
        new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=in_ch)
    return unet


@dataclass
class Components:
    tokenizer: object
    text_encoder: nn.Module
    vae: nn.Module
    noise_scheduler: object
    model: Pix2Gif
    device: str

    @property
    def scale(self):
        return self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_text(self, prompts):
        ids = self.tokenizer(prompts, padding="max_length", max_length=77, truncation=True,
                             return_tensors="pt").input_ids
        return self.text_encoder(ids.to(self.device))[0].float()


def assemble(cfg, tokenizer, text_encoder, vae, noise_scheduler, unet):
    unet = widen_conv_in(unet, 12 if cfg.use_warp else 8)
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    return Components(tokenizer, text_encoder.to(cfg.device), vae.to(cfg.device), noise_scheduler,
                      Pix2Gif(unet, cfg.use_warp).to(cfg.device), cfg.device)


def load_components(cfg):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    mid = cfg.model_id
    return assemble(
        cfg,
        CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer"),
        CLIPTextModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=cfg.dtype),
        AutoencoderKL.from_pretrained(cfg.vae_id, torch_dtype=torch.float32),
        DDPMScheduler.from_pretrained(mid, subfolder="scheduler"),
        UNet2DConditionModel.from_pretrained(mid, subfolder="unet"),
    )
