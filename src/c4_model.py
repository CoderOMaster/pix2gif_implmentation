#@title 4 · Model: Pix2Gif modules (ported from the official code) on a diffusers UNet
import inflect
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_inflect = inflect.engine()


def motion_prompt(caption, motion):
    """Official: caption + ' The optical flow is <words>.'  (words, not digits: CLIP handles them better)."""
    caption = caption.strip()
    caption = caption if caption.endswith(".") else caption + "."
    return f"{caption} The optical flow is {_inflect.number_to_words(int(motion))}."


# ---- util.py ----
def conv(i, o, norm=nn.BatchNorm2d, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, k, stride, (k - 1) // 2, bias=True), norm(o), nn.LeakyReLU(0.2, inplace=True))

def deconv(i, o, norm):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True), norm(o), nn.LeakyReLU(0.2, inplace=True))

def predict_flow(i):
    return nn.Sequential(nn.Conv2d(i, 2, 3, 1, 1, bias=True), nn.Tanh())

def i_conv(i, o, norm, k=3, stride=1, bias=True):
    return nn.Sequential(nn.Conv2d(i, o, k, stride, (k - 1) // 2, bias=bias), norm(o), nn.LeakyReLU(0.2, inplace=True))


# ---- openaimodel.py ----
class IntervalEmbedding(nn.Module):
    """Motion magnitude (int) -> learned 768-d vector, broadcast-added to all 77 CLIP tokens (c_L = c_T + c_M)."""
    def __init__(self, num_embeddings=1024, embedding_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, interval):
        return self.embedding(interval.long().unsqueeze(-1))  # (B, 1, 768)


class InjectionModule(nn.Module):
    """(B, 77, 768) text+motion condition -> (B, size, 16, 16) spatial feature map for FlowNet."""
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
    """Predicts a sampling grid (tanh, in [-1, 1]) over the 32×32 source latent, conditioned on c_L."""
    def __init__(self, ngf=64, norm=nn.BatchNorm2d, x=4):
        super().__init__()
        self.conv0 = conv(x, ngf, norm)
        self.conv1 = conv(ngf, ngf, norm, stride=2)
        self.conv1_1 = conv(ngf, ngf * 2, norm)
        self.conv2 = conv(ngf * 2, ngf * 2, norm, stride=2)
        self.conv2_1 = conv(ngf * 2, ngf * 2, norm)
        self.deconv1 = deconv(ngf * 2, ngf // 2, norm)
        self.deconv0 = deconv(ngf // 2 + ngf // 4 + 2, ngf // 4, norm)
        self.inter_conv1 = i_conv(ngf // 2 + ngf // 4 + 2, ngf // 2, norm)
        self.inter_conv0 = i_conv(ngf // 4 + 2, ngf // 4, norm)
        self.predict_flow2 = predict_flow(ngf * 2)
        self.predict_flow1 = predict_flow(ngf // 2)
        self.predict_flow0 = predict_flow(ngf // 4)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.injection1 = InjectionModule(ngf // 4)

    def forward(self, x, cond):
        out_conv0 = self.conv0(x)                                   # B×64×32×32
        out_conv1 = self.conv1_1(self.conv1(out_conv0))             # B×128×16×16
        out_conv2 = self.conv2_1(self.conv2(out_conv1))             # B×128×8×8
        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_deconv1, flow2_up, self.injection1(cond)), 1)  # condition injected at 16×16
        flow1 = self.predict_flow1(self.inter_conv1(concat1))
        flow1_up = self.upsampled_flow1_to_0(flow1)
        concat0 = torch.cat((self.deconv0(concat1), flow1_up), 1)
        flow0 = self.predict_flow0(self.inter_conv0(concat0))       # B×2×32×32
        return flow0, flow1, flow2


def warpnet(images, flow):  # WarpNet: flow is used directly as the grid_sample grid (as in the official code)
    return F.grid_sample(images, flow.permute(0, 2, 3, 1), mode="bilinear", align_corners=False)


# ---- perceptual.py (VGG19, first conv widened to 4 channels so it runs on latents) ----
class LatentPerceptual(nn.Module):
    def __init__(self, layers=(0, 5, 10, 19, 28)):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[: max(layers) + 1]
        first = vgg[0]
        new = nn.Conv2d(4, first.out_channels, first.kernel_size, first.stride, first.padding)
        with torch.no_grad():
            new.weight[:, :3] = first.weight
            new.weight[:, 3] = first.weight.mean(dim=1)
            new.bias.copy_(first.bias)
        vgg[0] = new
        for m in vgg:
            if isinstance(m, nn.ReLU):
                m.inplace = False  # features are tapped mid-network; keep them intact for autograd
        self.vgg, self.layers = vgg.eval().requires_grad_(False), set(layers)

    def forward(self, a, b):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            a, b = layer(a), layer(b)
            if i in self.layers:
                loss = loss + torch.mean((a - b) ** 2)
        return loss


# ---- the model (ddpm_edit.py: get_warped_image + DiffusionWrapper 'hybrid') ----
class Pix2Gif(nn.Module):
    def __init__(self, unet, use_warp=True):
        super().__init__()
        self.unet, self.use_warp = unet, use_warp
        self.interval_embedding = IntervalEmbedding()
        self.flownet = FlowNet(64, x=4)

    def forward(self, noisy, t, text_emb, interval, src_latent):
        ctx = text_emb + self.interval_embedding(interval)        # c_L = c_T + c_M; also the UNet's cross-attn context
        warped = None
        x = [noisy, src_latent]
        if self.use_warp:
            warped = warpnet(src_latent, self.flownet(src_latent, ctx)[0])
            x.append(warped)                                       # "Warp-concat": [z_t, E(c_I), z_W] -> 12 channels
        return self.unet(torch.cat(x, 1), t, encoder_hidden_states=ctx).sample, warped


def widen_conv_in(unet, in_ch):
    """Extra input channels start at zero, so the init model's behaviour is unchanged at step 0 (IP2P trick)."""
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


def load_components():
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    mid = MODEL_IDS[INIT]
    tok = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE).eval()
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet")
    sched = DDPMScheduler.from_pretrained(mid, subfolder="scheduler")
    unet = widen_conv_in(unet, 12 if USE_WARP else 8)
    model = Pix2Gif(unet, USE_WARP).to(DEVICE)
    return tok, text_encoder, vae, sched, model


tok, text_encoder, vae, noise_sched, model = load_components()
text_encoder.requires_grad_(False)
vae.requires_grad_(False)
SCALE = vae.config.scaling_factor  # 0.18215


@torch.no_grad()
def encode_text(prompts):
    ids = tok(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    return text_encoder(ids.to(DEVICE))[0].float()


n_train = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Pix2Gif ready · init={INIT} · warp={USE_WARP} · trainable params {n_train:.0f}M "
      f"(UNet + FlowNet {sum(p.numel() for p in model.flownet.parameters()) / 1e6:.2f}M + motion embedding)")
