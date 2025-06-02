import argparse
import os
import math
import torch
import numpy as np
from PIL import Image
from typing import List
from torch import nn
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------
# Latent Guidance Predictor
# ---------------------------
class LGP(nn.Module):
    def __init__(self, output_dim: int, input_dim: int, num_encodings: int):
        super().__init__()
        self.num_encodings = num_encodings
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pos_encoding = [torch.sin(2 * math.pi * t * (2 ** -l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=-1)
        x = torch.cat((x, t, pos_encoding), dim=-1).flatten(start_dim=0, end_dim=2)
        return self.layers(x)

# ---------------------------
# Preprocessing & Inference
# ---------------------------
@torch.no_grad()
def to_latents(img: Image.Image, vae: AutoencoderKL, device: str) -> torch.Tensor:
    np_img = (np.array(img).astype(np.float32) / 255.0) * 2.0 - 1.0
    torch_img = torch.from_numpy(np_img.transpose(2, 0, 1)).unsqueeze(0)
    latents = vae.encode(torch_img.to(vae.dtype).to(device)).latent_dist.sample()
    return latents * 0.18215

@torch.no_grad()
def to_image(latents: torch.Tensor, vae: AutoencoderKL, device: str) -> Image.Image:
    image = vae.decode(latents.to(vae.dtype).to(device)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    np_img = image.cpu().permute(0, 2, 3, 1).squeeze().numpy()
    return Image.fromarray((np_img * 255).astype(np.uint8))

def noisy_latent(latent: torch.Tensor, scheduler: DDPMScheduler, timestep: torch.Tensor, device: str):
    noise = torch.randn_like(latent)
    noisy = scheduler.add_noise(latent, noise, timestep)
    sqrt_alpha_prod = scheduler.alphas_cumprod[timestep].to(device).sqrt().flatten()
    for _ in range(len(latent.shape) - len(sqrt_alpha_prod.shape)):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    return noisy, noisy - (sqrt_alpha_prod * latent)

def extract_features(unet: UNet2DConditionModel, image: torch.Tensor, timestep: torch.Tensor,
                     text_emb: torch.Tensor, blocks: List[int]) -> List[torch.Tensor]:
    latent_input = torch.cat([image] * 2)
    hooks = []

    def save_hook(module, _, output):
        module.activations = output.detach().float()

    for idx, block in enumerate(unet.down_blocks + unet.up_blocks):
        if idx in blocks:
            hooks.append(block.register_forward_hook(save_hook))

    with torch.no_grad():
        unet(latent_input, timestep, encoder_hidden_states=text_emb)

    activations = [block.activations for block in unet.down_blocks + unet.up_blocks if hasattr(block, 'activations')]
    for hook in hooks:
        hook.remove()

    # Down 0-3, Up 0-3
    return [activations[i][0] if i < 4 else activations[i] for i in range(8)]

def resize_and_concatenate(activations: List[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    size = reference.shape[2:]
    resized = [nn.functional.interpolate(a[:1].transpose(1, 3), size=size, mode="bilinear") for a in activations]
    return torch.cat(resized, dim=3)

# ---------------------------
# Main Evaluation Pipeline
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate LGP on image sketch")
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument("--vae", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--unet", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--LGP_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--noise_strength", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    blocks = [0, 1, 2, 3]
    strength = args.noise_strength
    num_steps = 50
    batch_size = 1

    # Load LGP model
    model = LGP(output_dim=4, input_dim=7080, num_encodings=9).to(device)
    checkpoint = torch.load(args.LGP_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load VAE, UNet, tokenizer, scheduler
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.unet, subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.unet, subfolder="scheduler")
    scheduler.set_timesteps(num_steps)

    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = min(int(num_steps * strength) + offset, num_steps)
    timesteps = torch.tensor([scheduler.timesteps[-init_timestep]] * batch_size, device=device)

    # Text embedding
    text_input = tokenizer([args.caption], padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").to(device)
    uncond_input = tokenizer([""] * batch_size, padding="max_length",
                             max_length=text_input.input_ids.shape[-1], return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = torch.cat([
            text_encoder(uncond_input.input_ids)[0],
            text_encoder(text_input.input_ids)[0]
        ])

    # Process image
    img = Image.open(args.image_path).resize((512, 512))
    latents = to_latents(img, vae, device)
    noisy, noise_level = noisy_latent(latents, scheduler, timesteps, device)
    noise_level = noise_level.transpose(1, 3)

    features = extract_features(unet, noisy, timesteps, text_emb, blocks)
    features = resize_and_concatenate(features, latents)
    pred = model(features, noise_level).unflatten(0, (1, 64, 64)).transpose(3, 1)

    out_image = to_image(pred, vae, device)
    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_image.save(f"{img_name}-edge_map.jpg")
    print("Done!")

if __name__ == "__main__":
    main()
