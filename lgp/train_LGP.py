import os
import math
import torch
import argparse
import inspect
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from lgp import latent_guidance_predictor


# ──────────────────────── Argparser ────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Training the LGP")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--edge_maps_dir", type=str, required=True)
    parser.add_argument("--vae", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--unet", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--LGP_path", type=str, default="/workspace/trained_LGP")
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


# ──────────────────────── Utility ────────────────────────
@torch.no_grad()
def img_to_latents(img: Image.Image) -> torch.Tensor:
    np_img = (np.array(img).astype(np.float32) / 255.0) * 2.0 - 1.0
    torch_img = torch.from_numpy(np_img.transpose(2, 0, 1)[None])
    generator = torch.Generator(device).manual_seed(0)
    latents = vae.encode(torch_img.to(vae.dtype).to(device)).latent_dist.sample(generator=generator)
    return latents * 0.18215

def noisy_latent(image, scheduler):
    t = torch.randint(250, 900, (1,), device=device).long()
    noise = torch.randn_like(image)
    noisy = scheduler.add_noise(image, noise, t)
    sqrt_alpha = scheduler.alphas_cumprod[t].sqrt().flatten()
    while len(sqrt_alpha.shape) < len(image.shape):
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
    noise_level = noisy - (sqrt_alpha * image)
    return noisy, noise_level, t

def save_tensors(module: nn.Module, features, name: str):
    if isinstance(features, (list, tuple)):
        features = [f.detach().float() for f in features if isinstance(f, torch.Tensor)]
    elif isinstance(features, dict):
        features = {k: v.detach().float() for k, v in features.items()}
    else:
        features = features.detach().float()
    setattr(module, name, features)

def save_out_hook(module, inp, out): return save_tensors(module, out, 'activations')
def save_input_hook(module, inp, out): return save_tensors(module, inp[0], 'activations')

def extract_features(latent, blocks, text_emb, t):
    latent_input = torch.cat([latent] * 2)
    feature_blocks = []

    for i, block in enumerate(unet.down_blocks + unet.up_blocks):
        if i in blocks:
            block.register_forward_hook(save_out_hook)
            feature_blocks.append(block)

    with torch.no_grad():
        _ = unet(latent_input, t, encoder_hidden_states=text_emb)

    activations = [block.activations for block in feature_blocks]
    for block in feature_blocks: block.activations = None

    return [
        activations[0][0], activations[1][0], activations[2][0], activations[3][0],
        activations[4], activations[5], activations[6], activations[7]
    ]

def resize_and_concatenate(acts: List[torch.Tensor], ref: torch.Tensor):
    size = ref.shape[2:]
    resized = [F.interpolate(a[:1].transpose(1, 3), size=size, mode="bilinear") for a in acts]
    return torch.cat(resized, dim=3)


# ──────────────────────── Dataset ────────────────────────
class LGPDataset(Dataset):
    def __init__(self, dataset_dir, edge_maps_dir):
        self.items = []
        for subfolder in os.listdir(dataset_dir):
            sub_path = os.path.join(dataset_dir, subfolder)
            for img_file in os.listdir(sub_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(sub_path, img_file)
                    if Image.open(img_path).mode != "RGB":
                        continue
                    edge_path = os.path.join(edge_maps_dir, img_file.replace('.jpg', '.png'))
                    if os.path.exists(edge_path):
                        self.items.append((img_path, edge_path, subfolder))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, edge_path, caption = self.items[idx]
        img = Image.open(img_path).resize((512, 512))
        edge = Image.merge('RGB', [Image.open(edge_path)]*3).resize((512, 512))

        edge_latent = img_to_latents(edge).transpose(1, 3)
        img_latent = img_to_latents(img).to(device)
        noisy, noise_lvl, t = noisy_latent(img_latent, noise_scheduler)

        text_input = tokenizer([caption], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_emb = text_encoder(text_input.input_ids.to(device))[0]
            uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt")
            uncond_emb = text_encoder(uncond_input.input_ids.to(device))[0]

        text_embeddings = torch.cat([uncond_emb, text_emb])
        feats = extract_features(noisy, blocks, text_embeddings, t)
        feats = resize_and_concatenate(feats, img_latent)
        noise_lvl = noise_lvl.transpose(1, 3)

        return feats, edge_latent, noise_lvl


# ──────────────────────── Training ────────────────────────
if __name__ == "__main__":
    args = get_args()

    device = args.device
    batch_size = args.batch_size
    blocks = [0, 1, 2, 3]

    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.unet, subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    noise_scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    noise_scheduler.set_timesteps(50)

    dataset = LGPDataset(args.dataset_dir, args.edge_maps_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = latent_guidance_predictor(output_dim=4, input_dim=7080, num_encodings=9).to(device)
    model.init_weights()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    total_steps = len(dataloader) * args.epochs
    iterator = iter(dataloader)
    features, edge_map, noise_lvl = next(iterator)
    features, edge_map, noise_lvl = features.to(device), edge_map.to(device), noise_lvl.to(device)
    edge_map = edge_map.flatten(start_dim=0, end_dim=3)

    for step in tqdm(range(1, total_steps + 1), desc="Training"):
        optimizer.zero_grad()
        pred = model(features, noise_lvl)
        loss = loss_fn(pred, edge_map)
        loss.backward()
        optimizer.step()

        print(f"Step {step}/{total_steps} | Loss: {loss.item():.6f}")

        if step % args.epochs == 0 and step != total_steps:
            features, edge_map, noise_lvl = next(iterator)
            features, edge_map, noise_lvl = features.to(device), edge_map.to(device), noise_lvl.to(device)
            edge_map = edge_map.flatten(start_dim=0, end_dim=3)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, args.LGP_path + '.pt')
    print(f"Model saved to: {args.LGP_path + '.pt'}")
