from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from lgp.lgp import LGP
import torch

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # Creates LGP instance and load pre-trained weights
    lgp = LGP(output_dim=4, input_dim=7080, num_encodings=9).to(device)
    lgp_checkpoint = torch.load("./lgp/SDv1.5-trained_LGP.pt", map_location=device)
    lgp.load_state_dict(lgp_checkpoint["model_state_dict"], strict=True)

    diffusion = Diffusion(lgp=lgp).to(device)
    filtered_diffusion_state_dict = {
        k: v for k, v in state_dict['diffusion'].items() if not k.startswith("lgp.")
    }
    diffusion.load_state_dict(filtered_diffusion_state_dict, strict=False)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)


    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }