import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from generative.networks.nets import AutoencoderKL

# --- Configuration ---
output_dir = "../outputs/vae_generated/"
os.makedirs(output_dir, exist_ok=True)

n_samples = 10  # Number of images to generate
latent_shape = (1, 1, 8, 8)  # Match your VAE latent shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained VAE ---
vae = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 128, 256, 512),
    latent_channels=1,
    num_res_blocks=1,
)
vae.load_state_dict(torch.load("../trained_vae/vae_epoch_1.pt", map_location=device))
vae = vae.to(device).eval()

# --- Generate and save images ---
with torch.no_grad():
    for i in range(n_samples):
        z = torch.randn(latent_shape).to(device)  # Random latent vector
        decoded = vae.decode(z)[0, 0].cpu().numpy()

        # Save image
        img = (decoded * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/sample_{i}.png")

        # Print stats
        print(f"[Sample {i}] min={decoded.min():.3f}, max={decoded.max():.3f}, mean={decoded.mean():.3f}")

print(f"✅ {n_samples} images saved to {output_dir}")
