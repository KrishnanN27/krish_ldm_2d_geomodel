import torch
import numpy as np
from PIL import Image
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

device = torch.device("cpu")

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

# --- Load trained U-Net ---
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_res_blocks=1,
    num_channels=(64, 128, 256),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 128),
)
unet.load_state_dict(torch.load("../trained_unet/unet_epoch_1.pt", map_location=device))
unet = unet.to(device).eval()

# --- Load a real image and encode to latent ---
img = np.array(Image.open("../data/imgs/image_0.jpeg").convert("L")) / 255.0
x = torch.tensor(img[None, None], dtype=torch.float32).to(device)

with torch.no_grad():
    mu, sigma = vae.encode(x)
    z = vae.sampling(mu, sigma)

# --- Add noise to latent and ask U-Net to denoise ---
scheduler = DDPMScheduler(num_train_timesteps=1000)
timesteps = torch.randint(0, 1000, (1,), device=device).long()

noise = torch.randn_like(z)
z_noisy = scheduler.add_noise(z, noise, timesteps)

# --- Predict noise using U-Net ---
with torch.no_grad():
    predicted_noise = unet(x=z_noisy, timesteps=timesteps)

# --- Compute MSE ---
mse = torch.mean((predicted_noise - noise) ** 2).item()
print(f"🔍 U-Net noise prediction MSE: {mse:.6f}")
