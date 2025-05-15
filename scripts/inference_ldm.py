import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer

# --- Config ---
n_targets = 5
samples_per_target = 10
target_porosity_values = np.linspace(0.2, 0.4, n_targets)
output_dir = "../outputs/generated3/"
os.makedirs(output_dir, exist_ok=True)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load VAE ---
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

# --- Load U-Net ---
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

# --- Scheduler + Inferer ---
scheduler = DDPMScheduler(num_train_timesteps=100, schedule="linear_beta", beta_start=0.0001, beta_end=0.02)
with torch.no_grad():
    dummy = torch.randn(1, 1, 64, 64).to(device)
    scale_factor = 1 / torch.std(vae.encode_stage_2_inputs(dummy))
inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)

# --- Porosity Calculation ---
def compute_porosity(img_tensor):
    # Adaptive threshold = mid range
    t = (img_tensor.min() + img_tensor.max()) / 2
    return torch.mean((img_tensor < t).float()).item()

# --- Run Sampling + Porosity ---
actual_porosity = []
target_porosity = []

with torch.no_grad():
    sample_id = 0
    for target in target_porosity_values:
        for _ in range(samples_per_target):
            z_noise = torch.randn((1, 1, 8, 8)).to(device)
            z_denoised = inferer.sample(z_noise, diffusion_model=unet, autoencoder_model=vae)
            decoded = vae.decode(z_denoised).cpu()[0, 0]  # Shape: [64, 64]

            # Compute porosity
            porosity = compute_porosity(decoded)
            print(f"[Sample {sample_id}] Porosity: {porosity:.4f} | min={decoded.min():.3f}, max={decoded.max():.3f}, mean={decoded.mean():.3f}")
            actual_porosity.append(porosity)
            target_porosity.append(target)

            # Normalize image for viewing
            norm = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-5)
            img_array = (norm.numpy() * 255).astype(np.uint8)
            Image.fromarray(img_array).save(f"{output_dir}/sample_{sample_id}_p{porosity:.3f}.png")

            sample_id += 1

# --- Scatter Plot ---
plt.figure(figsize=(6, 5))
plt.scatter(target_porosity, actual_porosity, color='blue', alpha=0.7)
plt.xlabel("Target Porosity")
plt.ylabel("Actual Porosity")
plt.title("Target vs Actual Porosity (LDM Samples)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/porosity_scatter.png")
print(f"✅ Saved scatter plot to {output_dir}/porosity_scatter.png")