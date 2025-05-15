import torch
import numpy as np
from PIL import Image
from generative.networks.nets import AutoencoderKL

# --- Setup ---
device = torch.device("cpu")
img_path = "../data/imgs/image_0.jpeg"
output_path = "recon_image_0.png"

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

# --- Load and normalize input image ---
img = np.array(Image.open(img_path).convert("L")) / 255.0
x = torch.tensor(img[None, None], dtype=torch.float32).to(device)

# --- Encode and decode ---
with torch.no_grad():
    mu, sigma = vae.encode(x)
    z = vae.sampling(mu, sigma)
    recon = vae.decode(z)[0, 0].cpu().numpy()

# --- Save reconstruction ---
recon_img = (recon * 255).astype(np.uint8)
Image.fromarray(recon_img).save(output_path)

# --- Print stats ---
print(f"✅ Saved reconstructed image as {output_path}")
print(f"Reconstruction stats → min: {recon.min():.3f}, max: {recon.max():.3f}, mean: {recon.mean():.3f}")
