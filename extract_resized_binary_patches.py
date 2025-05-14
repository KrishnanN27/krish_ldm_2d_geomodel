import os
import xarray as xr
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.transform import resize
import random

input_dir = "tomo_R_SSw_SS_nc"
output_dir = "data/imgs111"
os.makedirs(output_dir, exist_ok=True)

# Parameters
large_patch_size = 300  # You can also try 300
final_patch_size = 128
stride = 128  # Set to large_patch_size for non-overlapping
max_images = 3000
saved = 0

for fname in os.listdir(input_dir):
    if not fname.endswith(".nc"):
        continue

    path = os.path.join(input_dir, fname)
    ds = xr.open_dataset(path)
    vol = ds["tomo"].values.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

    for z in range(vol.shape[0]):
        slice_ = vol[z]

        for _ in range(10):  # Try 10 random crops per slice
            if saved >= max_images:
                break

            y = random.randint(0, slice_.shape[0] - large_patch_size)
            x = random.randint(0, slice_.shape[1] - large_patch_size)
            patch = slice_[y:y+large_patch_size, x:x+large_patch_size]

            # Resize to 128×128
            patch_resized = resize(patch, (final_patch_size, final_patch_size), anti_aliasing=True)

            # Binarize
            thresh = threshold_otsu(patch_resized)
            patch_bin = (patch_resized > thresh).astype(np.uint8) * 255

            # Save
            out_path = os.path.join(output_dir, f"img_{saved:05}.png")
            Image.fromarray(patch_bin).save(out_path)
            saved += 1

        if saved >= max_images:
            break

print(f"✅ Saved {saved} resized binary images to '{output_dir}'")
