
'''
File: train_vae.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a variational autoencoder (VAE) to learn the mapping between geomodel space and low-dimensional latent space for latent diffusion models
Note: requires Python package "monai" or "monai-generative" to load VAE model and dataloaders
'''


# Import packages

# General imports
import os
import numpy as np
import shutil
import tempfile
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from PIL import Image 
import cv2
import matplotlib.pyplot as plt 

# Monai and diffusers modules
import monai
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

# Set directories
imgs_dir          =  '../data/imgs/'


# Load dataset
geomodels_dataset = [{"image": imgs_dir + img} for  img in os.listdir(imgs_dir)][:4000]
N_data            = len(geomodels_dataset)
image_size        = 64
device = torch.device("cpu")
device = torch.device("cuda")


# Split dataset
train_split       = 0.7
val_split         = 0.2
test_split        = 1 - train_split - val_split
batch_size        = 16

train_datalist    = geomodels_dataset[:int(N_data*train_split)]
val_datalist      = geomodels_dataset[int(len(train_datalist)):int(N_data*(1-test_split))+1]
test_datalist     = geomodels_dataset[int(-N_data*test_split):]

# Transform dataset

# Training set
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True)]
)

train_ds = Dataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# Validation set
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_ds = Dataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# Testing set
test_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)

test_ds = Dataset(data=test_datalist, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

# Set hard data conditioning points (first two coordinates are (x,y) points and third coordinate the pixel value)
hard_data_locations = np.array([[7,7, 255], 
                                [7,31, 255],
                                [7,55, 255],
                                [55,7,255],
                                [55,31,255],
                                [55,55,255]])


# Initiate variational autoendocder (VAE) model
autoencoderkl = AutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_channels=(128, 128, 256, 512),
                latent_channels=1,
                num_res_blocks=1, 
                                )
autoencoderkl = autoencoderkl.to(device)

# Train the VAE on three loss terms: (1) reconstruction loss, (2) K-L divergence loss, (3) hard data facies loss
trained_vae_dir = './trained_vae/'
if not os.path.exists(trained_vae_dir):
    os.makedirs(trained_vae_dir)
    


# Training parameters
n_epochs     = 100
val_interval = 10
autoencoder_warm_up_n_epochs = 10
epoch_losses = []
val_losses   = []
epoch_recon_losses = []
epoch_gen_losses = []
epoch_disc_losses = []
val_recon_losses = []
intermediary_images = []
num_example_images = 4
kl_weight = 1e-6
lambda_hd = 0

# Gradient parameters (optimizer and scaler)
optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
scaler_g = torch.cuda.amp.GradScaler()
scaler = GradScaler()

# Training loop

for epoch in range(n_epochs):
        
    autoencoderkl.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = autoencoderkl(images)
            reconstruction_hd = [reconstruction[...,loc[0],loc[1]] for loc in hard_data_locations]
            reconstruction_hd_vector =  torch.stack(reconstruction_hd, dim=0).flatten()
            images_hd = [images[...,loc[0],loc[1]] for loc in hard_data_locations]
            images_hd_vector = torch.stack(images_hd, dim=0).flatten()
            hd_loss =  F.mse_loss(images_hd_vector * lambda_hd, reconstruction_hd_vector * lambda_hd)

            recons_loss = F.l1_loss(reconstruction.float(), images.float())

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss_g = recons_loss + (kl_weight * kl_loss) + hd_loss


        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "kl_loss": kl_loss.item() / (step + 1),
                "hd_loss": hd_loss.item() / (step + 1),
            }
        )
    epoch_recon_losses.append(epoch_loss / (step + 1))
    epoch_gen_losses.append(gen_epoch_loss / (step + 1))
    if (epoch + 1) % 10 == 0:
        torch.save(autoencoderkl.state_dict(), f'{trained_vae_dir} + /vae_epoch_{epoch + 1}.pt')

    if (epoch + 1) % val_interval == 0:
        autoencoderkl.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)

                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = autoencoderkl(images)
                    if val_step == 1:
                        intermediary_images.append(reconstruction[:num_example_images, 0])

                    recons_loss = F.l1_loss(images.float(), reconstruction.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_losses.append(val_loss)
        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
progress_bar.close()

