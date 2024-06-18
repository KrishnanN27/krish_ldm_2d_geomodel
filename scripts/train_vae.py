
'''
File: train_vae.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a variational autoencoder (VAE) to learn the mapping between geomodel space and low-dimensional latent space for latent diffusion models
Note: requires Python package "monai" and "generative" to load VAE model and dataloaders
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
import diffusers
from datasets import Dataset as Dataset_diffusers
import datasets
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from generative.inferers import LatentDiffusionInferer
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss

# Set directories
imgs_dir          =  './imgs/'


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

