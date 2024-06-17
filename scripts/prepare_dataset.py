## Import packages

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

# Load dataset

train_split       = 0.7
val_split         = 0.2
test_split        = 1 - train_split - val_split

dataset_dir        = './data/diffusers_dataset'
imgs_dir           =  './imgs/'

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)
    
diffusers_dataset = datasets.Dataset.load_from_disk(dataset_dir)['image']
geomodels_numpy   = np.array([np.array(image) for image in diffusers_dataset]) 

for i in range(len(diffusers_dataset)):
    img_path = os.path.join(imgs_dir, f"image_{i}.jpeg")
    cv2.imwrite(img_path, geomodels_numpy[i,:,:])
