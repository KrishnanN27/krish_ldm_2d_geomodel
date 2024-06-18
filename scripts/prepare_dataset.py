## Import packages

# General imports
import os
import numpy as np
import cv2
from datasets import Datasets

# Load dataset
train_split       = 0.7
val_split         = 0.2
test_split        = 1 - train_split - val_split

# Set directories for diffusers dataset and jpg images folder
dataset_dir       = './data/diffusers_dataset/'
imgs_dir          =  './imgs/'

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)
    
# Load diffusers dataset and transform into NumPy
diffusers_dataset = Dataset.load_from_disk(dataset_dir)['image']
geomodels_numpy   = np.array([np.array(image) for image in diffusers_dataset]) 

# Transform every NumPy geomodel into .jpg image and save
for i in range(len(diffusers_dataset)):
    img_path = os.path.join(imgs_dir, f"image_{i}.jpeg")
    cv2.imwrite(img_path, geomodels_numpy[i,:,:])
