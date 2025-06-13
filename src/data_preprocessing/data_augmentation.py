import numpy as np
import scipy.io
import os
from torchvision.transforms import *  
import torch.nn.functional as F
from scipy.io import savemat
import copy

# Augmentation transforms
dirname = os.path.dirname(__file__)
root = '' # SET PROCESSED MATS DIR HERE   
dirs = os.listdir(root)
save_folder = "" # SET AUGMENTED DATASET DIR HERE

jitter_stds = [0.05, 0.1, 0.15]
cropping_probs = [0.05, 0.1, 0.15]

for matname in dirs:
  mat = scipy.io.loadmat(root + matname)
  name = matname.split('.')[0]
  data = mat['data']

  for idx, std in enumerate(jitter_stds):
    mask = std*np.random.randn(data.shape[0], data.shape[1])
    mat['data'] = data + mask
    savemat(f'{save_folder}/{name}_jitter_{idx}.mat', mat)
    
  for idx, prob in enumerate(cropping_probs):
    size = (data.shape[0], data.shape[1])
    indices1 = np.random.choice(np.arange(size[0]), replace=False, size=int(size[0]*prob))
    indices2 = np.random.choice(np.arange(size[1]), replace=False, size=int(size[1]*prob))

    data_copy = copy.deepcopy(data)

    data_copy[indices1[:, None], indices2] = 0

    mat['data'] = data_copy
    savemat(f'{save_folder}/{name}_rcop_{idx}.mat', mat)