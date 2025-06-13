import numpy as np
import scipy.io
from scipy.io import savemat
import os

dirname = os.path.dirname(__file__)
dataset_dir = "" # SET PROCESSED MATS DIR HERE
save_folder = "" # SET DATASET DIR FOR NORMALIZED MATS HERE

dirs = os.listdir(dataset_dir)
dirs.sort()

dataset = []
dataset_data = []

prev_id = -1
c = 0

for matname in dirs:
  mat = scipy.io.loadmat(dataset_dir + matname)
  dataset.append(mat)
  dataset_data.append(mat['data'])
  
  if (mat['data'][:,0].sum() == 0) and (mat['id'].item() != prev_id):
    c+=1
    prev_id = mat['id'].item()
  
np_dataset = np.stack(dataset_data)

mean = np.mean(np_dataset, axis=(0,1))
std = np.std(np_dataset, axis=(0,1))

for idx, (mat, frame) in enumerate(zip(dataset, dataset_data)):
  normalized_frame = (frame-mean)/std
  mat['data'] = normalized_frame
  
  savemat(f'{save_folder}/{idx}.mat', mat) 
  