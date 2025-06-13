# %%
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os
from torchvision.transforms import *  
import torch.nn.functional as F

def one_hot_encoding(label, num_classes):
    one_hot = torch.zeros(num_classes, dtype=torch.float32)
    one_hot[label.long()] = 1.0
    return one_hot

class JointsEMGS_Dataset(Dataset):
  def __init__(self, root, mode='train', subjects=None, stlo=None, shuffle=True, transform=None, target_transform=None,):
    
    self.root = root
    self.mode = mode
    self.subjects = subjects
    self.stlo = stlo # Subject to leave out
    self.shuffle = shuffle
    self.transform = transform
    self.target_transform = target_transform
    
    # List of paths
    dirs = os.listdir(root)
    dirs.sort()

    frames = []
    labels = []
    ids = []
  
    self.frames = []
    self.labels = []
    self.ids = []
    
    for matname in dirs:
      mat = scipy.io.loadmat(root + matname)
      frames.append(torch.from_numpy(mat['data']).float())
      labels.append(torch.from_numpy(mat['label']))
      ids.append(mat['id'])

    # Random shuffling
    if self.shuffle and self.mode == 'train':
      indices = np.arange(len(labels)) 
      np.random.shuffle(indices)
      frames = [frames[index] for index in indices]
      labels = [labels[index] for index in indices]
      ids = [ids[index] for index in indices]
      
    def check_id(frame_id, stlo):
      return frame_id in stlo
        
    for frame, label, frame_id in zip(frames, labels, ids):
      if (self.mode == 'test' and check_id(frame_id, stlo)) or (self.mode == 'train' and not check_id(frame_id, stlo)):
        if(frame_id in self.subjects):
          self.frames.append(frame)
          self.labels.append(label)
          self.ids.append(frame_id)

  def __len__(self):
    return len(self.labels)
    
  def __getitem__(self, idx):

    # 13 angles + 13 energies + 4 emgs/row
    frame = self.frames[idx].view(1, 180, 30)
    label = self.labels[idx]
    id = self.ids[idx][0][0]
    
    if (self.transform):
      frame = self.transform(frame)
      
    if (self.target_transform):
      label = self.target_transform(label)
      
    return frame, label, id