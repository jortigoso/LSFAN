# %%
from dataset import * 
from models import *
from loops import *

import os
import numpy as np
import glob

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

from torch.utils.data import DataLoader

from torchmetrics.classification import BinaryMatthewsCorrCoef
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score

NAME = 'L-SFAN'
dataset_dir = ""
weights_dir = "../weights/L_SFAN"

file_pattern = 'model*_segment0.pth'

# -------------------------- Load weights filenames -------------------------- #

matching_files = glob.glob(os.path.join(weights_dir, file_pattern))
file_names = [os.path.basename(file) for file in matching_files]

# Function to extract the number from filename
def extract_number(file_name):
    num = file_name.split('_')[0][len("model"):]  # Extract the number after 'model'
    return int(num)

file_names_sorted = sorted(file_names, key=extract_number)

# ----------------------------------- Model ---------------------------------- #
    
batch_size = 40
num_classes = 2

d_model = 30
nhead = 5
dim_feedforward = 30
dropout = 0.2
num_layers = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

model = CNN_TAP_Att(d_model, nhead, dim_feedforward, dropout, num_layers)

model.to(device)

NUM_SUBJECTS = 30
subjects = [*range(NUM_SUBJECTS)]
target_transform = Lambda(lambda y: one_hot_encoding(y, num_classes))

test_preds_list = []
test_probs_list = []
test_labels_list = []

for snum, (stlo) in enumerate(subjects):
    
    print(f'Testing subject {stlo}...')
    stlo = [stlo]
    
    test_data = JointsEMGS_Dataset(dataset_dir, mode='test', subjects=subjects, stlo=stlo, target_transform=target_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load model weights 
    model_weights = file_names_sorted[snum]
    state_dict = torch.load(f'{weights_dir}/{model_weights}')
    model.load_state_dict(state_dict)
    model.eval()
    
    test_predictions, test_probs, test_labels = test_loop(test_dataloader, model, device)

    test_preds_list.append(test_predictions)
    test_probs_list.append(test_probs)
    test_labels_list.append(test_labels)

y_pred = torch.cat(test_preds_list, dim=0).cpu()
y_prob = torch.cat(test_probs_list, dim=0).cpu()
y_true = torch.cat(test_labels_list, dim=0).cpu()

print(f'\nLOSO metrics for {NAME}:')
print('--------------------------')
# AUC
test_auc = metrics.roc_auc_score(y_true, y_prob[:,1])
print(f'Test AUC: {test_auc}')

# MCC
mcc = BinaryMatthewsCorrCoef()
test_MCC = mcc(y_pred, y_true).item()
print(f'Test MCC: {test_MCC}')

# Fm
precision_score_zero = precision_score(y_true, y_pred, pos_label=0)
recall_score_zero = recall_score(y_true, y_pred, pos_label=0)
precision_score_one = precision_score(y_true, y_pred, pos_label=1)
recall_score_one = precision_score(y_true, y_pred, pos_label=1)

f1_zero = 2*(precision_score_zero * recall_score_zero)/(precision_score_zero + recall_score_zero)
f1_one = 2*(precision_score_one * recall_score_one)/(precision_score_one + recall_score_one)

Fm = (f1_zero + f1_one)/2
print(f'Test Fm: {Fm}')



   
