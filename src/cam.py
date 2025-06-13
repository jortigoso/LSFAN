from dataset import * 
from models import *
from utils import *

import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

from torch.utils.data import DataLoader
from torchvision.transforms import * 
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import glob

NAME = "LSFAN"
LABEL_TYPE = 'true_label'
SUBJECT2PLOT = 8

dirname = ""
dataset = ""
dataset_dir = ""
models_folder = ""
plots_folder = ""
exports_folder = ""

file_pattern = 'model*_segment0.pth'

# -------------------------- Load weights filenames -------------------------- #

matching_files = glob.glob(os.path.join(dirname + models_folder, file_pattern))
file_names = [os.path.basename(file) for file in matching_files]

# Function to extract the number from filename
def extract_number(file_name):
    num = file_name.split('_')[0][5:]  # Extract the number after 'model'
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
embeddings = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if NAME == "LSFAN":
    model = CNN_TAP_Att(d_model, nhead, dim_feedforward, dropout, num_layers)
elif NAME == "CNNTAP":
    model = CNN_TAP()
elif NAME == "CNNSAP":
    model = CNN_SAP()
    
model.to(device)

model.to(device)

NUM_SUBJECTS = 30
subjects = [*range(NUM_SUBJECTS)]
target_transform = Lambda(lambda y: one_hot_encoding(y, num_classes))

for snum, (stlo) in enumerate(subjects):

    stlo = [SUBJECT2PLOT]
    snum = SUBJECT2PLOT

    test_data = JointsEMGS_Dataset(dataset_dir, mode='test', subjects=subjects, stlo=stlo, target_transform=target_transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Load model weights 
    model_weights = file_names_sorted[snum]
    state_dict = torch.load(dirname + models_folder + '/' + model_weights)
    model.load_state_dict(state_dict)
    model.eval()

    layer = [model.FrameFeatureExtractor]
    
    # layer = [model.encoder.layers[-1]]
    grad_cam = GradCAM(model=model, target_layers=layer)

    # Test pass
    cams = np.zeros((len(test_dataloader), 180, 30))

    cams = []
    correct = 0
    size = len(test_dataloader.dataset)

    for X, y, id in test_dataloader:
        X = X.to(device)
        y = y.to(device)

        scores = model(X)
        pred_probab = nn.Softmax(dim=1)(scores)
        y_pred = pred_probab.argmax(1)
        correct += (pred_probab.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        # Adapted target for gradcam with true label
        if LABEL_TYPE == 'true_label':
            targets = [ClassifierOutputSoftmaxTarget(y.argmax(1))]
        elif LABEL_TYPE == 'positive_label':
            targets = [ClassifierOutputSoftmaxTarget(1)]
        elif LABEL_TYPE == 'negative_label':
            targets = [ClassifierOutputSoftmaxTarget(0)]

        cam = grad_cam(input_tensor=X, targets=targets)  # (B, 180, 30)
        cams.append(cam)

    accuracy = correct/size

    cams = np.stack(cams)  # Stack all CAMs into one tensor
    average_cam = cams.mean(axis=0)  # Calculate average along the batch dimension
    
    np.savez(f"subject_{stlo[0]}_{NAME}.npz", average_cam=average_cam)

    dtype_fs = 27
    axis_fs = 24

    plt.figure(figsize=(10, 8), layout='compressed')
    im = plt.imshow(average_cam.mean(axis=0), cmap='jet', alpha=0.5, aspect='auto')
    colorbar = plt.colorbar() 
    colorbar.set_ticks([])

    if y.argmax(1) == 0:
        health = 'Healthy'
    else:
        health = 'CLBP'

    plt.title(f'Grad-CAM activation maps for 2D-{NAME}', weight='bold', y=1.02, fontsize=dtype_fs)  
    plt.xticks(ticks=range(0, 30, 2), labels=range(1, 31, 2), fontsize=axis_fs, rotation=45)
    plt.yticks(ticks=range(0, 181, 20), labels=range(0, 181, 20), fontsize=axis_fs)
    plt.gca().axhline(y=-0.5, linewidth=2, color='black')
    
    lw = 1.25
    plt.gca().axvline(x=12.5, linewidth=lw, color='black')
    plt.gca().axvline(x=25.5, linewidth=lw, color='black')
    plt.ylim(179, -16)

    energy_position = (6, -4)
    plt.text(energy_position[0], energy_position[1], 'Angles', fontsize=dtype_fs, ha='center', color='black')

    angle_position = (19, -4)
    plt.text(angle_position[0], angle_position[1], 'Energies', fontsize=dtype_fs, ha='center', color='black')

    emg_position = (27.5, -4)
    plt.text(emg_position[0], emg_position[1], 'sEMG', fontsize=dtype_fs, ha='center', color='black')
    plt.savefig(f'CAM_2D-{NAME}.pdf', dpi = 600)
    plt.show()
    plt.clf() 

    break
