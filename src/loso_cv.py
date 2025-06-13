from dataset import * 
from models import *
from loops import *
from utils import *

import os
import numpy as np

import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

from torch.utils.data import DataLoader
from torchvision.transforms import * 
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.nn as nn

from sklearn import metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gamma',type=float, default=1)
parser.add_argument('--step',type=int, default=15)
parser.add_argument('--n',type=int, default=0)
parser.add_argument('--model', type=int, default=0)
args = parser.parse_args()

experiment_name = "LSFAN"
RUN_NAME = ''

dirname = ""
dataset = ""
dataset_dir = dirname + '/../../' + dataset
models_folder = f'/../../paper_cam/models/{RUN_NAME}'
plots_folder = f'/../../paper_cam/graphs/{RUN_NAME}'

if not os.path.exists(dirname + models_folder):
  os.mkdir(dirname + models_folder)

if not os.path.exists(dirname + plots_folder):
  os.mkdir(dirname + plots_folder)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

epochs = args.epochs
epochs_checkpoint = 1
num_training_segments = epochs // epochs_checkpoint

batch_size = 40
learning_rate = args.lr
step_size = args.step
gamma = args.gamma
num_classes = 2

d_model = 30
nhead = 5
dim_feedforward = 30
dropout = 0.2
num_layers = 1
embeddings = False

target_transform = Lambda(lambda y: one_hot_encoding(y, num_classes))
  
NUM_SUBJECTS = 30
subjects = [*range(NUM_SUBJECTS)]
test_f1_list = np.zeros(NUM_SUBJECTS)
test_accuracy_list = np.zeros(NUM_SUBJECTS)
test_auc_list = np.zeros(NUM_SUBJECTS)
test_mcc_list = np.zeros(NUM_SUBJECTS)

test_shape = num_training_segments
test_F1 = np.zeros(test_shape)
test_F1_best = np.zeros(test_shape)
test_accuracy = np.zeros(test_shape)
test_AUC = np.zeros(test_shape)
test_MCC = np.zeros(test_shape)
test_F1_zero = np.zeros(test_shape)
test_F1_one = np.zeros(test_shape)
test_Fm = np.zeros(test_shape)
test_Fm_weighted = np.zeros(test_shape)
test_Fm_best = np.zeros(test_shape)
test_Fm_weighted_best = np.zeros(test_shape)
test_F1_zero_best = np.zeros(test_shape)
test_F1_one_best = np.zeros(test_shape)

data_shape = (NUM_SUBJECTS, epochs)
training_loss = np.zeros(data_shape)
validation_loss = np.zeros(data_shape)
training_accuracy = np.zeros(data_shape)
validation_accuracy = np.zeros(data_shape)
training_f1 = np.zeros(data_shape)
validation_f1 = np.zeros(data_shape)
training_auc = np.zeros(data_shape)
validation_auc = np.zeros(data_shape)
training_mcc = np.zeros(data_shape)
validation_mcc = np.zeros(data_shape)

subjects_preds = []
subjects_probs = []
subjects_labels = []

for segment in range(num_training_segments):

  segment_test_preds = []
  segment_test_probs = []
  segment_test_logits = []
  segment_test_labels = []

  for snum, (stlo) in enumerate(subjects):

    stlo = [stlo]
    
    print(f'Segment {segment} of model for {stlo} left out -------\n')
    
    # DATA
    training_data = JointsEMGS_Dataset(dataset_dir, mode='train', subjects=subjects, stlo=stlo, target_transform=target_transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    validation_data = JointsEMGS_Dataset(dataset_dir, mode='test', subjects=subjects, stlo=stlo, target_transform=target_transform)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    zero_count = 0
    one_count = 0
    total_count = 0
    for _, train_labels, _  in train_dataloader:
      zero_count += (train_labels.argmax(1) == 0).sum().item()
      one_count += (train_labels.argmax(1) == 1).sum().item()
      total_count += train_labels.shape[0]

    zero_weight = 1 - zero_count / total_count
    one_weight = 1 - one_count / total_count

    model = CNN_TAP_Att(d_model, nhead, dim_feedforward, dropout, num_layers)
    model.to(device)

    class_weights = torch.tensor([zero_weight, one_weight]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight = class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    if segment > 0:
      state_filename = dirname + models_folder + f'/{snum}_train_state.pth'
      model_filename = dirname + models_folder + f'/model{snum}_segment{segment-1}.pth'
      model, optimizer, init_epoch, lr_sched, scheduler = load_checkpoint(model, optimizer, learning_rate, device, state_filename, model_filename, scheduler)
    
    for epoch in range(epochs_checkpoint):
      current_epoch = segment*epochs_checkpoint + epoch
      
      print(f"Epoch {current_epoch+1}\n-------------------------------")
      train_loss, train_accuracy, train_f1, train_mcc, train_preds_probab, train_labels = train_loop(train_dataloader, model, loss_fn, optimizer, device) 
      val_loss, val_accuracy, val_f1, val_mcc, val_preds_probab, val_labels = validation_loop(validation_dataloader, model, loss_fn, device)
    
      training_loss[stlo, current_epoch] = train_loss
      training_accuracy[stlo, current_epoch] = train_accuracy
      training_f1[stlo, current_epoch] = train_f1
      training_mcc[stlo, current_epoch] = train_mcc

      validation_loss[stlo, current_epoch] = val_loss
      validation_accuracy[stlo, current_epoch] = val_accuracy    
      validation_f1[stlo, current_epoch] = val_f1
      validation_mcc[stlo, current_epoch] = val_mcc

      # In Sklearn's AUC the probability estimates correspond to the probability of the class with the greater label
      training_auc[stlo, current_epoch] = metrics.roc_auc_score(train_labels.argmax(1), train_preds_probab.detach()[:,1])
      
    print(f"Segment {segment} of model {snum} done.")

    # Saving subject model state
    checkpoint = {
      'stlo': snum,
      'segment': segment,
      'epoch': segment*epochs_checkpoint + epochs_checkpoint,
      'optimizer': optimizer.state_dict(),
      'lr_sched': learning_rate,
      'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, dirname + models_folder + f'/{snum}_train_state.pth')
    torch.save(model.state_dict(), dirname + models_folder + f'/model{snum}_segment{segment}.pth')

    # Testing
    test_predictions, test_probs, test_labels = test_loop(validation_dataloader, model, device)

    segment_test_preds.append(test_predictions)
    segment_test_probs.append(test_probs)
    # segment_test_logits.append(test_outputs)
    segment_test_labels.append(test_labels)

  segment_test_preds_tensor = torch.cat(segment_test_preds, dim=0)
  segment_test_probs_tensor = torch.cat(segment_test_probs, dim=0)
  segment_test_labels_tensor = torch.cat(segment_test_labels, dim=0)

  y_pred = segment_test_preds_tensor.cpu()
  y_prob = segment_test_probs_tensor.cpu()
  y_true = segment_test_labels_tensor.cpu()

  # Segment global metrics
  segment_test_F1, segment_best_test_F1, segment_test_accuracy, \
  segment_test_AUC, segment_test_MCC, segment_test_F1_zero, \
  segment_test_F1_one, segment_test_Fm, segment_test_Fm_weighted, \
  segment_test_Fm_best, segment_test_Fm_weighted_best, \
  segment_test_F1_zero_best, segment_test_F1_one_best = segment_global_metrics(y_prob, y_pred, y_true, segment)

  test_F1[segment] = segment_test_F1
  test_F1_best[segment] = segment_best_test_F1
  test_accuracy[segment] = segment_test_accuracy
  test_AUC[segment] = segment_test_AUC
  test_MCC[segment] = segment_test_MCC
  test_F1_zero[segment] = segment_test_F1_zero
  test_F1_one[segment] = segment_test_F1_one
  test_Fm[segment] = segment_test_Fm
  test_Fm_weighted[segment] = segment_test_Fm_weighted
  test_Fm_best[segment] = segment_test_Fm_best
  test_Fm_weighted_best[segment] = segment_test_Fm_weighted_best
  test_F1_zero_best[segment] = segment_test_F1_zero_best
  test_F1_one_best[segment] = segment_test_F1_one_best

  # Since at this point segments are complete we can plot 
  # confusion matrices for each one
  plt.figure(3, figsize=(8,7))
  cnf_matrix = confusion_matrix(y_true, y_pred)
  plot_confusion_matrix(cnf_matrix, classes = [0,1], normalize=False, title='Confusion matrix')
  if segment is num_training_segments-1:
    name = 'Final_CM'
  else:
    name = f'Segment_{segment}_CM'
  plt.savefig(dirname + f'{plots_folder}/{name}.png', bbox_inches='tight')
  plt.clf() 

# Global Plots

# Training/Validation. Segment vectors can be used since the last
# segmen overwrote the values and contain the last step outputs
for stlo in subjects:
  training_plots(stlo, training_loss[stlo,:], training_accuracy[stlo,:], 
                 training_f1[stlo,:], training_auc[stlo,:],
                 validation_loss[stlo,:], validation_accuracy[stlo,:],
                 validation_f1[stlo,:],
                 segment_test_labels[stlo], segment_test_preds[stlo],
                 dirname, plots_folder)
  
# Segment plot
segmentsvec = range(num_training_segments)
plt.figure(figsize=(20,10))

plt.subplot(2,4,1)
plt.title("F1")
plt.plot(segmentsvec, test_F1)
plt.ylim([0, 1])
plt.xlabel("Segments")

plt.subplot(2,4,2)
plt.title("Best F1")
plt.plot(segmentsvec, test_F1_best)
plt.ylim([0, 1])
plt.xlabel("Segments")

plt.subplot(2,4,3)
plt.title("Accuracy")
plt.plot(segmentsvec, test_accuracy)
plt.ylim([0, 1])
plt.xlabel("Segments")

plt.subplot(2,4,4)
plt.title("AUC")
plt.ylim([0, 1])
plt.xlabel("Segments")
plt.plot(segmentsvec, test_AUC)

plt.subplot(2,4,5)
plt.title("Fm")
plt.ylim([0, 1])
plt.xlabel("Segments")
plt.plot(segmentsvec, test_Fm)

plt.subplot(2,4,6)
plt.title("Weighted Fm")
plt.ylim([0, 1])
plt.xlabel("Segments")
plt.plot(segmentsvec, test_Fm_weighted)

plt.subplot(2,4,7)
plt.title("Best Fm")
plt.ylim([0, 1])
plt.xlabel("Segments")
plt.plot(segmentsvec, test_Fm_best)

plt.subplot(2,4,8)
plt.title("Best Weighted Fm")
plt.ylim([0, 1])
plt.xlabel("Segments")
plt.plot(segmentsvec, test_Fm_weighted_best)

plt.savefig(dirname + f'{plots_folder}/TotalTestMetrics.png')

# Average validation loss curve
epochsvec = range(epochs)
plt.figure(4)
mean_loss = validation_loss.mean(axis=0)
plt.ylim([0, 1])
plt.plot(epochsvec, mean_loss)
plt.savefig(dirname + f'{plots_folder}/Avg_Val_loss_Curve.png')
plt.close()