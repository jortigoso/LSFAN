import os
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryMatthewsCorrCoef

def train_loop(dataloader, model, loss_fn, optimizer, device, scheduler=None):

  predictions_probab = []
  labels = []
  
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  train_loss, correct, train_f1, train_mcc = 0, 0, 0, 0
  f1_metric = BinaryF1Score()
  mcc_metric = BinaryMatthewsCorrCoef()

  model.train()
  
  for batch, (X, y, id) in enumerate(dataloader):

    X = X.to(device)
    y = y.to(device)

    scores = model(X)
    loss = loss_fn(scores, y.argmax(1))
    pred_probab = nn.Softmax(dim=1)(scores)

    predictions_probab.append(pred_probab.cpu())
    labels.append(y.cpu())
        
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    train_loss += loss.item()
    correct += (pred_probab.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    f1_metric.update(pred_probab[:,1].cpu(), y.argmax(1).cpu())
    train_f1 += f1_metric.compute()
    mcc_value = mcc_metric(pred_probab.argmax(1).cpu(), y.argmax(1).cpu())
    train_mcc += mcc_value.item()

  if scheduler is not None: 
    scheduler.step()
      
  avg_loss = train_loss/num_batches
  accuracy = correct/size
  avg_f1 = train_f1/num_batches
  avg_mcc = train_mcc/num_batches
  
  predictions_probab_tensor = torch.cat(predictions_probab, dim= 0)
  labels_tensor = torch.cat(labels, dim=0)
  
  print(f'Train Error: \n Loss: {avg_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}, F1: {avg_f1:>8f}')
  
  return avg_loss, accuracy, avg_f1, avg_mcc, predictions_probab_tensor, labels_tensor

def validation_loop(dataloader, model, loss_fn, device):
  
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  val_loss, correct, val_f1, val_mcc = 0, 0, 0, 0
  f1_metric = BinaryF1Score()
  mcc_metric = BinaryMatthewsCorrCoef()
  
  predictions_probab = []
  labels = []

  model.eval()

  with torch.no_grad():
    for X, y, id in dataloader:
      
      X = X.to(device)
      y = y.to(device)

      scores = model(X)
      loss = loss_fn(scores, y.argmax(1))
      pred_probab = nn.Softmax(dim=1)(scores)
      
      predictions_probab.append(pred_probab.cpu())
      labels.append(y.cpu())
      
      val_loss += loss.item()
      correct += (pred_probab.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
      # f1_metric.update(pred_probab.argmax(1).cpu(), y.argmax(1).cpu())
      f1_metric.update(pred_probab[:,1].cpu(), y.argmax(1).cpu())
      # f1_metric.update(scores.cpu(), y.cpu())
      val_f1 += f1_metric.compute()
      mcc_value = mcc_metric(pred_probab.argmax(1).cpu(), y.argmax(1).cpu())
      val_mcc += mcc_value.item()

  avg_loss = val_loss/num_batches
  accuracy = correct/size
  avg_f1 = val_f1/num_batches
  avg_mcc = val_mcc/num_batches

  predictions_probab_tensor = torch.cat(predictions_probab, dim=0)
  labels_tensor = torch.cat(labels, dim=0)
  
  print(f'Validation Error: \n Loss: {avg_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}, F1: {avg_f1:>8f}')
  
  return avg_loss, accuracy, avg_f1, avg_mcc, predictions_probab_tensor, labels_tensor
  
def test_loop(dataloader, model, device):

    predictions = []
    labels = []
    probs =[]

    model.eval()
    
    with torch.no_grad():
      for index, (X, y, id) in enumerate(dataloader):
          X = X.to(device)
          scores = model(X)
          pred_probab = nn.Softmax(dim=1)(scores)
          y_pred = pred_probab.argmax(1)
    
          predictions.append(y_pred)
          labels.append(y)
          probs.append(pred_probab)
          
      predictions_tensor = torch.cat(predictions, dim=0)
      labels_tensor = torch.cat(labels, dim=0).argmax(1)
      probs_tensor = torch.cat(probs, dim=0)

    # Predictions are 1 dim
    return predictions_tensor, probs_tensor, labels_tensor


def load_checkpoint(model, optimizer,lr_sched, device, state_filename ,model_filename,scheduler = None):
  # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
  start_epoch = 0
  if os.path.isfile(state_filename):
      
    print("=> loading checkpoint '{}' & '{}'".format(state_filename,model_filename))
    checkpoint = torch.load(state_filename)
    start_epoch = checkpoint['epoch']+1
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    if device == "cuda":
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    lr_sched=checkpoint['lr_sched']
    model.load_state_dict(torch.load(model_filename))
    scheduler.load_state_dict(checkpoint['scheduler'])

    #Convert net to device
    model=model.to(device)
    print("=> loaded checkpoint '{}' & '{}' (epoch {})".format(state_filename, model_filename,checkpoint['epoch']))
  else:
    print("=> no checkpoint found at '{}' & '{}'".format(state_filename,model_filename))

  return model, optimizer, start_epoch, lr_sched, scheduler