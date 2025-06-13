import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryMatthewsCorrCoef
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve

def segment_global_metrics(y_prob, y_pred, y_true, segment):

  f1_metric = BinaryF1Score()
  f1_metric.update(y_pred, y_true)
  test_F1 = f1_metric.compute()
  print(f'Segment {segment} Test F1: {test_F1}')

  precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:,1])
  f1_scores = 2 * precision * recall / (precision + recall)
  best_threshold_idx = np.argmax(f1_scores[np.isfinite(f1_scores)])
  best_test_F1 = f1_scores[best_threshold_idx]
  print(f'Segment {segment} Best Test F1: {best_test_F1}')

  test_correct = (y_pred == y_true).type(torch.float).sum().item()
  test_accuracy = test_correct/len(y_true)
  print(f'Segment {segment} Test Accuracy: {test_accuracy}')

  test_auc = metrics.roc_auc_score(y_true, y_prob[:,1])
  print(f'Segment {segment} Test AUC: {test_auc}')

  mcc = BinaryMatthewsCorrCoef()
  test_MCC = mcc(y_pred, y_true).item()
  print(f'Segment {segment} Test MCC: {test_MCC}')

  # For FM
  precision_score_zero = precision_score(y_true, y_pred, pos_label=0)
  recall_score_zero = recall_score(y_true, y_pred, pos_label=0)
  precision_score_one = precision_score(y_true, y_pred, pos_label=1)
  recall_score_one = precision_score(y_true, y_pred, pos_label=1)

  f1_zero = 2*(precision_score_zero * recall_score_zero)/(precision_score_zero + recall_score_zero)
  f1_one = 2*(precision_score_one * recall_score_one)/(precision_score_one + recall_score_one)

  num_zero_samples = (y_true == 0).sum().item()
  num_one_samples = (y_true == 1).sum().item()
  weight_zero = num_zero_samples / (num_zero_samples + num_one_samples)
  weight_one = num_one_samples / (num_zero_samples + num_one_samples)

  Fm = (f1_zero + f1_one)/2
  Fm_weighted = weight_zero * f1_zero + weight_one * f1_one 

  precision_zero, recall_zero, thresholds_zero = precision_recall_curve(y_true, y_prob[:,0], pos_label=0)
  precision_one, recall_one, thresholds_one = precision_recall_curve(y_true, y_prob[:,1], pos_label=1)

  f1_zero_pr = 2*(precision_zero * recall_zero)/(precision_zero + recall_zero)
  f1_one_pr = 2*(precision_one * recall_one)/(precision_one + recall_one)

  best_threshold_zero = np.argmax(f1_zero_pr[np.isfinite(f1_zero_pr)])
  best_threshold_one = np.argmax(f1_one_pr[np.isfinite(f1_one_pr)])

  f1_zero_best = f1_zero_pr[best_threshold_zero]
  
  f1_one_best = f1_one_pr[best_threshold_one]
  f1_one_recall = recall_one[best_threshold_one]
  f1_one_precision = precision_one[best_threshold_one]

  Fm_best = (f1_zero_best + f1_one_best)/2
  Fm_weighted_best = weight_zero * f1_zero_best + weight_one * f1_one_best
    
  return test_F1, best_test_F1, test_accuracy, \
  test_auc, test_MCC, f1_zero, \
  f1_one, Fm, Fm_weighted, \
  Fm_best, Fm_weighted_best, \
  f1_zero_best, f1_one_best

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def training_plots(stlo, training_loss, training_accuracy, 
                 training_f1, training_auc, validation_loss, 
                 validation_accuracy, validation_f1,
                 segment_test_labels, segment_test_preds, 
                 dirname, plots_folder):
    
  epochsvec = range(training_loss.shape[0])

  plt.figure(figsize=(20,10))
  plt.subplot(2,4,1)
  plt.title("Average Train Loss")
  plt.plot(epochsvec,training_loss)
  plt.ylim([0, 1])
  plt.xlabel("Epochs")
  plt.subplot(2,4,2)
  plt.title("Train Accuracy")
  plt.plot(epochsvec, training_accuracy)
  plt.ylim([0, 1])
  plt.xlabel("Epochs")
  plt.subplot(2,4,3)
  plt.title("Train F1 score")
  plt.plot(epochsvec, training_f1)
  plt.ylim([0, 1])
  plt.xlabel("Epochs")
  plt.subplot(2,4,4)
  plt.title("Train AUC")
  plt.ylim([0, 1])
  plt.plot(training_auc)

  plt.subplot(2,4,5)
  plt.title("Average Validation (Test) Loss")
  plt.plot(epochsvec, validation_loss)
  plt.ylim([0, 1])
  plt.xlabel("Epochs")
  plt.subplot(2,4,6)
  plt.title("Average Validation (Test) Accuracy")
  plt.plot(epochsvec,validation_accuracy)
  plt.ylim([0, 1])
  plt.xlabel("Epochs")
  plt.subplot(2,4,7)
  plt.title("Validation F1 score")
  plt.plot(epochsvec, validation_f1)
  plt.ylim([0, 1])
  plt.xlabel("Epochs")
  plt.savefig(dirname + f'{plots_folder}/{stlo}_left_out.png')
  plt.clf() 
  
  plt.figure(2, figsize=(8,7))
  cnf_matrix = confusion_matrix(segment_test_labels.cpu(), segment_test_preds.cpu())
  plot_confusion_matrix(cnf_matrix, classes = [0,1], normalize=False, title='Confusion matrix, with normalization')
  plt.savefig(dirname + f'{plots_folder}/{stlo}_left_out_CM.png', bbox_inches='tight')
  plt.clf() 