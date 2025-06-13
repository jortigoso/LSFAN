import torch 
import torch.nn as nn
import torch.nn.functional as F

input_shape = (1, 180, 30) # CxHxW
frame_kernel = (3,3)
frame_pooling = (2,1)
frame_stride = (2,1)
padd = (0, frame_kernel[1]//2)
bias = True

feature_extractor_small = nn.Sequential(
  nn.Conv2d(input_shape[0], 2, frame_kernel, padding=padd, bias=bias),
  nn.BatchNorm2d(2),
  nn.ReLU(),

  nn.Conv2d(2, 4, frame_kernel, padding=padd, bias=bias),
  nn.BatchNorm2d(4),
  nn.ReLU(),

  nn.MaxPool2d(frame_pooling, frame_stride), 

  nn.Conv2d(4, 8, frame_kernel, padding=padd, bias=bias),
  nn.BatchNorm2d(8),
  nn.ReLU(),

  nn.Conv2d(8, 16, frame_kernel, padding=padd, bias=bias),
  nn.BatchNorm2d(16),
  nn.ReLU(),

  nn.MaxPool2d(frame_pooling, frame_stride), 
)

class CNN_SAP(nn.Module):
  def __init__(self):
    super(CNN_SAP, self).__init__()

    self.FrameFeatureExtractor = feature_extractor_small
    
    self.Classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(16*42, 2),
    )

  def forward(self, frame):

    frame_features = self.FrameFeatureExtractor(frame)  # Frame features are (B, C=16, H=42, W=30)
    extracted_features = nn.AdaptiveAvgPool2d((1, frame_features.shape[2]))(frame_features) # (B, 16, 1, 42)
    extracted_features = extracted_features.squeeze(2) # (B, 16, 42)
    logits = self.Classifier(extracted_features)

    return logits

class CNN_TAP(nn.Module):
  def __init__(self):
    super(CNN_TAP, self).__init__()

    self.FrameFeatureExtractor = feature_extractor_small
    
    self.Classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(16*30, 2),
    )

  def forward(self, frame):

    frame_features = self.FrameFeatureExtractor(frame)  # Frame features are (B, C=16, H=42, W=30)
    extracted_features = nn.AdaptiveAvgPool2d((1, frame_features.shape[3]))(frame_features) # (B, 16, 1, 30)
    extracted_features = extracted_features.squeeze(2) # Removes channel (B, 16, 30)
    logits = self.Classifier(extracted_features)

    return logits


class CNN_TAP_Att(nn.Module):
  def __init__(self, d_model=30, nhead=5, dim_feedforward=30, dropout=0.2, num_layers=1):
    super(CNN_TAP_Att, self).__init__()

    self.d_model = d_model

    self.FrameFeatureExtractor = feature_extractor_small
    
    self.encoderlayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True, norm_first=False)
    self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=num_layers)

    self.Classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(16*30, 2),
    )

  def forward(self, frame):

    frame_features = self.FrameFeatureExtractor(frame)  # Frame features are (B, C=16, H=42, W=30)
    extracted_features = nn.AdaptiveAvgPool2d((1, frame_features.shape[3]))(frame_features) # (B, 16, 1, 30)
    extracted_features = extracted_features.squeeze(2) # Removes channel (B, 16, 30)
    encoded_features = self.encoder(extracted_features)
    logits = self.Classifier(encoded_features)

    return logits