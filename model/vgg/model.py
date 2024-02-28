from torchvision import models # pretrained 모델을 가져오기 위한 import
import torch.nn as nn

def get_model(device):
  model = models.vgg11(pretrained=True) # pretrained=False로 설정되었을 경우 가중치는 가져오지 않음
  # vgg 11, 13, 16, 19
  for param in model.parameters():
    param.requires_grad = False  # 가중치 Freeze

  # Fully-Connected Layer를 Sequential로 생성하여 VGG pretrained 모델의 'Classifier'에 연결
  fc = nn.Sequential(
      nn.Linear(25088, 4096), 
      nn.ReLU(), 
      nn.Linear(4096, 64), 
      nn.ReLU(), 
      nn.Linear(64, 7), # 감정 분류 7개
  )

  model.classifier = fc
  model = model.to(device)
  print(device)
  return model