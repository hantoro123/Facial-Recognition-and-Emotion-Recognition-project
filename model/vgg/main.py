import torch
from torch import nn
from PIL import Image, UnidentifiedImageError,ImageFile
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from plot import one_item_plot
from time_save import time_save
from model import get_model
from dataset import get_dataloader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

def model_train(model, data_loader, loss_fn, optimizer, device):
  model.train()
  running_loss, corr = 0, 0      # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
  
  for img, lbl in tqdm(data_loader):
    img, lbl = img.to(device), lbl.to(device)
    
    optimizer.zero_grad()   # 누적 Gradient를 초기화 합니다.
    output = model(img)     # Forward Propagation을 진행하여 결과를 얻습니다.
    loss = loss_fn(output, lbl)   # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.    
    loss.backward()       # 오차역전파(Back Propagation)을 진행하여 미분 값을 계산합니다.
    optimizer.step()      # 계산된 Gradient를 업데이트 합니다.
    _, pred = output.max(dim=1)
    corr += pred.eq(lbl).sum().item()   # pred == lbl 인 것의 개수만 sum
    running_loss += loss.item() * img.size(0)     # 평균 loss * 배치 사이즈
      
  acc = corr / len(data_loader.dataset)
  avg_loss = running_loss / len(data_loader.dataset)
  return avg_loss, acc


def model_evaluate(model, data_loader, loss_fn, device):
  model.eval()
  
  with torch.no_grad():
    running_loss, corr = 0, 0      # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.

    for img, lbl in data_loader:
      img, lbl = img.to(device), lbl.to(device)
      
      output = model(img)
      _, pred = output.max(dim=1)
      corr += torch.sum(pred.eq(lbl)).item()
      running_loss += loss_fn(output, lbl).item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    avg_loss = running_loss / len(data_loader.dataset)
    return avg_loss, acc


def main(args):
  files_ = args.get("files")
  train_params = args.get("train_params")

  time_save_file = files_.get("time_save")
  model_name = files_.get("model_name")
  img_path = files_.get("img_path")

  num_epochs = train_params.get("num_epochs")
  learning_rate = train_params.get("learning_rate")
  batch_size = train_params.get("batch_size")


  train_data_loader, val_data_loader = get_dataloader(img_path, batch_size)

  model = get_model(device)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # 옵티마이저에는 model.parameters()를 지정
  loss_fn = nn.CrossEntropyLoss()     # Multi-Class Classification 이기 때문에 CrossEntropy

  start_time = time.time()      # 시작 시간
  if args.get("train"):
    min_loss = np.inf

    train_losses, train_acces, val_losses, val_acces = [], [], [], []

    for epoch in range(num_epochs):
      train_loss, train_acc = model_train(model, train_data_loader, loss_fn, optimizer, device)
      val_loss, val_acc = model_evaluate(model, val_data_loader, loss_fn, device)  
      
      # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
      if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(model.state_dict(), f'{model_name}_(batch{batch_size}_lr{learning_rate}_ep{num_epochs}).pth')
      
      # Epoch 별 결과를 출력합니다.
      print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

      train_losses.append(train_loss)
      train_acces.append(train_acc)
      val_losses.append(val_loss)
      val_acces.append(val_acc)

  end_time = time.time()        # 종료 시간

  time_save(start_time, end_time, time_save_file, model_name)
  one_item_plot(train_losses, val_losses, 'loss', num_epochs, learning_rate, batch_size, model_name)
  one_item_plot(train_acces, val_acces, 'acc', num_epochs, learning_rate, batch_size, model_name)


def get_args_parser(add_help=True):
  parser = argparse.ArgumentParser(description="vgg", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")
  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  print(args.config)
  exec(open(args.config).read())
  main(config)