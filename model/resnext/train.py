# 필요한 라이브러리 및 모듈 import
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import os
import csv

import matplotlib.pyplot as plt
from customdataset import CustomDataset
from model import *

# 사용자 정의 함수 및 클래스 정의
def train_model(cfg, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):
    since = time.time()
      # 각 파라미터
    train_params = cfg.get('train_params')
    model_params = cfg.get('resnext').get('model_params')

    device = train_params.get('device')
    resumed = False
    best_model_wts = model.state_dict()
    trn_losses = []
    trn_acces = []
    val_losses = []
    val_acces = []
    
    for epoch in range(model_params.get('start_epoch') + 1, num_epochs+1):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if model_params.get('start_epoch') > 0 and (not resumed):
                    scheduler.step(model_params.get('start_epoch') + 1)
                    resumed = True
                else:
                    scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # wrap them in Variable
                if model_params.get('use_gpu'):
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs).to(device)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i + 1) * model_params.get('batch_size'))
                batch_acc = running_corrects / ((i + 1) * model_params.get('batch_size'))

                if phase == 'train' and i % model_params.get('print_freq') == 0:
                    print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase] / model_params.get('batch_size')) - 1, scheduler.get_lr()[0], phase, batch_loss, batch_acc,
                          model_params.get('print_freq') / (time.time() - tic_batch)))
                    tic_batch = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # Save loss and accuracy
            if phase == 'train':
                trn_losses.append(epoch_loss)
                trn_acces.append(epoch_acc.cpu())
            else:
                val_losses.append(epoch_loss)
                val_acces.append(epoch_acc.cpu())

        if (epoch + 1) % model_params.get('save_epoch_freq') == 0:
            if not os.path.exists(model_params.get('save_path')):
                os.makedirs(model_params.get('save_path'))
            torch.save(model, os.path.join(model_params.get('save_path'), "epoch_" + str(epoch) + ".pth.tar"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, trn_losses, trn_acces, val_losses, val_acces

def model_save(model,cfg, start_time):

    train_params = cfg.get('train_params')
    model_params = cfg.get('resnext').get('model_params')
    optim_params = train_params.get('optim_params')

    # 모델 저장
    torch.save(model.state_dict(), f"ResNeXt_transfer_crop_(batch{model_params.get('batch_size')}_lr{optim_params.get('lr')}_ep{model_params.get('num_epochs')}).pth")

    end_time = time.time()
    time_taken = (end_time - start_time)/60
    print(f"걸린시간 {time_taken:.2f}분 걸렸습니다.")

    csv_file_name = '/home/KDT-admin/work/hsj/resnext/fig/training_time.csv'    # 시간 저장

    try:
        with open(csv_file_name, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Training Time'])

    except FileExistsError:
        with open(csv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"ResNeXt_transfer_crop_(batch{model_params.get('batch_size')}_lr{optim_params.get('lr')}_ep{model_params.get('num_epochs')}", time_taken])

