import time
import tqdm
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

from model import *


def train_model(cfg, model, criterion, optimizer, epochs, train_loader, val_loader, device):
  start_time = time.time()   
  batch_size = cfg.get('model').get('model_params').get('batch_size')
  lr = cfg.get('train_params').get('lr')
  
  pbar = tqdm(range(epochs), desc= f'Total ', position=2)

  trn_losses = []
  trn_acces = []
  val_losses = []
  val_acces = []
  min_loss = np.inf
  
  for epoch in pbar:
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for data, label in tqdm(train_loader, desc='training.. ', position=0, leave=True): 
      data = data.to(device)
      label = label.to(device)

      output = model(data)
      trn_loss = criterion(output, label)

      optimizer.zero_grad()
      trn_loss.backward()
      optimizer.step()

      trn_acc = (output.argmax(dim=1) == label).float().mean()
      epoch_accuracy += trn_acc.item() 
      epoch_loss += trn_loss.item()

      # print(f'accuracy : {trn_acc.item()}, loss : {trn_loss.item()}')
    
    epoch_accuracy /= len(train_loader)
    epoch_loss /= len(train_loader)
    
    trn_acces.append(epoch_accuracy)
    trn_losses.append(epoch_loss)

    model.eval()

    epoch_val_accuracy = 0
    epoch_val_loss = 0

    with torch.no_grad():
      for data, label in tqdm(val_loader, desc='validation.. ', position=2, leave=True):
        data = data.to(device)
        label = label.to(device)

        val_output = model(data)
        val_loss = criterion(val_output, label)

        val_acc = (val_output.argmax(dim=1) == label).float().mean()
        epoch_val_accuracy += val_acc.item()
        epoch_val_loss += val_loss.item()

        if val_loss < min_loss:
          print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
          min_loss = val_loss
          os.makedirs('./best', exist_ok=True)
          torch.save(model.state_dict(), f'./best/ViT_(epoch:{epochs}_lr:{lr}_batch:{batch_size}.pth')

    epoch_val_accuracy /= len(val_loader)
    epoch_val_loss /= len(val_loader)

    val_losses.append(epoch_val_loss)
    val_acces.append(epoch_val_accuracy)

    print(f"Epoch : {epoch+1} - trn_loss : {epoch_loss:.4f} - trn_acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
      
  pbar.close()
  end_time = time.time()    
  time_taken = (end_time - start_time)/60

  print(f"걸린 시간: {time_taken :.2f} 분")

  return model, trn_losses, trn_acces, val_losses, val_acces
