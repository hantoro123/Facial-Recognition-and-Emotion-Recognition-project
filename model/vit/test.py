import tqdm
from tqdm import tqdm
from model import *

def test_model(cfg, model, criterion, tst_loader, device):

  model.to(device)

  tst_losses = []
  tst_acces = []

  tst_accuracy_mean = 0
  tst_loss_mean = 0

  model.eval()

  pred = []
  true_label = []

  with torch.no_grad():
    for data, label in tqdm(tst_loader, desc='test.. ', leave=True):

      data = data.to(device)
      label = label.to(device)

      tst_output = model(data)
      tst_loss = criterion(tst_output, label)

      tst_acc = (tst_output.argmax(dim=1) == label).float().mean()
      tst_accuracy_mean += tst_acc.item()
      tst_loss_mean += tst_loss.item()

      pred.append(tst_output.argmax(dim=1).tolist())
      true_label.append(label.tolist())

  tst_accuracy_mean /= len(tst_loader)
  tst_loss_mean /= len(tst_loader)


  print(f"Test loss: {tst_loss_mean:.4f}, Test accuracy: {tst_accuracy_mean:.4f}\n")

  return pred, true_label