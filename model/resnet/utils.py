import torch
import tqdm
from tqdm.notebook import tqdm
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def Multi_confusion(model, tst_loader, device, criterion):
    tst_losses = []
    tst_acces = []

    tst_accuracy_mean = 0
    tst_loss_mean = 0
    model = model.to(device)
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

            tst_losses.append(tst_loss.item())
            tst_acces.append(tst_acc.item())


    tst_accuracy_mean /= len(tst_loader)
    tst_loss_mean /= len(tst_loader)


    print(f"Test loss: {tst_loss_mean:.4f}, Test accuracy: {tst_accuracy_mean:.4f}\n")
    return pred, true_label, tst_losses, tst_acces


def confusion_Matrix(pred_flat, true_label_flat):
    classes = ['anger', 'anxiety', 'embarrass', 'happy', 'normal', 'pain', 'sad']

    cm = confusion_matrix(pred_flat, true_label_flat, normalize='true')
    plt.figure(figsize=(10,8))
    plt.title('Resnet Confusion Matrix', fontsize=15)
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('True',fontsize=13)
    plt.show()