import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def one_item_plot(trn_data, tst_data, what, lr, batch_size, epochs):

  plot_start = 0
  epochs_to_plot = range(plot_start, epochs)

  plt.figure(figsize=(8, 5))
  plt.title(f"ViT_{what}(epoch:{epochs}_lr:{lr}_batch:{batch_size})")
  plt.plot(epochs_to_plot, trn_data[plot_start:], label=f'trn_{what}')
  plt.plot(epochs_to_plot, tst_data[plot_start:], label=f'tst_{what}')
  plt.xticks(range(epochs, 5))
  plt.legend()
  os.makedirs('./fig', exist_ok=True)
  plt.savefig(f'./fig/ViT_{what}(epoch:{epochs}_lr:{lr}_batch:{batch_size}).png')
  
  return plt.show()


def c_matrix(pred, true_label):

  pred_flat = [item for sublist in pred for item in sublist]
  true_label_flat = [item for sublist in true_label for item in sublist]

  classes = ['anger', 'anxiety', 'embarrass', 'happy', 'normal', 'pain', 'sad']

  cm = confusion_matrix(true_label_flat, pred_flat, normalize='true')
  plt.figure(figsize=(10,8))
  plt.title(f'ViT Confusion Matrix', fontsize=15)
  sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
  plt.xlabel('Predicted', fontsize=13)
  plt.ylabel('True',fontsize=13)
  os.makedirs('./fig', exist_ok=True)
  plt.savefig(f'./fig/ViT_Confusion_matrix.png')
  
  return plt.show()
