import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import confusion_matrix

# loss or accuracy 그래프 그리기
def one_item_plot(trn_data, tst_data,what, lr, batch_size, epochs):
  plot_start = 0     
  epochs_to_plot = range(plot_start, epochs)

  plt.figure(figsize=(8, 5))
  plt.title(f"ResNeXt_transfer_crop_{what}(batch: {batch_size})_lr: {lr}_epoch: {epochs}")
  plt.plot(epochs_to_plot, trn_data[plot_start:], label=f'trn_{what}')
  plt.plot(epochs_to_plot, tst_data[plot_start:], label=f'tst_{what}')
  plt.xticks(range(plot_start, epochs, 10))
  plt.legend()
  plt.savefig(f'./fig/ResNeXt_transfer_crop_{what}(batch{batch_size}_lr{lr}_ep{epochs}).png')
  return plt.show

def save_loss_acc(cfg,trn_losses, val_losses, trn_acces, val_acces):
  table_data = zip(trn_losses, val_losses, trn_acces, val_acces)

  # CSV 파일 경로
  csv_file_path = f'ResNeXt_transfer_crop_(batch{cfg.get('resnext').get('model_params').get('batch_size')}_lr{cfg.get('train_params').get('optim_params').get('lr')}_ep{cfg.get('train_params').get('epochs')}).csv'

  # CSV 파일 쓰기
  with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
      writer = csv.writer(csv_file)
      
      # 헤더 쓰기
      writer.writerow(['trn_losses', 'val_losses', 'trn_acces', 'val_acces'])
      
      # 데이터 쓰기
      writer.writerows(table_data)

def confusion_Matrix(pred_flat, true_label_flat):
    classes = ['anger', 'anxiety', 'embarrass', 'happy', 'normal', 'pain', 'sad']

    cm = confusion_matrix(pred_flat, true_label_flat, normalize='true')
    plt.figure(figsize=(10,8))
    plt.title('Resnet Confusion Matrix', fontsize=15)
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('True',fontsize=13)
    plt.show()