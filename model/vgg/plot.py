import matplotlib.pyplot as plt
# loss or accuracy 그래프 그리기

def one_item_plot(trn_data, tst_data, what, num_epochs, learning_rate, batch_size, model_name):
  plot_start = 0     
  epochs_to_plot = range(plot_start, num_epochs)
  
  plt.figure(figsize=(8, 5))
  plt.title(f"{model_name}_{what}(epoch: {num_epochs}_lr: {learning_rate}_batch: {batch_size})")
  plt.plot(epochs_to_plot, trn_data[plot_start:], label=f'trn_{what}')
  plt.plot(epochs_to_plot, tst_data[plot_start:], label=f'tst_{what}')
  plt.xticks(range(plot_start, num_epochs, 5))
  plt.legend()
  # plt.savefig(f'../fig/{model_name}_{what}(epoch: {num_epochs}_lr: {learning_rate}_batch: {batch_size}).png')
  plt.savefig(f'./fig/{model_name}_{what}(epoch: {num_epochs}_lr: {learning_rate}_batch: {batch_size}).png')
  return plt.show