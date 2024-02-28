import matplotlib.pyplot as plt
import os

def one_item_plot(trn_data, tst_data, what, epochs, lr, batch_size):
    plot_start = 0
    epochs_to_plot = range(plot_start, epochs)

    plt.figure(figsize=(8, 5))
    plt.title(f"resnet_{what}(epoch:{epochs}_lr:{lr}_batch:{batch_size})")
    plt.plot(epochs_to_plot, trn_data[plot_start:], label=f'trn_{what}')
    plt.plot(epochs_to_plot, tst_data[plot_start:], label=f'tst_{what}')
    plt.xticks(range(plot_start, epochs, 5))
    plt.legend()
    os.makedirs('./fig', exist_ok=True)
    plt.savefig(f'./fig/test_resnet_{what}(epoch:{epochs}_lr:{lr}_batch:{batch_size}).png')

    return plt.show