import os

from matplotlib import pyplot as plt


def save_loss_plot(num_epochs, filename, train_acc_file, test_acc_file=None, log_scale=False):
    train_acc_list = []
    with open(train_acc_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 1:
                train_acc_list.append(float(line.split()[0]))
    test_acc_list = []
    if test_acc_file is not None:
        with open(test_acc_file) as f:
            lines = f.readlines()
            for line in lines:
                test_acc_list.append(float(line))

    x = list(range(num_epochs))
    fig, ax = plt.subplots()
    ax.plot(x, train_acc_list, label='Training')
    if test_acc_file is not None:
        ax.plot(x, test_acc_list, label='Testing')
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(os.path.join(os.path.split(train_acc_file)[0], f'{filename}.png'))
