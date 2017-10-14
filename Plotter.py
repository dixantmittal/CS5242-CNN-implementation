import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def plot_graph(loss_history, train_acc_history, val_acc_history, model_key):
    plt.figure(1)
    plt.clf()
    plt.title(model_key)
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, 'o', ms=3, label=model_key)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, '-o')
    plt.plot(val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('./plots/' + model_key + '.png')

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, 'o', ms=3, label=model_key)
    plt.legend(loc='upper right', prop={'size': 6})
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, '-o', ms=3, label=model_key + '[train]')
    plt.plot(val_acc_history, '-o', ms=3, label=model_key + '[val]')
    plt.legend(loc='upper left', prop={'size': 6})
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('./plots/comparison.png')
