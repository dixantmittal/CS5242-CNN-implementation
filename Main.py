# A bit of setup
from __future__ import print_function

import matplotlib.pyplot as plt

from code_base.classifiers.cnn import *
from code_base.data_utils import *
from code_base.layers import *
from code_base.solver import Solver

settings.print_time_analysis(False)

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# Load the (preprocessed) CIFAR2 (airplane and bird) data.

data = get_CIFAR2_data()
for k, v in data.items():
    print('%s:        ' % k, v.shape)

np.random.seed(231)

num_train = 100
small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
}

to_load = input('\nDo you want to load pre-trained model? [y/n]: ')
if to_load == 'y' or to_load == 'Y':
    model = load_model('./models/cnn_model_conv64_fil5_fc512_numc2.p')
else:
    model = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=512, reg=0.0001, num_filters=64,
                              filter_size=5)

solver = Solver(model, data,
                num_epochs=20, batch_size=128,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                lr_decay=0.8,
                num_train_samples=100,
                verbose=True, print_every=1)
start = datetime.datetime.now()
solver.train()
end = datetime.datetime.now()
print('Total time taken: ', end - start)

to_save = input('\nDo you want to save this? [y/n]: ')
if to_save == 'y' or to_save == 'Y':
    save_model(model, './models/cnn_model_conv64_fil5_fc512_numc2.p')

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# Visualize Filters
from code_base.vis_utils import visualize_grid

grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
