# A bit of setup
from __future__ import print_function

import Models
import code_base.solver as slvr
from Plotter import *
from code_base.data_utils import *
from code_base.layers import *
from code_base.solver import Solver

settings.time_analysis['logger_enabled'] = False



# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# Load the (preprocessed) CIFAR2 (airplane and bird) data.

def getSolver(model, data, alpha, alpha_decay, epoch=10, batch_size=128):
    return Solver(model, data, num_epochs=epoch, batch_size=batch_size,
                  update_rule='adam',
                  optim_config={
                      'learning_rate': alpha,
                  }, lr_decay=alpha_decay, verbose=True, print_every=1)


def train_model(model_key):
    slvr._file.write('\n\n>>>> MODEL - ' + model_key + ' <<<<')
    model = Models.Models[model_key]
    solver = getSolver(model=model, data=data, alpha=1e-3, alpha_decay=0.8, epoch=3)
    start = datetime.datetime.now()
    solver.train()
    end = datetime.datetime.now()
    slvr._file.write('\nTotal time taken: ' + str(end - start))
    slvr._file.flush()

    plot_graph(solver.loss_history, solver.train_acc_history, solver.val_acc_history, model_key=model_key)
    save_model(model, './models/cnn_model_' + model_key + '.p')



data = pickle.load(open('./data.p', 'rb'), encoding='latin1')

# create augmented data - mirror image
# aug_X_train = np.flip(data['X_train'], 3)
# data['X_train'] = np.concatenate((data['X_train'], aug_X_train), 0)
# data['y_train'] = np.concatenate((data['y_train'], data['y_train']), 0)


for k, v in data.items():
    print('%s:  ' % k, v.shape)

train_model('conv64_filter5_fc512_drop0')
train_model('conv64_filter5_fc512_drop05')
train_model('conv128_filter3_fc1024_drop0')
train_model('conv128_filter3_fc1024_drop05')
train_model('conv32_filter7_fc256_drop0')
train_model('conv32_filter7_fc256_drop05')
