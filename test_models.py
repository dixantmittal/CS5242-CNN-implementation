# A bit of setup
from __future__ import print_function

from code_base.data_utils import *
# Load the (preprocessed) CIFAR2 (airplane and bird) data.
from code_base.solver import Solver

data = get_CIFAR2_data()
for k, v in data.items():
    print('%s:        ' % k, v.shape)

model = load_model('./models/cnn_model_conv64_fil5_fc512_numc2_dropout_0.5.p')
model.use_dropout = False
print('Test Accuracy: ', Solver(model, data=data).check_accuracy(data['X_val'], data['y_val'], batch_size=500))
