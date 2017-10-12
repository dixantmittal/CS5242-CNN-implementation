import numpy as np
import math as math
import datetime


def img2col(x):
    N, C, H, W = x.shape
    a = x.reshape(N, C * H * W)
    return a


def col2img(x, C, H, W):
    N, d = x.shape
    return x.reshape(N, C, H, W)


def get_receptive_field(x, HH, WW, stride, pad):
    N, C, H, W = x.shape
    pad_floor = int(math.floor(pad / 2))
    pad_ceil = int(math.ceil(pad / 2))
    x = np.pad(x, ((0, 0), (0, 0), (pad_floor, pad_ceil), (pad_floor, pad_ceil)), mode='constant', constant_values=0)
    _H = 1 + (H + pad - HH) // stride
    _W = 1 + (W + pad - WW) // stride

    receptive_field = np.array([])
    print('starting flattening')
    t1 = datetime.datetime.now()
    for i in range(_H):
        for j in range(_W):
            field = x[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            field = img2col(field)
            if len(receptive_field) == 0:
                receptive_field = field
            else:
                print('stacking: ', i)
                t = datetime.datetime.now()
                receptive_field = np.dstack((receptive_field, field))
                print('time: ', datetime.datetime.now() - t)
    t2 = datetime.datetime.now()
    print('time taken: ', t2 - t1)
    return receptive_field


def get_flattened_filters(w):
    F, C, HH, WW = w.shape
    w = w.reshape(F, C * HH * WW)
    return w
