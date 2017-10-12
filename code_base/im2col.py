import math

import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    out_height = 1 + (H + padding - field_height) // stride
    out_width = 1 + (W + padding - field_width) // stride

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    pad_floor = int(math.floor(padding / 2))
    pad_ceil = int(math.ceil(padding / 2))
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_floor, pad_ceil), (pad_floor, pad_ceil)), mode='constant',
                      constant_values=0)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def im2col_indices_different_images(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    pad_floor = int(math.floor(padding / 2))
    pad_ceil = int(math.ceil(padding / 2))
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_floor, pad_ceil), (pad_floor, pad_ceil)), mode='constant',
                      constant_values=0)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + padding, W + padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    pad_floor = int(math.floor(padding / 2))
    pad_ceil = int(math.ceil(padding / 2))
    return x_padded[:, :, pad_floor:-pad_ceil, pad_floor:-pad_ceil]


pass
