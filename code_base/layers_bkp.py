import numpy as np
import datetime
from code_base.matrix_transformation import *
from code_base.im2col import *
from code_base.classifiers import *


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_rs = np.reshape(x, (N, -1))
    out = x_rs.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Bias, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    x_rs = np.reshape(x, (N, -1))
    db = dout.sum(axis=0)
    dw = x_rs.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x >= 0) * dout
    return dx


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w, b, conv_param):
    """
    Forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input in each x-y direction.
         We will use the same definition in lecture notes 3b, slide 13 (ie. same padding on both sides).

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + pad - HH) / stride
      W' = 1 + (W + pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    _H = 1 + (H + pad - HH) // stride
    _W = 1 + (W + pad - WW) // stride

    w_flat = get_flattened_filters(w)
    # x_flat = get_receptive_field(x, HH, WW, stride, pad)
    x_flat = im2col_indices(x, HH, WW, pad, stride)
    activations = np.dot(w_flat, x_flat) + (b)[:, None]
    # activations = np.transpose(activations, axes=(1, 0, 2)) + b.reshape(b.shape[0], 1)
    out = activations.reshape(F, _H, _W, N)
    out = out.transpose(3, 0, 1, 2)
    # out = activations.reshape(N, F, _H, _W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param, x_flat)
    return out, cache


def conv_backward(dout, cache):
    """
    Backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param, x_flat = cache
    N, C, H, W = x.shape
    F, _, hh, ww = w.shape

    # print(dout)
    db = np.sum(dout, axis=(0, 2, 3))


    dout_res = dout.transpose(1, 2, 3, 0).reshape(F, -1)
    dw = np.dot(dout_res, x_flat.T)
    dw = dw.reshape(w.shape)

    w_res = w.reshape(F, -1)
    dx_flat = np.dot(w_res.T, dout_res)
    dx = col2im_indices(dx_flat, x.shape, hh, ww, conv_param['pad'], conv_param['stride'])

    # N, C, H, W = x.shape
    # F, _, HH, WW = w.shape
    # stride, pad = conv_param['stride'], conv_param['pad']
    # _H = 1 + (H + pad - HH) // stride
    # _W = 1 + (W + pad - WW) // stride
    #
    # x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    # dx = np.zeros_like(x)
    # dx_pad = np.zeros_like(x_pad)
    # dw = np.zeros_like(w)
    # db = np.zeros_like(b)
    #
    # db = np.sum(dout, axis=(0, 2, 3))
    #
    # x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    # for i in range(_H):
    #     for j in range(_W):
    #         x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
    #         for k in range(F):  # compute dw
    #             dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
    #         for n in range(N):  # compute dx_pad
    #             dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((w[:, :, :, :] *
    #                                                                                             (dout[n, :, i, j])[:,
    #                                                                                             None, None, None]),
    #                                                                                            axis=0)
    # dx = dx_pad[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    Forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    _H = 1 + (H - pool_height) // stride
    _W = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, _H, _W))

    for h in range(_H):
        for w in range(_W):
            field = x[:, :, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width]
            out[:, :, h, w] = np.max(field, axis=(2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    Backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    _H = 1 + (H - pool_height) // stride
    _W = 1 + (W - pool_width) // stride

    dx = np.zeros_like(x)

    for h in range(_H):
        for w in range(_W):
            field = x[:, :, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width]
            max = np.max(field, axis=(2, 3))
            mask = (field == (max)[:, :, None, None])
            dx[:, :, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width] += (dout[:, :, h, w])[:,
                                                                                                 :, None, None] * mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
