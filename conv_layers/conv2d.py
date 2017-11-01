import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

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
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    H_out = int(1 + (H + 2 * P - HH) / S)
    W_out = int(1 + (W + 2 * P - WW) / S)
    x = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant', constant_values=0)

    output = np.zeros((N, F, H_out, W_out))
    for n in range(N):
        for k in range(F):
            for i in range(W_out):
                for j in range(H_out):
                    output[n, k, j, i] = np.sum(
                        w[k] * x[n, :, S * j: S * j + HH, S * i: S * i + WW]) + \
                                         b[k]

    return output, (x, w, b, conv_param)