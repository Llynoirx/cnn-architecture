import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        N, _, H_in, W_in = A.shape
        H_out =  H_in - self.kernel_size + 1
        W_out =  W_in - self.kernel_size + 1

        Z = np.zeros((N, self.out_channels, H_out, W_out))

        for h, w in np.ndindex(H_out, W_out):
            Z[:, :, h, w] = np.tensordot(self.A[:, :, h:h+self.kernel_size, w:w+self.kernel_size], 
                                         self.W, axes=((1, 2, 3), (1, 2, 3))) + self.b
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        _, _, H_in, W_in = self.A.shape
        _, _, H_out, W_out = dLdZ.shape

        # Find dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        # Find dLdW
        self.dLdW = np.zeros(self.W.shape)
        for i, j in np.ndindex(self.kernel_size, self.kernel_size):
            self.dLdW[:, :, i, j] = np.tensordot(dLdZ, self.A[:, :, i:i+H_out, j:j+W_out], 
                                                 axes=((0, 2, 3), (0, 2, 3)))
        
        # Find dLdA
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), 
                                    (self.kernel_size-1, self.kernel_size-1), 
                                    (self.kernel_size-1, self.kernel_size-1)), 
                            'constant', constant_values=0)
        W_flipped = np.flip(self.W, axis=(2, 3))

        dLdA = np.zeros(self.A.shape)
        for h, w in np.ndindex(H_in, W_in):
            dLdA[:, :, h, w] = np.tensordot(dLdZ_padded[:, :, h:h+self.kernel_size, w:w+self.kernel_size], 
                                            W_flipped, axes=((1, 2, 3), (0, 2, 3)))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 
                         'constant', constant_values=(0,0))

        # Call Conv2d_stride1
        Z_stride1 = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ_stride1 = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ_stride1)

        # Unpad the gradient
        dLdA_unpadded = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA_unpadded