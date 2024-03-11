# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

# convolving the input with a kernel in just 1 direction
class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        N, _, input_size = A.shape
        output_size =  input_size - self.kernel_size + 1

        Z = np.zeros((N, self.out_channels, output_size))

        for s in range(output_size):
            Z[:, :, s] = np.tensordot(self.A[:, :, s:s+self.kernel_size], 
                                      self.W, axes=((1, 2), (1, 2))) + self.b
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        input_size = self.A.shape[2]
        output_size = dLdZ.shape[2]

        # Find dLdb
        self.dLdb = np.sum(dLdZ, axis=(0,2))

        # Find dLdW
        self.dLdW = np.zeros(self.W.shape)
        for i in range(self.kernel_size):
            self.dLdW[:, :, i] = np.tensordot(dLdZ, self.A[:, :, i:i+output_size], axes=((0,2), (0,2)))
        
        # Find dLdA
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), 
                                    'constant', constant_values=0)
        W_flipped = np.flip(self.W, axis=2)

        dLdA = np.zeros(self.A.shape)
        for s in range(input_size): 
            dLdA[:, :, s] = np.tensordot(dLdZ_padded[:, :, s:s+self.kernel_size], W_flipped, 
                                         axes=((1, 2), (0, 2)))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, 
                                             weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), 'constant', constant_values=(0,0))

        # Call Conv1d_stride1
        Z_stride1 = self.conv1d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample1d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ_stride1 = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ_stride1)

        # Unpad the gradient
        dLdA_unpadded = dLdA[:, :, self.pad:dLdA.shape[-1]-self.pad]

        return dLdA_unpadded
