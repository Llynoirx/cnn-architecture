import numpy as np
from resampling import *


# selects the largest from a pool of elements and is performed by “scanning” the input.
class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N, C, W_in, H_in = self.A.shape
        W_out = W_in-self.kernel + 1
        H_out = H_in-self.kernel + 1

        Z = np.zeros((N, C, W_out, H_out))
        for n, c, w, h in np.ndindex(N, C, W_out, H_out):
            Z[n, c, w, h] = np.max(A[n, c, w:w+self.kernel, h:h+self.kernel])
            
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C, W_out, H_out = dLdZ.shape
        dLdA = np.zeros(self.A.shape)
        for n, c, w, h in np.ndindex(N, C, W_out, H_out):
            A_window = self.A[n, c, w:w+self.kernel, h:h+self.kernel]
            is_max = (A_window == np.max(A_window)) # 1: max val, 0: ow
            dLdA[n, c, w:w+self.kernel, h:h+self.kernel] += is_max * dLdZ[n, c, w, h]
        return dLdA

#  takes the arithmetic mean of elements and is performed by “scanning” the input
class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N, C, W_in, H_in = self.A.shape
        W_out = W_in-self.kernel + 1
        H_out = H_in-self.kernel + 1

        Z = np.zeros((N, C, W_out, H_out))
        for n, c, w, h in np.ndindex(N, C, W_out, H_out):
            Z[n, c, w, h] = np.mean(A[n, c, w:w+self.kernel, h:h+self.kernel])
            
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        N, C, W_out, H_out = dLdZ.shape
        dLdA = np.zeros(self.A.shape)
        for n, c, w, h in np.ndindex(N, C, W_out, H_out):
            filters = 1/(self.kernel*self.kernel)
            dLdA[n, c, w:w+self.kernel, h:h+self.kernel] += filters * dLdZ[n, c, w, h]
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z_stride1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdZ_stride1 = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ_stride1)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)


    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        raise NotImplementedError

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        raise NotImplementedError
