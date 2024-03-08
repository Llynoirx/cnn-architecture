import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        """
            upsampling_factor = k
        """
        self.upsampling_factor = upsampling_factor 

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width) (N, C, W_in)
        Return:
            Z (np.array): (batch_size, in_channels, output_width) (N, C, W_out)
        """

        N, C, W_in = A.shape
        W_out = (W_in - 1)*self.upsampling_factor + 1
        Z = np.zeros((N, C, W_out))

        idx = np.ndindex(N, C, W_in)
        for n, c, w in idx:
            Z[n, c, w*self.upsampling_factor] = A[n, c, w]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width) (N, C, W_out)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width) (N, C, W_in)
        """

        N, C, W_out = dLdZ.shape
        W_in = (W_out - 1)//self.upsampling_factor + 1
        dLdA = np.zeros((N, C, W_in))

        idx = np.ndindex(N, C, W_in)
        for n, c, w in idx:
            dLdA[n, c, w] = dLdZ[n, c, w*self.upsampling_factor]

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        N, C, self.W_in = A.shape
        W_out = (self.W_in - 1)//self.downsampling_factor + 1
        Z = np.zeros((N, C, W_out))

        idx = np.ndindex(N, C, W_out)
        for n, c, w in idx:
            Z[n, c, w] = A[n, c, w*self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        N, C, W_out = dLdZ.shape
        dLdA = np.zeros((N, C, self.W_in))

        idx = np.ndindex(N, C, W_out)
        for n, c, w in idx:
            dLdA[n, c, w*self.downsampling_factor] = dLdZ[n, c, w]

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        N, C, H_in, W_in = A.shape
        H_out = (H_in - 1)*self.upsampling_factor + 1
        W_out = (W_in - 1)*self.upsampling_factor + 1
        Z = np.zeros((N, C, H_out, W_out))

        idx = np.ndindex(N, C, H_in, W_in)
        for n, c, h, w in idx:
            Z[n, c, h*self.upsampling_factor, w*self.upsampling_factor] = A[n, c, h, w]

        return Z
    

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        N, C, H_out, W_out = dLdZ.shape
        H_in = (H_out - 1)//self.upsampling_factor + 1
        W_in = (W_out - 1)//self.upsampling_factor + 1
        dLdA = np.zeros((N, C, H_in, W_in))

        idx = np.ndindex(N, C, H_in, W_in)
        for n, c, h, w in idx:
            dLdA[n, c, h, w] = dLdZ[n, c, h*self.upsampling_factor, w*self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        N, C, self.H_in, self.W_in = A.shape
        H_out = (self.H_in - 1)//self.downsampling_factor + 1
        W_out = (self.W_in - 1)//self.downsampling_factor + 1
        Z = np.zeros((N, C, H_out, W_out))

        idx = np.ndindex(N, C, H_out, W_out)
        for n, c, h, w in idx:
            Z[n, c, h, w] = A[n, c, h*self.downsampling_factor, w*self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        N, C, H_out, W_out = dLdZ.shape
        dLdA = np.zeros((N, C, self.H_in, self.W_in))

        idx = np.ndindex(N, C, H_out, W_out)
        for n, c, h, w in idx:
            dLdA[n, c, h*self.downsampling_factor, w*self.downsampling_factor] = dLdZ[n, c, h, w]

        return dLdA
