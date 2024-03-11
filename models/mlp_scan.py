# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # time steps(input length): 128
        # self.conv1 = input:24 dimen-vector; kernel:8 contiguous inputs/vector =>(8 * 24 neurons)
        #              stride: 4; output: 8 neurons 
        # self.conv2 = output: 16 neurons
        # self.conv3 = output: 4 neurons
        # outputs: (input length - kernel)/stride + 1 => (128 - 8)/4 + 1 = 31 
        # output length: 123 (4 neurons * 31 outputs)
        # all neurons use  ReLU except final output neurons
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]


    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        #    1. For each conv layer, have a look at the shape of its weight matrix
        #    2. Look at the shapes of w1, w2 and w3
        #    3. Figure out appropriate reshape and transpose operations

        # MLP weight: (input_neurons, out_channels) 
        # Conv weight: (out_channels (filters),  
        #               in_channels (input channels each filter spans),
        #               kernel_size (filter size))
        # 1) Transpose/switch MLP weights (cols) to align MLP's output neurons 
        #    w/ Conv weights's filters (rows)
        # 2) Reshape MLP weight to 3D structure
        # 3) Adjust dimensions (as necessary) to Conv weight structure (transpose)
        # w1: (8*24=192, 8) => (8, 192) => (8, 24, 8) => (8, 8, 24)
        # w2: (8, 16) => (16, 8) => (16, 8, 1) => (16, 1, 8)
        # w3: (16, 4) => (4, 16) => (4, 16, 1) => (4, 1, 16)

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = np.transpose(np.reshape(w1.T, (8, 8, 24)), axes=(0, 2, 1)) 
        self.conv2.conv1d_stride1.W = np.transpose(np.reshape(w2.T, (16, 1, 8)), axes=(0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(np.reshape(w3.T, (4, 1, 16)), axes=(0, 2, 1))


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = 
        # self.conv2 = 
        # self.conv3 = 
        # total neurons: 
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
       
        self.conv1.conv1d_stride1.W = np.transpose(np.reshape(w1[:,:2].T, (2, 8, 24))[:, :2, :], axes=(0,2,1))
        self.conv2.conv1d_stride1.W = np.transpose(np.reshape(w2[:,:8].T, (8, 4, 2))[:, :2, :], axes=(0,2,1))
        self.conv3.conv1d_stride1.W = np.transpose(np.reshape(w3[:,:].T, (4, 2, 8)), axes=(0,2,1))




        


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
