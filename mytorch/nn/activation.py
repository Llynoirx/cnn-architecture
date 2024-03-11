import numpy as np

# Copy your Linear class from HW1P1 here
# NOTE: HW1P1 HAS dLdA PASSED INTO BACKWARD WHILE THIS DOESNT
# RETURN dAdZ INSTEAD OF dLdZ; dLdZ=dLdA*dAdZ => dAdZ=dLdZ/dLdA
# ONLY NEED IDENTITY, SIGMOID, TANH, RELU (NO GELU, SOFTMAX)

import numpy as np
import scipy
from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z
        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")
        return dAdZ


class Sigmoid:
  
    def forward(self, Z):

        self.A = 1/(1 + np.exp(-Z)) 
        return self.A

    def backward(self):

        dAdZ = (self.A - self.A*self.A) 
        return dAdZ



class Tanh:

    def forward(self, Z):

        # self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
        self.A = np.tanh(Z) # to prevent overflow
        return self.A

    def backward(self):
        """
        Hint: tanh'(x) = 1 - tanh^2(x)
        dLdZ = dLdA * dAdZ; A = tanh(x)
        dLdZ = dLdA * (1-A^2)
        """

        dAdZ = (1 - self.A*self.A) 
        return dAdZ


class ReLU:

    def forward(self, Z):
        """
        If Z>0, then Z else 0 => A = max(0, Z)
        https://numpy.org/doc/stable/reference/generated/numpy.amax.html
        https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        https://stackoverflow.com/questions/33569668/numpy-max-vs-amax-vs-maximum
        """

        self.A = np.maximum(0, Z)
        return self.A

    def backward(self):
        """
        if dLdA > 0, then 1; else if dLdA <= 0, then 0 
        https://numpy.org/doc/stable/reference/generated/numpy.where.html
        """

        dAdZ =  np.where(self.A>0, 1, 0)
        return dAdZ
    

