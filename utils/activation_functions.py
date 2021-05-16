#Collection of activation functions and their backprop. representation
import numpy as np

def sigmoid(Z):
    """
    computes sigmoid activation
    Inputs:
        Z -- numpy array (any shape)
    
    Outputs:
        A -- result of sigmoid(z), (same shape as Z)
        Z -- return Z, for ease of use during backprop.
    """
    A = 1/(1+np.exp(-Z))
    return A, Z


def relu(Z):
    """
    computes RELU activation
    Inputs:
        Z -- Output of the linear layer, of any shape

    Outputs:
        A -- result of RELU(z), (same shape as Z)
        Z -- return Z, for ease of use during backprop.
    """
    A = np.maximum(0,Z)
    #check shape
    assert(A.shape == Z.shape)
    return A, Z

def softmax(Z):
    """
    computes softmax activation
    Inputs:
        Z -- Output of the linear layer, of any shape

    Outputs:
        A -- result of SOFTMAX(z), (same shape as Z)
        Z -- return Z, for ease of use during backprop.
    """

    A = np.exp(Z - np.max(Z,axis=0, keepdims=True))
    A = A / np.sum(A, axis=0, keepdims=True)
    #check shape
    # assert(A.shape == Z.shape)
    return A, Z

def relu_backprop(dA, Z):
    """
    compute back prop. through RELU unit

    Inputs:
        dA -- post-activation gradient, of any shape
        Z -- orig. Z used to compute sigmoid activation

    Outputs:
        dZ -- Gradient of cost fcn. (w.r.t Z)
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    #check shape
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backprop(dA, Z):
    """
    compute back prop. through sigmoid unit
    Inputs:
        dA -- post-activation gradient, of any shape
        Z -- orig. Z used to compute sigmoid activation

    Outputs:
        dZ -- Gradient of cost fcn. w.r.t. Z
    """    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    #check shape    
    assert (dZ.shape == Z.shape)
    return dZ

def softmax_backprop(dA, Z):
    """
    compute back prop. through SOFTMAX unit

    Inputs:
        dA -- post-activation gradient, of any shape
        Z -- orig. Z used to compute sigmoid activation

    Outputs:
        dZ -- Gradient of cost fcn. (w.r.t Z)
    """
    dZ = np.array(dA, copy=True)
    # for i in range(len(Z)):
    #     for j in range(len(Z)):
    #         if i == j:
    #             dZ[i] = Z[i] * (1-Z[i])
    #         else: 
    #              dZ[i] = -Z[i]*Z[j]
    #check shape
    assert (dZ.shape == Z.shape)
    return dZ