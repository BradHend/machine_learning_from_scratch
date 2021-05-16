"""Class for managing different Layer types
"""
import numpy as np

# Machine Learning from Scratch packages
from utils import activation_functions


# class Conv2D():

# class Pooling():
    

class FullyConnected():
    """fully-connected (Dense) layer type
    """
    def __init__(self, 
                 input_shape=None,
                 output_shape=None,
                 activation=None,
                 dropout=1.,
                 lambd=0,
                 ):
        self.input_shape = int(input_shape)
        self.output_shape = int(output_shape)
        self.activation = activation
        self.dropout = dropout
        self.lambd = lambd
        self.weights = None
        self.bias = None
        self.dZ = None
        self.dA = None
        self.db = None
        self.dW = None
        self.inputs = None
        self.outputs = None

    
    def set_weights(self,weights):
        self.weights = weights
    
    def set_bias(self,bias):
        self.bias = bias
    
    def initialize_layer(self):
        self.set_weights(weights=np.random.randn(self.output_shape, self.input_shape) * np.sqrt(2 / self.input_shape))
        self.set_bias(bias=np.zeros((self.output_shape, 1)))
        
    def apply_activation(self):
        #apply activation function to result of linear feed-forward
        if self.activation == "sigmoid":
            outputs, _ = activation_functions.sigmoid(self.Z)
        elif self.activation == "relu":
            outputs, _ = activation_functions.relu(self.Z)
        elif self.activation == "softmax":
            outputs, _ = activation_functions.softmax(self.Z)
        self.outputs = outputs
        
    def layer_forward(self,layer_inputs):
        self.inputs = layer_inputs
        #linear portion of feed-forward
        self.Z = np.dot(self.weights, self.inputs) + self.bias
        #activation of linear 
        self.apply_activation()
        #check if using dropout, zero-out neurons if so
        if self.dropout < 1.:  
            drop_msk = np.random.rand(self.weights.shape[0], self.inputs.shape[-1])
            self.drop_msk = (drop_msk < self.dropout).astype(int)
            self.outputs *= self.drop_msk #zero out neurons according to msk
            self.outputs /= self.dropout #rescale neurons output based on dropout
        

    def linear_backprop(self):
        #compute backprop through linear portion of feed-forward
        m = self.outputs.shape[-1]
        
        self.dA = np.dot(self.weights.T,self.dZ)
        
        #compute derivative of weights
        dW = 1./m * np.dot(self.dZ,self.inputs.T)
        
        #check if using L2 regularization
        if self.lambd != 0:
            dW += ((self.lambd/m)*self.weights)
        self.dW = dW
        
        #compute derivative of bias term
        self.db = 1./m * np.sum(self.dZ, axis = -1, keepdims = True)

    def layer_backprop(self, dA):   
        #apply dropout before taking gradient
        if self.dropout < 1.:
            dA *= self.drop_msk #apply drop out
            dA /= self.dropout  #rescale

        #compute gradient of activation w.r.t. output
        if self.activation == "relu":
            dZ = activation_functions.relu_backprop(dA, self.Z)            
        elif self.activation == "sigmoid":
            dZ = activation_functions.sigmoid_backprop(dA, self.Z)
        elif self.activation == 'softmax':
            # dZ = softmax_backprop(dA, activation_cache)
            dZ = np.array(dA, copy=True)
        self.dZ = dZ
        
        #compute backprop through linear portion of feed-forward
        self.linear_backprop()