#python packages
import math
import numpy as np

def make_sub_batches(X, Y, batch_size = 64, seed = 0):
    """
    Creates a list randomized batches
    Inputs:
        X -- input dataset, shape=(input size, number of examples)
        Y -- truth label vect., shape=(1, number of examples)
        batch_size -- size of the mini-batches, integer
    
    Outputs:
        mini_batches -- python list of sub-batches (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    num_classes = Y.shape[0]
    
    #shuffle dataset
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((num_classes,m))

    #partition dataset
    num_complete_batches  = math.floor(m/batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_batches):
        mini_batch_X = shuffled_X[:, k*batch_size  : (k+1)*batch_size ]
        mini_batch_Y = shuffled_Y[:,k*batch_size  : (k+1)*batch_size ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # final batch (last mini-batch < mini_batch_size)
    if m % batch_size  != 0:
        mini_batch_X = shuffled_X[:, num_complete_batches*batch_size  : -1]
        mini_batch_Y = shuffled_Y[:, num_complete_batches*batch_size  : -1]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def update_layers_with_gradient_descent(layers, learning_rate):
    """
    Update training objectives (W,b) using Gradient Descent
    Inputs:
        layers - list of layer objects to update
        learning_rate -- the learning rate (float)
    
    Outputs:
        None (layer object(s) with updated weights/biases)
    """
    # apply GD
    for layer in layers:
        # update training parameters
        layer.set_weights(layer.weights - learning_rate*layer.dW)
        layer.set_bias(layer.bias - learning_rate*layer.db)


def initialize_velocity(layers):
    """
    Initialize velocity python dict
    Inputs:
        layers - list of layer objects to update
    
    Outputs:
        None (layer object(s) with self.v_dW, self.v_db initialized)
    """
    # Initialize velocity
    for layer in layers:
        layer.v_dW = np.zeros(layer.weights.shape)
        layer.v_db = np.zeros(layer.bias.shape)


def update_parameters_with_momentum(layers, beta=0.90, learning_rate=0.005):
    """
    Update training parameters with Momentum
    Inputs:
        layers -- list of layer objects (with self.v initialized)
        beta -- momentum hyperparameter (float)
        learning_rate -- learning rate ((float))
    
    Outputs:
        None (layer object(s) with updated weights/biases)
    """   
    # Perform Momentum update for each parameter
    for layer in layers:
        # compute velocities
        layer.v_dW = beta*layer.v_dW + (1.-beta)*layer.dW
        layer.v_db = beta*layer.v_db + (1.-beta)*layer.db
        # update training parameters
        layer.weights = layer.weights - learning_rate*layer.v_dW
        layer.bias = layer.bias - learning_rate*layer.v_db


def initialize_adam(layers):
    """
    Initialize velocity python dict
    Inputs:
        layers - list of layer objects to update
    
    Outputs:
        None (layer object(s) with self.v_dW, self.s_dW, self.v_db, self.s_db initialized)
            v -- exponentially weighted average of the gradient
            s -- exponentially weighted average of the squared gradient
    """
    for layer in layers:
        layer.v_dW = np.zeros(layer.weights.shape)
        layer.v_db = np.zeros(layer.bias.shape)
        
        layer.s_dW = np.zeros(layer.weights.shape)
        layer.s_db = np.zeros(layer.bias.shape)


def update_parameters_with_adam(layers, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    Inputs:
        layers -- list of layer objects (with adam parameters as properties)
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
        None (layer object(s) with updated weights/biases)
               - v -- Adam variable, moving average of the first gradient
               - s -- Adam variable, moving average of the squared gradient
    """
    # Perform Adam update on all layers
    for layer in layers:
        # Moving average of the gradients
        layer.v_dW = beta1*layer.v_dW + (1.-beta1)*layer.dW
        layer.v_db = beta1*layer.v_db + (1.-beta1)*layer.db
        
        # Compute bias-corrected first moment estimate
        v_corrected_dW = layer.v_dW/(1-(beta1**2))
        v_corrected_db = layer.v_db/(1-(beta1**2))
        
        # Moving average of the squared gradients
        layer.s_dW = beta2*layer.s_dW + (1.-beta2)*np.square(layer.dW)
        layer.s_db = beta2*layer.s_db + (1.-beta2)*np.square(layer.db)
        # Compute bias-corrected second raw moment estimate
        s_corrected_dW = layer.s_dW/(1-np.power(beta2,t))
        s_corrected_db = layer.s_db/(1-np.power(beta2,t))
        
        # update training parameters
        layer.weights = layer.weights - learning_rate*(v_corrected_dW/(np.sqrt(s_corrected_dW)+epsilon))
        layer.bias = layer.bias - learning_rate*(v_corrected_db/(np.sqrt(s_corrected_db)+epsilon))