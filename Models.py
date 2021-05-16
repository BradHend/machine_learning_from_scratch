"""classes and methods for different model architectures
"""
#python packages
import numpy as np

# Machine Learning from Scratch packages
from Layers import FullyConnected
from utils.optimizers import *


class NeuralNet():
    """
    Linear stack of layers.
    """
    def __init__(self, layers=None):
        # Add any layers passed into constructor to the model
        if layers:
            for layer in layers:
                self.layers.append(layer)
        else:
            self.layers = []
        self.output = None
    
    def add_layer(self, layer_type=None,
                     input_shape=None,
                     output_shape=None,
                     activation=None,
                     dropout=1.,
                     lambd=0,):
        """Adds a Layer class to model
        """
        #only FullyConnected layer type supported right now
        if layer_type=="FullyConnected":
            layer = FullyConnected(input_shape=input_shape,
                                output_shape=output_shape,
                                activation=activation,
                                dropout=dropout,
                                lambd=lambd
                         )
        #append layer to model Class
        self.layers.append(layer)
        
    def model_forward(self,X,training=False):
        """ Perform forward evaluation of model on given data
        Inputs:
            X -- input data to be evaluated by model vector shape=(len(Wl_1), number of examples)
            training -- training flag, no layer dropout if True
        Outputs:
            predictions -- model prediction(s) for given data
        """
        layer_inputs = X
        for layer in self.layers:
            if training==False: #only use dropout when training
                layer.dropout=1.
            #loop over all layers, using the output of previous layer as input 
            layer.layer_forward(layer_inputs=layer_inputs)
            #update "layer_inputs" for next iteration
            layer_inputs = layer.outputs
        #predictions will be layer.output of the last layer
        predictions = layer_inputs
        return predictions
    
    def model_backprop(self,Y):
        """ Perform back-prop. of prediction error through model
        Inputs:
            Y -- truth "label" vector shape=(n_y, number of examples)
        Outputs:
            None -- updates Layer properties
        """
        # output_layer = self.layers[-1]
        dZ = self.compute_loss_grad(Y)
        #backprop output layer results through the network
        for layer in reversed(self.layers):
            #loop over all layers, using following layerdZ 
            layer.layer_backprop(dZ)
            #update "dZ" for next iteration, set to current layer's Activation gradient
            dZ = layer.dA
    
    def compute_cost(self,predictions,Y):
        """ compute "cost" for given predictions/truth
        Inputs:
            predictions -- model predictions vector shape=(n_y, number of examples)
            Y -- truth "label" vector shape=(n_y, number of examples)
        Outputs:
            cost - gradient of output layer's activation
        """
        m = Y.shape[1]
        # Compute loss from predictions and y.
        predictions = np.clip(predictions, 1e-13, 1 - 1e-13)
        if self.loss == 'binary-crossentropy':
            cost = np.multiply(-np.log(predictions),Y) + np.multiply(-np.log(1 - predictions), 1 - Y)
        elif self.loss == 'categorical-crossentropy':
            #Categorical Crossentropy
            cost =  np.sum(np.multiply(Y, -np.log(predictions)),axis=0,keepdims=False)
        else: 
            return None
        return cost
    
    def compute_loss_grad(self,Y):
        """    
        Inputs:
            Y -- truth "label" vector shape=(n_y, number of examples)
        Outputs:
            dZ - gradient of output layer's loss
        """
        output_layer = self.layers[-1]
        # outputs = output_layer.outputs
        predictions = np.clip(output_layer.outputs, 1e-13, 1 - 1e-13)
        if self.loss == 'binary-crossentropy':
            #gradient of sigmoid (for now)
            # print("outputs: ", output_layer.outputs)
            # print(1 - output_layer.outputs)
            dZ = - (np.divide(Y, predictions) - np.divide(1 - Y, 1 - predictions))
    
        elif self.loss == 'categorical-crossentropy':
            #gradient of softmax
            dZ = predictions - Y
        return dZ
                    
    def predict(self, X):
        predictions = self.model_forward(X,training=False)
        return predictions
    
    def train(self, X, Y,
              optimizer="gd",
              loss=None,
              learning_rate = 0.007,
              mini_batch_size = [],
              num_epochs = 100,
              print_cost=True):
        """    
        Inputs:
            X -- input data, of shape=(n_x, number of examples)
            Y -- truth "label" vector shape=(n_y, number of examples)
            loss -- loss function to use
            optimizer -- optimizer to use to update trainable params.
            learning_rate -- the learning rate, scalar.
            mini_batch_size -- the size of each dataset mini batch
            num_epochs -- number of epochs
            print_cost -- True to print the cost every 1000 epochs
    
        """
        
        self.loss = loss
        
        if print_cost:
            #print at every 1% of training completion, or at every epoch if num_epoch <= 100
            print_interval = np.max([1,int(0.01*num_epochs)])
            
        m = X.shape[1]                   # number of training examples
        if not mini_batch_size:
            mini_batch_size = m #make the mini-batch the entire dataset
    
        costs = []                       # to keep track of the cost
        accuracy_lst = []                # keep track of acc. for multi-class problems
        seed = 10
        
        # Initialize layers (weights & bias vectors)
        for layer in self.layers:
            layer.initialize_layer()
            if layer.dropout > 1.: #check that inputs make sense
                layer.dropout = 1.
            #if true, dropout was requested, override/ignore user's L2 reg. request (as of this commit)
            if layer.dropout < 1.:
                layer.lambd = 0
        
        # Initialize the optimizer
        if optimizer == "gd":
            pass # no initialization needed
        elif optimizer == "momentum":
            initialize_velocity(self.layers)
            beta = 0.90
        elif optimizer == "adam":
            t = 0 #counter required for Adam update
            #use values from the ADAM paper 
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-7
            learning_rate = 0.01
            initialize_adam(self.layers)
        
        # Optimization loop
        for i in range(num_epochs):
            
            # Define the random minibatches, change seed each time
            seed = seed + 1
            minibatches = make_sub_batches(X, Y, mini_batch_size, seed)
    
            #init cost summation variable    
            cost_total = 0.
            #init accuracy summation variable 
            training_correct = 0.
            
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
    
                # Forward prop
                predictions = self.model_forward(minibatch_X, training=True)
                
                # Compute cost (for printing) and add to the running total
                cost_total += np.nansum(self.compute_cost(predictions, minibatch_Y))

                #compute train set acc. for multi-class class. problems
                if (predictions.shape[0] > 1) | (self.loss == ('categorical-crossentropy')):
                    #compute number of examples correctly classified, assuming only one class can present right now
                    training_correct += np.sum(np.argmax(predictions,axis=0)==np.argmax(minibatch_Y,axis=0),keepdims=False)
    
                # Backprop
                self.model_backprop(Y=minibatch_Y)
                
                # Update weights/bias
                if optimizer == "gd":
                    update_layers_with_gradient_descent(self.layers, learning_rate)
                elif optimizer == "momentum":
                    update_parameters_with_momentum(self.layers, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    update_parameters_with_adam(self.layers, t, learning_rate, beta1, beta2,  epsilon)
                    
            #compute training stats. for this epoch
            cost_avg = cost_total / m
            if predictions.shape[0] > 1: #for multi-class class. problems show accuracy
                accuracy_percent = 100.*(training_correct/m)
                
            # Print the cost every epoch
            # if print_cost and i % print_interval == 0:
            if print_cost and i % 1 == 0:
                if predictions.shape[0] > 1: #for multi-class class. problems show accuracy
                    print("Cost after epoch %i: %f, Acc.: %f" %(i, cost_avg, accuracy_percent))
                    accuracy_lst.append(accuracy_percent)
                else:
                    print(("Cost after epoch %i: %f" %(i, cost_avg)))
                costs.append(cost_avg)
            
            #will need to implement better convergence detection..
            if self.loss == ('categorical-crossentropy'):
                pass
            elif cost_avg < 0.17:
                break