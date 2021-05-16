#example of training & evaluating Deep Neural Network (N-layer deep)
#   for multi-class classification using MNIST

#python packages
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
#ML_from_scratch packages
from Models import NeuralNet


def to_one_hot_array(labels,num_labels):
    #assumed input shape=(N,)
    one_hot = np.zeros(shape=(labels.shape[0],num_labels))
    for i in range(len(labels)):
        one_hot[i][int(labels[i])] = 1
    return one_hot

def load_MNIST(train_fraction, seed=2021):
    np.random.seed(seed)
    #load MNIST dataset using sklearn utilities
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    #random sort the dataset
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    #reshape images to 784xN
    X = X.reshape((X.shape[0], -1))
    #create one-hot arrays for multi-class classification
    num_labels = 10
    y_one_hot = to_one_hot_array(y,num_labels)
    #normalize images to 0.-1. from 0-255
    X_norm = (X/255.).T
    y_one_hot_T = y_one_hot.T
    
    #split dataset into train/test sets
    num_examples = X.shape[0]
    X_train = X_norm[:,0:int(train_fraction*num_examples)]
    X_test = X_norm[:,int(train_fraction*num_examples)::]
    y_train = y_one_hot_T[:,0:int(train_fraction*num_examples)]
    y_test = y_one_hot_T[:,int(train_fraction*num_examples)::]
    return (X_train, y_train, X_test, y_test)


def predict_class(model, X):
    # Predict using forward propagation and a classification threshold of 0.5
    y_hat = model.predict(X)
    # predictions = (y_hat > 0.5)
    predictions = y_hat
    return predictions





#load MNIST data
num_labels = 10
train_fraction = 0.7 #fraction of dataset to use for training
X_train, y_train, X_test, y_test = load_MNIST(train_fraction,seed=20210306)

#build deep neural net model
model = NeuralNet()
model.add_layer(layer_type="FullyConnected", input_shape=X_train.shape[0], output_shape=256, activation="relu", dropout=0.9)
model.add_layer(layer_type="FullyConnected", input_shape=256, output_shape=128, activation="relu", dropout=0.8)
model.add_layer(layer_type="FullyConnected", input_shape=128, output_shape=64, activation="relu")
model.add_layer(layer_type="FullyConnected", input_shape=64, output_shape=num_labels, activation="softmax")
#train model
model.train(X_train, y_train, optimizer="adam", loss='categorical-crossentropy', learning_rate = 0.007,
            mini_batch_size = 1024, num_epochs = 10, print_cost=True)


#let's look at a example image from the test dataset
test_im = X_test[:,80,None]
plt.imshow(test_im.reshape(28,28),cmap='gray')
y_pred = predict_class(model,test_im)
print("Model says example image is Class: %i, with Confidence: %f" %(np.argmax(y_pred),y_pred[np.argmax(y_pred)]))

#evaluate the testing dataset
m_test = y_test.shape[-1]
test_predicitons = predict_class(model,X_test)
num_test_correct = np.sum(np.argmax(test_predicitons,axis=0)==np.argmax(y_test,axis=0),keepdims=False)
test_acc = num_test_correct / m_test
print("Test set Acc. is: ", test_acc)
