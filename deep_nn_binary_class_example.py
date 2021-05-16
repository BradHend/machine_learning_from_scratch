#example of training & evaluating Deep Neural Network (N-layer deep)
#python packages
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

#ML_from_scratch packages
from Models import NeuralNet

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title("Learned Decision Boundary")
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def load_moons_dataset(seed=2021):
    np.random.seed(seed)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    # fig = plt.figure()
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y

def load_petal_dataset(seed=2021):
    np.random.seed(seed)
    m = 400 # number of examples
    pnts_per_class = int(m/2) # equal number of points per 0/1 class
    X = np.zeros((m,2)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0/red, 1/blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(pnts_per_class*j,pnts_per_class*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,pnts_per_class) + np.random.randn(pnts_per_class)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(pnts_per_class)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    X = X.T
    Y = Y.T
    return X, Y

def predict_class(model, X):
    # Predict using forward propagation and a classification threshold of 0.5
    y_hat = model.predict(X)
    predictions = (y_hat > 0.5)
    return predictions




#load example data
# train_X, train_Y = load_moons_dataset()
train_X, train_Y = load_petal_dataset(seed=20210306)

#build deep neural net model
model = NeuralNet()
model.add_layer(layer_type="FullyConnected", input_shape=2, output_shape=5, activation="relu", dropout=0.9)
model.add_layer(layer_type="FullyConnected", input_shape=5, output_shape=3, activation="relu", dropout=0.8)
model.add_layer(layer_type="FullyConnected", input_shape=3, output_shape=1, activation="sigmoid")
#train model
model.train(train_X, train_Y, optimizer="adam", loss='binary-crossentropy', learning_rate = 0.007,
            mini_batch_size = 32, num_epochs = 1000, print_cost=True)


plot_decision_boundary(lambda x: predict_class(model, x.T), train_X, train_Y)