import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def compute_cost_logistic(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = -1*np.matrix(y)
    first = np.multiply(y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 + y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def grad_logistic(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

def predict_logistic(theta,X):
    probability = sigmoid(X@theta.T)
    response = np.round(probability)
    return response

def accuracy(pred, y):
    return (1-np.sum(np.abs(y-pred))/y.shape[0])

def plot_logistic(theta):
    plot_x = np.arange(100)
    plot_y = (-1./theta[2,0])*(theta[0,0] + theta[1,0]*plot_x)

    plt.plot(plot_x, plot_y , '-', label='Decision Boundary', linewidth=2, markersize=12, color='#FF0000')


    