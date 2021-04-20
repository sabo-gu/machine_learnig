import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import sympy as sp

sp.init_printing()

def data_scale(X):
    return (X-np.mean(X,0))/np.std(X,0)

def sigmod_function(X,W):
    return 1/(1+np.exp(-X@W))

def cost_function(X,W,Y):
    return -Y.T@np.log(sigmod_function(X,W))+(1-Y).T@np.log(1-sigmod_function(X,W))

def derivative_of_cost_function(X,W,Y):
    return -(Y-sigmod_function(X,W)).T@X

def init_w(n_features):
    w=np.zeros(n_features+1).reshape(n_features+1,1)
    return w

if __name__ == '__main__':
    data = datasets.make_blobs(centers=2)
    X = data[0]

    n_features = X.shape[1]
    n_samples = X.shape[0]
    Y = data[1].reshape(n_samples, 1)
    X = data_scale(X)
    X = np.hstack((X, np.ones(n_samples).reshape(n_samples, 1)))

    learning_rate = 0.001

    W = init_w(n_features)
    sns.set_style('whitegrid')

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ax.plot(Y)
    line, = ax.plot(sigmod_function(X, W), 'r-', linewidth=2)


    def update(i):
        global W
        y_pre = sigmod_function(X, W)
        delta_w = learning_rate * derivative_of_cost_function(X, W, Y)
        W = W - delta_w.T
        label = 'timestep{0}'.format(i)
        print(label)
        print(W)
        line.set_ydata(sigmod_function(X, W))
        ax.set_xlabel(label)
        return line, ax


    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
    plt.show()

