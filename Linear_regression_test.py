import numpy as np
import random
from sklearn.preprocessing import scale
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import sympy as sp

sp.init_printing()

def data_scale(X):
    return (X-np.mean(X,0))/np.std(X,0)

def predict_function(X,W):
    return X@W

def cost_function(X,W,Y):
    return (Y-X@W).T@(Y-X@W)/Y.shape[1]/2

def derivative_of_cost_function( NX,W,Y):
    return (W.T@X.T@X - Y.T@X)/Y.shape[1]

def init_w(n_features):
    w=np.zeros(n_features+1).reshape(n_features+1,1)
    return w

if __name__=='__main__':
    data=datasets.load_boston()
    X=data['data']

    n_features=len(data['feature_names'])
    n_samples=X.shape[0]
    Y = data['target'].reshape(n_samples,1)
    X = data_scale(X)
    X=np.hstack((X,np.ones(n_samples).reshape(n_samples,1)))



    learning_rate = 0.0001

    W=init_w(n_features)
    sns.set_style('whitegrid')

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ax.plot(Y)
    line, = ax.plot(X@ W, 'r-', linewidth=2)


    def update(i):
        global W
        y_pre = predict_function(X,W)
        delta_w = learning_rate * derivative_of_cost_function(X,W,Y)
        W = W - delta_w.T
        label = 'timestep{0}'.format(i)
        print(label)
        print(W)
        line.set_ydata((X @ W))
        ax.set_xlabel(label)
        return line, ax


    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
    plt.show()


