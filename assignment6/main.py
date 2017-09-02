import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import svmTrain as svmt
import dataset3Params as dp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_seq_items', None)

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples
    y = y.flatten()
    pos = y==1
    neg = y==0

    # Plot Examples
    plt.plot(X[:,0][pos], X[:,1][pos], "k+", markersize=10)
    plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=10)
    plt.show()
    plt.interactive(False)

def visualizeBoundary(X, y, model, varargin=0):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    #   boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
       this_X = np.column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = model.predict(svmt.gaussianKernelGramMatrix(this_X, X))

    # Plot the SVM boundary
    plt.contour(X1, X2, vals)
    plt.show()



def visualizeBoundaryLinear(X, y, model):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    #SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    #   learned by the SVM and overlays the data on it

    # plot decision boundary
    # right assignments from http://stackoverflow.com/a/22356267/583834
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yp = - (w[0] * xp + b) / w[1]

    plt.plot(xp, yp, 'b-')

    # plot training data
    plotData(X, y)


dataset1 = loadmat('D:\\Computer Vision\\coursera\\assignment6\\ex6data1.mat')
dataset2 = loadmat('D:\\Computer Vision\\coursera\\assignment6\\ex6data2.mat')
dataset3 = loadmat('D:\\Computer Vision\\coursera\\assignment6\\ex6data3.mat')

X = dataset1["X"]
y = dataset1["y"]

# plotData(X, y)

print('------ Training Linear SVM ----- ')

# change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 2
model = svmt.svmTrain(X, y, C, "linear", 1e-3, 20)
plt.close()
visualizeBoundaryLinear(X, y, model)


# print('------ Training SVM with RBF Kernel with Dataset 2 ------');
#
# X = dataset2["X"]
# y = dataset2["y"]
#
# # SVM Parameters
# C = 1
# sigma = 0.1
#
# model = svmt.svmTrain(X, y, C, "gaussian")
#
# plt.close()
# visualizeBoundary(X, y, model)

print('------ Training SVM with RBF Kernel with Dataset 3 ------');

X = dataset3["X"]
y = dataset3["y"]
Xval = dataset3["Xval"]
yval = dataset3["yval"]

# Try different SVM Parameters here
C, sigma = dp.dataset3Params(X, y, Xval, yval)

# train model on training corpus with current sigma and C
model = svmt.svmTrain(X, y, C, "gaussian", sigma=sigma)
plt.close()
visualizeBoundary(X, y, model)

