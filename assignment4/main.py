import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

# import to load MATLAB files
from scipy.io import loadmat


import nnCostFunction as cost
import checkNNGradients as cnng
import displayData as dataplot
import predict as pred
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_seq_items', None)

def randomInitializeParam(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))

    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init) - epsilon_init
    return W



# returns dictionary with variable names as keys, and loaded matrices as values
dataset = loadmat("D:\\Computer Vision\\coursera\\assignment4\\ex4data1.mat")

# keys of dictionary
dataset.keys()

parameters = loadmat("D:\\Computer Vision\\coursera\\assignment4\\ex4weights.mat")
parameters.keys()

# getting variable y
y = dataset['y']

# example of np.c_
# np.c_[np.array([1,2,3]), np.array([4,5,6])]
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
X = dataset['X']
print('X: {}'.format(X.shape))
print('y: {}'.format(y.shape))

theta1, theta2 = parameters['Theta1'], parameters['Theta2']
params = np.concatenate((theta1.reshape(theta1.size, order='F'), theta2.reshape(theta2.size, order='F')))

print('theta1: {}'.format(theta1.shape))
print('theta2: {}'.format(theta2.shape))
print('param: {}'.format(params.shape))


# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10


cost2, _ = cost.nnCostFunction(params, input_layer_size, hidden_layer_size, num_labels, X, y, 1)

print('Result of cost function is ', cost2)


g = cost.computeSigmoidGradient( np.array([1, -0.5, 0, 0.5, 1]) )
print('Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1]:', g)

# randomly initialize parameters for neural networks
initial_Theta1 = randomInitializeParam(input_layer_size, hidden_layer_size)
initial_Theta2 = randomInitializeParam(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'), initial_Theta2.reshape(initial_Theta2.size, order='F')))

# checking gradients
cnng.checkNNGradients()

# ------ TRAINING NEURAL NETWORK ------ #
print('Training Neural Network...')

maxiter = 20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
results = minimize(cost.nnCostFunction, x0=params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, input_layer_size + 1), order='F')

Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                 (num_labels, hidden_layer_size + 1), order='F')

dataplot.displayData(Theta1[:, 1:])

# Accuracy prediction

pred = pred.predict(Theta1, Theta2, X)

print('Training Set Accuracy: {:f}'.format( ( np.mean(pred == y)*100) ) )
