import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import to load MATLAB files
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_seq_items', None)


def computeSigmoid(z):
	return (1 / (1 + np.exp(-z)))

def computeCostFunction(theta, regParam, X, y):
	m = y.size
	regTerm = (regParam / (2*m))*np.sum(np.square(theta[1:]))
	h = computeSigmoid(X.dot(theta))
	
	J = ((-1)/m) * (np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + regTerm
	return (J[0])

def gradient(theta, regParam, X, y):
	m = y.size
	h = computeSigmoid(X.dot(theta.reshape(-1,1)))
	regTerm = (regParam / (m))*np.r_[[[0]],theta[1:].reshape(-1,1)]

	gradient = (1/m)*X.T.dot(h-y) + regTerm
	return (gradient.flatten())

def classify_oneVsRest(features, classes, nolabels, reg):
    initialTheta = np.zeros((X.shape[1],1))  # 401x1
    all_theta = np.zeros((nolabels, X.shape[1])) #10x401

    for c in np.arange(1, nolabels+1):
        res = minimize(computeCostFunction, initialTheta, args=(reg, features, (classes == c)*1), method=None,
                       jac=gradient, options={'maxiter':50})
        all_theta[c-1] = res.x
    return(all_theta)


def predict_OneVsRest(all_theta, features):
	probs = computeSigmoid(X.dot(all_theta.T))
	return (np.argmax(probs, axis=1) + 1)


def predict_NeuralNetwork(theta_1, theta_2, features):
	z2 = theta_1.dot(features.T)
	a2 = np.c_[np.ones((dataset['X'].shape[0], 1)), computeSigmoid(z2).T]
	
	z3 = a2.dot(theta_2.T)
	a3 = computeSigmoid(z3)
	return (np.argmax(a3, axis=1) + 1)


# returns dictionary with variable names as keys, and loaded matrices as values
dataset = loadmat("D:\\Computer Vision\\coursera\\assignment3\\ex3data1.mat")

# keys of dictionary
dataset.keys()

parameters = loadmat("D:\\Computer Vision\\coursera\\assignment3\\ex3weights.mat")
parameters.keys()

# getting variable y
y = dataset['y']

# example of np.c_
# np.c_[np.array([1,2,3]), np.array([4,5,6])]
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
X = np.c_[np.ones((dataset['X'].shape[0], 1)), dataset['X']]
# print('X: {}'.format(X.shape))
# print('y: {}'.format(y.shape))

theta1, theta2 = parameters['Theta1'], parameters['Theta2']

# print('theta1: {}'.format(theta1.shape))
# print('theta2: {}'.format(theta2.shape))

# checking the dataset by looking at random samples
sample = np.random.choice(X.shape[0], 20)
# result will be a 1-D array of length (transposed)
plt.imshow(X[sample, 1:].reshape(-1, 20).T)
plt.axis('off')
plt.savefig('myfig.png')

# prediction of classification with Logistic Regression
clf = LogisticRegression(C=10, penalty='l2', solver='liblinear')
clf.fit(X[:, 1:], y.ravel())

LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                    penalty='l2', random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

logistic_pred = clf.predict(X[:, 1:])
print('Training set accuracy by using Linear Regression: {} %'.format(np.mean(logistic_pred == y.ravel())*100))


# implementation of one-vs-rest classification
theta = classify_oneVsRest(X, y, 10, 0.1)
OnevsRest_pred = predict_OneVsRest(theta, X)
print('Training set accuracy by using One-vs-Rest Classification: {} %'.format(np.mean(OnevsRest_pred == y.ravel())*100))

# implementation of neural network classification
neuralNetwork_pred = predict_NeuralNetwork(theta1, theta2, X)
print('Training set accuracy by using Neural Network: {} %'.format(np.mean(neuralNetwork_pred == y.ravel())*100))




