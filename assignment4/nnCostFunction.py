# %NNCOSTFUNCTION Implements the neural network cost function for a two layer
# %neural network which performs classification
# %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
# %   X, y, lambda) computes the cost and gradient of the neural network. The
# %   parameters for the neural network are "unrolled" into the vector
# %   nn_params and need to be converted back into the weight matrices.
# %
# %   The returned parameter grad should be a "unrolled" vector of the
# %   partial derivatives of the neural network.
# %
#
import numpy as np

def computeSigmoid(z):
	return (1 / (1 + np.exp(-z)))

def computeSigmoidGradient(z):
	grad = computeSigmoid(z)
	grad = grad * (1 - grad)
	return grad

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, regParam):

	# % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	# % for our 2 layer neural network
	Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
	                    (hidden_layer_size, input_layer_size + 1), order='F')
	
	Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
	                    (num_labels, hidden_layer_size + 1), order='F')
	
	# Setup some useful variables
	m =  len(X)
	
	# You need to return the following variables correctly
	J = 0
	Theta1_grad = np.zeros(Theta1.shape)
	Theta2_grad = np.zeros(Theta2.shape)

	# bias unit from input layer to second layer
	X = np.column_stack((np.ones((m, 1)), X))  # = a1

	# compute second layer (a2) by using input layer
	a2 = computeSigmoid(np.dot(X, Theta1.T))
	
	# add bias unit to second layer (1 - hidden layer)
	a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

	# compute third layer (a3) by using a2 layer
	a3 = computeSigmoid(np.dot(a2, Theta2.T))
	
	# ------ Cost Function (without regularization)-------- #
	labels = y
	# set y to be matrix of size m x k (number of labels)
	y = np.zeros((m, num_labels))

	for i in range(m):
		y[i, labels[i] - 1] = 1
	
	# at this point, both a3 and y are m x k matrices, where m is the number of inputs
	# and k is the number of hypotheses. Given that the cost function is a sum
	# over m and k, loop over m and in each loop, sum over k by doing a sum over the row
	
	cost = 0
	for i in range(m):
		cost += np.sum(y[i] * np.log(a3[i]) + (1 - y[i]) * np.log(1 - a3[i]))

	J = -(1.0 / m) * cost
	
	# ------ Regularized Cost Function -------- #
	sumTheta1 = np.sum(np.sum((Theta1[:, 1:]**2)))
	sumTheta2 = np.sum(np.sum((Theta2[:, 1:]**2)))

	J += ((regParam/(2*m))*(sumTheta1+sumTheta2))
	
	bigDelta1 = 0
	bigDelta2 = 0
	
	# ------ For each training example ------#
	# Set the input layer's values (a(1)) to the t-th training example x(t).
	# Perform a feedforward pass
	for t in range(m):
		x = X[t]
		
		a2 = computeSigmoid(np.dot(x, Theta1.T))
		
		# add column of ones as bias unit from second layer to third layer
		a2 = np.concatenate((np.array([1]), a2))
		
		# calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
		a3 = computeSigmoid(np.dot(a2, Theta2.T))

		delta3 = np.zeros((num_labels))
		
		# indicates whether the current training example belongs
		#  to class k (yk = 1), or if it belongs to a different class (yk = 0).
		for k in range(num_labels):
			y_k = y[t, k]
			delta3[k] = a3[k] - y_k
		
		# bias element of theta2 will not be used here!!
		delta2 = (np.dot(Theta2[:, 1:].T, delta3).T) * computeSigmoidGradient(np.dot(x, Theta1.T))
		
		bigDelta1 += np.outer(delta2, x)
		bigDelta2 += np.outer(delta3, a2)
		
	Theta1_grad = bigDelta1 / m
	Theta2_grad = bigDelta2 / m
	

	# only regularize for j >= 1, so skip the first column
	Theta1_grad_unregularized = np.copy(Theta1_grad)
	Theta2_grad_unregularized = np.copy(Theta2_grad)
	
	Theta1_grad += (float(regParam) / m) * Theta1
	Theta2_grad += (float(regParam) / m) * Theta2
	Theta1_grad[:, 0] = Theta1_grad_unregularized[:, 0]
	Theta2_grad[:, 0] = Theta2_grad_unregularized[:, 0]
	
	
	# Unroll gradients
	grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))
	
	return J, grad
