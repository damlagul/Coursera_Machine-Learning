from sklearn import svm
import numpy as np


def gaussianKernel(x1, x2, sigma=0.1):
    # RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim
    
    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()
    
    # You need to return the following variables correctly.
    sim = 0
    
    sim = np.exp(- np.sum(np.power((x1 - x2), 2)) / float(2 * (sigma ** 2)))
    return sim


def gaussianKernelGramMatrix(X1, X2, K_function=gaussianKernel, sigma=0.1):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2, sigma)
    return gram_matrix


def svmTrain(X, y, C, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    """Trains an SVM classifier"""
    y = y.flatten() # prevents warning

    # alternative to emulate mapping of 0 -> -1 in svmTrain.m
    #  but results are identical without it
    # also need to cast from unsigned int to regular int
    # otherwise, contour() in visualizeBoundary.py doesn't work as expected
    # y = y.astype("int32")
    # y[y==0] = -1

    if kernelFunction == "gaussian":
        clf = svm.SVC(C=C, kernel="precomputed", tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(gaussianKernelGramMatrix(X, X, sigma=sigma), y)

    else:  # works with "linear", "rbf"
        clf = svm.SVC(C=C, kernel=kernelFunction, tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(X, y)
