# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    x = np.clip(x, -1e100, 1e100)

    #raise NotImplementedError('This function must be implemented by the student.')

    if derivative == False:
        return x
    else:
        return np.ones(x.shape)


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    #raise NotImplementedError('This function must be implemented by the student.')

    x = np.clip(x, -1e100, 1e100)

    # This should look something like 1 / (1 + np.exp(-x))
    sigmoid_val = np.zeros(np.shape(x))
    for row_index, row_val in enumerate(x):
        for col_index, col_val in enumerate(row_val):
            if col_val >= 0:
                sigmoid_val[row_index, col_index] = 1 / (1 + np.exp(-col_val))
            else:
                sigmoid_val[row_index, col_index] = np.exp(col_val) / (1 + np.exp(col_val))

    if derivative == False:
        return sigmoid_val
    else:
        return sigmoid_val * (1 - sigmoid_val)


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    x = np.clip(x, -1e100, 1e100)

    # raise NotImplementedError('This function must be implemented by the student.')
    tanh_val = np.tanh(x)

    if derivative == False:
        return tanh_val
    else:
        # The derivative of tanh(x) is 1-tanh^2(x) (it is also (1 / cosh^2(x)), but since we have tanh_val already, it is "cheaper" to compute it the original way)
        return 1 - (tanh_val**2)


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    x = np.clip(x, -1e100, 1e100)
    #raise NotImplementedError('This function must be implemented by the student.')

    # ReLu is either 0 or a positive increasing value depending on input x
    relu_val = np.zeros(x.shape)
    for row_index, row in enumerate(x):
        for col_index, col_val in enumerate(row):
            if derivative == False:
                relu_val[row_index, col_index] = max(0, col_val)
            else:
                if col_val > 0:
                    relu_val[row_index, col_index] = 1
                else:
                    relu_val[row_index, col_index] = 0

    return relu_val


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    #raise NotImplementedError('This function must be implemented by the student.')
    # Cross entropy is bounded by the sum of the one-hot encoding values times their log probabilities (from softmax, all then multiplied by minus 1)
    cross_entropy_val = np.zeros(y.shape)
    # Clips the values for the probabilities to prevent log(0) (undefined value).
    p = np.clip(p, 1e-100, 1e100)
    for i in range(0, len(y)):
        cross_entropy_val[i] = y[i] * np.log(p[i])

    return -1 * cross_entropy_val


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """

    #raise NotImplementedError('This function must be implemented by the student.')

    # This code assumes classes are given from 0 to n-1 classes.
    number_of_unique_labels = len(np.unique(y))
    one_hot_encoded_data = np.zeros((len(y), number_of_unique_labels))

    for sample_index, sample in enumerate(y):
        one_hot_encoded_data[sample_index, sample] = 1

    return one_hot_encoded_data