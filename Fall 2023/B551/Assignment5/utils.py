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
    mySum = 0
    for i in range(len(x1)):
        mySum += np.power(x1[i] - x2[i], 2)

    return np.linalg.norm(x1 - x2)



def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    mySum = 0
    for i in range(len(x1)):
        mySum += np.abs(x1[i] - x2[i])

    return np.sum(np.abs(x1 - x2))



def identity(x, derivative=False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    return np.ones(x.shape) if derivative else x


def sigmoid(x, derivative=False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))



def tanh(x, derivative=False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    return (1 - np.tanh(x) ** 2) if derivative else np.tanh(x)


def relu(x, derivative=False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    return np.where(x > 0, 1, 0) if derivative else np.where(x > 0, x, 0)



def softmax(x, derivative=False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis=1, keepdims=True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis=1, keepdims=True)))
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
    y_pred = softmax(p)
    loss = 0
    for i in range(len(y_pred)):
        loss += -1 * y[i] * np.log(y_pred[i])

    return loss



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

    unique_classes, encoded_y = np.unique(y, return_inverse=True)
    one_hot_matrix = np.eye(len(unique_classes))

    return one_hot_matrix[encoded_y]

