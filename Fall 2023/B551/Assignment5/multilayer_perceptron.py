# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import (
    identity,
    sigmoid,
    tanh,
    relu,
    softmax,
    cross_entropy,
    one_hot_encoding,
)


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(
        self,
        n_hidden=16,
        hidden_activation="sigmoid",
        n_iterations=1000,
        learning_rate=0.01,
    ):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {
            "identity": identity,
            "sigmoid": sigmoid,
            "tanh": tanh,
            "relu": relu,
        }

        # Check if the provided arguments are valid
        if (
            not isinstance(n_hidden, int)
            or hidden_activation not in activation_functions
            or not isinstance(n_iterations, int)
            or not isinstance(learning_rate, float)
        ):
            raise ValueError(
                "The provided class parameter arguments are not recognized."
            )

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None


    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self._y = one_hot_encoding(y)
        self.n_input_features = self._X.shape[1]
        self.n_output_features = self._y.shape[1]

        # print()
        # print(f'Number of input units: {self.n_input_features}')
        # print(f'Number of hidden units: {self.n_hidden}')
        # print(f'Number of output units: {self.n_output_features}')
        # print()

        # This is the same seed passed to MLP for scikit-learn
        np.random.seed(42)

        self._h_weights = np.random.normal(0, 1, (self.n_input_features, self.n_hidden))
        self._h_bias = np.random.normal(0, 1, (1, self.n_hidden))
        self._o_weights = np.random.normal(
            0, 1, (self.n_hidden, self.n_output_features)
        )
        self._o_bias = np.random.normal(0, 1, (1, self.n_output_features))

    def __forward_propagation(self, X):
        # input layer
        Z = X.dot(self._h_weights) + self._h_bias
        A = self.hidden_activation(Z)
        # output layer
        Z_o = A.dot(self._o_weights) + self._o_bias
        A_o = self._output_activation(Z_o)

        return Z, A, Z_o, A_o

    def __backward_propagation(self, Z, A, Z_o, A_o, X, y):
        m = y.size
        error_Z_o = A_o - one_hot_encoding(y)
        error_weights_o = (1 / m) * A.T.dot(error_Z_o)
        error_bias_o = (1 / m) * np.sum(error_Z_o)
        error_Z = error_Z_o.dot(self._o_weights.T) * self.hidden_activation(Z, True)
        error_weights = (1 / m) * X.T.dot(error_Z)
        error_bias = (1 / m) * np.sum(error_Z)

        return error_weights, error_bias, error_weights_o, error_bias_o

    def __update_weights(
        self, error_weights, error_bias, error_weights_o, error_bias_o
    ):
        self._h_weights -= self.learning_rate * error_weights
        self._h_bias -= self.learning_rate * error_bias
        self._o_weights -= self.learning_rate * error_weights_o
        self._o_bias -= self.learning_rate * error_bias_o


    def fit(self, X, y):
        self._initialize(X, y)

        for epoch in range(self.n_iterations):
            # Forward Propagation
            Z, A, Z_o, A_o = self.__forward_propagation(X)

            # Calculate Loss
            if (epoch % 20) == 0:
                self._loss_history.append(np.sum(self._loss_function(self._y, Z_o)))

            # Backpropagation
            (
                error_weights,
                error_bias,
                error_weights_o,
                error_bias_o,
            ) = self.__backward_propagation(Z, A, Z_o, A_o, X, y)

            # Update Weights
            self.__update_weights(
                error_weights, error_bias, error_weights_o, error_bias_o
            )

        self._loss_history.append(
            np.sum(
                self._loss_function(
                    self._y,
                    self._output_activation(
                        np.dot(
                            self.hidden_activation(
                                np.dot(self._X, self._h_weights) + self._h_bias
                            ),
                            self._o_weights,
                        )
                        + self._o_bias
                    ),
                )
            )
        )


    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        # raise NotImplementedError('This function must be implemented by the student.')

        return_predictions_one_hot_encoded = self._output_activation(
            np.matmul(
                self.hidden_activation(np.matmul(X, self._h_weights) + self._h_bias),
                self._o_weights,
            )
            + self._o_bias
        )


        # Convert the one-hot output from the network into a vector of class labels
        return_predictions = np.zeros(X.shape[0])
        for return_index, row in enumerate(return_predictions_one_hot_encoded):
            return_predictions[return_index] = np.argmax(row)

        # print(return_predictions)

        return return_predictions

