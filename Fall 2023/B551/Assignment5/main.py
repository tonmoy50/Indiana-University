# main.py: Driver program to test machine learning implementations on various datasets.
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff
#
# Feel free to edit this driver program file in order to test other datasets or change specific parameters. When
# grading, we will use our own driver program and test cases to test your machine learning implementations and functions
# made from scratch.

import sys

import numpy as np
import pandas as pd

from itertools import product

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from utils import euclidean_distance, manhattan_distance
from queue import PriorityQueue as PQ
from dataclasses import dataclass, field
from typing import Any 

# create class used to wrap data before pushing it into queue
# class adapted from: https://docs.python.org/3/library/queue.html?highlight=priorityqueue#queue.PriorityQueue
@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)
    def __getitem__(self):
         return self.item
    def __getpriority__(self):
         return self.priority
# end of adapted code

import warnings
from sklearn.exceptions import ConvergenceWarning

from k_nearest_neighbors import KNearestNeighbors
from multilayer_perceptron import MultilayerPerceptron


def test_knn():
    print("Testing K-Nearest Neighbors Classification...")
    param_dict = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["l1", "l2"],
    }
    params_list = [
        dict(zip(param_dict.keys(), v)) for v in product(*param_dict.values())
    ]

    params_len = len(params_list)

    knn_iris_results = pd.DataFrame(
        columns=[
            "  Model",
            "  # of Neighbors",
            "  Weight Function",
            "  Distance Metric",
            "  Accuracy Score",
        ]
    )

    for i, params in enumerate(params_list):
        sys.stdout.write("\r")
        sys.stdout.write(
            f" - Iris Dataset Progress:   [{'=' * int(60 * (i + 1) / params_len):60s}]"
            f" {int(100 * (i + 1) / params_len)}%"
        )
        sys.stdout.flush()

        sklearn_model = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            algorithm="brute",
            p=1 if params["metric"] == "l1" else 2,
        )
        sklearn_model.fit(X_train_i, y_train_i)
        y_pred = sklearn_model.predict(X_test_i)


        knn_iris_results.loc[len(knn_iris_results)] = ['Sklearn',
                                                       params['n_neighbors'],
                                                       params['weights'],
                                                       params['metric'],
                                                       round(accuracy_score(y_test_i, y_pred), 3)]

        my_model = KNearestNeighbors(params['n_neighbors'], params['weights'], params['metric'])
        train = my_model.fit(X_train_i, y_train_i)
        
        y_temp = my_model.predict(X_test_i)
        #print(len(y_temp[0]))
        myQ = PQ()
        myDir = {'0': 0, '1': 0, '2': 0}
        y_pred1 = []
        for i in range(len(y_temp)):
            myDir['0'] = 0
            myDir['1'] = 0
            myDir['2'] = 0
            myQ = PQ()
            for j in range(len(train)):
                if params['metric'] == 'l2':
                    myQ.put( PrioritizedItem(euclidean_distance(y_temp[i], train[j][0]), train[j][1]) )
                if params['metric'] == 'l1':
                    myQ.put( PrioritizedItem(manhattan_distance(y_temp[i], train[j][0]), train[j][1]) )
        
            for k in range(params['n_neighbors']):
                temp = myQ.get()
                if params['weights'] == 'uniform':
                    myDir[str(temp.item)] += 1
                if params['weights'] == 'distance':
                    if temp.priority == 0:
                        myDir[str(temp.item)] += np.inf
                    else:
                        myDir[str(temp.item)] += 1 / temp.priority
            Keymax = max(zip(myDir.values(), myDir.keys()))[1]
        #print(Keymax)   
            y_pred1.append(int(Keymax))
        
       # print(len(y_pred1))
        knn_iris_results.loc[len(knn_iris_results)] = ['My Model',
                                                       params['n_neighbors'],
                                                       params['weights'],
                                                       params['metric'],
                                                       round(accuracy_score(y_test_i, y_pred1), 3)]

        knn_iris_results.loc[len(knn_iris_results)] = ["", "", "", "", ""]

    print()

    knn_digits_results = pd.DataFrame(
        columns=[
            "  Model ",
            "  # of Neighbors",
            "  Weight Function",
            "  Distance Metric",
            "  Accuracy Score",
        ]
    )

    for i, params in enumerate(params_list):
        sys.stdout.write("\r")
        sys.stdout.write(
            f" - Digits Dataset Progress: [{'=' * int(60 * (i + 1) / params_len):60s}]"
            f" {int(100 * (i + 1) / params_len)}%"
        )
        sys.stdout.flush()

        sklearn_model = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            algorithm="brute",
            p=1 if params["metric"] == "l1" else 2,
        )
        sklearn_model.fit(X_train_d, y_train_d)
        y_pred = sklearn_model.predict(X_test_d)


        knn_digits_results.loc[len(knn_digits_results)] = ['Sklearn',
                                                           params['n_neighbors'],
                                                           params['weights'],
                                                           params['metric'],
                                                           round(accuracy_score(y_test_d, y_pred), 3)]

        my_model = KNearestNeighbors(params['n_neighbors'], params['weights'], params['metric'])
        train = my_model.fit(X_train_d, y_train_d)
        
        y_temp = my_model.predict(X_test_d)
        #print(len(y_temp[0]))
        myQ = PQ()
        myDir = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        y_pred1 = []
        for i in range(len(y_temp)):
            myDir['0'] = 0
            myDir['1'] = 0
            myDir['2'] = 0
            myDir['3'] = 0
            myDir['4'] = 0
            myDir['5'] = 0
            myDir['6'] = 0
            myDir['7'] = 0
            myDir['8'] = 0
            myDir['9'] = 0

            myQ = PQ()
            for j in range(len(train)):
                if params['metric'] == 'l2':
                    myQ.put( PrioritizedItem(euclidean_distance(y_temp[i], train[j][0]), train[j][1]) )
                if params['metric'] == 'l1':
                    myQ.put( PrioritizedItem(manhattan_distance(y_temp[i], train[j][0]), train[j][1]) )
        
            for k in range(params['n_neighbors']):
                temp = myQ.get()
                if params['weights'] == 'uniform':
                    myDir[str(temp.item)] += 1
                if params['weights'] == 'distance':
                    if temp.priority == 0:
                        myDir[str(temp.item)] += np.inf
                    else:
                        myDir[str(temp.item)] += 1 / temp.priority
            Keymax = max(zip(myDir.values(), myDir.keys()))[1]
        #print(Keymax)   
            y_pred1.append(int(Keymax))

        knn_digits_results.loc[len(knn_digits_results)] = ['My Model',
                                                          params['n_neighbors'],
                                                           params['weights'],
                                                           params['metric'],
                                                           round(accuracy_score(y_test_d, y_pred1), 3)]
        knn_digits_results.loc[len(knn_digits_results)] = ['', '', '', '', '']

    print('\n')
    print('Exporting KNN Results to HTML Files...\n')
    knn_iris_results.to_html('knn_iris_results.html')
    knn_digits_results.to_html('knn_digits_results.html')

    print("Done Testing K-Nearest Neighbors Classification!\n")


def test_mlp():
    print("Testing Multilayer Perceptron Classification...")
    param_dict = {
        "n_hidden": [8, 16, 32],
        "hidden_activation": ["identity", "sigmoid", "tanh", "relu"],
        "n_iterations": [500, 1000, 2000],
        "learning_rate": [0.1, 0.01, 0.001],
    }
    params_list = [
        dict(zip(param_dict.keys(), v)) for v in product(*param_dict.values())
    ]
    params_len = len(params_list)

    mlp_iris_results = pd.DataFrame(
        columns=[
            "  Model",
            "  # of Hidden Neurons",
            "  Hidden Activation Function",
            "  # of Iterations",
            "  Learning Rate",
            "  Accuracy Score",
        ]
    )

    for i, params in enumerate(params_list):
        sys.stdout.write("\r")
        sys.stdout.write(
            f" - Iris Dataset Progress:   [{'=' * int(60 * (i + 1) / params_len):60s}]"
            f" {int(100 * (i + 1) / params_len)}%"
        )
        sys.stdout.flush()

        param_logistic = params["hidden_activation"] == "sigmoid"
        sklearn_model = MLPClassifier(
            (params["n_hidden"],),
            params["hidden_activation"] if not param_logistic else "logistic",
            solver="sgd",
            alpha=0,
            batch_size=X_train_i.shape[0],
            learning_rate_init=params["learning_rate"],
            max_iter=params["n_iterations"],
            shuffle=False,
            random_state=42,
        )
        sklearn_model.fit(X_train_i, y_train_i)
        y_pred = sklearn_model.predict(X_test_i)

        mlp_iris_results.loc[len(mlp_iris_results)] = [
            "Sklearn",
            params["n_hidden"],
            params["hidden_activation"],
            params["n_iterations"],
            params["learning_rate"],
            round(accuracy_score(y_test_i, y_pred), 3),
        ]

        my_model = MultilayerPerceptron(
            params["n_hidden"],
            params["hidden_activation"],
            params["n_iterations"],
            params["learning_rate"],
        )
        my_model.fit(X_train_i, y_train_i)
        y_pred = my_model.predict(X_test_i)

        mlp_iris_results.loc[len(mlp_iris_results)] = [
            "My Model",
            params["n_hidden"],
            params["hidden_activation"],
            params["n_iterations"],
            params["learning_rate"],
            round(accuracy_score(y_test_i, y_pred), 3),
        ]

        mlp_iris_results.loc[len(mlp_iris_results)] = ["", "", "", "", "", ""]

    print()

    mlp_digits_results = pd.DataFrame(
        columns=[
            "  Model",
            "  # of Hidden Neurons",
            "  Hidden Activation Function",
            "  # of Iterations",
            "  Learning Rate",
            "  Accuracy Score",
        ]
    )

    for i, params in enumerate(params_list):
        sys.stdout.write("\r")
        sys.stdout.write(
            f" - Digits Dataset Progress: [{'=' * int(60 * (i + 1) / params_len):60s}]"
            f" {int(100 * (i + 1) / params_len)}%"
        )
        sys.stdout.flush()

        param_logistic = params["hidden_activation"] == "sigmoid"
        sklearn_model = MLPClassifier(
            (params["n_hidden"],),
            params["hidden_activation"] if not param_logistic else "logistic",
            solver="sgd",
            alpha=0,
            batch_size=X_train_i.shape[0],
            learning_rate_init=params["learning_rate"],
            max_iter=params["n_iterations"],
            shuffle=False,
            random_state=42,
        )
        sklearn_model.fit(X_train_d, y_train_d)
        y_pred = sklearn_model.predict(X_test_d)

        mlp_digits_results.loc[len(mlp_digits_results)] = [
            "Sklearn",
            params["n_hidden"],
            params["hidden_activation"],
            params["n_iterations"],
            params["learning_rate"],
            round(accuracy_score(y_test_d, y_pred), 3),
        ]

        my_model = MultilayerPerceptron(
            params["n_hidden"],
            params["hidden_activation"],
            params["n_iterations"],
            params["learning_rate"],
        )
        my_model.fit(X_train_d, y_train_d)
        y_pred = my_model.predict(X_test_d)

        mlp_digits_results.loc[len(mlp_digits_results)] = [
            "My Model",
            params["n_hidden"],
            params["hidden_activation"],
            params["n_iterations"],
            params["learning_rate"],
            round(accuracy_score(y_test_d, y_pred), 3),
        ]

        mlp_digits_results.loc[len(mlp_digits_results)] = ["", "", "", "", "", ""]

    print("\n")

    print("Exporting MLP Results to HTML Files...\n")
    mlp_iris_results.to_html("mlp_iris_results.html")
    mlp_digits_results.to_html("mlp_digits_results.html")

    print("Done Testing Multilayer Perceptron Classification!\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Incorrect number of arguments have been specified.")

    if sys.argv[1] not in ["knn", "mlp", "all"]:
        raise Exception(
            f"The command line argument {sys.argv[1]} is not allowed and must be either"
            f" 'knn', 'mlp', or 'all'."
        )

    np.random.seed(42)
    pd.set_option(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    )
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

    print("Loading the Iris and Digits Datasets...\n")
    data_i, data_d = load_iris(), load_digits()
    X_i, X_d = data_i.data, data_d.data
    y_i, y_d = data_i.target, data_d.target

    print("Splitting the Datasets into Train and Test Sets...\n")
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_i, y_i, test_size=0.3, random_state=42
    )
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_d, y_d, test_size=0.3, random_state=42
    )

    print("Standardizing the Train and Test Datasets...\n")
    sc_i = StandardScaler()
    sc_d = StandardScaler()

    X_train_i = sc_i.fit_transform(X_train_i)
    X_test_i = sc_i.transform(X_test_i)
    X_train_d = sc_d.fit_transform(X_train_d)
    X_test_d = sc_d.transform(X_test_d)

    if sys.argv[1] == "knn":
        test_knn()
    elif sys.argv[1] == "mlp":
        test_mlp()
    else:
        test_knn()
        test_mlp()

    print("Program Finished! Exiting the Program...")
