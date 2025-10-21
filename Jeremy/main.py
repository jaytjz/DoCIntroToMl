import numpy as np
import os
from numpy.random import default_rng
from decisionTree import DecisionTree

def split_dataset(data, test_proportion, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(data))
    n_test = round(len(data) * test_proportion)
    n_train = len(data) - n_test
    data_train = data[shuffled_indices[:n_train]]
    data_test = data[shuffled_indices[n_train:]]
    return data_train, data_test

def compute_accuracy(y_gold, y_prediction):
    assert len(y_gold) == len(y_prediction)

    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clean_dataset = os.path.join(base, 'wifi_db', 'clean_dataset.txt')
    data = np.loadtxt(clean_dataset)
    data_train, data_test = split_dataset(data, test_proportion=0.75)
    decisionTree = DecisionTree(data_train)
    predicted = decisionTree.predict(data_test[:, :-1])
    print("clean accuracy", compute_accuracy(data_test[:, -1], predicted))
    #decisionTree.print_tree()

if __name__ == '__main__':
    main()