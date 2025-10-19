import numpy as np
import os
from numpy.random import default_rng
from treeNode import TreeNode

def load_dataset(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data

def split_dataset(data, test_proportion, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(data))
    n_test = round(len(data) * test_proportion)
    n_train = len(data) - n_test
    data_train = data[shuffled_indices[:n_train]]
    data_test = data[shuffled_indices[n_train:]]
    return data_train, data_test

def H(dataset):
    y = dataset[:, -1]
    labels = np.unique(y)
    n = y.shape[0]

    res = 0
    for label in labels:
        p_k = np.sum(y == label) / n
        res -= p_k * np.log2(p_k)

    return res

def remainder(s_left, s_right):
    samples_left = s_left.shape[0]
    samples_right = s_right.shape[0]
    total_sample = samples_left + samples_right
    return (samples_left / (total_sample)) * H(s_left) + (samples_right / total_sample) * H(s_right)

def gain(s_all, s_left, s_right):
    return H(s_all) - remainder(s_left, s_right)

def findSplit(data, min_samples_leaf):
    x = data[:, :-1]
    y = data[:, -1]
    ROWS, COLS = x.shape

    best = {
        "feature": None, 
        "threshold": None,
        "gain": -np.inf,
        "left": None,
        "right": None
    }

    for c in range(COLS):
        x_i = x[:, c]
        order = np.argsort(x_i)
        x_sorted = x_i[order]

        distinct =  ~np.isclose(x_sorted[:-1], x_sorted[1:])
        if not np.any(distinct):
            continue
        idxs = np.flatnonzero(distinct)

        for i in idxs:
            thresh = (x_sorted[i] + x_sorted[i+1]) * 0.5
            left_mask = x[:, c] <= thresh
            right_mask = x[:, c] > thresh
            nL = int(left_mask.sum())
            nR = int(right_mask.sum())
            if nL < min_samples_leaf or nR < min_samples_leaf:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]
            g = gain(y, y_left, y_right)

            if g > best["gain"]:
                best.update({
                    "feature": c,
                    "threshold": thresh,
                    "gain": g,
                    "left": data[left_mask],
                    "right":data[right_mask]
                })

        if best["feature"] is None:
            best["gain"] = 0.0

    return best["left"], best["right"]


def decision_tree_learning(training_dataset, depth):
    x, y = training_dataset[:, :-1], training_dataset[:, -1]
    labels = np.unique(y)

    if len(labels) == 1:
        return labels[0], depth
    else:
        l_dataset, r_dataset = findSplit(training_dataset)
        node = TreeNode()
        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth = decision_tree_learning(l_dataset, depth+1)
        return node, max(l_depth, r_depth)

def _demo():
    # locate dataset relative to this script
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clean_dataset = os.path.join(base, 'wifi_db', 'clean_dataset.txt')
    data = load_dataset(clean_dataset)
    data_train, data_test = split_dataset(data, test_proportion=0.75)
    decision_tree_learning(data_train, depth=10)#depth placeholder

if __name__ == '__main__':
    _demo()