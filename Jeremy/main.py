import numpy as np
import os
from numpy.random import default_rng
from treeNode import TreeNode

def load_dataset(file_path):
    data = np.loadtxt(file_path)
    return data

def split_dataset(data, test_proportion, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(data))
    n_test = round(len(data) * test_proportion)
    n_train = len(data) - n_test
    data_train = data[shuffled_indices[:n_train]]
    data_test = data[shuffled_indices[n_train:]]
    return data_train, data_test

def H(y):
    _, counts = np.unique(y, return_counts=True)
    p_k = counts / counts.sum()
    return -np.sum(p_k * np.log2(p_k))

def remainder(s_left, s_right):
    samples_left = s_left.shape[0]
    samples_right = s_right.shape[0]
    total_sample = samples_left + samples_right
    return (samples_left / (total_sample)) * H(s_left) + (samples_right / total_sample) * H(s_right)

def gain(s_all, s_left, s_right):
    return H(s_all) - remainder(s_left, s_right)

def findSplit(data, min_samples_leaf=1):
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

    return best


def decision_tree_learning(training_dataset, depth):
    x, y = training_dataset[:, :-1], training_dataset[:, -1]
    labels = np.unique(y)

    if labels.size == 1:
        return TreeNode(prediction=labels[0]), depth
    else:
        best = findSplit(training_dataset)
        l_dataset, r_dataset = best["left"], best["right"]
        node = TreeNode(feature=best["feature"], threshold=best["threshold"])
        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
        node.left, node.right = l_branch, r_branch
        return node, max(l_depth, r_depth)

def is_leaf(node):
    return node.left is None and node.right is None

def predict_one(node, x):
    while not is_leaf(node):
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction

def predict(node, x):
    return np.array([predict_one(node, row) for row in x])

    
def visualize_tree(node, feature_names=None, decimals=3):
    def fname(idx):
        if idx is None:
            return "?"
        return feature_names[idx] if feature_names is not None else f"X[{idx}]"

    def nthr(n):
        t = getattr(n, "threshold", None)
        return None if t is None else f"{t:.{decimals}f}"

    lines = []

    def is_leaf(n):
        if n is None:
            return True
        if hasattr(n, "is_leaf"):
            return bool(n.is_leaf)
        return n.left is None and n.right is None

    def rec(n, prefix="", branch_label=""):
        if n is None:
            lines.append(prefix + branch_label + "<empty>")
            return
        if is_leaf(n):
            pred = getattr(n, "prediction", None)
            if pred is not None:
                lines.append(prefix + branch_label + f"Predict: {pred}")
            else:
                lines.append(prefix + branch_label + "Leaf")
            return

        f = fname(getattr(n, "feature", None))
        t = nthr(n)
        cond = f if t is None else f"{f} ≤ {t}"
        lines.append(prefix + branch_label + f"[{cond}]")

        child_prefix = prefix + "    "
        rec(n.left,  child_prefix, "├─ True : ")
        rec(n.right, child_prefix, "└─ False: ")

    rec(node)
    return "\n".join(lines)

def save_tree_txt(node, file_path, feature_names=None, decimals=3):
    """Render the ASCII tree and write it to file_path."""
    # In case caller passes (node, depth)
    if isinstance(node, tuple) and len(node) == 2:
        node = node[0]

    txt = visualize_tree(node, feature_names=feature_names, decimals=decimals)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return file_path

def compute_accuracy(y_gold, y_prediction):
    assert len(y_gold) == len(y_prediction)

    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0

def _demo():
    # locate dataset relative to this script
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clean_dataset = os.path.join(base, 'wifi_db', 'clean_dataset.txt')
    data = load_dataset(clean_dataset)
    data_train, data_test = split_dataset(data, test_proportion=0.75)
    root, depth = decision_tree_learning(data_train, depth=0)
    out_path = os.path.join(base, "tree.txt")
    save_tree_txt(root, out_path)
    print(f"Wrote ASCII tree to: {out_path}")
    predicted = predict(root, data_test[:, :-1])
    print("clean accuracy", compute_accuracy(data_test[:, -1], predicted))

    ###NOISY
    noisy_path = os.path.join(base, 'wifi_db', 'noisy_dataset.txt')
    noisy_data = load_dataset(noisy_path)
    data_train, data_test = split_dataset(data, test_proportion=0.75)
    root, depth = decision_tree_learning(data_train, depth=0)
    predicted = predict(root, data_test[:, :-1])
    print("noisy accuracy", compute_accuracy(data_test[:, -1], predicted))

if __name__ == '__main__':
    _demo()