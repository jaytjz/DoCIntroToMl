import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
import os


def plot_tree(root, depth, x=0, y=0, dx=None, dy=3, ax=None, current_depth=0):

    if ax is None:
        fig_width = min(60, max(25, depth * 3))
        fig_height = min(40, max(12, depth * 2.5))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        if dx is None:
            dx = max(2 ** depth, 2 * (2 ** depth))
        plot_tree(root, depth, x, y, dx, dy, ax, current_depth)
        plt.show()
        return

    # ---- leaf case: not a node ----
    if not hasattr(root, "left") and not hasattr(root, "right"):
        fontsize = max(6, 10 - current_depth // 2)
        ax.text(x, y, f"Class: {root}", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", edgecolor="black"),
                fontsize=fontsize)
        return

    # ---- draw current node ----
    fontsize = max(6, 10 - current_depth // 2)
    label = f"X[{root.feature}] <= {root.cond:.2f}"
    ax.text(x, y, label, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black"),
            fontsize=fontsize)

    # ---- recurse to children ----
    if depth > 1:
        offset = max(dx / 2, 2.5)  # Ensure minimum spacing of 2.5
        if root.left is not None:
            ax.plot([x, x - offset], [y - 0.3, y - dy + 0.3], "k-", linewidth=1.0)
            plot_tree(root.left, depth - 1, x - offset, y - dy, offset, dy, ax, current_depth + 1)
        if root.right is not None:
            ax.plot([x, x + offset], [y - 0.3, y - dy + 0.3], "k-", linewidth=1.0)
            plot_tree(root.right, depth - 1, x + offset, y - dy, offset, dy, ax, current_depth + 1)


def plot_confusion(cm):
    labels = ['A', 'B', 'C', 'D']

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.colorbar(im, ax=ax)
    plt.title('Confusion Matrix')
    plt.show()


def shuffle(data, max_depth):
    shuffled_indices = default_rng().permutation(len(data))
    n_test = round(len(data) * 0.1)
    n_train = len(data) - n_test
    df_train = data[shuffled_indices[:n_train]]
    df_test = data[shuffled_indices[n_train:]]

    tree = DecisionTree()
    tree.fit(df_train, max_depth)
    print(f"Tree depth: {tree.depth}")
    plot_tree(tree.root, tree.depth)

    return df_test, tree
    

def testing(df_test, tree):
    correct = 0
    cm = np.zeros((4, 4), dtype=int)

    for i in range(len(df_test)):
        sample = df_test[i, :7]
        true_label = int(df_test[i, 7])
        predicted_label = int(tree.predict(sample))
        cm[true_label-1, predicted_label-1] += 1
    
        if predicted_label == true_label:
            correct += 1
    # plot_confusion(cm)

    return correct
    

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path_clean = os.path.join(script_dir, '..', 'wifi_db', 'clean_dataset.txt')
    data_path_noisy = os.path.join(script_dir, '..', 'wifi_db', 'noisy_dataset.txt')
    data_clean = np.loadtxt(data_path_clean)
    data_noisy = np.loadtxt(data_path_noisy)

    clean_max_depth = 15
    noisy_max_depth = 5
    
    df_test_clean, tree_clean = shuffle(data_clean, clean_max_depth)
    correct_clean = testing(df_test_clean, tree_clean)

    df_test, tree = shuffle(data_noisy, noisy_max_depth)
    correct_noisy = testing(df_test, tree)

    print(f"\nAccuracy on 200 clean samples: {correct_clean}/200 = {100*correct_clean/200:.1f}%")
    print(f"\nAccuracy on 200 noisy samples: {correct_noisy}/200 = {100*correct_noisy/200:.1f}%")

if __name__ == '__main__':
    main()