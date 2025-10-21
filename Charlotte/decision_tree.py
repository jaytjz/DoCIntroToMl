from platform import node
import numpy as np
import itertools
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Node:
    attribute: int = None
    value: float = None
    left: 'Node' = None
    right: 'Node' = None
    terminal: bool = False
    depth: int = 0

class DecisionTree:
    def __init__(self, n_classes):
        """Initializes the DecisionTree."""
        self.root = None
        self.depth = 0
        self.n_classes = n_classes
        
    def predict(self, X_test):
        """Predict the class labels for the given test data."""
        n = X_test.shape[0]
        y_hat = np.empty(n, dtype=int)

        # Traverse the tree for each test instance
        for i, x in enumerate(X_test):
            node = self.root
            while not node.terminal:
                node = node.left if x[node.attribute] <= node.value else node.right
            y_hat[i] = node.value

        return y_hat

    def decision_tree_learning(self, ds, depth):
        """Recursively builds the decision tree using the ID3 algorithm."""
        X, y = ds[:, :-1], ds[:, -1].astype(int)
        
        # Return a leaf node if all labels are the same
        if np.all(y == y[0]):
            return Node(value=int(y[0]), terminal=True, depth=depth), depth

        # Find the best split
        split_attribute, split_value, left, right = self.find_split(X, y)
        if split_attribute is None or left.size == 0 or right.size == 0:
            majority_class = int(np.argmax(np.bincount(y, minlength=self.n_classes + 1)))
            return Node(value=majority_class, terminal=True, depth=depth), depth

        # Recursively build the left and right branches
        l_branch, l_depth = self.decision_tree_learning(ds[left], depth + 1)
        r_branch, r_depth = self.decision_tree_learning(ds[right], depth + 1)

        return Node(attribute=split_attribute, value=split_value, left=l_branch, right=r_branch, terminal=False, depth=depth), max(l_depth, r_depth)

    def find_split(self, X, y):
        """Finds the best attribute and value to split the data."""
        n, d = X.shape
        if n == 0 or d == 0:
            return None, None, None, None
        
        # Calculate the entropy of the parent node
        H_parent = self.entropy(y)
        if H_parent == 0:
            return None, None, None, None
        
        best_ig, best_attribute, best_split_value = -np.inf, None, None

        for j in range(d):
            # Sort the data by the j-th attribute
            col = X[:, j]
            order = np.argsort(col)
            xj, yj = col[order], y[order]
            
            # Identify candidate split points
            candidates = np.flatnonzero(np.diff(xj) != 0) + 1
            if candidates.size == 0:
                continue

            # Compute class counts for left splits using one-hot encoding
            one_hot = np.eye(self.n_classes)[yj - 1]
            prefix_sums = np.cumsum(one_hot, axis=0)

            # Calculate entropy of the left split
            n_left = candidates
            l_counts = prefix_sums[candidates - 1]
            H_left = self.vector_entropy(n_left, l_counts)
            
            # Calculate entropy of the right split
            n_right = n - n_left
            r_counts = prefix_sums[-1] - l_counts
            H_right = self.vector_entropy(n_right, r_counts)
            
            # Calculate information gain and update best split if necessary
            ig = H_parent - (n_left / n) * H_left - (n_right / n) * H_right
            k = np.argmax(ig)
            if ig[k] > best_ig:
                best_ig = ig[k]
                i = candidates[k]
                best_attribute = j
                best_split_value = (xj[i - 1] + xj[i]) / 2

        # Return None if no valid split found
        if best_attribute is None or best_ig <= 0 or not np.isfinite(best_split_value):
            return None, None, None, None
        
        # Determine the indices for left and right splits
        best_left = np.nonzero(X[:, best_attribute] <= best_split_value)[0]
        best_right = np.nonzero(X[:, best_attribute] > best_split_value)[0]

        return best_attribute, best_split_value, best_left, best_right
    
    def prune(self, val_ds, node, acc_func, cm_func, print_prunes=False):
        if not node.terminal:
            self.prune(val_ds, node.left, acc_func, cm_func, print_prunes)
            self.prune(val_ds, node.right, acc_func, cm_func, print_prunes)

            X_val, y_val = val_ds[:, :-1], val_ds[:, -1].astype(int)
            old_node = node
            
            def try_prune_to(X_val, y_val, new_value, old_node):
                old_accuracy = acc_func(cm_func((y_val, self.predict(X_val))))

                new_node = Node(value=new_value, terminal=True)
                new_accuracy = acc_func(cm_func((y_val, self.predict(X_val))))

                if new_accuracy < old_accuracy:
                    return old_node
                else:
                    if print_prunes:
                        print("Succesful prune")

                return new_node

            if node.left and node.left.terminal:
                node.left = try_prune_to(X_val, y_val, node.left.value, old_node)
            if node.right and node.right.terminal:
                node.right = try_prune_to(X_val, y_val, node.right.value, old_node)

    def vector_entropy(self, n, counts):
        """Calculates the entropy for a batch of class-count rows."""
        p = np.divide(counts, n[:, None], out=np.zeros_like(counts, dtype=float), where=n[:, None] > 0)
        logp = np.zeros_like(p)
        np.log2(p, out=logp, where=p > 0)
        return -(p * logp).sum(axis=1)

    def entropy(self, y):
        """Calculates the entropy of the class labels."""
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    def visualise_tree(self, figsize):
        """Plots the decision tree."""
        fig, ax = plt.subplots(figsize=figsize)
        root = self.root
        counter = itertools.count()
        pos = {}
        
        def traverse_inorder(n, depth):
            """In-order traversal to assign positions to nodes."""
            if n is None:
                return
            if not n.terminal:
                traverse_inorder(n.left, depth + 1)
            pos[id(n)] = (next(counter), -depth)
            if not n.terminal:
                traverse_inorder(n.right, depth + 1)

        traverse_inorder(root, 0)

        # Normalize x-coordinates
        xs = [x for x, _ in pos.values()]
        x_min, x_max = min(xs), max(xs)
        span = max(1, x_max - x_min + 1)
        for k, (x, y) in pos.items():
            pos[k] = ((x - x_min + 0.5) / span, y)

        
        def label(n):
            """Generates the label for a node."""
            if n.terminal:
                return f"R{int(n.value)}"
            return f"X{n.attribute} ≤ {n.value}"

        def draw_edges(n):
            """Draws the edges of the tree."""
            if n is None or n.terminal:
                return
            x0, y0 = pos[id(n)]
            for child in [n.left, n.right]:
                x1, y1 = pos[id(child)]
                ax.plot([x0, x1], [y0, y1], color='black')
                draw_edges(child)

        def collect(n, out):
            """Collects all nodes in the tree."""
            if n is None:
                return
            out.append(n)
            collect(n.left, out)
            collect(n.right, out)

        nodes = []
        draw_edges(root)
        collect(root, nodes)

        # Draw nodes
        for n in nodes:
            x, y = pos[id(n)]
            ax.text(x, y, label(n), ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1.2), fontsize=6)

        ax.axis('off')
        plt.tight_layout()
        plt.show()
