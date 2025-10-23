import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Node:
    """Class representing a node in the decision tree."""
    attribute: int = None
    value: float = None
    left: 'Node' = None
    right: 'Node' = None
    terminal: bool = False

class DecisionTree:
    def __init__(self, n_classes):
        """Initializes the DecisionTree."""
        self.root = None
        self.depth = 0
        self.n_classes = n_classes
        self.pruned = False
        
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
            return Node(value=int(y[0]), terminal=True), depth

        # Find the best split
        split_attribute, split_value, left, right = self.find_split(X, y)
        if split_attribute is None or left.size == 0 or right.size == 0:
            majority_class = int(np.argmax(np.bincount(y, minlength=self.n_classes + 1)))
            return Node(value=majority_class, terminal=True), depth

        # Recursively build the left and right branches
        l_branch, l_depth = self.decision_tree_learning(ds[left], depth + 1)
        r_branch, r_depth = self.decision_tree_learning(ds[right], depth + 1)

        return Node(attribute=split_attribute, value=split_value, left=l_branch, right=r_branch, terminal=False), max(l_depth, r_depth)

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
        """Prunes the decision tree using least-error pruning."""
        if not node or node.terminal:
            return

        # Recursively prune left and right subtrees
        self.prune(val_ds, node.left, acc_func, cm_func, print_prunes)
        self.prune(val_ds, node.right, acc_func, cm_func, print_prunes)

        # Evaluate accuracy before pruning
        X_val, y_val = val_ds[:, :-1], val_ds[:, -1].astype(int)
        old_accuracy = acc_func(cm_func((y_val, self.predict(X_val))))

        def try_to_prune(new_value):
            """Tries to prune the node to a terminal node with new_value."""
            old_state = (node.attribute, node.value, node.left, node.right, node.terminal)

            # Modify the node to be a terminal node
            node.attribute = None
            node.value = new_value
            node.left = None
            node.right = None
            node.terminal = True

            # Evaluate accuracy after pruning and revert if accuracy decreases
            new_accuracy = acc_func(cm_func((y_val, self.predict(X_val))))
            if new_accuracy < old_accuracy:
                (node.attribute, node.value, node.left, node.right, node.terminal) = old_state
            else:
                if print_prunes:
                    print(f"Successful prune.")

        # Try to prune the current node
        if node.left and node.left.terminal:
            try_to_prune(node.left.value)
        if node.right and node.right.terminal:
            try_to_prune(node.right.value)

    def recompute_depth(self):
        """Recomputes the depth of the tree."""
        def compute_depth(node):
            if not node or node.terminal:
                return 0
            return 1 + max(compute_depth(node.left), compute_depth(node.right))
        
        self.depth = compute_depth(self.root)

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
    
    def draw_tree(self):
        """Draws the decision tree."""
        positions = self.compute_positions(self.root, 0)
        scale_x, scale_y = 2.0, 3.0
        
        # Determine figure size based on tree dimensions
        max_x = max(x for _, x, _ in positions.values())
        max_y = max(y for _, _, y in positions.values())
        fig_width = (max_x + 2) * scale_x / 2
        fig_height = (max_y + 2) * scale_y / 2

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        def draw_edges(node):
            """Recursively draws edges between nodes."""
            if node.terminal:
                return 
            
            _, x, y = positions[id(node)]
            x1, y1 = x * scale_x, y * scale_y

            # Draw edges from parent to each childs
            for child in (node.left, node.right):
                if not child:
                    continue

                _, x, y = positions[id(child)]
                x2, y2 = x * scale_x, y * scale_y
                ax.plot([x1, x2], [y1, y2], 'k-', zorder=1)
                draw_edges(child)

        draw_edges(self.root)

        # Draw the nodes and their labels
        for n, x, y in positions.values():
            x1, y1 = x * scale_x, y * scale_y
            ax.add_patch(plt.Circle((x1, y1), 0.6, color='skyblue', ec='black', zorder=2))
            label = f"Class:\n{n.value}" if n.terminal else f"Attr {n.attribute}\n< {n.value:.1f}"
            ax.text(x1, y1, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

        plt.tight_layout()
        plt.show()

    def compute_positions(self, node, depth, positions=None):
        """Computes the (x, y) positions for each node in the tree."""
        if not node:
            return positions

        if positions is None:
            positions = {}

        # Recursively compute positions for child nodes
        if not node.terminal:
            self.compute_positions(node.left, depth + 1, positions)
            self.compute_positions(node.right, depth + 1, positions)

            # Compute x position based on child nodes
            left_x = positions.get(id(node.left), (None, 0, 0))[1]
            right_x = positions.get(id(node.right), (None, 0, 0))[1]
            x = (left_x + right_x) / 2
        else:
            # Compute x position for terminal nodes
            x = sum(1 for n, _, _ in positions.values() if n.terminal) 

        # Store the position of the current node
        positions[id(node)] = (node, x, depth)
        return positions
