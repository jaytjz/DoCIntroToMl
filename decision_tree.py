import numpy as np
import itertools
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, n_classes):
        """Initializes the DecisionTree."""
        self.tree = None
        self.depth = 0
        self.n_classes = n_classes
        
    def predict(self, X_test):
        """Predict the class labels for the given test data."""
        n = X_test.shape[0]
        y_hat = np.empty(n, dtype=int)

        # Traverse the tree for each test instance
        for i in range(n):
            node = self.tree
            while not node['terminal']:
                node = node['left'] if X_test[i, node['attribute']] <= node['value'] else node['right']
            y_hat[i] = node['value']

        return y_hat

    def decision_tree_learning(self, ds, depth):
        """Recursively builds the decision tree using the ID3 algorithm."""
        X, y = ds[:, :-1], ds[:, -1].astype(int)
        
        # Return a leaf node if all labels are the same
        if np.all(y == y[0]):
            return {'attribute': None, 'value': int(y[0]), 'left': None, 'right': None, 'terminal': True}, depth

        # Find the best split
        split_attribute, split_value, left_indices, right_indices = self.find_split(X, y)
        if split_attribute is None or left_indices.size == 0 or right_indices.size == 0:
            majority_class = int(np.argmax(np.bincount(y, minlength=self.n_classes + 1)))
            return {'attribute': None, 'value': majority_class, 'left': None, 'right': None, 'terminal': True}, depth

        # Recursively build the left and right branches
        l_branch, l_depth = self.decision_tree_learning(ds[left_indices], depth + 1)
        r_branch, r_depth = self.decision_tree_learning(ds[right_indices], depth + 1)

        return {'attribute': split_attribute, 'value': split_value, 'left': l_branch, 'right': r_branch, 'terminal': False}, max(l_depth, r_depth)

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
            p_left = np.divide(l_counts, n_left[:, None], out=np.zeros_like(l_counts, dtype=float), where=n_left[:, None] > 0)
            log_p_left = np.zeros_like(p_left)
            np.log2(p_left, out=log_p_left, where=(p_left > 0))
            H_left  = -np.sum(p_left * log_p_left, axis=1)

            # Calculate entropy of the right split
            n_right = n - n_left
            r_counts = prefix_sums[-1] - l_counts
            p_right = np.divide(r_counts, n_right[:, None], out=np.zeros_like(r_counts, dtype=float), where=n_right[:, None] > 0)
            log_p_right = np.zeros_like(p_right)
            np.log2(p_right, out=log_p_right, where=(p_right > 0))
            H_right = -np.sum(p_right * log_p_right, axis=1)

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

    def entropy(self, y):
        """Calculates the entropy of the class labels."""
        counts = np.bincount(y, minlength=self.n_classes + 1)
        probabilities = counts[1:] / counts.sum()
        probabilities = probabilities[probabilities > 0]
        return -np.dot(probabilities, np.log2(probabilities))

    def visualise_tree(self, figsize):
        """Plots the decision tree."""
        fig, ax = plt.subplots(figsize=figsize)
        tree = self.tree
        counter = itertools.count()
        pos = {}
        
        def traverse_inorder(n, depth):
            """In-order traversal to assign positions to nodes."""
            if n is None:
                return
            if not n['terminal']:
                traverse_inorder(n['left'], depth + 1)
            pos[id(n)] = (next(counter), -depth)
            if not n['terminal']:
                traverse_inorder(n['right'], depth + 1)

        traverse_inorder(tree, 0)
        
        # Normalize x-coordinates
        xs = [x for x, _ in pos.values()]
        x_min, x_max = min(xs), max(xs)
        span = max(1, x_max - x_min + 1)
        for k, (x, y) in pos.items():
            pos[k] = ((x - x_min + 0.5) / span, y)

        
        def label(n):
            """Generates the label for a node."""
            if n['terminal']:
                return f"Leaf {int(n['value'])}"
            return f"X{n['attribute']} ≤ {n['value']}"
        
        def draw_edges(n):
            """Draws the edges of the tree."""
            if n is None or n['terminal']:
                return
            x0, y0 = pos[id(n)]
            for child in [n['left'], n['right']]:
                x1, y1 = pos[id(child)]
                ax.plot([x0, x1], [y0, y1], color='black')
                draw_edges(child)

        def collect(n, out):
            """Collects all nodes in the tree."""
            if n is None:
                return
            out.append(n)
            collect(n['left'], out)
            collect(n['right'], out)

        nodes = []
        draw_edges(tree)
        collect(tree, nodes)

        # Draw nodes
        for n in nodes:
            x, y = pos[id(n)]
            ax.text(x, y, label(n), ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1.2), fontsize=6)

        ax.axis('off')
        plt.tight_layout()
        plt.show()
