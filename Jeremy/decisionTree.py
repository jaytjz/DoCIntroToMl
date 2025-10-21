import numpy as np
from treeNode import TreeNode

class DecisionTree:
    def __init__(self, data):
        self.root, self.depth = self.decision_tree_learning(data, depth=0)

    def decision_tree_learning(self, training_dataset, depth):
        y = training_dataset[:, -1]
        labels = np.unique(y)

        if labels.size == 1:
            return TreeNode(prediction=labels[0]), depth
        else:
            best = self.findSplit(training_dataset)
            l_dataset, r_dataset = best["left"], best["right"]
            node = TreeNode(feature=best["feature"], threshold=best["threshold"])
            l_branch, l_depth = self.decision_tree_learning(l_dataset, depth+1)
            r_branch, r_depth = self.decision_tree_learning(r_dataset, depth+1)
            node.left, node.right = l_branch, r_branch
            return node, max(l_depth, r_depth)
        
    def findSplit(self, data):
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

            distinct =  ~np.isclose(x_sorted[:-1], x_sorted[1:]) #find distinct values of x
            idxs = np.flatnonzero(distinct)

            for i in idxs:
                thresh = (x_sorted[i] + x_sorted[i+1]) * 0.5
                left_mask = x[:, c] <= thresh
                right_mask = x[:, c] > thresh

                y_left = y[left_mask]
                y_right = y[right_mask]
                g = self.gain(y, y_left, y_right)

                if g > best["gain"]:
                    best.update({
                        "feature": c,
                        "threshold": thresh,
                        "gain": g,
                        "left": data[left_mask],
                        "right":data[right_mask]
                    })

        return best
    
    def gain(self, s_all, s_left, s_right):
        def H(y):
            _, counts = np.unique(y, return_counts=True)
            p_k = counts / counts.sum()
            return -np.sum(p_k * np.log2(p_k))
        
        def remainder(s_left, s_right):
            samples_left = s_left.shape[0]
            samples_right = s_right.shape[0]
            total_sample = samples_left + samples_right
            return (samples_left / (total_sample)) * H(s_left) + (samples_right / total_sample) * H(s_right)
        
        return H(s_all) - remainder(s_left, s_right)

    def predict(self, x):
        def predict_one(node, x):
            while not node.leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            return node.prediction
        return np.array([predict_one(self.root, row) for row in x])
    
    def print_tree(self, node= None, indent=""):
        if node is None:
            node = self.root

        if node.leaf:
            print(f"{indent}Prediction: {node.prediction}")
        else:
            print(f"{indent}Feature: {node.feature}, Threshold: {node.threshold}")
            print(f"{indent}--> Left Branch:")
            self.print_tree(node.left, indent + "\t")
            print(f"{indent}--> Right Branch:")
            self.print_tree(node.right, indent + "\t")