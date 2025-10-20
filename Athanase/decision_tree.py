import numpy as np
from node import node


class DecisionTree:
    def __init__(self):
        self.root = None
        self.depth = 0

    def entropy(self, df):
        """Calculate entropy of a dataset based on the labels (column 7)"""
        df = np.unique(df[:, 7], return_counts=True)

        p = df[1] / np.sum(df[1])
        p = p[p > 0]  # only consider non-zero probabilities
        H = -np.sum(p * np.log2(p), axis=0)
        return H

    def find_split(self, train_set):
        """Find the best split for the dataset"""
        set = {}
        for i in range(train_set.shape[1]-1):
            sorted_df = train_set[train_set[:,i].argsort()]
            save = [0,0,0]

            for j in range(sorted_df.shape[0]-1):
                # Skip if values are the same (no point in splitting here)
                if sorted_df[j][i] == sorted_df[j+1][i]:
                    continue

                threshold = (sorted_df[j][i] + sorted_df[j+1][i])/2
                l_data = sorted_df[:j+1, :]
                r_data = sorted_df[j+1:, :]

                H_l = self.entropy(l_data)
                H_r = self.entropy(r_data)

                H = self.entropy(sorted_df)

                Remainder = ((j+1)/sorted_df.shape[0])*H_l + ((sorted_df.shape[0]-j-1)/sorted_df.shape[0])*H_r

                Gain = H - Remainder

                if Gain > save[0]:
                    save = [Gain, threshold, j]
            set[i] = save

        feature = max(set.items(), key=lambda x: x[1][0])[0]
        cond = set[feature][1]
        j = set[feature][2]

        # Resort by the best feature to get correct split
        sorted_df = train_set[train_set[:,feature].argsort()]
        l_data = sorted_df[:j+1, :]
        r_data = sorted_df[j+1:, :]

        return l_data, r_data, feature, cond

    def build_tree(self, data, depth):
        if len(np.unique(data[:,7])) == 1:
            return data[0, 7], depth

        l_dataset, r_dataset, feature, cond = self.find_split(data)
        l_branch, l_depth = self.build_tree(l_dataset, depth+1)
        r_branch, r_depth = self.build_tree(r_dataset, depth+1)
        return node(feature, cond, l_branch, r_branch), max(l_depth, r_depth)

    def fit(self, data):
        self.root, self.depth = self.build_tree(data, 0)
        return self

    def predict(self, x):
        return self.evaluate(self.root, x)

    def evaluate(self, tree, x):
        if not isinstance(tree, node):
            return tree

        if x[tree.feature] <= tree.cond:
            return self.evaluate(tree.left, x)
        else:
            return self.evaluate(tree.right, x)
