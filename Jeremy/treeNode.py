class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.threshold = threshold
        self.prediction = prediction 

    @property
    def leaf(self):
        return self.prediction is not None