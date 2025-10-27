import numpy as np
from decision_tree import DecisionTree

class KFoldValidator:
    def __init__(self, ds_filename, n_classes=4, k=10, prune=True):
        """Initializes the KFoldValidator."""
        self.ds_filename = ds_filename
        self.k = k
        self.n_classes = n_classes
        self.prune = prune

        # Load and shuffle the dataset
        data = np.loadtxt(ds_filename)
        self.data = data[np.random.default_rng(42).permutation(data.shape[0])]

        self.splits = self.split_data(self.data, k)
        self.models = []

    def split_data(self, data, k):
        """Splits the data into k folds for cross-validation."""
        n = data.shape[0]

        # Determine the size of each fold
        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[:n % k] += 1
        
        splits = []
        current = 0
        for fold_size in fold_sizes:
            # Get the test data for the current fold
            start, stop = current, current + fold_size
            test = data[start:stop]

            # Get training data by masking out the test indices
            mask = np.ones(n, dtype=bool)
            mask[start:stop] = False
            train = data[mask]

            # Update splits and current index
            splits.append((train, test))
            current = stop

        return splits
    
    def split_validation(self, data):
        split = 0.9 * data.shape[0]
        train = data[:int(split)]
        test = data[int(split):]
        return train, test
    
    def validate(self):
        """Validates the model with or without pruning."""
        return self.k_fold_validation(prune=self.prune)

    def k_fold_validation(self, prune=True):
        """Performs k-fold cross-validation."""
        self.best_model = None
        self.best_accuracy = 0
        cms = []
        
        for train, test in self.splits:
            X_test, y_test = test[:, :-1], test[:, -1].astype(int)

            # Train pruned decision tree models using internal k-fold validation
            if prune:   
                pruned_cms = self.prune_k_fold_validation(train, ki=self.k - 1) 
                cms += pruned_cms
            # Train the unpruned decision tree model
            else:
                # Train the decision tree model
                model = DecisionTree(n_classes=self.n_classes)
                model.root, model.depth = model.decision_tree_learning(train, 0)
                self.models.append(model)

                # Evaluate the unpruned model
                y_hat = model.predict(X_test)
                cm = self.confusion_matrix((y_test, y_hat))
                cms.append(cm)

                acc = self.compute_accuracy(cm)
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    self.best_model = model

        return self.evaluate(cms, False)
    
    def prune_k_fold_validation(self, data, ki=9):
        """Performs internal k-fold cross-validation with pruning."""
        internal_splits = self.split_data(data, ki)
        cms = []

        for train, val in internal_splits:
            X_val, y_val = val[:, :-1], val[:, -1].astype(int)
            
            # Train and prune the decision tree model
            model = DecisionTree(n_classes=self.n_classes)
            model.root, model.depth = model.decision_tree_learning(train, 0)
            model.prune(val, model.root, acc_func=self.compute_accuracy, cm_func=self.confusion_matrix)
            model.recompute_depth()
            model.pruned = True
            self.models.append(model)

            # Evaluate the pruned model
            y_hat = model.predict(X_val)
            cm = self.confusion_matrix((y_val, y_hat))
            cms.append(cm)

            acc = self.compute_accuracy(cm)
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_model = model
            
        return cms

    def confusion_matrix(self, data_to_evaluate):
        """Computes the confusion matrix for the given true and predicted labels."""
        y_true = np.asarray(data_to_evaluate[0]).astype(int)
        y_pred = np.asarray(data_to_evaluate[1]).astype(int)
        
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        if y_true.size:
            # Populate the confusion matrix, normalise to 0-based indexing
            np.add.at(cm, (y_true - 1, y_pred - 1), 1)
        return cm

    def compute_accuracy(self, confusion_matrix):
        """Computes the accuracy from the confusion matrix."""
        total = np.sum(confusion_matrix)
        correct = np.sum(np.diag(confusion_matrix))
        return (correct / total) if total > 0 else 0

    def evaluate(self, confusion_matrices, single_fold=True):
        """Evaluates the model performance using the confusion matrices."""
        cm = confusion_matrices if single_fold else np.sum(confusion_matrices, axis=0)
        averaged_cm = np.array2string(np.mean(confusion_matrices, axis=0), formatter={'float_kind': lambda x: "%.2f" % x})
        tp = np.diag(cm)
        col_sum = np.sum(cm, axis=0)
        row_sum = np.sum(cm, axis=1)
        total = np.sum(cm)

        # Calculate overall accuracy, precision, recall, and F1-score
        accuracy = (np.sum(tp) / total) if total > 0 else 0
        precision_per_class = np.divide(tp.astype(float), col_sum, out=np.zeros_like(tp, dtype=float), where=(col_sum != 0))
        recall_per_class = np.divide(tp.astype(float), row_sum, out=np.zeros_like(tp, dtype=float), where=(row_sum != 0))
        precision_recall_sum = precision_per_class + recall_per_class
        precision_recall_prod = 2.0 * precision_per_class * recall_per_class
        f1_score_per_class = np.divide(precision_recall_prod, precision_recall_sum, out=np.zeros_like(precision_recall_prod, dtype=float), where=(precision_recall_sum != 0))
        
        return {'confusion_matrix': cm, 'average_confusion_matrix': averaged_cm, 'accuracy': accuracy, 'precision_per_class': precision_per_class, 'recall_per_class': recall_per_class, 'f1_score_per_class': f1_score_per_class}

    def get_average_model_depth(self):
        """Returns the average depth of the models."""
        return np.mean([model.depth for model in self.models])