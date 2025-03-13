import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=None, max_features=None, bootstrap=True):
        """
        Custom Random Forest classifier.
        Parameters:
            n_estimators (int): Number of trees in the forest.
            min_samples_split (int): Minimum samples required to split an internal node.
            max_depth (int): Maximum depth of each tree (None for no limit).
            max_features (int or None): Number of features to consider at each split (default sqrt of total features if None).
            bootstrap (bool): Whether to use bootstrap sampling for each tree.
        """
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth if max_depth is not None else float('inf')  # if None, treat as no limit
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []               # list to hold trained DecisionTree instances
        self.samples_indices = []     # store bootstrap sample indices for each tree (for feature importance calc)
        self.n_features_ = None       # total number of features (set during training)
        self.feature_importances_ = None  # to be computed after training

    def train(self, X, y):
        """
        Train the Random Forest on data.
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        # Determine number of features to use for splits in each tree:
        features_per_tree = self.max_features or int(np.sqrt(n_features))  # default to sqrt(n_features) if max_features not set
        features_per_tree = max(1, features_per_tree)  # at least 1 feature
        self.trees = []
        self.samples_indices = []
        for _ in range(self.n_estimators):
            # Sample data for this tree (bootstrap if enabled)
            if self.bootstrap:
                sample_idx = np.random.choice(n_samples, n_samples, replace=True)
            else:
                sample_idx = np.arange(n_samples)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            # Initialize and train a decision tree on the sample
            tree = DecisionTree(min_samples_split=self.min_samples_split, 
                                max_depth=self.max_depth, 
                                n_features=features_per_tree)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)
            self.samples_indices.append(sample_idx)
        # Compute feature importances after all trees are trained
        self._calculate_feature_importances(X, y)

    def _calculate_feature_importances(self, X, y):
        """
        Calculate feature importance based on mean decrease in impurity (entropy reduction).
        """
        n_features = self.n_features_
        importances = np.zeros(n_features)
        # Sum impurity decrease from each feature over all trees
        for tree, sample_idx in zip(self.trees, self.samples_indices):
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            # Recursively traverse the tree to accumulate impurity reduction for each feature
            def recurse(node, X_subset, y_subset):
                if node.is_leaf_node():
                    return  # no split at leaf
                feat = node.feature
                thresh = node.threshold
                # Compute parent node entropy
                if len(y_subset) == 0:
                    return
                # Entropy of current node (parent)
                hist = np.bincount(y_subset, minlength=len(np.unique(y_sample)))
                ps = hist / len(y_subset)
                parent_entropy = -np.sum([p * np.log(p) for p in ps if p > 0])
                # Split data based on the node's threshold
                left_idx = np.where(X_subset[:, feat] <= thresh)[0]
                right_idx = np.where(X_subset[:, feat] > thresh)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    return  # no valid split (should not happen in a trained tree)
                # Entropy of left child
                n_left = len(left_idx)
                hist_left = np.bincount(y_subset[left_idx], minlength=len(np.unique(y_sample)))
                ps_left = hist_left / n_left
                entropy_left = -np.sum([p * np.log(p) for p in ps_left if p > 0])
                # Entropy of right child
                n_right = len(right_idx)
                hist_right = np.bincount(y_subset[right_idx], minlength=len(np.unique(y_sample)))
                ps_right = hist_right / n_right
                entropy_right = -np.sum([p * np.log(p) for p in ps_right if p > 0])
                # Weighted child entropy
                n_total = len(y_subset)
                weighted_child_entropy = (n_left/n_total) * entropy_left + (n_right/n_total) * entropy_right
                # Impurity decrease (information gain) for this split:
                impurity_decrease = parent_entropy - weighted_child_entropy
                importances[feat] += impurity_decrease
                # Recurse into children
                recurse(node.left, X_subset[left_idx], y_subset[left_idx])
                recurse(node.right, X_subset[right_idx], y_subset[right_idx])
            recurse(tree.root, X_sample, y_sample)
        # Normalize importances to sum to 1 (optional for interpretability)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        self.feature_importances_ = importances

    def predict(self, X):
        """
        Predict class labels for given input samples X.
        """
        X = np.array(X)
        # Collect predictions from each tree
        all_preds = [tree.test(X) for tree in self.trees]  # DecisionTree.test returns predictions for X
        # all_preds is a list of arrays (each of shape [n_samples]), convert to matrix [n_samples x n_estimators]
        all_preds = np.array(all_preds).T
        # Majority vote for each sample
        y_pred = []
        for preds in all_preds:
            # most common class among predictions
            most_common = Counter(preds).most_common(1)[0][0]
            y_pred.append(most_common)
        return np.array(y_pred)

    def accuracy(self, X, y):
        """
        Compute accuracy of the classifier on a given dataset (X, y).
        """
        y = np.array(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def confusion_matrix(self, X, y, positive_label):
        """
        Compute confusion matrix components for binary classification.
        Treat 'positive_label' as the positive class, and all other labels as negative.
        Returns a dictionary with TP, TN, FP, FN counts.
        """
        y = np.array(y)
        y_pred = self.predict(X)
        # Define positive/negative as per the given positive_label
        positive = positive_label
        negative = None  # represents "not positive"
        # True/False Positive/Negative calculations
        TP = np.sum((y == positive) & (y_pred == positive))
        TN = np.sum((y != positive) & (y_pred != positive))
        FP = np.sum((y != positive) & (y_pred == positive))
        FN = np.sum((y == positive) & (y_pred != positive))
        return {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}