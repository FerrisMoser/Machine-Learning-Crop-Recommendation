import numpy as np
import graphviz
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = float('inf')  # Infinite depth
        self.n_features = n_features
        self.root = None

    def train(self, array, y):
        self.n_features = array.shape[1] if not self.n_features else min(array.shape[1], self.n_features)
        self.root = self.grow_tree(array, y)

    def grow_tree(self, array, y, depth=0):
        n_samples, n_features = array.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self.best_split(array, y, feat_idxs)

        left_idxs, right_idxs = self.split(array[:, best_feature], best_threshold)
        left = self.grow_tree(array[left_idxs, :], y[left_idxs], depth+1)
        right = self.grow_tree(array[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def best_split(self, array, y, feat_idxs):
        best_gain = -1
        split_index, split_threshold = None, None

        for feat_idx in feat_idxs:
            array_column = array[:, feat_idx]
            thresholds = np.unique(array_column)

            for threshold in thresholds:
                gain = self.information_gain(y, array_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feat_idx
                    split_threshold = threshold

        return split_index, split_threshold

    def information_gain(self, y, array_column, threshold):
        parent_entropy = self.entropy(y)
        left_idxs, right_idxs = self.split(array_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def split(self, array_column, split_threshold):
        left_idxs = np.argwhere(array_column <= split_threshold).flatten()
        right_idxs = np.argwhere(array_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def test(self, array):
        return np.array([self.traverse_tree(x, self.root) for x in array])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def visualize_tree(self):
        """ Generate a Graphviz representation of the decision tree. """
        def add_nodes_edges(dot, node, parent_name=None, edge_label=""):
            if node is None:
                return

            node_label = f"Feature {node.feature} ≤ {node.threshold:.2f}" if not node.is_leaf_node() else f"Class: {node.value}"
            node_name = f"Node_{id(node)}"
            dot.node(node_name, node_label, shape="ellipse" if not node.is_leaf_node() else "box")

            if parent_name:
                dot.edge(parent_name, node_name, label=edge_label)

            add_nodes_edges(dot, node.left, node_name, "Yes")
            add_nodes_edges(dot, node.right, node_name, "No")

        dot = graphviz.Digraph(format="png")
        add_nodes_edges(dot, self.root)
        return dot
