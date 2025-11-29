import numpy as np
from tree_node import Node

# DecisionTree should only take in numpy arrays
class DecisionTree:
    
    def __init__(self, min_samples_split=5_000, max_depth=10):
        self.tree = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    
    def predict(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        return np.array([self.predict_one(x) for x in X])
        
    # handles building out the tree itself
    def fit(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be NumPy arrays.")
        self.tree = self._build_tree(X, y)
    
    def predict_one(self, x):
        node = self.tree
        while node.value == None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value
        
    def _build_tree(self, X, y, depth=0):
        # ----- OPTIONAL DEBUG PRINTS -----
        indent = "  " * depth
        print(f"{indent}[BUILD] depth={depth}, samples={len(y)}")
    
        # 1. Stopping: max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            labels, counts = np.unique(y, return_counts=True)
            majority_label = labels[np.argmax(counts)]
            # print(f"{indent}[LEAF] max_depth → {majority_label}")
            return Node(value=majority_label)
    
        # 2. Stopping: not enough samples to split
        if len(y) < self.min_samples_split:
            labels, counts = np.unique(y, return_counts=True)
            majority_label = labels[np.argmax(counts)]
            # print(f"{indent}[LEAF] min_samples → {majority_label}")
            return Node(value=majority_label)
    
        # 3. Stopping: pure node
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            # print(f"{indent}[LEAF] pure → {unique_labels[0]}")
            return Node(value=unique_labels[0])
    
        # 4. Compute best split
        gain, feature, threshold, left, right = self._best_split(X, y)
        # print(f"{indent}[SPLIT] feature={feature}, threshold={threshold}, gain={gain}")
    
        # 5. Stopping: no useful split
        if gain <= 0:
            labels, counts = np.unique(y, return_counts=True)
            majority_label = labels[np.argmax(counts)]
            # print(f"{indent}[LEAF] zero gain → {majority_label}")
            return Node(value=majority_label)
    
        # 6. Recurse into children
        X_left, y_left = left
        X_right, y_right = right
        # print(f"{indent}[CHILDREN] left={len(y_left)}, right={len(y_right)}")
    
        node = Node(feature=feature, threshold=threshold)
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)
    
        return node


    def _entropy(self, y):
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        
        H = (-probs * np.log2(probs)).sum()
        return H
        
    def _best_split(self, X, y, n_quantiles=100):
        n_samples, n_features = X.shape
        parent_entropy = self._entropy(y)
    
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None
    
        # Loop over features
        for feature in range(n_features):
            col = X[:, feature]
    
            # Sort once for this feature
            sorted_idx = np.argsort(col)
            sorted_col = col[sorted_idx]
            sorted_y = y[sorted_idx]
    
            # Compute quantile thresholds
            # e.g. 50 evenly spaced thresholds between min and max
            quantile_positions = np.linspace(0, n_samples - 1, n_quantiles + 2)[1:-1]
            quantile_positions = quantile_positions.astype(int)
            candidate_thresholds = sorted_col[quantile_positions]
    
            # Evaluate thresholds
            for thr in candidate_thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
    
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
    
                y_left = y[left_mask]
                y_right = y[right_mask]
    
                H_left = self._entropy(y_left)
                H_right = self._entropy(y_right)
                w_left = len(y_left) / n_samples
                w_right = 1 - w_left
    
                weighted_entropy = w_left * H_left + w_right * H_right
                gain = parent_entropy - weighted_entropy
    
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = thr
                    best_left = (X[left_mask], y_left)
                    best_right = (X[right_mask], y_right)
    
        return best_gain, best_feature, best_threshold, best_left, best_right

    
    def print_tree(self, feature_names):
        if self.tree is None:
            print("Tree has not been fit yet.")
            return
        
        self._print_tree_recursive(self.tree, feature_names, depth=0)
        
    def _print_tree_recursive(self, node, feature_names, depth):
        indent = "    " * depth  # 4 spaces per depth level
        
        # Leaf node
        if node.value is not None:
            print(f"{indent}→ Predict: {node.value}")
            return
    
        # Internal node
        feature_name = feature_names[node.feature]
        threshold = node.threshold
        
        print(f"{indent}● If {feature_name} ≤ {threshold:.4f}:")
        self._print_tree_recursive(node.left, feature_names, depth + 1)
        
        print(f"{indent}● Else ({feature_name} > {threshold:.4f}):")
        self._print_tree_recursive(node.right, feature_names, depth + 1)
