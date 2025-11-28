import numpy as np
# top down, greedy algorithm, using recursive binary splitting
# start with the predictor that leads to the greatest reduction in gini index
    # gini index is a measure of variance across classes
    # or we could also use entropy
# next, we split either of the two new resulting spaces
# repeat until we get to a predefined stopping criteria
# for this we will say a terminal node needs a minimum number of observations

# after the basic implementation, add some form of pruning
# for this, we will use cost complexity pruning as a function of alpha
# k fold cross validation to choose alpha
class DecisionTree:
    
    
    def __init__(self):
        pass
        
    def predict(self):
        pass
        
    # handles building out the tree itself
    def fit(self, X, y):
        pass
    
    def _entropy(self, y):
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        
        H = (-probs * np.log2(probs)).sum()
        return H
        
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_entropy = self._entropy(y)
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None
        
        for feature in range(n_features):
            sorted_idxs = np.argsort(X[:, feature])
            sorted_features = X[sorted_idxs, feature]
            sorted_labels = y[sorted_idxs]
            
            candidate_thresholds = []
            # go through the list of sorted features to obtain candidate thresholds
            for i in range(len(sorted_features) - 1):
                if sorted_labels[i] != sorted_labels[i+1]:
                    candidate_thresholds.append((sorted_features[i] + sorted_features[i+1]) / 2)
            
            # skip if we have a homogeneous split
            if not candidate_thresholds:
                continue
                
            # loop through candidate thresholds to see if they beat our current bests
            for threshold in candidate_thresholds:
                # split into two
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                
                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[right_mask], y[right_mask]
                
                # entropy calc doesn't work if one of the sides are empty (entropy divide by zero)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                H_left = self._entropy(y_left)
                H_right = self._entropy(y_right)
                w_left, w_right = len(y_left) / n_samples, len(y_right) / n_samples
                weighted_entropy = w_left * H_left + w_right * H_right
                gain = parent_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left = (X_left, y_left)
                    best_right = (X_right, y_right)
                
        return best_feature, best_threshold, best_left, best_right

