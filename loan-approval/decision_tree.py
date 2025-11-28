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
    
    def _entropy(y) -> int:
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        
        H = (-probs * np.log2(probs)).sum()
        return H
        
    def _best_split(X, y):
        pass

