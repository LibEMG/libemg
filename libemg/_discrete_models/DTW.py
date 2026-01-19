from tslearn.metrics import dtw_path
import numpy as np
from collections import Counter

"""
A temporal Dynamic Time Warping Classifier using tslearn dtw_path.
Supports K-Nearest Neighbors logic.
"""
class DTWClassifier:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.templates = None
        self.labels = None

    def fit(self, features, labels):
        self.templates = features
        self.labels = labels

    def predict(self, samples): 
        predictions = []
        for s in samples:
            distances = []
            for t in self.templates:
                # dtw_path returns (path, distance)
                _, dist = dtw_path(t, s)
                distances.append(dist)
            
            # Convert distances to a numpy array for easy indexing
            distances = np.array(distances)
            
            # Get the indices of the n_neighbors smallest distances
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get the labels for these neighbors
            neighbor_labels = self.labels[neighbor_indices]
            
            # Use Counter to find the most common label (majority vote)
            most_common = Counter(neighbor_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)