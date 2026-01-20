from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats

"""
A simple majority vote classifier using LDA. 
"""
class MVLDA:
    def __init__(self):
        self.model = None  
        self.classes_ = None

    def fit(self, x, y):
        self.model = LinearDiscriminantAnalysis()
        # Create a flat array of labels corresponding to every frame in x
        labels = np.hstack([[v] * x[i].shape[0] for i, v in enumerate(y)])
        self.model.fit(np.vstack(x), labels)
        # Store classes for consistent probability mapping
        self.classes_ = self.model.classes_
    
    def predict(self, y):
        preds = []
        for s in y:
            frame_predictions = self.model.predict(s)
            # Majority vote on the labels
            majority_vote = stats.mode(frame_predictions, keepdims=False)[0]
            preds.append(majority_vote)
        return np.array(preds)

    def predict_proba(self, y):
        """
        Calculates probabilities by averaging the frame-level probabilities 
        for each sample (Soft Voting).
        """
        probas = []
        for s in y:
            # Get probabilities for each frame: shape (n_frames, n_classes)
            frame_probas = self.model.predict_proba(s)
            
            # Average probabilities across all frames in this sample
            sample_proba = np.mean(frame_probas, axis=0)
            probas.append(sample_proba)
            
        return np.array(probas)