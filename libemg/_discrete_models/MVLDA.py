from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats

"""
A simple majority vote classifier using LDA. 
"""
class MVLDA:
    def __init__(self):
        self.model = None  

    def fit(self, x, y):
        self.model = LinearDiscriminantAnalysis()
        labels = np.hstack([[v] * x[i].shape[0] for i, v in enumerate(y)])
        self.model.fit(np.vstack(x), labels)
    
    def predict(self, y):
        preds = []
        for s in y:
            frame_predictions = self.model.predict(s)
            # Use scipy.stats.mode which handles ties and returns the smallest value
            majority_vote = stats.mode(frame_predictions, keepdims=False)[0]
            preds.append(majority_vote)
        return np.array(preds)