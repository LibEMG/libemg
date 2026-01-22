from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats


class MVLDA:
    """Majority Vote Linear Discriminant Analysis classifier.

    A classifier that uses Linear Discriminant Analysis (LDA) on individual frames
    and aggregates predictions using majority voting. This is designed for discrete
    gesture recognition where each sample contains multiple frames.

    Attributes
    ----------
    model: LinearDiscriminantAnalysis
        The underlying LDA model.
    classes_: ndarray
        The class labels known to the classifier.
    """

    def __init__(self):
        """Initialize the MVLDA classifier."""
        self.model = None
        self.classes_ = None

    def fit(self, x, y):
        """Fit the MVLDA classifier on training data.

        Parameters
        ----------
        x: list of ndarray
            A list of samples where each sample is a 2D array of shape
            (n_frames, n_features). Each sample can have a different number of frames.
        y: array-like
            The target labels for each sample.
        """
        self.model = LinearDiscriminantAnalysis()
        # Create a flat array of labels corresponding to every frame in x
        labels = np.hstack([[v] * x[i].shape[0] for i, v in enumerate(y)])
        self.model.fit(np.vstack(x), labels)
        # Store classes for consistent probability mapping
        self.classes_ = self.model.classes_

    def predict(self, y):
        """Predict class labels using majority voting.

        Performs frame-level LDA predictions and returns the majority vote
        for each sample.

        Parameters
        ----------
        y: list of ndarray
            A list of samples where each sample is a 2D array of shape
            (n_frames, n_features).

        Returns
        -------
        ndarray
            Predicted class labels for each sample.
        """
        preds = []
        for s in y:
            frame_predictions = self.model.predict(s)
            # Majority vote on the labels
            majority_vote = stats.mode(frame_predictions, keepdims=False)[0]
            preds.append(majority_vote)
        return np.array(preds)

    def predict_proba(self, y):
        """Predict class probabilities using soft voting.

        Calculates probabilities by averaging the frame-level probabilities
        for each sample (soft voting).

        Parameters
        ----------
        y: list of ndarray
            A list of samples where each sample is a 2D array of shape
            (n_frames, n_features).

        Returns
        -------
        ndarray
            Predicted class probabilities of shape (n_samples, n_classes).
        """
        probas = []
        for s in y:
            # Get probabilities for each frame: shape (n_frames, n_classes)
            frame_probas = self.model.predict_proba(s)
            
            # Average probabilities across all frames in this sample
            sample_proba = np.mean(frame_probas, axis=0)
            probas.append(sample_proba)
            
        return np.array(probas)