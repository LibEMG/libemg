from tslearn.metrics import dtw_path
import numpy as np


class DTWClassifier:
    """Dynamic Time Warping k-Nearest Neighbors classifier.

    A classifier that uses Dynamic Time Warping (DTW) distance for template
    matching with k-nearest neighbors. Suitable for discrete gesture recognition
    where temporal alignment between samples varies.

    Parameters
    ----------
    n_neighbors: int, default=1
        Number of neighbors to use for k-nearest neighbors voting.

    Attributes
    ----------
    templates: list of ndarray
        The training templates stored after fitting.
    labels: ndarray
        The labels corresponding to each template.
    classes_: ndarray
        The unique class labels known to the classifier.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the DTW classifier.

        Parameters
        ----------
        n_neighbors: int, default=1
            Number of neighbors to use for k-nearest neighbors voting.
        """
        self.n_neighbors = n_neighbors
        self.templates = None
        self.labels = None
        self.classes_ = None

    def fit(self, features, labels):
        """Fit the DTW classifier by storing training templates.

        Parameters
        ----------
        features: list of ndarray
            A list of training samples (templates) where each sample is
            a 2D array of shape (n_frames, n_features).
        labels: array-like
            The target labels for each template.
        """
        self.templates = features
        self.labels = np.array(labels)
        self.classes_ = np.unique(labels)

    def predict(self, samples):
        """Predict class labels for samples.

        Parameters
        ----------
        samples: list of ndarray
            A list of samples to classify where each sample is a 2D array
            of shape (n_frames, n_features).

        Returns
        -------
        ndarray
            Predicted class labels for each sample.
        """
        # We can reuse predict_proba logic to get the class with highest probability
        probas = self.predict_proba(samples)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, samples, gamma=None, eps=1e-12):
        """Predict class probabilities using DTW distance-weighted voting.

        Computes DTW distances to all templates, selects k-nearest neighbors,
        and computes class probabilities using exponentially weighted voting.

        Parameters
        ----------
        samples: list of ndarray
            A list of samples to classify where each sample is a 2D array
            of shape (n_frames, n_features).
        gamma: float, default=None
            The kernel bandwidth for distance weighting. If None, automatically
            computed based on median neighbor distance.
        eps: float, default=1e-12
            Small constant to prevent division by zero.

        Returns
        -------
        ndarray
            Predicted class probabilities of shape (n_samples, n_classes).
        """
        if self.templates is None:
            raise ValueError("Call fit() before predict_proba().")

        X = np.asarray(samples, dtype=object)
        out = np.zeros((len(X), len(self.classes_)), dtype=float)

        for i, s in enumerate(X):
            # DTW distances to templates
            dists = np.array([dtw_path(t, s)[1] for t in self.templates], dtype=float)

            # kNN
            nn_idx = np.argsort(dists)[:self.n_neighbors]
            nn_dists = dists[nn_idx]
            nn_labels = self.labels[nn_idx]

            # choose gamma if not provided (scale to typical distance)
            g = gamma
            if g is None:
                scale = np.median(nn_dists) if len(nn_dists) else 1.0
                g = 1.0 / max(scale, eps)

            weights = np.exp(-g * nn_dists)  # closer -> bigger weight

            # accumulate per class
            for cls_j, cls in enumerate(self.classes_):
                out[i, cls_j] = weights[nn_labels == cls].sum()

            # normalize to probabilities
            z = out[i].sum()
            out[i] = out[i] / max(z, eps)

        return out
