from tslearn.metrics import dtw_path
import numpy as np

class DTWClassifier:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.templates = None
        self.labels = None
        self.classes_ = None

    def fit(self, features, labels):
        self.templates = features
        self.labels = np.array(labels)
        self.classes_ = np.unique(labels)

    def predict(self, samples):
        # We can reuse predict_proba logic to get the class with highest probability
        probas = self.predict_proba(samples)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, samples, gamma=None, eps=1e-12):
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
