import numpy as np
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class OfflineMetrics:
    """
    Offline Metrics class is used for extracting offline performance metrics.
    """
    
    def _ignore_rejected(self, y_predictions, y_true):
        # ignore rejections
        valid_samples = y_predictions != -1
        y_predictions = y_predictions[valid_samples]
        y_true        = y_true[valid_samples]
        return y_predictions, y_true


    def get_common_metrics(self):
        """Gets a list of the common metrics used for assessing EMG performance.

        Returns
        ----------
        list
            A list of common metrics (CA, AER, and INS).
        """
        return [
            'CA',
            'AER',
            'INS'
        ]

    def get_available_metrics(self):
        """Gets a list of all available offline performance metrics.

        Returns
        ----------
        list
            A list of all available metrics.
        """
        return [
            'CA',
            'AER',
            'INS',
            'REJ_RATE',
            'CONF_MAT',
            'RECALL',
            'PREC',
            'F1',
            'R2',
            'MSE',
            'MAPE',
            'RMSE',
            'NRMSE',
            'MAE'
        ]
    
    def extract_common_metrics(self, y_true, y_predictions, null_label=None):
        """Extracts the common set of offline performance metrics (CA, AER, and INS).

        Parameters
        ----------
        y_true: numpy.ndarray 
            A list of the true labels associated with each prediction.
        y_predictions: numpy.ndarray
            A list of predicted outputs from a classifier.
        null_label: int (optional)
            A null label used for the AER metric - this should correspond to the label associated
            with the No Movement or null class. 

        Returns
        ----------
        dictionary
            A dictionary containing the metrics, where each metric is a key in the dictionary.
        self.extract_offline_metrics(self.get_common_metrics(), y_true, y_predictions, null_label)
        """
        return self.extract_offline_metrics(self.get_common_metrics(), y_true, y_predictions, null_label)
        
    def extract_offline_metrics(self, metrics, y_true, y_predictions, null_label=None):
        """Extracts a set of offline performance metrics.

        Parameters
        ----------
        metrics: list
            A list of the metrics to extract. A list of metrics can be found running the 
            get_available_metrics function.
        y_true: numpy.ndarray 
            A list of the true labels associated with each prediction.
        y_predictions: numpy.ndarray
            A list of predicted outputs from a classifier.
        null_label: int (optional)
            A null label used for the AER metric - this should correspond to the label associated
            with the No Movement or null class. 

        Returns
        ----------
        dictionary
            A dictionary containing the metrics, where each metric is a key in the dictionary.
        
        Examples
        ---------
        >>> y_true = np.array([1,1,1,2,2,2,3,3,3,4,4,4])
        >>> y_predictions = np.array([1,1,2,2,2,1,3,3,3,4,1,4])
        >>> metrics = ['CA', 'AER', 'INS', 'REJ_RATE', 'CONF_MAT', 'RECALL', 'PREC', 'F1']
        >>> om = OfflineMetrics()
        >>> computed_metrics = om.extract_offline_metrics(metrics, y_true, y_predictions, 2))
        """
        assert len(y_true) == len(y_predictions)
        og_y_preds = y_predictions.copy()
        is_classification = np.all(y_predictions.astype(int) == y_predictions) and np.all(y_true.astype(int) == y_true)
        if -1 in y_predictions and is_classification:
            # Only apply to classification data
            rm_idxs = np.where(y_predictions == -1)
            y_predictions = np.delete(y_predictions, rm_idxs)
            y_true = np.delete(y_true, rm_idxs)

        offline_metrics = {}            
        for metric in metrics:
            method_to_call = getattr(self, 'get_' + metric)
            if metric in ['AER']:
                if not null_label is None:
                    offline_metrics[metric] = method_to_call(y_true, y_predictions, null_label)
                else:
                    print("AER not computed... Please input the null_label parameter.")
            elif metric in ['REJ_RATE']:
                offline_metrics[metric] = method_to_call(og_y_preds)
            else:
                # Assume all other metrics have the signature (y_true, y_predictions)
                offline_metrics[metric] = method_to_call(y_true, y_predictions)
        return offline_metrics

    def get_CA(self, y_true, y_predictions):
        """Classification Accuracy.

        The number of correct predictions normalized by the total number of predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        float
            Returns the classification accuracy.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        if len(y_true) == 0:
            print("No test samples - check the rejection rate.")
            return 1.0
        return sum(y_predictions == y_true)/len(y_true)

    def get_AER(self, y_true, y_predictions, null_class):
        """Active Error.

        Classification accuracy on active classes (i.e., all classes but no movement/rest). Rejected samples are ignored.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.
        null_class: int
            The null class that shouldn't be considered.

        Returns
        ----------
        float
            Returns the active error.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        nm_predictions = [i for i, x in enumerate(y_predictions) if x == null_class]
        return 1 - self.get_CA(np.delete(y_true, nm_predictions), np.delete(y_predictions, nm_predictions))

    def get_INS(self, y_true, y_predictions):
        """Instability.

        The number of subsequent predicitons that change normalized by the total number of predicitons.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        float
            Returns the instability.
        """
        num_gt_changes = np.count_nonzero(y_true[:-1] != y_true[1:])
        pred_changes = np.count_nonzero(y_predictions[:-1] != y_predictions[1:])
        ins = (pred_changes - num_gt_changes) / len(y_predictions)
        return ins if ins > 0 else 0.0

    def get_REJ_RATE(self, y_predictions):
        """Rejection Rate.

        The number of rejected predictions, normalized by the total number of predictions.

        Parameters
        ----------
        y_predictions: numpy.ndarray
            A list of predicted labels. -1 in the list correspond to rejected predictions.

        Returns
        ----------
        float
            Returns the rejection rate.
        """
        return sum(y_predictions == -1)/len(y_predictions)
    
    def get_CONF_MAT(self, y_true, y_predictions):
        """Confusion Matrix.

        A NxN matric where N is the number of classes. Each column represents the predicted class and 
        each row represents the true class. 

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns the confusion matrix.
        """
        classes = np.sort(np.unique(y_true))
        conf_mat = np.zeros(shape=(len(classes), len(classes)))
        for row in range(0, len(classes)):
            c_true = np.where(y_true == classes[row])
            for col in range(0, len(classes)):
                conf_mat[row,col] = len(np.where(y_predictions[c_true] == classes[col])[0])
        return conf_mat

    def get_RECALL(self, y_true, y_predictions):
        """Recall Score.

        The recall is simply: True Positives / (True Positives + False Negatives).
        This metric takes into account the corresponding weights of each class.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the recall for each class.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        recall, weights = self._get_RECALL_helper(y_true, y_predictions)
        return np.average(recall, weights=weights)

    def _get_RECALL_helper(self, y_true, y_predictions):
        classes = np.sort(np.unique(y_true))
        recall = np.zeros(shape=(len(classes)))
        weights = np.zeros(shape=(len(classes)))
        for c in range(0, len(classes)):
            c_true = np.where(y_true == classes[c])
            tp = len(np.where(y_predictions[c_true] == classes[c])[0])
            fn = len(np.where(y_predictions[c_true] != classes[c])[0])
            weights[c] = len(c_true[0])/len(y_true)
            recall[c] = tp / (tp + fn)
        return recall, weights

    def get_PREC(self, y_true, y_predictions):
        """Precision Score.

        The precision is simply: True Positives / (True Positives + False Positive).
        This metric takes into account the corresponding weights of each class.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the precision for each class.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        precision, weights = self._get_PREC_helper(y_true, y_predictions)
        return np.average(precision, weights=weights)
    
    def _get_PREC_helper(self, y_true, y_predictions):
        classes = np.sort(np.unique(y_true))
        precision = np.zeros(shape=(len(classes)))
        weights = np.zeros(shape=(len(classes)))
        for c in range(0, len(classes)):
            c_true = np.where(y_true == classes[c])
            c_false = np.where(y_true != classes[c])
            tp = len(np.where(y_predictions[c_true] == classes[c])[0])
            fp = len(np.where(y_predictions[c_false] == classes[c])[0])
            weights[c] = len(c_true[0])/len(y_true)
            precision[c] = tp / (tp + fp)
        return precision, weights

    def get_F1(self, y_true, y_predictions):
        """F1 Score.

        The f1 score is simply: 2 * (Precision * Recall)/(Precision + Recall).
        This metric takes into account the corresponding weights of each class.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the f1 score for each class.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        prec, weights = self._get_PREC_helper(y_true, y_predictions)
        recall, _ = self._get_RECALL_helper(y_true, y_predictions)
        f1 = 2 * (prec * recall) / (prec + recall)
        return np.average(f1, weights=weights)  
    
    def get_R2(self, y_true, y_predictions):
        """R2 score.

        The R^2 score measures how well a regression model captures the variance in the predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the R2 score for each DOF.
        """
        ssr = np.sum((y_predictions - y_true) ** 2, axis=0)
        sst = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
        r2 = 1 - ssr/sst
        return r2
    
    def get_MSE(self, y_true, y_predictions):
        """Mean squared error.

        The MSE measures the averages squared errors between the true labels and predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the MSE score for each DOF.
        """
        values = (y_true - y_predictions) ** 2
        mse = np.sum(values, axis=0) / y_true.shape[0]
        return mse

    def get_MAPE(self, y_true, y_predictions):
        """Mean absolute percentage error.

        The MAPE measures the average error between the true labels and predictions as a percentage of the true value.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the MAPE score for each DOF.
        """
        values = np.abs((y_true - y_predictions) / np.maximum(np.abs(y_true), np.finfo(np.float64).eps))    # some values could be 0, so take epsilon if that's the case to avoid inf
        mape = np.sum(values, axis=0) / y_true.shape[0]
        return mape

    def get_RMSE(self, y_true, y_predictions):
        """Root mean square error.

        The RMSE measures the square root of the MSE.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the RMSE score for each DOF.
        """
        values = (y_true - y_predictions) ** 2
        mse = np.sum(values, axis=0) / y_true.shape[0]
        rmse = np.sqrt(mse)
        return rmse

    def get_NRMSE(self, y_true, y_predictions):
        """Normalized root mean square error.

        The NRMSE measures the RMSE normalized by the range of possible values.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the RMSE score for each DOF.
        """
        values = (y_true - y_predictions) ** 2
        mse = np.sum(values, axis=0) / y_true.shape[0]
        nrmse = np.sqrt(mse) / (y_true.max(axis=0) - y_true.min(axis=0))
        return nrmse

    def get_MAE(self, y_true, y_predictions):
        """Mean absolute error.

        The MAE measures the average L1 error between the true labels and predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the MAE score for each DOF.
        """
        residuals = np.abs(y_predictions - y_true)
        mae = np.mean(residuals, axis=0)
        return mae 
    
    def visualize(self, dic, y_axis=[0,1]):
        """Visualize the computed metrics in a bar chart.

        Parameters
        ----------
        dic: dict
            The output from the extract_offline_metrics function.
        y_axis: dict (optional), default=[0,1]
            A dictionary for lower and upper bounds of y axis. 
        """
        plt.style.use('ggplot')
        plt.title('Offline Metrics')
        plt.gca().set_ylim(y_axis)
        x = []
        y = []
        for key in dic:
            if key != 'CONF_MAT':
                x.append(key)
                y.append(dic[key])
        plt.bar(x,y)
        plt.show()
    
    def visualize_conf_matrix(self, mat, labels=None):
        """Visualize the 2D confusion matrix.

        Parameters
        ----------
        mat: list (2D)
            A NxN confusion matrix.
        """
        disp = ConfusionMatrixDisplay(confusion_matrix = mat, display_labels = labels)
        disp.plot()
        plt.show()


class UsabilityMetrics:
    """Training-data quality metrics for predicting myoelectric control usability.

    Based on: Nawfel et al., "A Multi-Variate Approach to Predicting Myoelectric Control
    Usability," IEEE Trans. Neural Syst. Rehabil. Eng., vol. 29, pp. 1312-1327, 2021.

    These metrics characterize the EMG feature space populated during training and have
    been shown to predict online usability better than classification accuracy alone.

    Input conventions
    -----------------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix. For most metrics, exclude the null/no-movement class from X and y
        before calling — all formulas define N as the number of *active* motion classes.
    y : np.ndarray, shape (n_samples,)
        Integer class labels corresponding to rows of X.
    reps : np.ndarray, shape (n_samples,)
        Repetition (trial) indices. Required by rep-dependent metrics (RI, mwRI, swRI,
        swSI, CD, mwSI, DS).
    null_class : int
        No-movement class label. Include null class in X/y when calling ACA and supply
        null_class so the formula can treat those predictions correctly.

    Notes
    -----
    Nine metrics from the paper (CDM, PU, NS, CE, C, rPU, rNS, rCE, rC) depend on
    external algorithms not fully specified in this paper and are stubbed as
    NotImplementedError.
    """

    _REG = 1e-6  # covariance regularisation to avoid singular matrices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _class_stats(self, X, y):
        """Returns dict: class_label -> (centroid, regularised covariance)."""
        D = X.shape[1]
        stats = {}
        for c in np.unique(y):
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            if Xc.shape[0] > 1:
                cov = np.cov(Xc.T)
                cov = np.atleast_2d(cov)
            else:
                cov = np.zeros((D, D))
            cov = cov + self._REG * np.eye(D)
            stats[c] = (mu, cov)
        return stats

    def _rep_centroids(self, X, y, reps):
        """Returns dict: (class, rep) -> centroid."""
        centroids = {}
        for c in np.unique(y):
            for r in np.unique(reps):
                mask = (y == c) & (reps == r)
                if mask.any():
                    centroids[(c, r)] = X[mask].mean(axis=0)
        return centroids

    def _rep_data(self, X, y, reps):
        """Returns dict: (class, rep) -> data array."""
        data = {}
        for c in np.unique(y):
            for r in np.unique(reps):
                mask = (y == c) & (reps == r)
                if mask.any():
                    data[(c, r)] = X[mask]
        return data

    @staticmethod
    def _mahal(diff, cov_inv):
        """0.5 * sqrt(diff^T cov_inv diff)."""
        return 0.5 * np.sqrt(float(diff @ cov_inv @ diff))

    def _rep_mwri_values(self, X, y, reps):
        """Per-rep mwRI contributions used by both get_mwRI and get_swRI."""
        classes = np.unique(y)
        rep_list = np.unique(reps)
        class_st = self._class_stats(X, y)
        rep_cent = self._rep_centroids(X, y, reps)
        rep_data = self._rep_data(X, y, reps)

        per_rep = []
        for r in rep_list:
            class_vals = []
            for c in classes:
                if (c, r) not in rep_data:
                    continue
                Xcr = rep_data[(c, r)]
                mu_r = rep_cent[(c, r)]
                cov_inv = np.linalg.inv(class_st[c][1])
                dists = [self._mahal(mu_r - x, cov_inv) for x in Xcr]
                class_vals.append(np.mean(dists))
            if class_vals:
                per_rep.append(np.mean(class_vals))
        return np.array(per_rep)

    def _trial_mwsi_values(self, X, y, reps):
        """Per-trial mwSI contributions used by both get_mwSI and get_swSI."""
        D = X.shape[1]
        per_trial = []
        for r in np.unique(reps):
            mask = reps == r
            Xt, yt = X[mask], y[mask]
            classes_t = np.unique(yt)
            if len(classes_t) < 2:
                continue
            stats_t = {}
            for c in classes_t:
                Xc = Xt[yt == c]
                mu = Xc.mean(axis=0)
                cov = np.atleast_2d(np.cov(Xc.T)) if Xc.shape[0] > 1 else np.zeros((D, D))
                cov = cov + self._REG * np.eye(D)
                stats_t[c] = (mu, np.linalg.inv(cov))
            vals = []
            for c in classes_t:
                mu_j, inv_j = stats_t[c]
                dists = [self._mahal(mu_j - stats_t[c2][0], inv_j)
                         for c2 in classes_t if c2 != c]
                vals.append(min(dists))
            per_trial.append(np.mean(vals))
        return np.array(per_trial)

    # ------------------------------------------------------------------
    # Variability metrics
    # ------------------------------------------------------------------

    def get_RI(self, X, y, reps):
        """Repeatability Index.

        Measures the reproducibility of EMG patterns between repetitions. Lower values
        indicate more consistent feature-space representations across repetitions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (active classes only).
        y : np.ndarray, shape (n_samples,)
            Class labels.
        reps : np.ndarray, shape (n_samples,)
            Repetition indices.

        Returns
        -------
        float
        """
        classes = np.unique(y)
        rep_list = np.unique(reps)
        class_st = self._class_stats(X, y)
        rep_cent = self._rep_centroids(X, y, reps)

        total = 0.0
        for c in classes:
            mu_tr, cov_tr = class_st[c]
            cov_inv = np.linalg.inv(cov_tr)
            dists = [self._mahal(mu_tr - rep_cent[(c, r)], cov_inv)
                     for r in rep_list if (c, r) in rep_cent]
            if dists:
                total += np.mean(dists)
        return total / len(classes)

    def get_mwRI(self, X, y, reps):
        """Mean Within-Repetition Repeatability Index.

        Mean distance from each data point to its per-repetition class centroid,
        measured in Mahalanobis terms using the global class covariance.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        reps : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        vals = self._rep_mwri_values(X, y, reps)
        return float(np.mean(vals)) if len(vals) else 0.0

    def get_swRI(self, X, y, reps):
        """Standard Deviation Within-Repetition Repeatability Index.

        Variation of the within-repetition repeatability across repetitions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        reps : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        vals = self._rep_mwri_values(X, y, reps)
        if len(vals) < 2:
            return 0.0
        mwri = np.mean(vals)
        return float(np.sqrt(np.sum((vals - mwri) ** 2) / (len(vals) - 1)))

    def get_swSI(self, X, y, reps):
        """Standard Deviation Within-Trial Separability Index.

        Variability of the distinguishability of EMG patterns across trials.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        reps : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        vals = self._trial_mwsi_values(X, y, reps)
        if len(vals) < 2:
            return 0.0
        mwsi = np.mean(vals)
        return float(np.sqrt(np.sum((vals - mwsi) ** 2) / (len(vals) - 1)))

    def get_MSA(self, X, y):
        """Mean Semi-Principal Axes.

        Quantifies the average size of each class's training ellipsoid using the
        geometric mean of its semi-principal axes (square roots of covariance eigenvalues).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        D = X.shape[1]
        total = 0.0
        for c in classes:
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                continue
            cov = np.atleast_2d(np.cov(Xc.T))
            eigvals = np.linalg.eigvalsh(cov)
            axes = np.sqrt(np.maximum(eigvals, 0.0))
            geom_mean = np.prod(axes) ** (1.0 / D)
            total += geom_mean
        return total / len(classes)

    def get_CD(self, X, y, reps):
        """Centroid Drift.

        Sum of Euclidean distances between successive per-repetition centroids,
        averaged over classes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        reps : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        rep_list = np.sort(np.unique(reps))
        rep_cent = self._rep_centroids(X, y, reps)

        total = 0.0
        for c in classes:
            prev = None
            drift = 0.0
            for r in rep_list:
                if (c, r) not in rep_cent:
                    continue
                if prev is not None:
                    drift += np.linalg.norm(rep_cent[(c, r)] - rep_cent[(c, prev)])
                prev = r
            total += drift
        return total / len(classes)

    def get_MAV(self, X, y):
        """Mean Absolute Value.

        Average amplitude of the feature data per class, averaged over classes.
        When X contains raw EMG windows, this is the classical MAV; when X is a
        feature matrix it provides a magnitude measure of the feature space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        return float(np.mean([np.mean(np.abs(X[y == c])) for c in classes]))

    # ------------------------------------------------------------------
    # Separability metrics
    # ------------------------------------------------------------------

    def get_SI(self, X, y):
        """Separability Index.

        Minimum Mahalanobis distance (using each class's own covariance) to the nearest
        other class centroid, averaged over classes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        class_st = self._class_stats(X, y)
        total = 0.0
        for c in classes:
            mu_j, cov_j = class_st[c]
            inv_j = np.linalg.inv(cov_j)
            dists = [self._mahal(mu_j - class_st[c2][0], inv_j)
                     for c2 in classes if c2 != c]
            total += min(dists)
        return total / len(classes)

    def get_mSI(self, X, y):
        """Modified Separability Index.

        Like SI but uses the average of the two class covariances (pooled) instead of
        only the source class covariance.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        class_st = self._class_stats(X, y)
        total = 0.0
        for c in classes:
            mu_j, cov_j = class_st[c]
            best = np.inf
            for c2 in classes:
                if c2 == c:
                    continue
                mu_i, cov_i = class_st[c2]
                S = (cov_j + cov_i) / 2.0
                d = self._mahal(mu_j - mu_i, np.linalg.inv(S))
                best = min(best, d)
            total += best
        return total / len(classes)

    def get_mwSI(self, X, y, reps):
        """Mean Within-Trial Separability Index.

        Distinguishability of EMG patterns computed per trial and averaged.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        reps : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        vals = self._trial_mwsi_values(X, y, reps)
        return float(np.mean(vals)) if len(vals) else 0.0

    def get_BD(self, X, y):
        """Bhattacharyya Distance.

        Statistical similarity between class distributions. Higher values indicate
        better separation. Averaged over classes using each class's nearest neighbour.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        class_st = self._class_stats(X, y)
        total = 0.0
        for c in classes:
            mu_j, cov_j = class_st[c]
            _, ld_j = np.linalg.slogdet(cov_j)
            best = np.inf
            for c2 in classes:
                if c2 == c:
                    continue
                mu_i, cov_i = class_st[c2]
                S = (cov_j + cov_i) / 2.0
                S_inv = np.linalg.inv(S)
                _, ld_S = np.linalg.slogdet(S)
                _, ld_i = np.linalg.slogdet(cov_i)
                diff = mu_j - mu_i
                term1 = (1.0 / 8.0) * float(diff @ S_inv @ diff)
                term2 = -0.5 * (ld_S - 0.5 * (ld_j + ld_i))
                best = min(best, term1 + term2)
            total += best
        return total / len(classes)

    def get_KLD(self, X, y):
        """Kullback-Leibler Divergence.

        Asymmetric KL divergence from class j to its nearest class. Averaged over classes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        D = X.shape[1]
        class_st = self._class_stats(X, y)
        total = 0.0
        for c in classes:
            mu_j, cov_j = class_st[c]
            inv_j = np.linalg.inv(cov_j)
            _, ld_j = np.linalg.slogdet(cov_j)
            best = np.inf
            for c2 in classes:
                if c2 == c:
                    continue
                mu_i, cov_i = class_st[c2]
                _, ld_i = np.linalg.slogdet(cov_i)
                diff = mu_j - mu_i
                trace_term = np.trace(inv_j @ cov_i)
                quad_term = float(diff @ inv_j @ diff)
                kld = 0.5 * (trace_term + quad_term - D + (ld_j - ld_i))
                best = min(best, kld)
            total += best
        return total / len(classes)

    def get_HD(self, X, y):
        """Hellinger Distance.

        Quantifies similarity between class distributions. Values in [0, 1] where 0
        means identical distributions. Averaged over classes using each nearest neighbour.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        class_st = self._class_stats(X, y)
        total = 0.0
        for c in classes:
            mu_j, cov_j = class_st[c]
            _, ld_j = np.linalg.slogdet(cov_j)
            best = np.inf
            for c2 in classes:
                if c2 == c:
                    continue
                mu_i, cov_i = class_st[c2]
                _, ld_i = np.linalg.slogdet(cov_i)
                S = (cov_j + cov_i) / 2.0
                S_inv = np.linalg.inv(S)
                _, ld_S = np.linalg.slogdet(S)
                diff = mu_j - mu_i
                log_amp = 0.25 * ld_i + 0.25 * ld_j - 0.5 * ld_S
                amp = np.exp(np.clip(log_amp, -500, 500))
                exp_val = np.exp(-0.125 * float(diff @ S_inv @ diff))
                hd = 1.0 - amp * exp_val
                best = min(best, hd)
            total += best
        return total / len(classes)

    def get_VOR(self, X, y):
        """Volume of Overlap Region.

        Fraction of the combined bounding-box volume shared between the most overlapping
        class pair (per class), as a product over feature dimensions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        bounds = {c: (X[y == c].min(axis=0), X[y == c].max(axis=0)) for c in classes}

        total = 0.0
        for c in classes:
            lo_j, hi_j = bounds[c]
            best = -np.inf
            for c2 in classes:
                if c2 == c:
                    continue
                lo_i, hi_i = bounds[c2]
                overlap_num = np.minimum(hi_j, hi_i) - np.maximum(lo_j, lo_i)
                total_range = np.maximum(hi_j, hi_i) - np.minimum(lo_j, lo_i)
                ratio = np.clip(overlap_num, 0.0, None) / np.maximum(total_range, 1e-12)
                vor = float(np.prod(ratio))
                best = max(best, vor)
            total += best
        return total / len(classes)

    def get_FE(self, X, y):
        """Feature Efficiency.

        Fraction of points separable by the single best feature dimension, for the most
        separable class pair. Averaged over classes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        total = 0.0
        for c in classes:
            Xc = X[y == c]
            n_j = len(Xc)
            best = -np.inf
            for c2 in classes:
                if c2 == c:
                    continue
                Xi = X[y == c2]
                n_i = len(Xi)
                denom = n_j + n_i
                best_k = -np.inf
                for k in range(X.shape[1]):
                    lo = max(Xc[:, k].min(), Xi[:, k].min())
                    hi = min(Xc[:, k].max(), Xi[:, k].max())
                    if hi <= lo:
                        n_sk = 0
                    else:
                        in_j = np.sum((Xc[:, k] >= lo) & (Xc[:, k] <= hi))
                        in_i = np.sum((Xi[:, k] >= lo) & (Xi[:, k] <= hi))
                        n_sk = int(in_j + in_i)
                    best_k = max(best_k, (denom - n_sk) / denom)
                best = max(best, best_k)
            total += best
        return total / len(classes)

    def get_TSM(self, X, y):
        """Trace of Within-Class and Between-Class Scatter Matrices.

        Tr(S_w^{-1} S_b). Higher values indicate better class discriminability.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        N = len(classes)
        D = X.shape[1]
        mu_global = X.mean(axis=0)

        Sw = np.zeros((D, D))
        Sb = np.zeros((D, D))
        for c in classes:
            Xc = X[y == c]
            mu_c = Xc.mean(axis=0)
            diff_w = Xc - mu_c
            Sw += diff_w.T @ diff_w
            diff_b = (mu_c - mu_global).reshape(-1, 1)
            Sb += len(Xc) * (diff_b @ diff_b.T)
        Sw = Sw / N + self._REG * np.eye(D)
        Sb = Sb / N
        return float(np.trace(np.linalg.inv(Sw) @ Sb))

    def get_FDR(self, X, y):
        """Fisher's Discriminant Ratio.

        Minimum inter-class discriminability using the pooled within-class covariance,
        averaged over classes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        classes = np.unique(y)
        N = len(classes)
        D = X.shape[1]

        Sw = np.zeros((D, D))
        for c in classes:
            Xc = X[y == c]
            diff = Xc - Xc.mean(axis=0)
            Sw += diff.T @ diff / len(Xc)
        Sw = Sw / N + self._REG * np.eye(D)
        Sw_inv = np.linalg.inv(Sw)

        class_st = self._class_stats(X, y)
        total = 0.0
        for c in classes:
            mu_j = class_st[c][0]
            dists = [float((mu_j - class_st[c2][0]) @ Sw_inv @ (mu_j - class_st[c2][0]))
                     for c2 in classes if c2 != c]
            total += min(dists)
        return total / N

    def get_DS(self, X, y, reps):
        """Desirability Score.

        DS = SI / (RI * MSA). Combines separability, repeatability, and class size into
        a single quality score. Higher is better.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        reps : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        si = self.get_SI(X, y)
        ri = self.get_RI(X, y, reps)
        msa = self.get_MSA(X, y)
        denom = ri * msa
        return si / denom if denom != 0.0 else np.inf

    # ------------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------------

    def get_CA(self, y_true, y_predictions):
        """Classification Accuracy (leave-one-trial-out).

        Fraction of predictions the classifier labelled correctly. In the paper this is
        computed via leave-one-trial-out cross-validation. Provide the cross-validated
        predictions and ground-truth labels.

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            Ground-truth class labels.
        y_predictions : np.ndarray, shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        float
        """
        return float(np.mean(y_predictions == y_true))

    def get_ACA(self, y_true, y_predictions, null_class):
        """Active Classification Accuracy.

        Fraction of predictions that are correct OR predicted as no-movement. This
        excludes misclassifications caused by the no-movement class from the error count.

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            Ground-truth class labels.
        y_predictions : np.ndarray, shape (n_samples,)
            Predicted class labels (leave-one-trial-out CV output recommended).
        null_class : int
            Label of the no-movement / null class.

        Returns
        -------
        float
        """
        correct = (y_predictions == y_true) | (y_predictions == null_class)
        return float(np.mean(correct))

    def get_UD(self, y_true, y_predictions):
        """Usable Data.

        Fraction of correctly classified windows over the entire training period
        (e.g. from an adaptive classifier). Equivalent to classification accuracy.

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
        y_predictions : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        return float(np.mean(y_predictions == y_true))

    # ------------------------------------------------------------------
    # Neighbourhood metrics
    # ------------------------------------------------------------------

    def get_ICF(self, X, y):
        """Inter-Class Fraction.

        Fraction of samples whose nearest neighbour belongs to a different class.
        Lower values indicate better class separation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        nn = NearestNeighbors(n_neighbors=2, algorithm='auto')
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        nn_labels = y[indices[:, 1]]  # index 0 is the sample itself
        return float(np.mean(nn_labels != y))

    def get_IIF(self, X, y):
        """Intra-Inter Fraction.

        Ratio of the average intra-class nearest-neighbour distance to the average
        inter-class nearest-neighbour distance. For each sample the nearest same-class
        neighbour and the nearest different-class neighbour are found independently.
        Lower values indicate better class separation (tight, well-separated clusters).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        n = len(X)
        idx = np.arange(n)
        intra_dists = []
        inter_dists = []
        for i in range(n):
            same = (y == y[i]) & (idx != i)
            diff = y != y[i]
            if same.any():
                intra_dists.append(np.min(np.linalg.norm(X[same] - X[i], axis=1)))
            if diff.any():
                inter_dists.append(np.min(np.linalg.norm(X[diff] - X[i], axis=1)))
        mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
        mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
        return mean_intra / mean_inter if mean_inter > 0.0 else np.inf

    # ------------------------------------------------------------------
    # Stubs — require external algorithms
    # ------------------------------------------------------------------

    def get_CDM(self, X, y):
        """Class Discriminability Measure (stub).

        Requires the adaptive partitioning algorithm from:
        Kohn et al. (1996), Pattern Recognit. 29(5):873-887.
        """
        raise NotImplementedError(
            "CDM requires the adaptive partitioning algorithm described in "
            "Kohn et al. (1996), Pattern Recognit. 29(5):873-887."
        )

    def get_PU(self, X, y):
        """Purity (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "PU requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    def get_NS(self, X, y):
        """Neighbourhood Separability (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "NS requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    def get_CE(self, X, y):
        """Collective Entropy (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "CE requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    def get_C(self, X, y):
        """Compactness (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "C requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    def get_rPU(self, X, y):
        """Rescaled Purity (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "rPU requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    def get_rNS(self, X, y):
        """Rescaled Neighbourhood Separability (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "rNS requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    def get_rCE(self, X, y):
        """Rescaled Collective Entropy (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "rCE requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    def get_rC(self, X, y):
        """Rescaled Compactness (stub). Requires the PRISM framework (Singh 2003)."""
        raise NotImplementedError(
            "rC requires the PRISM framework described in "
            "Singh (2003), Pattern Anal. Appl. 6(2):134-149."
        )

    # ------------------------------------------------------------------
    # Listing and extraction
    # ------------------------------------------------------------------

    def get_available_usability_metrics(self):
        """All 32 metric names from Nawfel et al. (2021).

        Returns
        -------
        list of str
            Metrics marked with (*) raise NotImplementedError — their formulas
            require algorithms from external papers.
        """
        return [
            # variability (7)
            'RI', 'mwRI', 'swRI', 'swSI', 'MSA', 'CD', 'MAV',
            # separability (11)
            'SI', 'mSI', 'mwSI', 'BD', 'KLD', 'HD', 'VOR', 'FE', 'TSM', 'FDR', 'DS',
            # classification (3)
            'CA', 'ACA', 'UD',
            # neighbourhood (2)
            'ICF', 'IIF',
            # stubs — require external algorithm implementations (9)
            'CDM', 'PU', 'NS', 'CE', 'C', 'rPU', 'rNS', 'rCE', 'rC',
        ]

    # Metrics that require `reps`
    _REPS_REQUIRED = frozenset({'RI', 'mwRI', 'swRI', 'swSI', 'CD', 'mwSI', 'DS'})
    # Metrics that use (y_true, y_pred) instead of (X, y)
    _CLASSIFICATION = frozenset({'CA', 'ACA', 'UD'})
    # ACA additionally needs null_class
    _NULL_REQUIRED = frozenset({'ACA'})

    def extract_usability_metrics(self, metrics, X, y, reps=None, null_class=None):
        """Extract a set of usability metrics.

        For feature-space metrics pass X (feature matrix) and y (class labels).
        Classification metrics ACA and UD expect X=y_true and y=y_predictions;
        call them directly via get_ACA / get_UD for clarity.

        Parameters
        ----------
        metrics : list of str
            Metric names to compute. See get_available_usability_metrics().
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (active classes only for most metrics).
        y : np.ndarray, shape (n_samples,)
            Class labels.
        reps : np.ndarray, shape (n_samples,), optional
            Repetition indices. Required for RI, mwRI, swRI, swSI, CD, mwSI, DS.
        null_class : int, optional
            No-movement class label. Required for ACA.

        Returns
        -------
        dict
            Mapping of metric name to computed value.

        Examples
        --------
        >>> um = UsabilityMetrics()
        >>> results = um.extract_usability_metrics(
        ...     ['SI', 'BD', 'FE', 'ICF'],
        ...     X_train, y_train
        ... )
        >>> results_with_reps = um.extract_usability_metrics(
        ...     ['RI', 'mwRI', 'CD', 'DS'],
        ...     X_train, y_train, reps=rep_labels
        ... )
        """
        results = {}
        for m in metrics:
            method = getattr(self, 'get_' + m)
            if m in self._NULL_REQUIRED:
                if null_class is None:
                    print(f"{m} not computed: null_class parameter is required.")
                    continue
                results[m] = method(X, y, null_class)
            elif m in self._REPS_REQUIRED:
                if reps is None:
                    raise ValueError(f"Metric '{m}' requires the reps parameter.")
                results[m] = method(X, y, reps)
            else:
                results[m] = method(X, y)
        return results