import numpy as np
from sklearn.metrics import *
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