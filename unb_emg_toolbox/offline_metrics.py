import numpy as np
from sklearn.metrics import *
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

class OfflineMetrics:
    """Offline Metrics class is used for extracting offline performance metrics.
    """

    def get_available_metrics(self):
        """Gets a list of all available offline performance metrics.

        Returns
        ----------
        array_like
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
            'F1'
        ]
        
    def extract_offline_metrics(self, metrics, y_true, y_predictions, null_label=None):
        """Extracts a set of offline performance metrics.

        Parameters
        ----------
        metrics: array_like
            A list of the metrics to extract. A list of metrics can be found running the 
            get_available_metrics function.
        y_true: array_like 
            A list of the true labels associated with each prediction.
        y_predictions: array_like
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
        if -1 in y_predictions:
            rm_idxs = np.where(y_predictions == -1)
            y_predictions = np.delete(y_predictions, rm_idxs)
            y_true = np.delete(y_true, rm_idxs)

        offline_metrics = {}            
        for metric in metrics:
            method_to_call = getattr(self, 'get_' + metric)
            if metric in ['CA', 'INS', 'CONF_MAT', 'RECALL', 'PREC', 'F1']:
                offline_metrics[metric] = method_to_call(y_true, y_predictions)
            elif metric in ['AER'] and not null_label is None:
                offline_metrics[metric] = method_to_call(y_true, y_predictions, null_label)
            elif metric in ['REJ_RATE']:
                offline_metrics[metric] = method_to_call(og_y_preds)
        return offline_metrics

    def get_CA(self, y_true, y_predictions):
        """Classification Accuracy.

        The number of correct predictions normalized by the total number of predictions.

        Parameters
        ----------
        y_true: array_like
            A list of ground truth labels.
        y_predictions: array_like
            A list of predicted labels.

        Returns
        ----------
        float
            Returns the classification accuracy.
        """
        return sum(y_predictions == y_true)/len(y_true)

    def get_AER(self, y_true, y_predictions, null_class):
        """Active Error.

        Classification accuracy without considering null_label (No Movement) predictions.

        Parameters
        ----------
        y_true: array_like
            A list of ground truth labels.
        y_predictions: array_like
            A list of predicted labels.
        null_class: int
            The null class that shouldn't be considered.

        Returns
        ----------
        float
            Returns the active error.
        """
        nm_predictions = [i for i, x in enumerate(y_predictions) if x == null_class]
        return 1 - self.get_CA(np.delete(y_true, nm_predictions), np.delete(y_predictions, nm_predictions))

    def get_INS(self, y_true, y_predictions):
        """Instability.

        The number of subsequent predicitons that change normalized by the total number of predicitons.

        Parameters
        ----------
        y_true: array_like
            A list of ground truth labels.
        y_predictions: array_like
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
        y_predictions: array_like
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
        y_true: array_like
            A list of ground truth labels.
        y_predictions: array_like
            A list of predicted labels.

        Returns
        ----------
        array_like
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
        y_true: array_like
            A list of ground truth labels.
        y_predictions: array_like
            A list of predicted labels.

        Returns
        ----------
        array_like
            Returns a list consisting of the recall for each class.
        """
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
        y_true: array_like
            A list of ground truth labels.
        y_predictions: array_like
            A list of predicted labels.

        Returns
        ----------
        array_like
            Returns a list consisting of the precision for each class.
        """
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
        y_true: array_like
            A list of ground truth labels.
        y_predictions: array_like
            A list of predicted labels.

        Returns
        ----------
        array_like
            Returns a list consisting of the f1 score for each class.
        """
        prec, weights = self._get_PREC_helper(y_true, y_predictions)
        recall, _ = self._get_RECALL_helper(y_true, y_predictions)
        f1 = 2 * (prec * recall) / (prec + recall)
        return np.average(f1, weights=weights)  
    
    def getRELIAB():
        #TODO: Evan - I am going to leave this for you since I am 100% sure what you want.
        pass
    
    def visualize(self, dic):
        """Visualize the computed metrics in a bar chart.

        Parameters
        ----------
        dic: dictionary
            The output from the extract_offline_metrics function.
        """
        plt.style.use('ggplot')
        plt.title('Offline Metrics')
        x = []
        y = []
        for key in dic:
            if key != 'CONF_MAT':
                x.append(key)
                y.append(dic[key])
        plt.bar(x,y)
        plt.show()
    
    def visualize_conf_matrix(self, mat):
        """Visualize the 2D confusion matrix.

        Parameters
        ----------
        mat: array_like (2D)
            A NxN confusion matrix.
        """
        plt.style.use('ggplot')
        _, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(mat, alpha=0.3)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(x=j, y=i,s=int(mat[i, j]), va='center', ha='center', size='x-large')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

