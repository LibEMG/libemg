import numpy as np

# TODO: Evan Review
def get_CA(y_true, y_predictions):
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

def get_AER(y_true, y_predictions, null_class):
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
    return 1 - get_CA(np.delete(y_true, nm_predictions), np.delete(y_predictions, nm_predictions))

def get_INS(y_true, y_predictions):
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

def get_REJ_RATE(y_predictions):
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

#TODO: Add additional metrics