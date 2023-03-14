import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.offline_metrics import OfflineMetrics

if __name__ == "__main__" :
    y_true = np.array([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3])
    y_preds = np.array([1,0,0,0,0,2,1,1,1,1,1,1,1,2,3,3,3,3,1,1,2,3,3,3])

    om = OfflineMetrics()

    # Get and extract all available metrics:
    metrics = om.get_available_metrics()
    offline_metrics = om.extract_offline_metrics(metrics=metrics, y_true=y_true, y_predictions=y_preds, null_label=2)
    print(offline_metrics)
    om.visualize_conf_matrix(offline_metrics['CONF_MAT'], labels=["open","close","rest","test"])

    # Get and extract a subset of metrics:
    metrics = ['AER', 'CA', 'INS']
    offline_metrics = om.extract_offline_metrics(metrics=metrics, y_true=y_true, y_predictions=y_preds, null_label=2)
    print(offline_metrics)