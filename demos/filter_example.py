import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler, OnlineDataHandler
from libemg.filtering import Filter
# from libemg.utils import mock_emg_stream
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from libemg.datasets import _3DCDataset
from libemg.emg_classifier import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter

from libemg.offline_metrics import OfflineMetrics

if __name__ == "__main__" :
    y_true = np.array([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3])
    y_preds = np.array([1,0,0,0,0,2,1,1,1,1,1,1,1,2,3,3,3,3,1,1,2,3,3,3])

    om = OfflineMetrics()

    # Get and extract all available metrics:
    metrics = om.get_available_metrics()
    offline_metrics = om.extract_offline_metrics(metrics=metrics, y_true=y_true, y_predictions=y_preds, null_label=2)
    om.visualize(offline_metrics)

    # Get and extract a subset of metrics:
    metrics = ['AER', 'CA', 'INS', 'CONF_MAT']
    offline_metrics = om.extract_offline_metrics(metrics=metrics, y_true=y_true, y_predictions=y_preds, null_label=2)
    om.visualize_conf_matrix(offline_metrics['CONF_MAT'])
    print(offline_metrics)