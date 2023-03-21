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
    dataset = _3DCDataset()

    odh = dataset.prepare_data(OfflineDataHandler, subjects_values=["1"])

    fi = Filter(sampling_frequency=1000)
    fi.install_common_filters()
    fi.install_filters({"name":"notch",
                            "cutoff": 60,
                            "bandwidth": 3})
    #odh = fi.filter(odh)


    # get a bit of data -- just for making a figure

    data = odh.data[31][:,1].reshape(-1,1)
    fi.visualize_affect(data)
    fi.visualize_filters()

    A = 1
