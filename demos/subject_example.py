import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.datasets import OneSubjectMyoDataset
from libemg.emg_classifier import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter

if __name__ == "__main__":
    dataset = OneSubjectMyoDataset(save_dir="demos/data/", redownload=False)
    dataset.print_info()
    # take the downloaded dataset and load it as an offlinedatahandler
    odh = dataset.prepare_data(format=OfflineDataHandler)
    new_odh = odh.isolate_channels([0,1,2,3,4,5])
    print("Here")