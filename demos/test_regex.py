import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.datasets import OneSubjectMyoDataset
from libemg.emg_classifier import EMGClassifier
from libemg.filtering import Filter
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler
from libemg.offline_metrics import OfflineMetrics

import numpy as np

def main():
    # get only subject 1 from 3dcdataset
    # this shouldnt get 11
    subject_values = ["1"]
    # odh = OfflineDataHandler()
    # subject_values=["1"]
    # subject_regex = make_regex(left_bound="Participant", right_bound="/t", values=subject_values)
    # sets_values = ["train","test"]
    # sets_regex = make_regex(left_bound="/", right_bound="/EMG", values=sets_values)
    # reps_values = ["0","1","2","3"]
    # reps_regex = make_regex(left_bound="gesture_", right_bound="_", values=reps_values)
    # classes_values = ["0","1","2","3","4","5","6","7","8","9","10"]
    # classes_regex = make_regex(left_bound="_", right_bound=".txt", values=classes_values)
    # dic = {
    #     "subject": subject_values,
    #     "subject_regex": subject_regex,
    #     "sets": sets_values,
    #     "sets_regex": sets_regex,
    #     "reps": reps_values,
    #     "reps_regex": reps_regex,
    #     "classes": classes_values,
    #     "classes_regex": classes_regex}
    # odh.get_data("example_data/_3DCDataset",
    #              filename_dic = dic,
    #              delimiter=",")
    dataset = OneSubjectMyoDataset(save_dir="example_data/", redownload=False)
    odh = dataset.prepare_data(OfflineDataHandler)
    # fi = Filter(sampling_frequency=200)
    # filter_dictionary={"name":"notch",
    #                     "cutoff": 60,
    #                     "bandwidth": 3 }
    # fi.install_filters(filter_dictionary=filter_dictionary)
    # fi.filter(odh)
    odh.active_threshold(num_std=3, nm_label=2, class_attribute="classes")
    
    print(np.unique(np.vstack(odh.subject)))

    windows, metadata = odh.parse_windows(50,10)
    A = 1


if __name__ == "__main__":
    main()