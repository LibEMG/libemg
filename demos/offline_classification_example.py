import os
import sys
from os import walk
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.emg_classifier import EMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import get_windows
from unb_emg_toolbox.utils import create_folder_dictionary
from unb_emg_toolbox.data_handler import OfflineDataHandler


# Currently this file is for only one individual
if __name__ == "__main__" :
    # Process and Read Dataset:
    training_folder = 'demos/data/myo_dataset/training/'
    testing_folder = 'demos/data/myo_dataset/testing/'
    train_dic = create_folder_dictionary(left_bound="C_", right_bound="_E", class_values=["0","1","2","3","4"], rep_values=["0"])
    test_dic = create_folder_dictionary(left_bound="C_", right_bound="_E", class_values=["0","1","2","3","4"], rep_values=["0","1","2","3"])
    odh = OfflineDataHandler(5, train_folder=training_folder, train_dic=train_dic, test_folder=testing_folder, test_dic=test_dic)
    odh.get_data()
    odh.parse_windows(50, 25)

    # Extract features from data set
    fe = FeatureExtractor(num_channels=8)
    training_features = fe.extract_feature_group('TD4', odh.training_windows)
    testing_features = fe.extract_feature_group('TD4', odh.testing_windows)

    # Create data set dictionary 
    data_set = {}
    data_set['testing_features'] = testing_features
    data_set['training_features'] = training_features
    data_set['testing_labels'] = odh.testing_labels
    data_set['training_labels'] = odh.training_labels
    data_set['null_label'] = 2

    # Create Classifier and Evaluate
    args = {'n_neighbors': 5}
    classifier = EMGClassifier("QDA", data_set, arguments=args)
    offline_metrics = classifier.offline_evaluation()
    print("Offline Metrics:")
    print(offline_metrics)
