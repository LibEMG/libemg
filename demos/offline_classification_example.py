import os
import sys
from os import walk
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.emg_classifier import EMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import get_windows
from unb_emg_toolbox.utils import create_folder_dictionary, _make_regex
from unb_emg_toolbox.data_handler import OfflineDataHandler


# Currently this file is for only one individual
if __name__ == "__main__" :
    # Process and Read Dataset:
    dataset_folder = 'demos/data/myo_dataset/training/'
    testing_folder = 'demos/data/myo_dataset/testing/'
    train_dic = create_folder_dictionary(left_bound="C_", right_bound="_E", class_values=["0","1","2","3","4"], rep_values=["0"])
    test_dic = create_folder_dictionary(left_bound="C_", right_bound="_E", class_values=["0","1","2","3","4"], rep_values=["0","1","2","3"])
    odh = OfflineDataHandler()
    # add the training data
    odh.get_data(dataset_folder=dataset_folder, dictionary=train_dic, delimiter=",")
    train_windows, train_metadata = odh.parse_windows(50,25)
    print(f"num train windows={train_windows.shape[0]}")
    # add the testing data
    odh = OfflineDataHandler()
    odh.get_data(dataset_folder=testing_folder, dictionary=test_dic, delimiter=",")
    test_windows, test_metadata = odh.parse_windows(50,25)
    print(f"num test windows={test_windows.shape[0]}")

    # Extract features from data set
    fe = FeatureExtractor(num_channels=8)
    training_features = fe.extract_feature_group('TD4', train_windows)
    testing_features = fe.extract_feature_group('TD4', test_windows)

    # Create data set dictionary 
    data_set = {}
    data_set['testing_features'] = testing_features
    data_set['training_features'] = training_features
    data_set['testing_labels'] = test_metadata['classes']
    data_set['training_labels'] = train_metadata['classes']
    data_set['null_label'] = 2

    # Create Classifier and Evaluate
    args = {'n_neighbors': 5}
    classifier = EMGClassifier("LDA", data_set, arguments=args)
    offline_metrics = classifier.offline_evaluation()
    print("Offline Metrics:")
    print(offline_metrics)






    # example isolating data
    # you can probably see how this is used for things like k-fold cross-validation
    # we could have instead gotten the all the data in the folder (train and test at once)
    # Note: this contains another set of demo data, but because of our regex we won't collect it!

    # manual dictionary making (showing how you can add more than the default metadata [classes, reps])
    dataset_folder = 'demos/data'
    sets_values = ["training", "testing"]
    sets_regex = _make_regex(left_bound = "dataset\\\\", right_bound="\\\\", values = sets_values)
    classes_values = ["0","1","2","3","4"]
    classes_regex = _make_regex(left_bound = "_C_", right_bound="_EMG.csv", values = classes_values)
    reps_values = ["0","1","2","3"]
    reps_regex = _make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "sets": sets_values,
        "sets_regex": sets_regex,
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }
    odh = OfflineDataHandler()
    odh.get_data(dataset_folder=dataset_folder, dictionary = dic, delimiter=",")

    # values=[0] corresponds to training since the 0th element of sets_values is training
    train_odh = odh.isolate_data(key="sets", values=[0])
    train_windows, train_metadata = train_odh.parse_windows(50,25)
    test_odh = odh.isolate_data(key="sets", values=[1])
    test_windows, test_metadata = test_odh.parse_windows(50,25)

    print(f"num train windows={train_windows.shape[0]}")
    print(f"num test windows={test_windows.shape[0]}")
    