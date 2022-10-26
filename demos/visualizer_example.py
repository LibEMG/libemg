
import os
import sys
import socket
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.emg_classifier import EMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import get_windows, make_regex, mock_emg_stream
from unb_emg_toolbox.data_handler import OfflineDataHandler, OnlineDataHandler
        

if __name__ == "__main__" :
    # manual dictionary making (showing how you can add more than the default metadata [classes, reps])
    dataset_folder = 'demos/data/myo_dataset'
    sets_values = ["training", "testing"]
    sets_regex = make_regex(left_bound = "dataset\\\\", right_bound="\\\\", values = sets_values)
    classes_values = ["0","1","2","3","4"]
    classes_regex = make_regex(left_bound = "_C_", right_bound="_EMG.csv", values = classes_values)
    reps_values = ["0","1","2","3"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
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

    fe = FeatureExtractor(num_channels=8, feature_group="HTD")

    data_set = {}
    data_set['training_windows'] = train_windows # used for velocity control
    data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
    data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
    data_set['testing_labels'] = test_metadata['classes']
    data_set['training_labels'] = train_metadata['classes']
    data_set['null_label'] = 2

    classifier = EMGClassifier("SVM", data_set.copy())
    classifier.run()
    classifier.visualize()

    data = np.loadtxt("demos/data/myo_dataset/training/R_0_C_0_EMG.csv", delimiter=",")
    windows = get_windows(data, 50, 25)
    features = fe.extract_feature_group('HTD', windows)
    fe.visualize(features)

    odh = OnlineDataHandler(emg_arr=True)
    odh.get_data()
    mock_emg_stream("demos/data/stream_data.csv", 8, sampling_rate=100)
    odh.visualize(num_samples=200)

    # plot_pca(classifier.data_set['testing_features'], data_set['testing_labels'])
    
    # data = np.loadtxt("demos/data/myo_dataset/training/R_0_C_0_EMG.csv", delimiter=",")
    # plot_raw_emg(data, channels=[1,2,3,4,5])
    # windows = get_windows(data, 50, 25)
    # plot_features(windows, ['MAV', 'ZC', 'SSC', 'WL'], 8)

    # p = multiprocessing.Process(target=worker, daemon=True)
    # p.start()
    # odh = OnlineDataHandler(emg_arr=True)
    # odh.get_data()
    # classifier = OnlineEMGClassifier(model="LDA", data_set=data_set, window_size=50, window_increment=25, 
    #         online_data_handler=odh, feature_extractor=fe, std_out=True)
    # classifier.run(block=False)
    # plot_live_decisions()

