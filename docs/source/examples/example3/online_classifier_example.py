import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from unb_emg_toolbox.datasets import OneSubjectMyoDataset
from unb_emg_toolbox.data_handler import OfflineDataHandler, OnlineDataHandler
from unb_emg_toolbox.offline_metrics import OfflineMetrics
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.emg_classifier import EMGClassifier, OnlineEMGClassifier
from unb_emg_toolbox.utils import mock_emg_stream

if __name__ == "__main__":
    # setup variables
    window_size = 50
    increment_size = 25
    num_channels = 8

    # get the predefined one subject myo dataset
    dataset = OneSubjectMyoDataset(save_dir='example_data',
                          redownload=False)
    # take the downloaded dataset and load it as an offlinedatahandler
    odh = dataset.prepare_data(format=OfflineDataHandler)

    # split the data into training and testing
    train_data = odh.isolate_data("sets",[0])
    test_data  = odh.isolate_data("sets",[1])

    # from the standardized data, perform windowing
    train_windows, train_metadata = train_data.parse_windows(window_size,increment_size)
    test_windows, test_metadata = test_data.parse_windows(window_size,increment_size)

    # extract hudgin's time domain features 
    fe = FeatureExtractor(num_channels=8)
    train_features = fe.extract_feature_group('HTD', train_windows)
    test_features = fe.extract_feature_group('HTD', test_windows)

    # get the dataset ready for the classifier
    data_set = {}
    data_set['testing_features'] = test_features
    data_set['training_features'] = train_features
    data_set['testing_labels'] = test_metadata["classes"]
    data_set['training_labels'] = train_metadata["classes"]
    
    # offline metrics 
    om = OfflineMetrics()

    # iterate through four classifiers and select the most performant
    models = ['LDA', 'SVM', 'NB', 'RF']
    metrics = ['CA']
    accuracies = []
    for model in models:
        classifier = EMGClassifier(model, data_set.copy())
        preds = classifier.run()
        metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
        accuracies.append(metrics['CA'])

    # choose model with the best offline accuracy on the testing data
    model = models[accuracies.index(max(accuracies))]

    # run mock online classifier 
    mock_emg_stream("example_data/OneSubjectMyoDataset/stream/raw_emg.csv", num_channels=num_channels, sampling_rate=200)
    online_dh = OnlineDataHandler(emg_arr=True)
    online_dh.get_data()
    online_classifier = OnlineEMGClassifier(model=model, data_set=data_set.copy(), num_channels=num_channels, window_size=window_size, window_increment=25, online_data_handler=online_dh, features=fe.get_feature_groups()['HTD'], std_out=True)
    online_classifier.run(block=True)