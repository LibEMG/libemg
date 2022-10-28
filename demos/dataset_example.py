import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.emg_classifier import EMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.data_handler import OfflineDataHandler
from unb_emg_toolbox.dataset import _3DCDataset
from unb_emg_toolbox.offline_metrics import OfflineMetrics
from unb_emg_toolbox.filtering import Filter




def main():
    # get the 3DC Dataset using toolbox handle - this downloads the dataset
    dataset = _3DCDataset(save_dir='example_data',
                          redownload=False)
    # take the downloaded dataset and load it as an offlinedatahandler
    odh = dataset.prepare_data(format=OfflineDataHandler)



    # Perform an analysis where we test all the features available to the toolbox individually for within-subject
    # classification accuracy.

    # setup out model type and output metrics
    model = "LDA"
    om = OfflineMetrics()
    metrics = ['CA']

    # get the subject list
    subject_list = np.unique(odh.subjects)
    
    # initialize our feature extractor
    fe = FeatureExtractor(num_channels=10)
    feature_list = fe.get_feature_list()

    # get the variable ready for where we save the results
    results = np.zeros((len(feature_list), len(subject_list)))


    for s in subject_list:
        subject_data = odh.isolate_data("subjects",[s])
        subject_train = subject_data.isolate_data("sets",[0])
        subject_test  = subject_data.isolate_data("sets",[1])

        # apply a standardization on the raw data (x - mean)/std
        filter = Filter(sampling_frequency=1000)
        filter_dic = {
            "name": "standardize",
            "data": subject_train
        }
        filter.install_filters(filter_dic)
        filter.filter(subject_train)
        filter.filter(subject_test)

        # from the standardized data, perform windowing
        train_windows, train_metadata = subject_train.parse_windows(200,100)
        test_windows, test_metadata = subject_test.parse_windows(200,100)

        # for each feature in the feature list
        for i, f in enumerate(feature_list):
            train_features = fe.extract_features([f], train_windows)
            test_features  = fe.extract_features([f], test_windows)

            # check there are no invalid feature values
            # print(f"S{s} training features check")
            # print("-"*25)
            # fe.check_features(train_features)
            # print(f"S{s} testing features check")
            # print("-"*25)
            # fe.check_features(test_features)

            # get the dataset ready for the classifier
            data_set = {}
            data_set['testing_features'] = test_features
            data_set['training_features'] = train_features
            data_set['testing_labels'] = test_metadata["classes"]
            data_set['training_labels'] = train_metadata["classes"]
            # setup the classifier
            classifier = EMGClassifier(model, data_set.copy())

            # running the classifier analyzes the test data we already passed it
            preds = classifier.run()
            # get the CA: classification accuracy offline metric and add it to the results
            results[i,s] = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds)[metrics[0]] * 100
            # print(f"S{s} {f}: {results[i,s]}%")
    # the feature accuracy is represented by the mean accuracy across the subjects
    mean_feature_accuracy = results.mean(axis=1)
    std_feature_accuracy  = results.std(axis=1)


    plt.bar(feature_list, mean_feature_accuracy, yerr=std_feature_accuracy)
    plt.grid()
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.show()



if __name__ == "__main__":
    main()