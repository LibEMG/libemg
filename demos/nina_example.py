import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libemg.datasets import _3DCDataset, NinaDB1
from libemg.emg_classifier import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.data_handler import OfflineDataHandler
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter

if __name__ == "__main__" :
    nina_1 = NinaDB1(save_dir='example_data', redownload=False, subjects=[1,2])
    odh = nina_1.prepare_data(format=OfflineDataHandler)


    # setup out model type and output metrics
    model = "LDA"
    om = OfflineMetrics()
    metrics = ['CA']

    # get the subject list
    subject_list = np.unique(odh.subjects)
    classset_list = np.unique(odh.classset)

    
    # initialize our feature extractor
    fe = FeatureExtractor(num_channels=10)
    feature_list = fe.get_feature_list()

    # get the variable ready for where we save the results
    results = np.zeros((len(classset_list), len(feature_list), len(subject_list)))
    for cs in classset_list:

        cs_data = odh.isolate_data("classset", [cs])
        for s in subject_list:
            subject_data = cs_data.isolate_data("subjects",[s])
            subject_train = subject_data.isolate_data("reps",list(range(0,8)))
            subject_test  = subject_data.isolate_data("reps",list(range(8,11)))

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
                print(f)
                train_features = fe.extract_features([f], train_windows)
                test_features  = fe.extract_features([f], test_windows)
                train_metadata_ = train_metadata.copy()
                test_metadata_ = test_metadata.copy()
                # hack to accomodate ninapro sensors holding last value:
                invalid_train = []
                invalid_test  = []
                for key in train_features:
                    out = np.where(train_features[key] == -1*np.inf)
                    invalid_train.extend(out[0])
                    out = np.where(train_features[key] == np.inf)
                    invalid_train.extend(out[0])
                    out = np.where(train_features[key] == np.nan)
                    invalid_train.extend(out[0])
                    out = np.where([ not(float('-inf') < i < float('inf')) for i in np.sum(train_features[key],axis=1)])
                    invalid_train.extend(out[0])
                    out = np.where(test_features[key] == -1*np.inf)
                    invalid_test.extend(out[0])
                    out = np.where(test_features[key] == np.inf)
                    invalid_test.extend(out[0])
                    out = np.where(test_features[key] == np.nan)
                    invalid_test.extend(out[0])
                    out = np.where([ not(float('-inf') < i < float('inf')) for i in np.sum(test_features[key],axis=1)])
                    invalid_test.extend(out[0])
                # remove duplicates
                invalid_train = set(invalid_train)
                invalid_test  = set(invalid_test)
                for key in train_features:
                    train_features[key] = np.delete(train_features[key], list(invalid_train),axis=0)
                    test_features[key] = np.delete(test_features[key], list(invalid_test),axis=0)
                for key in train_metadata:
                    train_metadata_[key] = np.delete(train_metadata_[key], list(invalid_train),axis=0)
                    test_metadata_[key] = np.delete(test_metadata_[key], list(invalid_test), axis=0)

                # get the dataset ready for the classifier
                data_set = {}
                data_set['testing_features'] = test_features
                data_set['training_features'] = train_features
                data_set['testing_labels'] = test_metadata_["classes"]
                data_set['training_labels'] = train_metadata_["classes"]
                # setup the classifier
                classifier = EMGClassifier()
                classifier.fit(model, data_set.copy())

                # running the classifier analyzes the test data we already passed it
                preds = classifier.run()
                # get the CA: classification accuracy offline metric and add it to the results
                results[cs, i,s] = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds)[metrics[0]] * 100
            # print(f"S{s} {f}: {results[i,s]}%")
    # the feature accuracy is represented by the mean accuracy across the subjects
    mean_feature_accuracy = results.mean(axis=2)
    std_feature_accuracy  = results.std(axis=2)

    fig, ax = plt.subplots(len(classset_list),1)
    for cs in classset_list:
        
        ax[cs].bar(feature_list, mean_feature_accuracy[cs,:], yerr=std_feature_accuracy[cs,:])
        ax[cs].grid()
        ax[cs].set_xlabel("Features")
        ax[cs].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()