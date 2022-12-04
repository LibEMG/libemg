import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.datasets import GRABMyo
from libemg.emg_classifier import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter

def plot(ax, row, col, features, mean, std, label):
    ax[row].set_ylim([0,100])
    ax[row].grid()
    ax[row].set_xlabel("Features")
    ax[row].set_ylabel(label)
    ax[row].bar(features, mean, yerr=std)

if __name__ == "__main__":
    # get the GRABMyo Dataset using toolbox handle - this downloads the dataset
    dataset = GRABMyo(save_dir="demos/data/", redownload=False)
    dataset.print_info()
    # take the downloaded dataset and load it as an offlinedatahandler
    i_odh = dataset.prepare_data(format=OfflineDataHandler, sessions=["1"])

    # Perform an analysis where we test all the features available to the toolbox individually for within-subject
    # classification accuracy.

    # setup out model type and output metrics
    model = "LDA"
    om = OfflineMetrics()
    metrics = ['CA', 'AER', 'INS']

    # get the subject list
    subject_list = np.unique(i_odh.subjects)
    
    # initialize our feature extractor
    fe = FeatureExtractor()
    feature_list = fe.get_feature_list()
    feature_list.remove('SAMPEN')
    feature_list.remove('FUZZYEN')

    fig, ax = plt.subplots(2)
    
    for fore in range(0,2):
        if fore == 0:
            # Forearm EMG:
            odh = i_odh.isolate_channels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        else:
            # Wrist EMG
            odh = i_odh.isolate_channels([17,18,19,20,21,22,25,26,27,28,29,30])

        # get the variable ready for where we save the results
        ca = np.zeros((len(feature_list), len(subject_list)))
        aer = np.zeros((len(feature_list), len(subject_list)))
        ins = np.zeros((len(feature_list), len(subject_list)))

        for s in subject_list:
            print("Subject #" + str(s) + ":")
            subject_data = odh.isolate_data("subjects",[s])
            subject_train = subject_data.isolate_data("reps",[0,1,2,3])
            subject_test  = subject_data.isolate_data("reps",[4,5,6])

            # from the standardized data, perform windowing
            train_windows, train_metadata = subject_train.parse_windows(200,100)
            test_windows, test_metadata = subject_test.parse_windows(200,100)

            # for each feature in the feature list
            for i, f in enumerate(feature_list):
                print("Features: " + f)
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
                classifier = EMGClassifier()
                classifier.fit(model, feature_dictionary=data_set.copy())

                # running the classifier analyzes the test data we already passed it
                preds = classifier.run()
                # get the CA: classification accuracy offline metric and add it to the results
                mets = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, null_label=16)
                ca[i,s] = mets['CA'] * 100
                # aer[i,s] = mets['AER'] * 100
                # ins[i,s] = mets['INS'] * 100
                
                # print(f"S{s} {f}: {results[i,s]}%")
        # the feature accuracy is represented by the mean accuracy across the subjects
        mean_feature_accuracy = ca.mean(axis=1)
        std_feature_accuracy  = ca.std(axis=1)
        plot(ax,fore,0,feature_list,mean_feature_accuracy,std_feature_accuracy,'Accuracy')
        # mean_feature_aer = aer.mean(axis=1)
        # std_feature_aer  = aer.std(axis=1)
        # plot(ax,1,fore,feature_list,mean_feature_aer,std_feature_aer,'AER')
        # mean_feature_ins = ins.mean(axis=1)
        # std_feature_ins  = ins.std(axis=1)
        # plot(ax,2,fore,feature_list,mean_feature_ins,std_feature_ins,'Instability')

    plt.show()
