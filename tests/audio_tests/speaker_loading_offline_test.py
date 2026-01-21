from libemg._datasets.speaker_distortion import SpeakerDistortionDataset
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
import numpy as np
import os
import matplotlib.pyplot as plt
import mrmr

# Test Session Settings
WINDOW_SIZE = 1067 
WINDOW_INCREMENT = 1067
METRICS = ['CA', 'PREC', 'R2', 'RECALL'] # AER may be worth considering for distance if regression is explored
CLASS_LIST = ["distance", "loads"] # Classify seperately.
DISC_ANALYSIS = ['']

# Load in dataset and set up objects
dataset = SpeakerDistortionDataset()
fe = FeatureExtractor()
om = OfflineMetrics()
clf = EMGClassifier('LDA') # Only for initial feature selection

# Prepare data and windows
data = dataset.prepare_data()
train_data = data['Train'] # Anechoic Data
test_data = data['Test'] # Non-Anechoic Data, Recorded seperately
train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)

# Get all features, manually remove incompatible features.
feature_list = fe.get_feature_list()
feature_list.remove("FUZZYEN")
feature_list.remove("SAMPEN")
feature_list.remove("RMSPHASOR")
feature_list.remove("WLPHASOR")
feature_list.remove("WENG")
feature_list.remove("WV")
feature_list.remove("WWL")
feature_list.remove("WENT")
feature_group_list = fe.get_feature_groups()
feature_group_list.pop("TSTD")
feature_group_list.pop("MSWT")
feature_group_list.pop("COMB")


#results = np.zeros((1,len(feature_list)+len(feature_group_list)))
# if not os.path.exists("results.npy"):

#     for f in range(len(feature_list)+len(feature_group_list)):
#         if f < len(feature_list):
#             feature = feature_list[f]
#             train_features = fe.extract_features([feature], train_windows)
#             test_features = fe.extract_features([feature], test_windows)
#         else:
#             feature = list(feature_group_list.keys())[f-len(feature_list)]
#             train_features = fe.extract_feature_group(feature, train_windows)
#             test_features = fe.extract_feature_group(feature, test_windows)
#         feature_dictionary = {
#             "training_features": train_features,
#             "training_labels": train_meta["distance"]
#         }
#         # train classifier
#         clf.fit(feature_dictionary.copy())

#         preds = clf.run(test_features)
        
#         # test classifier
#         results[0,f] = om.extract_offline_metrics(METRICS, test_meta["distance"], preds[0])[METRICS[0]] * 100
#         print("Feature: {}, Accuracy: {}".format(feature, results[0,f]))

#     #np.save("results.npy", results)
# else:
#     results = np.load("results.npy")
# mean_feature_accuracy = results.mean(axis=0)
# std_feature_accuracy  = results.std(axis=0)

# plt.figure(figsize=(10,5))
# plt.bar(feature_list+list(feature_group_list.keys()), mean_feature_accuracy, yerr=std_feature_accuracy)
# plt.grid()
# plt.xlabel("Features")
# plt.ylabel("Accuracy")
# plt.xticks(feature_list+list(feature_group_list.keys()), rotation=90)

# plt.tight_layout()
# plt.savefig("LDA_Distance_Accuracy.png")
# plt.show()