import libemg
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# classes_values = ["0","1","2","3","4"]#["Hand_Close","Hand_Open","No_Motion","Pronation","Supination","Wrist_Extension","Wrist_Flexion"]
# reps_values    = ["0","1","2","3","4"]
# odh = libemg.data_handler.OfflineDataHandler()
# odh.get_data("data/",
#              filename_dic = {"classes":classes_values,
#                              "classes_regex": libemg.utils.make_regex("data/C_","_R", values=classes_values),
#                              "reps": reps_values,
#                              "reps_regex": libemg.utils.make_regex("R_",".csv", values=reps_values)})

# windows, metadata = odh.parse_windows(250, 30)
# fe = libemg.feature_extractor.FeatureExtractor()
# features = fe.extract_feature_group("LS4", windows, feature_dic={"WAMP_threshold": 1e-5})
# fe.visualize_feature_space(features, "PCA", metadata["classes"])

reps_values    = ["0","1","2","3","4"]
odh = libemg.data_handler.OfflineDataHandler()
odh.get_data("data/",
             filename_dic = {"reps": reps_values,
                             "reps_regex": libemg.utils.make_regex("R_",".csv", values=reps_values)})
odh.add_regression_labels('media/class_file.txt',['timestamp', 'hand','dof1','dof2'])
metadata_operations = {
        "timestamp": np.mean,
        "hand": [np.int64, np.bincount, np.argmax],
        "dof1": np.mean,
        "dof2": np.mean
    }
windows, metadata = odh.parse_windows(250, 30, metadata_operations)
fe = libemg.feature_extractor.FeatureExtractor()
features = fe.extract_feature_group("HTD", windows)

features = np.hstack([features[e] for e in features.keys()])
features = (features - np.mean(features,0))/np.std(features,0)
pca = PCA(2)
tf = pca.fit_transform(features)

fig = plt.figure(figsize=(12, 12))
plt.subplot(2,1,1)
plt.scatter(tf[:,0], tf[:,1], c=metadata["dof1"])
plt.subplot(2,1,2)
plt.scatter(tf[:,0], tf[:,1], c=metadata["dof2"])
plt.show()
A = 1


features = fe.extract_feature_group("HTD", windows)
train_labels = np.hstack((np.expand_dims(metadata["dof1"],axis=1), 
                                   np.expand_dims(metadata["dof2"],axis=1)))

feature_dic = {
        'training_features': features,
        'training_labels': train_labels
}

reg = libemg.emg_classifier.EMGRegressor()
reg.fit(MultiOutputRegressor(GradientBoostingRegressor()), feature_dic)
predictions = reg.run(features, train_labels)
score = r2_score(train_labels[:,0], predictions[:,0])
print(score)
score = r2_score(train_labels[:,1], predictions[:,1])
print(score)