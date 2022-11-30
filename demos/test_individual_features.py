import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.emg_classifier import EMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.data_handler import OfflineDataHandler
from unb_emg_toolbox.offline_metrics import OfflineMetrics


# Currently this file is for only one individual
if __name__ == "__main__" :
    # example isolating data
    # you can probably see how this is used for things like k-fold cross-validation
    # we could have instead gotten the all the data in the folder (train and test at once)
    # Note: this contains another set of demo data, but because of our regex we won't collect it!
    
    feature_dict = {"DFTR_fs":200, "ignorethisplease_":100, "MDF_fs":200, "AR_order":9}

    # manual dictionary making (showing how you can add more than the default metadata [classes, reps])
    dataset_folder = 'demos/data/myo_dataset'
    sets_values = ["training", "testing"]
    sets_regex = make_regex(left_bound = "dataset/", right_bound="/", values = sets_values)
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
    odh.get_data(folder_location=dataset_folder, filename_dic = dic, delimiter=",")
    
    # values=[0] corresponds to training since the 0th element of sets_values is training
    train_odh = odh.isolate_data(key="sets", values=[0])
    train_windows, train_metadata = train_odh.parse_windows(50,25)
    test_odh = odh.isolate_data(key="sets", values=[1])
    test_windows, test_metadata = test_odh.parse_windows(50,25)

    fe = FeatureExtractor(num_channels=8)

    data_set = {}
    features = ["MAV","ZC","SAMPEN"]

    tr_features = fe.extract_features(features, train_windows, feature_dict)
    te_features = fe.extract_features(features, test_windows, feature_dict)
    fe.visualize_feature_space(tr_features, projection="PCA",classes=train_metadata["classes"], savedir="f2", render=True, test_feature_dic=te_features, t_classes=test_metadata["classes"])
    
    data_set['training_windows'] = train_windows # used for velocity control
    data_set['testing_features'] = fe.extract_features(te_features, test_windows,feature_dict)
    data_set['training_features'] = fe.extract_features(tr_features, train_windows,feature_dict)
    data_set['testing_labels'] = test_metadata['classes']
    data_set['training_labels'] = train_metadata['classes']

    om = OfflineMetrics()
    metrics = ['CA', 'AER', 'INS', 'REJ_RATE', 'CONF_MAT', 'RECALL', 'PREC', 'F1']
    # Normal Case - Test all different classifiers
    
    classifier = EMGClassifier()
    classifier.fit("LDA", data_set.copy())
    preds = classifier.run()
    y_true = data_set['testing_labels']
    metrics = om.extract_offline_metrics(['CONF_MAT'], preds, y_true)
    metrics = om.extract_offline_metrics(['CA'], preds, y_true)