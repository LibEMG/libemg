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
    data_set['training_windows'] = train_windows # used for velocity control
    data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
    data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
    data_set['testing_labels'] = test_metadata['classes']
    data_set['training_labels'] = train_metadata['classes']

    om = OfflineMetrics()
    metrics = ['CA', 'AER', 'INS', 'REJ_RATE', 'CONF_MAT', 'RECALL', 'PREC', 'F1']
    # Normal Case - Test all different classifiers
    for model in ['LDA', 'QDA', 'SVM', 'KNN', 'RF', 'NB', 'GB', 'MLP']:
        classifier = EMGClassifier(model, data_set.copy())
        preds = classifier.run()
        y_true = data_set['testing_labels']
        metrics = om.extract_offline_metrics(['CONF_MAT'], preds, y_true)
        om.visualize_conf_matrix(metrics['CONF_MAT'])
        print("Classifier: " + model)
        out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
        print(str(out_metrics) + "\n")
        # om.visualize(out_metrics)
        # om.visualize_conf_matrix(out_metrics['CONF_MAT'])

    # Rejection Case
    rejection_classifier = EMGClassifier("SVM", data_set.copy(), rejection_type="CONFIDENCE", rejection_threshold=0.5)
    preds = rejection_classifier.run()
    out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
    print("Offline Rejection Metrics:")
    print(out_metrics)

    # Majority Vote Case
    mv_classifier = EMGClassifier("SVM", data_set.copy(), majority_vote=1)
    preds = mv_classifier.run()
    out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
    print("Offline Majority Vote Metrics:")
    print(out_metrics)

    # Custom classifier case
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    classifier = EMGClassifier(clf, data_set.copy())
    classifier.run()

    # another example
    # get metadata from columns in a csv file
    odh = OfflineDataHandler()
    dataset_folder = 'demos/data/column_metadata_dataset'
    # when getting classes and reps from the csv file, you don't need to give a template for the values, just specify where they will appear.
    classes_values=[]
    reps_values = []
    subject_values=["S001"]
    subject_regex = make_regex(left_bound="SGT_", right_bound="_EMG.csv", values=subject_values)
    dic = {
        "subject": subject_values,
        "subject_regex": subject_regex,
        "classes": [],
        "classes_column":[18],
        "reps": [],
        "reps_column": [19],
        "data_column": [2,3,4,5,6,7,8,9] # optionally, you can specify the data columns if you don't want all the columns to be collected
    }
    odh.get_data(folder_location=dataset_folder, filename_dic=dic, delimiter=",")
    train_odh = odh.isolate_data(key="reps", values=[0])
    train_windows, train_metadata = train_odh.parse_windows(50,25)
    test_odh = odh.isolate_data(key="reps", values=[1])
    test_windows, test_metadata = test_odh.parse_windows(50,25)

    fe = FeatureExtractor(num_channels=8)

    data_set = {}
    data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
    data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
    data_set['testing_labels'] = test_metadata['classes']
    data_set['training_labels'] = train_metadata['classes']

    classifier = EMGClassifier("SVM", data_set)
    preds = classifier.run()
    output = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
    print("Offline Metrics:")
    print(output)
