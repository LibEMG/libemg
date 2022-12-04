import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from libemg.feature_extractor import FeatureExtractor
from libemg.data_handler import OfflineDataHandler
from libemg.feature_selector import FeatureSelector
from libemg.datasets import OneSubjectMyoDataset


if __name__ == "__main__" :
    # load one subject myo dataset
    dataset = OneSubjectMyoDataset(save_dir='example_data', redownload=False)
    odh = dataset.prepare_data(format=OfflineDataHandler)

    # get the training data from the dataset
    train_data = odh.isolate_data("sets",[0])
    train_windows, train_metadata = train_data.parse_windows(50, 25)

    # we want to get all the features our library can extract
    fe = FeatureExtractor()
    feature_list = fe.get_feature_list()
    
    # extract every feature in the library on the training data
    training_features = fe.extract_features(feature_list, train_windows)

    # first we initialize the feature selector class
    fs = FeatureSelector()
    
    # demo for accuracy (view documentation for all the metrics you can use)
    metric="accuracy"
    class_var = train_metadata["classes"].astype(int)
    crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
    accuracy_results, accuracy_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    print("Accuracy: \n")
    fs.print(metric, accuracy_results, accuracy_fs)

    # demo if you don't have a cross-validation variable and just want to randomly split the dataset (suboptimal choice, but available if necessary)
    class_var = train_metadata["classes"].astype(int)
    crossvalidation_var = {"crossval_amount": 5,
                           "crossval_percent": 0.75}
    ss_accuracy_results, ss_accuracy_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    print("Accuracy on Subset: \n")
    fs.print(metric, ss_accuracy_results, ss_accuracy_fs)

    # these results are easy to use with the rest of the library! if you wanted to extract the feature set you just got:
    # if you want the best 5 features from one of these selections
    top_feature_set = accuracy_fs[:5]
    top_training_features = fe.extract_features(top_feature_set, train_windows)
    # and continue the pipeline normally from here!