import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.data_handler import OfflineDataHandler
from unb_emg_toolbox.feature_selector import FeatureSelector


if __name__ == "__main__" :
    # import a dataset to work with (as was done in the past demos)
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
    
    # Let's grab the saved test set (it has multiple reps to perform cross-validation against)
    train_odh = odh.isolate_data(key="sets", values=[1])
    train_windows, train_metadata = train_odh.parse_windows(50,25)


    # we want to get all the features our toolbox can extract, so call the get feature list method to return a list of all computable features
    fe = FeatureExtractor(num_channels=8)
    feature_list = fe.get_feature_list()
    # and extract those features. this returns a dictionary
    training_features = fe.extract_features(feature_list, train_windows)

    # perform a feature selection
    # there are 4 supported feature selection metrics currently:
    # 1. accuracy,
    # 2. active error,
    # 3. mean semi principal axis,
    # 4. feature efficiency.

    # first we initialize the feature selector class
    fs = FeatureSelector()
    
    # # demo for accuracy!
    metric="accuracy"
    class_var = train_metadata["classes"].astype(int)
    crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
    accuracy_results, accuracy_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    fs.print(metric, accuracy_results, accuracy_fs)
    
    # demo for active error!
    metric="activeerror"
    crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
    class_var = train_metadata["classes"].astype(int)
    aerror_results, aerror_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    fs.print(metric, aerror_results, aerror_fs)
    
    # demo for mean semi principal axis length!
    metric="meansemiprincipalaxis"
    crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
    class_var = train_metadata["classes"].astype(int)
    msa_results, msa_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    fs.print(metric, msa_results, msa_fs)

    # demo for feature efficiency!
    metric="featureefficiency"
    crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
    class_var = train_metadata["classes"].astype(int)
    fe_results, fe_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    fs.print(metric, fe_results, fe_fs)

    # demo if you don't have a cross-validation variable and just want to randomly split the dataset (suboptimal choice, but available if necessary)
    class_var = train_metadata["classes"].astype(int)
    crossvalidation_var = {"crossval_amount": 5,
                           "crossval_percent": 0.75}
    accuracy_results, accuracy_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)




    # these results are easy to use with the rest of the toolbox! if you wanted to extract the feature set you just got:
    # if you want the best 5 features from one of these selections
    top_feature_set = accuracy_fs[:5]
    top_training_features = fe.extract_features(top_feature_set, train_windows)
    # and continue the pipeline normally from here!
