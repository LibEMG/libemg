import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler
from libemg.feature_selector import FeatureSelector
from libemg.datasets import _3DCDataset

if __name__ == "__main__" :
    # import a dataset to work with (as was done in the past demos)
    dataset = _3DCDataset()
    train_odh = dataset.prepare_data(subjects_values=["1"], sets_values=["train"])
    
    
    # Let's grab the saved test set (it has multiple reps to perform cross-validation against)
    #train_odh = odh.isolate_data(key="sets", values=[0])
    train_windows, train_metadata = train_odh.parse_windows(200,50)


    # we want to get all the features our toolbox can extract, so call the get feature list method to return a list of all computable features
    fe = FeatureExtractor()
    feature_list = fe.get_feature_list()
    feature_list.remove("SAMPEN")
    feature_list.remove("FUZZYEN")
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
    accuracy_results, accuracy_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var, num_features=6)
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

    # demo for repeatability! -- note repeatability takes a long time for large number of reps!
    metric = "repeatability"
    class_var = train_metadata["classes"].astype(int)
    crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
    repeatability_results, repeatability_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    fs.print(metric, repeatability_results, repeatability_fs)

    # demo for separability!
    metric = "separability"
    class_var = train_metadata["classes"].astype(int)
    crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
    separability_results, separability_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
    fs.print(metric, separability_results, separability_fs)


    # demo if you don't have a cross-validation variable and just want to randomly split the dataset (suboptimal choice, but available if necessary)
    class_var = train_metadata["classes"].astype(int)
    crossvalidation_var = {"crossval_amount": 5}
    accuracy_results, accuracy_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)


    # these results are easy to use with the rest of the toolbox! if you wanted to extract the feature set you just got:
    # if you want the best 5 features from one of these selections
    top_feature_set = accuracy_fs[:5]
    top_training_features = fe.extract_features(top_feature_set, train_windows)
    # and continue the pipeline normally from here!
