import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

class FeatureSelector:
    """
    Feature selector class including different feature space metrics and sequential feature selection.
    """
    def __init__(self):
        # This will have methods for the evaluation of 33 metrics... eventually 
        #------|-----------------------------------------------------------
        # RI   | Repeatability index
        # mwRI | Mean Within-Repetition Repeatability Index
        # swRI | Std. Dev. Within-Repetition Repeatability Index
        # swSI | Std. Dev. Within-Trial Separability Index
        # MSA  | Mean Semi-Principal Axes
        # CD   | Centroid Drift
        # MAV  | Mean Absolute Value
        # SI   | Separability Index
        # mSI  | Modified Separability Index
        # mwSI | Mean Within-Trial Separability Index
        # BD   | Bhattacharyya Distance
        # KLD  | Kullback-Leibler Divergence
        # HD   | Hellinger Distance Squared
        # FDR  | Fisher's Discriminant Analysis
        # VOR  | Volume of Overlap Region
        # FE   | Feature Efficiency
        # TSM  | Trace of Scatter Matrices
        # DS   | Desirability Score
        # CDM  | Class Discriminability Measure
        # PU   | Purity
        # rPU  | Rescaled Purity
        # NS   | Neighborhood Separability
        # rNS  | Rescaled Neighborhood Separability
        # CE   | Collective Entropy
        # rCE  | Rescaled Collective Entropy
        # C    | Compactness
        # rC   | Rescaled Compactness
        # CA   | Classification Accuracy
        # ACA  | Active Error
        # UD   | Usable Data
        # ICF  | Inter-Class Fraction
        # IIF  | Intra-Inter Fraction
        # TT   | Training time
        # PU   | Purity
        # MAV  | Mean Absolute Value
        # RI   | Repeatability index
        # SI   | Separability Index

        # Implemented:
        # CA   | Classification Accuracy
        # AE   | Active Error
        # MSA  | Mean Semi-Principal Axes
        # FE   | Feature Efficiency
        pass
        
    

    def run_selection(self, data={}, metric="accuracy", class_var = [], crossvalidation_var={}, verbose=0):
        metric_callback = getattr(self, "_get_" + metric)
        metric_type, metric_objective = _get_metric_type(metric)
        
        if metric_type == "wrapper":
            wrapper_results, feature_order = self._get_wrapper_selection_results(metric_callback, metric_objective, data, class_var, crossvalidation_var)
            if verbose:
                self._print_wrapper_results(wrapper_results, feature_order)
            return wrapper_results, feature_order
        elif metric_type == "filter":
            filter_results, feature_order = self._get_filter_selection_results(metric_callback, metric_objective, data, class_var)
            if verbose:
                self._print_filter_results(filter_results)
            return filter_results, feature_order

        
    
    def _get_wrapper_selection_results(self, metric_callback, metric_objective, data, class_var, crossvalidation_var={"var": [],
                                                                                                                      "crossval_amount": 5,
                                                                                                                      "crossval_percent":0.75}):
        features = list(data.keys())
        features_included = []
        dic={}
        features_remaining = features.copy()
        wrapper_results = np.empty((len(features), len(features)))
        wrapper_results[:] = np.nan
        best_features = []
        if "var" in list(crossvalidation_var.keys()):
            crossvalidation_vars = np.unique(crossvalidation_var["var"])
            assert crossvalidation_vars.shape[0] > 1
            crossvalidation = "by_reps"
        elif "crossval_percent" in list(crossvalidation_var.keys()):
            crossval_percent = crossvalidation_var["crossval_percent"]
            crossval_amount  = crossvalidation_var["crossval_amount"]
            crossvalidation = "random"


        prior_features = np.array([])
        for i, fi in enumerate(features):
            for j, fj in enumerate(features):
                if fj in features_included:
                    continue
                metric_value = []
                features_for_iteration = np.hstack((prior_features, data[fj])) if prior_features.size else data[fj]
                if crossvalidation == "by_reps":
                    for cv in crossvalidation_vars:
                        dic["train_ids"] = [i != cv for i in crossvalidation_var["var"]]
                        dic["test_ids"] = [i == cv for i in crossvalidation_var["var"]]
                        dic["labels"] = class_var
                        dic["features"] = features_for_iteration
                        metric_value.append(metric_callback(dic))
                elif crossvalidation == "random":
                    # deal with not passing in a cross-validation set here
                    train_amount = int(crossval_percent * class_var.shape[0])
                    ids = np.arange(class_var.shape[0])
                    for cv in range(crossval_amount):
                        np.random.shuffle(ids)
                        dic["train_ids"] = ids[:train_amount]
                        dic["test_ids"] = ids[train_amount:]
                        dic["labels"] = class_var
                        dic["features"] = features_for_iteration
                        metric_value.append(metric_callback(dic))

                wrapper_results[i,j] = np.mean(metric_value)
            best_feature = metric_objective(wrapper_results[i,:])
            best_features.append(best_feature)
            features_included.append(features[best_feature])
            features_remaining.remove(features[best_feature])
            prior_features = np.hstack((prior_features, data[features[best_feature]])) if prior_features.size else data[features[best_feature]]
        # resort columns
        wrapper_results = wrapper_results[:,best_features]
        return wrapper_results, features_included

    def _get_filter_selection_results(self, metric_callback, metric_objective, data, class_var):
        features = list(data.keys())
        features_included = []
        dic={}
        dic["labels"] = class_var
        filter_results = np.empty((len(features)))
        filter_results[:] = np.nan
        for i, fi in enumerate(features):
            dic["features"] = data[fi]
            filter_results[i] = metric_callback(dic)
        
        fs_order = np.argsort(filter_results)
        if 'max' in str(metric_objective):
            fs_order = fs_order[::-1]
        filter_results = filter_results[fs_order]
        fs_order = [features[i] for i in fs_order]
        return filter_results, fs_order

        


    def _get_accuracy(self, dictionary={}):
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        assert "train_ids" in keys
        assert "test_ids" in keys
        lda = LinearDiscriminantAnalysis()
        train_features = dictionary["features"][dictionary["train_ids"],:]
        train_labels = dictionary["labels"][dictionary["train_ids"]]
        lda.fit(train_features, train_labels)
        test_features = dictionary["features"][dictionary["test_ids"],:]
        test_labels = dictionary["labels"][dictionary["test_ids"]]
        predictions = lda.predict(test_features)
        return 100*sum(predictions == test_labels)/predictions.shape[0]


    def _get_activeerror(self, dictionary={}):
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        assert "train_ids" in keys
        assert "test_ids" in keys
        lda = LinearDiscriminantAnalysis()
        train_features = dictionary["features"][dictionary["train_ids"],:]
        train_labels = dictionary["labels"][dictionary["train_ids"]]
        lda.fit(train_features, train_labels)
        test_features = dictionary["features"][dictionary["test_ids"],:]
        test_labels = dictionary["labels"][dictionary["test_ids"]]
        predictions = lda.predict(test_features)
        errors = (predictions != test_labels).astype(int)
        activepredictions = (predictions != 0)
        return 100*sum((errors+activepredictions==2))/predictions.shape[0]

    def _get_meansemiprincipalaxis(self, dictionary={}):
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        # first normalize the features
        norm_features = (dictionary["features"] - dictionary["features"].mean())/ dictionary["features"].std()
        classes = np.unique(dictionary["labels"])
        msa = [0]*classes.shape[0]
        for c, ci in enumerate(classes):
            class_features = norm_features[dictionary["labels"]==c,:]
            pca = PCA()
            pca.fit(class_features)
            for comp in pca.components_:
                msa[ci] += np.abs(comp).prod() ** (1.0 / len(comp))
        
        return sum(msa)/len(msa)

    def _get_featureefficiency(self, dictionary={}):
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        classes = np.unique(dictionary["labels"])
        sum_class_efficiency = 0
        for ci_val  in classes:
            ci_ids = dictionary["labels"]==ci_val
            cardinality_ci = sum(ci_ids)
            max_class_efficiency = 0
            for cj_val  in classes:
                if ci_val == cj_val:
                    continue
                cj_ids = dictionary["labels"] == cj_val
                cardinality_cj = sum(cj_ids)
                max_feature_class_efficiency = 0
                for k in range(dictionary["features"].shape[1]):
                    min_i = min(dictionary["features"][ci_ids,k])
                    min_j = min(dictionary["features"][cj_ids,k])
                    max_of_mins = max([min_i, min_j])

                    max_i = max(dictionary["features"][ci_ids,k])
                    max_j = max(dictionary["features"][cj_ids,k])
                    min_of_maxs = min([max_i, max_j])

                    confused_points = (dictionary["features"][ci_ids+cj_ids,k] < min_of_maxs).astype(np.int32) + \
                                        (dictionary["features"][ci_ids+cj_ids,k] > max_of_mins).astype(np.int32) 
                    Sk = sum(confused_points == 2)
                    class_efficiency = (cardinality_ci + cardinality_cj - Sk) / (cardinality_ci+cardinality_cj)
                    if class_efficiency > max_feature_class_efficiency:
                        max_feature_class_efficiency = class_efficiency
                if max_feature_class_efficiency > max_class_efficiency:
                    max_class_efficiency = max_feature_class_efficiency
            sum_class_efficiency += max_class_efficiency
        return sum_class_efficiency / (classes.shape[0]-1)
    
    def print(self, metric, results, fs):
        metric_type, _ = _get_metric_type(metric)
        if metric_type == "wrapper":
            self._print_wrapper_results(results, fs)
        elif metric_type == "filter":
            self._print_filter_results(results, fs)

    def _print_wrapper_results(self, wrapper_results, fs):
        # longest feature name
        longest = 11
        for f in fs:
            if longest < len(f):
                longest = len(f)
        header_row = "iter".center(longest)
        for f in range(len(fs)):
            header_row += "|" + fs[f].center(longest)
        print(header_row)
        print('='*((longest+1)*(len(fs)+1)))

        for i in range(len(fs)):
            row = str(i).center(longest)

            for j in range(i):
                row += "|" + " ".center(longest)
            for ii,f in enumerate(range(i, len(fs))):
                row += "|" + ("{:.1f}".format(wrapper_results[i,f])).center(longest)
            print(row)


    def _print_filter_results(self, filter_results, fs):
        # longest feature name
        longest = 11
        for f in fs:
            if longest < len(f):
                longest = len(f)
        header_row = "iter".center(longest)
        for f in range(len(fs)):
            header_row += "|" + fs[f].center(longest)
        print(header_row)
        print('='*((longest+1)*(len(fs)+1)))

        row = str(0).center(longest)
        for i in range(len(fs)):
            row += "|" + ("{:.1f}".format(filter_results[i])).center(longest)
        print(row)


def _get_metric_type(metric="accuracy"):
    metric_type = {
        "accuracy": "wrapper",
        "activeerror": "wrapper",
        "meansemiprincipalaxis": "wrapper",
        "featureefficiency":"filter"
    }
    objective = {
        "accuracy": np.nanargmax,
        "activeerror": np.nanargmin,
        "meansemiprincipalaxis": np.nanargmax, 
        "featureefficiency": np.nanargmax
    }
    assert metric in list(metric_type.keys())
    return metric_type[metric], objective[metric]
