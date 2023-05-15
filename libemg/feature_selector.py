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

        # Implemented:
        # CA   | Classification Accuracy
        # AE   | Active Error
        # MSA  | Mean Semi-Principal Axes
        # FE   | Feature Efficiency
        # RI   | Repeatability index
        # SI   | Separability Index
        pass
        

    def run_selection(self, data={}, metric="accuracy", class_var = [], crossvalidation_var={}, verbose=False, early_stop=None, nm_class=2):
        """The main method for running feature selection on a dataset. Selection will choose features that result in the ranking 
        of features according to the specified metric and the metric values found through the selection process. This is an entry
        point function that calls the _get_sequential_selection_results and _get_metric_selection_results methods, after finding the
        metric callback (function handle for extracting the metric from data) and the metric type (sequential or metric).

        Parameters
        ----------
        data: dict
            Dictionary of features where every feature has its own key. The element associated with each key is a np.ndarray
            of size N samples x F features. N is consistent across all features in the dictionary, but F can vary (eg., AR can
            have 4 features for channel and MAV has 1 feature per channel). 
        metric: str
            A string specifying what metric should be used as the basis of the selection.
        class_var: list
            a np.ndarray containing the class labels associated with the features.
        crossvalidation_var: dict
            A dictionary containing the method by which crossvalidation is performed (most metrics support crossvalidation). Either specify the 
            "crossval_amount" key <int> to perform k-fold cross validation with k random folds of size (1-1/k) training (1/k) testing, or provide 
            the "var" key to run leave-one-key-out cross validation with the np.ndarray you pass in.
        verbose: bool (optional), default = False
            If True, automatically calls the print method after getting the results.
        early_stop: int (optional), default = None
            The number of features to return from the selection -- if None is passed, all features are returned. This only influences the computation
            time of wrapper-based selection.
        nm_class: int (optioal), default=2
            The integer value of class label that corresponds to the no motion class. This is used only for active error.
        Returns
        ------- 
        filter_results/wrapper_results: np.ndarray
            filter_results is a 1D np.ndarray of metric values for each feature in the data dictionary
            wrapper_results is a 2D upper-triangle np.ndarray containing the metric values for the sequential selection processes.
        feature_order:  list
            The optimal order of features according to the metric specified. This is a list of feature names.
        """
        metric_callback = getattr(self, "_get_" + metric)
        metric_type, metric_objective = self._get_metric_type(metric)
        
        if metric_type == "sequential":
            wrapper_results, feature_order = self._get_sequential_selection_results(metric_callback, metric_objective, data, class_var, crossvalidation_var, early_stop, nm_class)
            if verbose:
                self._print_wrapper_results(wrapper_results, feature_order)
            return wrapper_results, feature_order
        elif metric_type == "metric":
            filter_results, feature_order = self._get_metric_selection_results(metric_callback, metric_objective, data, class_var, crossvalidation_var)
            if verbose:
                self._print_filter_results(filter_results)
            return filter_results, feature_order
    
    def _get_sequential_selection_results(self, metric_callback, metric_objective, data, class_var, crossvalidation_var={"var": [],
                                                                                                                      "crossval_amount": 5},
                                                                                                                      early_stop=None,
                                                                                                                      nm_class=2):
        """The selection procedure carried out for metrics where the selection is sequential (F)(F-1)/2 metric calls to form an upper-triangle
        matrix of metric values. These metric values are sorted according to their optimality criterion and used for sorting the features in
        the "best" order.

        Parameters
        ----------
        metric_callback: function callback
            A callback to compute the metric.
        metric_objective: function callback
            A callback to either np.nanmax or np.nanmin to specify whether the better features are given by a low or high metric value.
        data: dict
            Dictionary of features where every feature has its own key. The element associated with each key is a np.ndarray
            of size N samples x F features. N is consistent across all features in the dictionary, but F can vary (eg., AR can
            have 4 features for channel and MAV has 1 feature per channel). 
        class_var: list
            a np.ndarray containing the class labels associated with the features.
        crossvalidation_var: dict
            A dictionary containing the method by which crossvalidation is performed (most metrics support crossvalidation). Either specify the 
            "crossval_amount" key <int> to perform k-fold cross validation with k random folds of size (1-1/k) training (1/k) testing, or provide 
            the "var" key to run leave-one-key-out cross validation with the np.ndarray you pass in.
        early_stop: int (optional), default = None
            The number of features to stop at early in the selection process. If None is passed, all features are returned.
        nm_class: int (optional), default=2
            The class that corresponds to no motion. Used for active error
        Returns
        ------- 
        wrapper_results: np.ndarray
            wrapper_results is a 2D upper-triangle np.ndarray containing the metric values for the sequential selection processes.
        feature_order:  list
            The optimal order of features according to the metric specified. This is a list of feature names.
        """
        features = list(data.keys())
        features_included = []
        dic={}
        features_remaining = features.copy()
        wrapper_results = np.empty((len(features), len(features)))
        wrapper_results[:] = np.nan
        #TODO: add wrapper std?
        best_features = []
        if "var" in list(crossvalidation_var.keys()):
            crossvalidation_vars = np.unique(crossvalidation_var["var"])
            crossvalidation_var = crossvalidation_var["var"]
            assert crossvalidation_vars.shape[0] > 1
        elif "crossval_amount" in list(crossvalidation_var.keys()):
            crossval_amount  = crossvalidation_var["crossval_amount"]
            crossvalidation_var = np.zeros((data[features[0]].shape[0]))
            crossval_width = int(data[features[0]].shape[0]/crossval_amount)
            # split dataset as evenly as possible
            for k in range(crossval_amount):
                crossvalidation_var[k*crossval_width:(k+1)*crossval_width] = k
            # assign remainder of features to one crossval set
            crossvalidation_var[(k+1)*crossval_width:]=k
            # we need to shuffle the data to make sure that we have equal class labels in each set
            shuffle_id = np.arange(data[features[0]].shape[0])
            np.random.shuffle(shuffle_id)
            for f in features:
                data[f] = data[f][shuffle_id,:]
            class_var = class_var[shuffle_id]
            
        prior_features = np.array([])
        for i, fi in enumerate(features):
            for j, fj in enumerate(features):
                if fj in features_included:
                    continue
                features_for_iteration = np.hstack((prior_features, data[fj])) if prior_features.size else data[fj]
                dic["features"] = features_for_iteration
                dic["labels"]   = class_var
                dic["crossval"] = crossvalidation_var
                dic["nm"] = nm_class
                wrapper_results[i,j] = np.mean(metric_callback(dic))
            best_feature = metric_objective(wrapper_results[i,:])
            best_features.append(best_feature)
            features_included.append(features[best_feature])
            features_remaining.remove(features[best_feature])
            prior_features = np.hstack((prior_features, data[features[best_feature]])) if prior_features.size else data[features[best_feature]]
            if early_stop is not None and i == early_stop-1:
                break
        # resort columns
        wrapper_results = wrapper_results[:,best_features]
        return wrapper_results, features_included

    def _get_metric_selection_results(self, metric_callback, metric_objective, data, class_var, crossvalidation_var={"var": [],
                                                                                                                     "crossval_amount": 5}):
        """The selection procedure carried out for metrics where the selection is done once per feature (F) metric calls.
        These metric values are sorted according to their optimality criterion and used for sorting the features in
        the "best" order.

        Parameters
        ----------
        metric_callback: function callback
            A callback to compute the metric.
        metric_objective: function callback
            A callback to either np.nanmax or np.nanmin to specify whether the better features are given by a low or high metric value.
        data: dict
            Dictionary of features where every feature has its own key. The element associated with each key is a np.ndarray
            of size N samples x F features. N is consistent across all features in the dictionary, but F can vary (eg., AR can
            have 4 features for channel and MAV has 1 feature per channel). 
        class_var: list
            a np.ndarray containing the class labels associated with the features.
        crossvalidation_var: dict
            A dictionary containing the method by which crossvalidation is performed (most metrics support crossvalidation). Either specify the 
            "crossval_amount" key <int> to perform k-fold cross validation with k random folds of size (1-1/k) training (1/k) testing, or provide 
            the "var" key to run leave-one-key-out cross validation with the np.ndarray you pass in.
        
        Returns
        ------- 
        filter_results: list
            filter_results is a 1D np.ndarray of metric values for each feature in the data dictionary
        feature_order:  list
            The optimal order of features according to the metric specified. This is a list of feature names.
        """
        features = list(data.keys())
        dic={}
        dic["labels"] = class_var
        filter_results = np.empty((len(features)))
        filter_results[:] = np.nan

        if "var" in list(crossvalidation_var.keys()):
            crossvalidation_vars = np.unique(crossvalidation_var["var"])
            crossvalidation_var = crossvalidation_var["var"]
            assert crossvalidation_vars.shape[0] > 1
        elif "crossval_amount" in list(crossvalidation_var.keys()):
            crossval_amount  = crossvalidation_var["crossval_amount"]
            crossvalidation_var = np.zeros((data[features[0]].shape[0]))
            crossval_width = int(data[features[0]].shape[0]/crossval_amount)
            # split dataset as evenly as possible
            for k in range(crossval_amount):
                crossvalidation_var[k*crossval_width:(k+1)*crossval_width] = k
            # assign remainder of features to one crossval set
            crossvalidation_var[(k+1)*crossval_width:]=k
            # we need to shuffle the data to make sure that we have equal class labels in each set
            shuffle_id = np.arange(data[features[0]].shape[0])
            np.random.shuffle(shuffle_id)
            for f in features:
                data[f] = data[f][shuffle_id,:]
            class_var = class_var[shuffle_id]


        for i, fi in enumerate(features):
            dic["features"] = data[fi]
            dic["labels"]   = class_var
            dic["crossval"] = crossvalidation_var
            filter_results[i] = metric_callback(dic)
        
        fs_order = np.argsort(filter_results)
        if 'max' in str(metric_objective):
            fs_order = fs_order[::-1]
        filter_results = filter_results[fs_order]
        fs_order = [features[i] for i in fs_order]
        return filter_results, fs_order

    def _get_accuracy(self, dictionary={}):
        """The helper function metric callback to compute accuracy.

        Parameters
        ----------
        dictionary: dict
            A dictionary that contains features, labels, and crossval keys. These keys provide the data necessary to
            train a LDA classifier and determine accuracy.
        
        Returns
        ------- 
        float: the accuracy on the data passed in.
        """
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        assert "crossval" in keys

        k = np.unique(dictionary["crossval"])
        metric_value = []
        for ki in k:
            train_ids = [i != ki for i in dictionary["crossval"]]
            test_ids =  [i == ki for i in dictionary["crossval"]]
            lda = LinearDiscriminantAnalysis()
            train_features = dictionary["features"][train_ids,:]
            train_labels = dictionary["labels"][train_ids]
            lda.fit(train_features, train_labels)
            test_features = dictionary["features"][test_ids,:]
            test_labels = dictionary["labels"][test_ids]
            predictions = lda.predict(test_features)
            metric_value.append(100*sum(predictions == test_labels)/predictions.shape[0])
        return np.mean(metric_value)


    def _get_activeerror(self, dictionary={}):
        """The helper function metric callback to compute active error.

        Parameters
        ----------
        dictionary: dict
            A dictionary that contains features, labels, and crossval keys. These keys provide the data necessary to
            train a LDA classifier and determine active error.
        
        Returns
        ------- 
        float: the active error on the data passed in.
        """
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        assert "crossval" in keys
        assert "nm" in keys

        k = np.unique(dictionary["crossval"])
        metric_value = []
        for ki in k:
            train_ids = [i != ki for i in dictionary["crossval"]]
            test_ids =  [i == ki for i in dictionary["crossval"]]
            lda = LinearDiscriminantAnalysis()
            train_features = dictionary["features"][train_ids,:]
            train_labels = dictionary["labels"][train_ids]
            lda.fit(train_features, train_labels)
            test_features = dictionary["features"][test_ids,:]
            test_labels = dictionary["labels"][test_ids]
            predictions = lda.predict(test_features)
            errors = (predictions != test_labels).astype(int)
            activepredictions = (predictions != dictionary["nm"])
            metric_value.append(100*sum((errors+activepredictions==dictionary["nm"]))/predictions.shape[0])

        return np.mean(metric_value)

    def _get_meansemiprincipalaxis(self, dictionary={}):
        """The helper function metric callback to compute mean semi-principal axis length.

        Parameters
        ----------
        dictionary: dict
            A dictionary that contains features, labels, and crossval keys. These keys provide the data necessary to
            fit a PCA model for each class and determine component length.
        
        Returns
        ------- 
        float: the mean within-class component length on the data passed in.
        """
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        assert "crossval" in keys
        
        k = np.unique(dictionary["crossval"])
        metric_value = []
        for ki in k:
            train_ids = [i != ki for i in dictionary["crossval"]]
            k_features = dictionary["features"][train_ids,:]
            k_classes  = dictionary["labels"][train_ids]
            norm_features = (k_features - k_features.mean())/ k_features.std()
            classes = np.unique(k_classes)
            msa = [0]*classes.shape[0]
            for c, ci in enumerate(classes):
                class_features = norm_features[k_classes==c,:]
                pca = PCA()
                pca.fit(class_features)
                for comp in pca.components_:
                    msa[ci] += np.abs(comp).prod() ** (1.0 / len(comp))
            metric_value.append(sum(msa)/len(msa))
        return np.mean(metric_value)

    def _get_featureefficiency(self, dictionary={}):
        """The helper function metric callback to compute feature efficiency.

        Parameters
        ----------
        dictionary: dict
            A dictionary that contains features, labels, and crossval keys. These keys provide the data necessary to
            the number of points that lie between the regions of confusion of each class.
        
        Returns
        ------- 
        float: the feature efficiency on the data passed in.
        """
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        assert "crossval" in keys

        k = np.unique(dictionary["crossval"])
        metric_value = []

        for ki in k:
            train_ids = [i != ki for i in dictionary["crossval"]]
            k_features = dictionary["features"][train_ids,:]
            k_classes  = dictionary["labels"][train_ids]
            classes = np.unique(k_classes)
            sum_class_efficiency = 0
            for ci_val  in classes:
                ci_ids = k_classes==ci_val
                cardinality_ci = sum(ci_ids)
                max_class_efficiency = 0
                for cj_val  in classes:
                    if ci_val == cj_val:
                        continue
                    cj_ids = k_classes == cj_val
                    cardinality_cj = sum(cj_ids)
                    max_feature_class_efficiency = 0
                    for ki in range(k_features.shape[1]):
                        min_i = min(k_features[ci_ids,ki])
                        min_j = min(k_features[cj_ids,ki])
                        max_of_mins = max([min_i, min_j])

                        max_i = max(k_features[ci_ids,ki])
                        max_j = max(k_features[cj_ids,ki])
                        min_of_maxs = min([max_i, max_j])

                        confused_points = (k_features[ci_ids+cj_ids,ki] < min_of_maxs).astype(np.int32) + \
                                            (k_features[ci_ids+cj_ids,ki] > max_of_mins).astype(np.int32) 
                        Sk = sum(confused_points == 2)
                        class_efficiency = (cardinality_ci + cardinality_cj - Sk) / (cardinality_ci+cardinality_cj)
                        if class_efficiency > max_feature_class_efficiency:
                            max_feature_class_efficiency = class_efficiency
                    if max_feature_class_efficiency > max_class_efficiency:
                        max_class_efficiency = max_feature_class_efficiency
                sum_class_efficiency += max_class_efficiency
            metric_value.append(sum_class_efficiency / (classes.shape[0]-1))
        return np.mean(metric_value)
    
    def _get_repeatability(self, dictionary={}):
        """The helper function metric callback to compute repeatability index.

        Parameters
        ----------
        dictionary: dict
            A dictionary that contains features, labels, and crossval keys. These keys provide the data necessary to
            find the centroids of each class for each rep and compare it between reps.
        
        Returns
        ------- 
        float: the repeatability index on the data passed in.
        """
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        assert "crossval" in keys
        r = np.unique(dictionary["crossval"])
        n = np.unique(dictionary["labels"])
        repeatability_index = 0
        for ni in n:
            class_repeatability = 0
            class_ids = [i == ni for i in dictionary["labels"]]
            for ri in r:
                rep_ids = [i == ri for i in dictionary["crossval"]]
                other_reps = [i != ri for i in dictionary["crossval"]]
                tr_ids = [a and b for a, b in zip(class_ids, other_reps)]
                te_ids = [a and b for a, b in zip(class_ids, rep_ids)]
                tr_mean = np.mean(dictionary["features"][tr_ids,:],axis=0)
                te_mean = np.mean(dictionary["features"][te_ids,:],axis=0)
                tr_cov_i = np.linalg.pinv(np.cov(dictionary["features"][tr_ids,:], rowvar=False))
                class_repeatability += np.sqrt((tr_mean-te_mean).T @ tr_cov_i @ (tr_mean-te_mean)) / (2*n.shape[0])
            repeatability_index += class_repeatability / n.shape[0]
        return repeatability_index

    
    def _get_separability(self, dictionary={}):
        """The helper function metric callback to compute separability index.

        Parameters
        ----------
        dictionary: dict
            A dictionary that contains features, labels, and crossval keys. These keys provide the data necessary to
            find the mean between-class Mahalanobis distance for the data provided.
        
        Returns
        ------- 
        float: the mean within-class component length on the data passed in
        """
        keys = list(dictionary.keys())
        assert "features" in keys
        assert "labels" in keys
        # we don't really need crossval here 
        n = np.unique(dictionary["labels"])
        separability_index = 0
        for ni in n:
            class_sep_list = []
            tr_ids = [i == ni for i in dictionary["labels"]]
            for nj in n:
                if ni == nj:
                    continue
                te_ids = [i == nj for i in dictionary["labels"]]
                tr_mean = np.mean(dictionary["features"][tr_ids,:],axis=0)
                te_mean = np.mean(dictionary["features"][te_ids,:],axis=0)
                tr_cov_i = np.linalg.pinv(np.cov(dictionary["features"][tr_ids,:], rowvar=False))
                class_sep_list.append(np.sqrt((tr_mean-te_mean).T @ tr_cov_i @ (tr_mean-te_mean))/2)
            separability_index += min(class_sep_list)/n.shape[0]
        return separability_index

    def print(self, metric, results, fs):
        """The function to print the selection results in a table format to the console. This is the 
        outward-facing function that calls helper functions for the sequential and metric print functions.

        Parameters
        ----------
        metric: str
            The metric that was used in the selection
        results: np.ndarray
            The metric values returned (wrapper or filter) from the selection procedure
        fs: list
            The list containing the optimal feature order according to the metric used in the selection.
        """
        metric_type, _ = self._get_metric_type(metric)
        if metric_type == "sequential":
            self._print_sequential_results(results, fs)
        elif metric_type == "metric":
            self._print_metric_results(results, fs)

    def _print_sequential_results(self, wrapper_results, fs):
        """The helper function to print the selection results in a table format to the console for sequential metrics.

        Parameters
        ----------
        results: np.ndarray
            The metric values returned (wrapper or filter) from the selection procedure
        fs: list
            The list containing the optimal feature order according to the metric used in the selection.
        """
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


    def _print_metric_results(self, filter_results, fs):
        """The helper function to print the selection results in a table format to the console for "metric" metrics.

        Parameters
        ----------
        results: np.ndarray
            The metric values returned (wrapper or filter) from the selection procedure
        fs: list
            The list containing the optimal feature order according to the metric used in the selection.
        """
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


    def _get_metric_type(self, metric="accuracy"):
        """The helper function to recover the metric handles (function to compute metric) and objectives (whether a high or low value is good).

            Parameters
            ----------
            metric: str
                The metric to query.
            fs: list
                The list containing the optimal feature order according to the metric used in the selection.
            
            Returns
            ------- 
            metric_handle: handle
                The function handle to compute the desired metric
            optimal_criterion: handle
                The function handle that indicates whether low of high is good for feature values.
            """
        metric_type = {
            "accuracy": "sequential",
            "activeerror": "sequential",
            "meansemiprincipalaxis": "sequential",
            "featureefficiency":"metric",
            "repeatability": "sequential",
            "separability": "sequential"
        }
        objective = {
            "accuracy": np.nanargmax,
            "activeerror": np.nanargmin,
            "meansemiprincipalaxis": np.nanargmax, 
            "featureefficiency": np.nanargmax,
            "repeatability": np.nanargmin,
            "separability": np.nanargmax
        }
        assert metric in list(metric_type.keys())
        return metric_type[metric], objective[metric]

    def get_metrics(self):
        """The function to get the list of all possible metrics that can be used in feature selection.
        
            Returns
            ------- 
            metric_list: list
                A list of strings that are the supported selection metrics
            """
        metric_list = [
            "accuracy",
            "activeerror",
            "meansemiprincipalaxis",
            "featureefficiency",
            "repeatability",
            "separability"
        ]
        return metric_list 