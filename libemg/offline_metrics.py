import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import Counter
from itertools import combinations
import math

class OfflineMetrics:
    """
    Offline Metrics class is used for extracting offline performance metrics.
    """
    
    def _ignore_rejected(self, y_predictions, y_true):
        # ignore rejections
        valid_samples = y_predictions != -1
        y_predictions = y_predictions[valid_samples]
        y_true        = y_true[valid_samples]
        return y_predictions, y_true

    def get_common_metrics(self):
        """Gets a list of the common metrics used for assessing EMG performance.

        Returns
        ----------
        list
            A list of common metrics (CA, AER, and INS).
        """
        return [
            'CA',
            'AER',
            'INS'
        ]

    def get_available_metrics(self):
        """Gets a list of all available offline performance metrics.

        Returns
        ----------
        list
            A list of all available metrics.
        """
        return [
            'RI',
            'mwRI',
            'swRI',
            'swSI',
            'MSA',
            'CD',
            'SI',
            'mSI',
            'mwSI',
            'BD',
            'KLD',
            'HD',
            'VOR',
            'FE',
            'TSM',
            'DS',
            'FDR',
            'CDM',
            'PU',
            'NS',
            'CE',
            'C',
            'CA',
            'ACA',
            'ICF',
            'IIF',
            'AER',
            'INS',
            'REJ_RATE',
            'CONF_MAT',
            'RECALL',
            'PREC',
            'F1',
            'R2',
            'MSE',
            'MAPE',
            'RMSE',
            'NRMSE',
            'MAE'
        ]
    
    def extract_common_metrics(self, y_true, y_predictions, null_label=None):
        """Extracts the common set of offline performance metrics (CA, AER, and INS).

        Parameters
        ----------
        y_true: numpy.ndarray 
            A list of the true labels associated with each prediction.
        y_predictions: numpy.ndarray
            A list of predicted outputs from a classifier.
        null_label: int (optional)
            A null label used for the AER metric - this should correspond to the label associated
            with the No Movement or null class. 

        Returns
        ----------
        dictionary
            A dictionary containing the metrics, where each metric is a key in the dictionary.
        self.extract_offline_metrics(self.get_common_metrics(), y_true, y_predictions, null_label)
        """
        return self.extract_offline_metrics(self.get_common_metrics(), y_true, y_predictions, null_label)
        
    def extract_offline_metrics(self, metrics, y_true, y_predictions, null_label=None):
        """Extracts a set of offline performance metrics.

        Parameters
        ----------
        metrics: list
            A list of the metrics to extract. A list of metrics can be found running the 
            get_available_metrics function.
        y_true: numpy.ndarray 
            A list of the true labels associated with each prediction.
        y_predictions: numpy.ndarray
            A list of predicted outputs from a classifier.
        null_label: int (optional)
            A null label used for the AER metric - this should correspond to the label associated
            with the No Movement or null class. 

        Returns
        ----------
        dictionary
            A dictionary containing the metrics, where each metric is a key in the dictionary.
        
        Examples
        ---------
        >>> y_true = np.array([1,1,1,2,2,2,3,3,3,4,4,4])
        >>> y_predictions = np.array([1,1,2,2,2,1,3,3,3,4,1,4])
        >>> metrics = ['CA', 'AER', 'INS', 'REJ_RATE', 'CONF_MAT', 'RECALL', 'PREC', 'F1']
        >>> om = OfflineMetrics()
        >>> computed_metrics = om.extract_offline_metrics(metrics, y_true, y_predictions, 2))
        """
        assert len(y_true) == len(y_predictions)
        og_y_preds = y_predictions.copy()
        is_classification = np.all(y_predictions.astype(int) == y_predictions) and np.all(y_true.astype(int) == y_true)
        if -1 in y_predictions and is_classification:
            # Only apply to classification data
            rm_idxs = np.where(y_predictions == -1)
            y_predictions = np.delete(y_predictions, rm_idxs)
            y_true = np.delete(y_true, rm_idxs)

        offline_metrics = {}            
        for metric in metrics:
            method_to_call = getattr(self, 'get_' + metric)
            if metric in ['AER']:
                if not null_label is None:
                    offline_metrics[metric] = method_to_call(y_true, y_predictions, null_label)
                else:
                    print("AER not computed... Please input the null_label parameter.")
            elif metric in ['REJ_RATE']:
                offline_metrics[metric] = method_to_call(og_y_preds)
            else:
                # Assume all other metrics have the signature (y_true, y_predictions)
                offline_metrics[metric] = method_to_call(y_true, y_predictions)
        return offline_metrics

    def get_reps_dictionary_rest_zero(self, reps, number_for_zero):
        """
            Parameters
            ----------
            reps: dictionary
                A dictionary containing the indices of each window of data.
            number_for_zero: integer
                Value for the key that is the non-active class - rest.

            Returns
            ----------
            dictionary
                Returns the new reps dictionary, organized in the following way:
                    CASE I: No class 0
                        Now the key 0 has the values for what key 'number_for_zero' used to have.
                        Key 'number_for_zero' has the values for what the last key used to have, and there is no last key anymore.
                        EXAMPLE:
                            Given: 
                                reps = {
                                    1: [(0, 10), (15, 20)],
                                    2: [(30, 40)],
                                    3: [(50, 60), (70, 80)]
                                }
                                number_for_zero = 2
                            Returns:
                            {
                                1: [(0, 10), (15, 20)], 
                                2: [(50, 60), (70, 80)], 
                                0: [(30, 40)]
                            }
                    CASE II: There is a class 0, but it is not rest
                        Now the key 0 has the values for what key 'number_for_zero' used to have.
                        Key 'number_for_zero' has the values for what the key 0 used to have.
                        EXAMPLE:
                            Given: 
                                reps = {
                                    0: [(90,120), (120,130)],
                                    1: [(0, 10), (15, 20)],
                                    2: [(30, 40)],
                                    3: [(50, 60), (70, 80)]
                                }
                                number_for_zero = 2
                            Returns:
                            {
                                0: [(30, 40)], 
                                1: [(0, 10), (15, 20)], 
                                2: [(90, 120), (120, 130)], 
                                3: [(50, 60), (70, 80)]
                            }
            """
        if number_for_zero != 0:
            if reps.get(0) is None:
                last_key = next(reversed(reps))
                reps[0] = reps[number_for_zero]
                if last_key != number_for_zero:
                    reps[number_for_zero] = reps[last_key]
                del reps[last_key]
            else:
                reps[0], reps[number_for_zero] = (
                    reps[number_for_zero],
                    reps[0]
                )
                
        return reps

    def get_CA(self, y_true, y_predictions):
        """Classification Accuracy.

        The number of correct predictions normalized by the total number of predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        float
            Returns the classification accuracy.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        if len(y_true) == 0:
            print("No test samples - check the rejection rate.")
            return 1.0
        return sum(y_predictions == y_true)/len(y_true)

    def get_ACA(self, y_true, y_predictions, non_active):
        """Active Classification Accuracy.

        The number of correct predictions if predicted label matches true label or if predicted label equals 
        to the non-active class normalized by the total number of predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.
        non_active: int
            An integer that represents the non active class - usually 0.

        Returns
        ----------
        float
            Returns the active classification accuracy.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        if len(y_true) == 0:
            print("No test samples - check the rejection rate.")
            return 1.0
        return sum((y_predictions == y_true) | (y_predictions == non_active))/len(y_true)
    
    def get_RI(self, features_train, features_test, reps_train, reps_test, n):
        """Repeatability Index.

        A measure of the reproducibility of EMH patterns between repetitions.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        features_test: list
            A list of testing data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        reps_test: dictionary
            A dictionary containing the indices of each window of the testing data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the repeatability index.
        """
        total_sum = 0
        for x in range(1,n):
            reps_sum = 0
            for j in range(len(reps_train[0][x])):
                curr_train = np.empty((0,features_train[0].shape[1]))
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][j]
                    curr_train = np.vstack([curr_train, features_train[i][start:end,:]])
                covariance_matrix = np.cov(curr_train, rowvar=False)
                inverse_covariance = np.linalg.pinv(covariance_matrix)
                curr_test = np.empty((0,features_test[0].shape[1]))
                for i in range(len(features_test)):
                    start,end = reps_test[i][x][j]
                    curr_test = np.vstack([curr_test, features_test[i][start:end,:]])
                centroid_train = [np.mean(curr_train[:,k]) for k in range(curr_train.shape[1])]
                centroid_test = [np.mean(curr_test[:,k]) for k in range(curr_test.shape[1])]
                mu = distance.mahalanobis(centroid_train,centroid_test,inverse_covariance)
                reps_sum += mu/2
            total_sum += reps_sum/len(reps_train[0][x])
        return total_sum/(n-1)
    
    def get_mwRI(self, features_train, reps_train, n):
        """Mean Within-Repetition Repeatability Index.

        An interpretation of Bunderson and Kuiken's repeatability index.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the mean within-repetition repeatability index.
        """
        total_sum = 0
        for r in range(len(reps_train[0][1])):
            class_sum = 0
            for x in range(1,n):
                curr_train = np.empty((0,features_train[0].shape[1]))
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][r]
                    curr_train = np.vstack([curr_train, features_train[i][start:end,:]])
                covariance_matrix = np.cov(curr_train, rowvar=False)
                inverse_covariance = np.linalg.pinv(covariance_matrix)
                centroid_train = [np.mean(curr_train[:,k]) for k in range(curr_train.shape[1])]
                points_sum = 0
                for p in range(len(curr_train)):
                    mu = distance.mahalanobis(centroid_train,curr_train[p],inverse_covariance)
                    points_sum += mu/2
                class_sum += points_sum/len(curr_train)
            total_sum += class_sum/(n-1)
        return total_sum/len(reps_train[0][1])

    def get_swRI(self, features_train, reps_train, n):
        """Standard Deviation Within-Repetition Repeatability Index.

        A measure of the variation of the within-repetition repeatability across repetition.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the standard deviation within-repetition repeatability index.
        """
        total_sum = 0
        mwri = self.get_mwRI(features_train, reps_train, n) 
        for r in range(len(reps_train[0][1])):
            class_sum = 0
            for x in range(1,n):
                curr_train = np.empty((0,features_train[0].shape[1]))
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][r]
                    curr_train = np.vstack([curr_train, features_train[i][start:end,:]])
                covariance_matrix = np.cov(curr_train, rowvar=False)
                inverse_covariance = np.linalg.pinv(covariance_matrix)
                centroid_train = [np.mean(curr_train[:,k]) for k in range(curr_train.shape[1])]
                points_sum = 0
                for p in range(len(curr_train)):
                    mu = distance.mahalanobis(centroid_train,curr_train[p],inverse_covariance)
                    points_sum += mu/2
                class_sum += points_sum/len(curr_train)
                
            total_sum += (class_sum/4 - mwri) ** 2
        return math.sqrt(total_sum/(len(reps_train[0][1])-1))
    
    def _smallest_magnitude(self, centroid, array, size):
        temp = [0] * size
        i = 0
        for j in range(size):
            temp[j] = np.linalg.norm(np.array(centroid) - np.array(np.mean(array[j], axis=0)))
            if temp[j] < temp[i]:
                i = j
        return array[i]

    def get_mwSI(self, features_train, reps_train, n):
        """Mean Within-Trial Separability Index.

        A measure of distinguishability of EMG patterns within a trial.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the mean within-trial seperability index.
        """
        total_sum = 0
        for t in range(len(reps_train[0][1])):
            class_sum = 0
            for x in range(1,n):
                arr = [0] * (n-2)
                index = 0
                curr_train = np.empty((0,features_train[0].shape[1]))
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][t]
                    curr_train = np.vstack([curr_train, features_train[i][start:end,:]])
                centroid = np.mean(curr_train, axis=0)
                
                for j in range(1,n):
                    if j != x:
                        temp = np.empty((0,features_train[0].shape[1]))
                        for i in range(len(features_train)):
                            start,end = reps_train[i][j][t]
                            temp = np.vstack([temp, features_train[i][start:end,:]])
                        arr[index] = temp
                        index += 1
                train_smallest = self._smallest_magnitude(centroid,arr,index)
                centroid_small = np.mean(train_smallest, axis=0)
                
                covariance_matrix = np.cov(curr_train, rowvar=False)
                inverse_covariance = np.linalg.pinv(covariance_matrix)
                
                mu = distance.mahalanobis(centroid,centroid_small,inverse_covariance)
                class_sum += mu/2
            total_sum += class_sum/(n-1)
        return total_sum/len(reps_train[0][1])

    def get_swSI(self, features_train, reps_train, n):
        """Standard Deviation Within-Trial Separability Index.

        A measure of the variability of the distinguishability of EMG patterns across trials.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the standard deviation within-trial seperability index.
        """
        mwSI = self.get_mwSI(features_train, reps_train, n) 
        total_sum = 0
        for t in range(len(reps_train[0][1])):
            class_sum = 0
            for x in range(1,n):
                arr = [0] * (n-2)
                index = 0
                curr_train = np.empty((0,features_train[0].shape[1]))
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][t]
                    curr_train = np.vstack([curr_train, features_train[i][start:end,:]])
                centroid = np.mean(curr_train, axis=0)
                
                for j in range(1,n):
                    if j != x:
                        temp = np.empty((0,features_train[0].shape[1]))
                        for i in range(len(features_train)):
                            start,end = reps_train[i][j][t]
                            temp = np.vstack([temp, features_train[i][start:end,:]])
                        arr[index] = temp
                        index += 1
                train_smallest = self._smallest_magnitude(centroid,arr,index)
                centroid_small = np.mean(train_smallest, axis=0)
                
                covariance_matrix = np.cov(curr_train, rowvar=False)
                inverse_covariance = np.linalg.pinv(covariance_matrix)
                
                mu = distance.mahalanobis(centroid,centroid_small,inverse_covariance)
                class_sum += mu/2
            total_sum += ((class_sum/(n-1)) - mwSI) ** 2 
        return math.sqrt(total_sum/(len(reps_train[0][1])-1))

    def get_MSA(self, features_train, reps_train, n):
        """Mean Semi-Principal Axes.

        A measure that quantifies the size of a training ellipsoid.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the mean semi-principal axes.
        """
        total_sum = 0
        for x in range(1,n):
            train_perClass = []
            for r in range(len(reps_train[0][x])):
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][r]
                    train_perClass.append(features_train[i][start:end,:])
            train_perClass = np.vstack(train_perClass)
            covariance_matrix = np.cov(train_perClass, rowvar=False)
            eigenvalues, _ = np.linalg.eig(covariance_matrix)
            axes = np.sqrt(eigenvalues)
            total_product = np.prod(axes)
            total_sum += total_product ** (1/train_perClass.shape[1])
        return total_sum/(n-1)

    def get_CD(self, features_train, reps_train, n):
        """Centroid Drift.

        A measure that quantifies the variation in centroid location of 
        a training ellipsoid across subsequent repetitions.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the centroid drift.
        """
        total_sum = 0
        for x in range(1,n):
            reps_sum = 0
            for r in range(len(reps_train[0][x])-1):
                curr_train = np.empty((0,features_train[0].shape[1]))
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][r]
                    curr_train = np.vstack([curr_train, features_train[i][start:end,:]])

                centroid_1 = [np.mean(curr_train[:,k]) for k in range(curr_train.shape[1])]
                
                curr_train = np.empty((0,features_train[0].shape[1]))
                for i in range(len(features_train)):
                    start,end = reps_train[i][x][r+1]
                    curr_train = np.vstack([curr_train, features_train[i][start:end,:]])

                centroid_2 = [np.mean(curr_train[:,k]) for k in range(curr_train.shape[1])]
                
                reps_sum += np.linalg.norm(np.array(centroid_1) - np.array(centroid_2))
            total_sum += reps_sum
        return total_sum/(n-1)

    def _find_training(self,features_train,reps_train,n):
        train_combined = []
        for i in range(len(features_train)):
            for start,end in reps_train[i][n]:
                train_combined.append(features_train[i][start:end,:])
        return np.vstack(train_combined)

    def get_SI(self, features_train, reps_train, n):
        """Separability Index.

        A measure of interclass distance.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the separability index.
        """
        total_sum = 0
        for x in range(1,n):
            arr = [0] * (n-2)
            i = 0
            
            for j in range (1,n):
                if j != x:
                    arr[i] = self._find_training(features_train,reps_train,j)
                    i += 1
            curr_train = self._find_training(features_train,reps_train,x)
            cn = np.mean(curr_train, axis=0)
            
            covariance_matrix = np.cov(curr_train, rowvar=False)
            inverse_covariance = np.linalg.pinv(covariance_matrix)
            
            train_small = self._smallest_magnitude(cn,arr,(n-2))
            centroid_small = np.mean(train_small, axis=0)
            
            total_sum += (distance.mahalanobis(cn,centroid_small,inverse_covariance))/2
        return total_sum/(n-1)

    def get_mSI(self, features_train, reps_train, n):
        """Modified Separability Index.

        A measure similar to the separability index, except that it accounts for the covariance matrix of both distributions being compared.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the modified separability index.
        """
        total_sum = 0
        
        for x in range(1,n):
            arr = [0] * (n-2)
            i = 0
            for j in range(1,n):
                if j != x:
                    arr[i] = self._find_training(features_train,reps_train,j)
                    i += 1
            curr_train = self._find_training(features_train, reps_train,x)
            centroid = np.mean(curr_train, axis=0)
            
            training_small = self._smallest_magnitude(centroid,arr,(n-2))
            centroid_small = np.mean(training_small, axis=0)
            
            covariance_matrix_j = np.cov(curr_train, rowvar=False)
            covariance_matrix_i = np.cov(training_small, rowvar=False)
            avg_cov = (covariance_matrix_j + covariance_matrix_i) / 2
            inverse_covariance = np.linalg.pinv(avg_cov)
            
            mu = distance.mahalanobis(centroid,centroid_small,inverse_covariance)
            total_sum += mu/2
        return total_sum/(n-1)

    def _get_bhattacharyya_distance(self, train_j, train_i):
        mu_j = np.mean(train_j, axis=0)
        mu_i = np.mean(train_i, axis=0)

        cov_j = np.cov(train_j, rowvar=False)
        cov_i = np.cov(train_i, rowvar=False)

        S = (cov_j + cov_i) / 2
        S_inv = np.linalg.pinv(S)

        diff = mu_j - mu_i
        term1 = (1/8) * (diff @ S_inv @ diff.T)

        _, logdet_S = np.linalg.slogdet(S)
        _, logdet_j = np.linalg.slogdet(cov_j)
        _, logdet_i = np.linalg.slogdet(cov_i)
        term2 = 0.5 * (logdet_S - 0.5 * (logdet_j + logdet_i))

        return term1 + term2

    def get_BD(self, features_train, reps_train, n):
        """Bhattacharyya Distance.

        A measure of the statistical similarity between two distributions.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the bhattacharyya distance.
        """
        total_sum = 0

        for x in range(1, n):
            train_j = self._find_training(features_train,reps_train,x)

            distances = []
            for i in range(1, len(reps_train[0][x])):
                if i != x:
                    train_i = self._find_training(features_train, reps_train ,i)

                    bd = self._get_bhattacharyya_distance(train_j,train_i)

                    distances.append(bd)

            total_sum += min(distances)

        return total_sum / (n-1)

    def _get_KL_divergence(self,train_j, train_i):
        mu_j = np.mean(train_j, axis=0)
        mu_i = np.mean(train_i, axis=0)

        cov_j = np.cov(train_j, rowvar=False)
        cov_i = np.cov(train_i, rowvar=False)

        S_inv = np.linalg.pinv(cov_j)
        
        term1 = np.trace(S_inv @ cov_i)
        
        diff = mu_j - mu_i
        term2 = (diff @ S_inv @ diff.T)
        
        term3 = 24
        
        _, logdet_j = np.linalg.slogdet(cov_j)
        _, logdet_i = np.linalg.slogdet(cov_i)
        term4 = logdet_j - logdet_i
        
        return 0.5 * (term1 + term2 - term3 + term4)

    def get_KLD(self, features_train, reps_train, n):
        """Kullback-Leibler Divergence.

        A measure of how well a distribution can be approximated by a reference distribution.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the kullback-leibler divergence.
        """
        total_sum = 0

        for x in range(1, n):
            train_j = self._find_training(features_train,reps_train,x)

            distances = []

            for i in range(1, n):
                if i != x:
                    train_i = self._find_training(features_train,reps_train,i)

                    kl = self._get_KL_divergence(train_j,train_i)

                    distances.append(kl)

            total_sum += min(distances)

        return total_sum / (n-1)

    def _get_hellinger_distance(self,train_j, train_i):
        mu_j = np.mean(train_j, axis=0)
        mu_i = np.mean(train_i, axis=0)

        cov_j = np.cov(train_j, rowvar=False)
        cov_i = np.cov(train_i, rowvar=False)
        S = (cov_j + cov_i) / 2
        S_inv = np.linalg.pinv(S)
        
        temp = (np.linalg.det(cov_i) ** (1/4)) * (np.linalg.det(cov_j) ** (1/4))
        term1 = temp / (np.linalg.det(S) ** (1/2))
        
        diff = mu_j - mu_i
        term = (diff @ S_inv @ diff.T)
        
        term2 = np.exp(-(1/8) * term)
        
        return 1 - (term1 * term2)

    def get_HD(self, features_train, reps_train, n):
        """Hellinger Distance.

        A measure that quantifies the similarity between two probability distributions.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.
        Returns
        ----------
        float
            Returns the hellinger distance.
        """
        total_sum = 0

        for x in range(1, n):
            train_j = self._find_training(features_train,reps_train,x)

            distances = []

            for i in range(1, n):
                if i != x:
                    train_i = self._find_training(features_train,reps_train,i)

                    hd = self._get_hellinger_distance(train_j,train_i)

                    distances.append(hd)

            total_sum += min(distances)
        return total_sum / (n-1)

    def _get_volume_overlap_region(self,train_i,train_j):
        total_product = 1 
        for k in range(train_i.shape[1]):
            temp = [0] * 2
            temp[0] = max(train_j[:,k])
            temp[1] = max(train_i[:,k])
            term1 = min(temp)
            term3 = max(temp)
            
            temp[0] = min(train_j[:,k])
            temp[1] = min(train_i[:,k])
            term2 = max(temp)
            term4 = min(temp)
            
            total_product = total_product * ((term1-term2)/(term3-term4))
        return total_product

    def get_VOR(self, features_train, reps_train, n):
        """Volume of Overlap Region.

        A measure of the degree of overlap between the tails of the two class conditional distributions.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the volume of overlap region.
        """
        total_sum = 0

        for x in range(1, n):
            train_j = self._find_training(features_train,reps_train,x)

            regions = []

            for i in range(1, n):
                if i != x:
                    train_i = self._find_training(features_train,reps_train,i)

                    vor = self._get_volume_overlap_region(train_j,train_i)

                    regions.append(vor)

            total_sum += max(regions)

        return total_sum / (n-1)

    def _get_feature_efficiency(self,train_i, train_j):
        fek = np.zeros(train_i.shape[1])
        
        max_i = np.max(train_i, axis= 0)
        min_i = np.min(train_i, axis= 0)
        max_j = np.max(train_j, axis= 0)
        min_j = np.min(train_j, axis= 0)
        
        for i in range(train_i.shape[1]):
            upper_bound = min(max_i[i],max_j[i])
            lower_bound = max(min_i[i],min_j[i])
            
            overlap_i = np.sum((train_i[:,i] <= upper_bound) & (train_i[:,i] >= lower_bound))
            overlap_j = np.sum((train_j[:,i] <= upper_bound) & (train_j[:,i] >= lower_bound))
            
            total_overlap = overlap_i + overlap_j
            totalPoints = len(train_i) + len(train_j)
            
            fek[i] = (totalPoints - total_overlap)/totalPoints
        
        return max(fek) 

    def get_FE(self, features_train, reps_train, n):
        """Feature Efficiency.

        A measure of the fraction of points separable by a particular feature.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the feature efficiency.
        """
        total_sum = 0
        for x in range(1,n):
            train_i = self._find_training(features_train,reps_train,x)
            val = []
            
            for j in range(1,n):
                if(j != x):
                    train_j = self._find_training(features_train,reps_train,j)
                    val.append(self._get_feature_efficiency(train_i,train_j))
            total_sum += max(val)
            
        return total_sum/(n-1)

    def _get_withinClass_scatter(self,features_train,reps_train,n):
        total_sum = 0
        for j in range(1,n):
            train_j = self._find_training(features_train,reps_train,j)
            mu_j = np.mean(train_j,axis=0)
            n_sum = 0
            for i in range(len(train_j)):
                x = train_j[i]
                diff = (x - mu_j).reshape(-1,1)
                n_sum += diff @ diff.T

            total_sum += n_sum
        return total_sum/(n-1)

    def _get_betweenClass_scatter(self,features_train,reps_train,n):
        total_sum = 0
        data = []
        for x in range(1,n):
            data.append(self._find_training(features_train,reps_train,x))
        data = np.vstack(data)
        mean_j = np.mean(data,axis=0)
        
        for j in range(1,n):
            train_j = self._find_training(features_train,reps_train,j)
            mu_j = np.mean(train_j,axis=0)
            temp = len(train_j)
            diff = (mu_j - mean_j).reshape(-1,1)
            total_sum += temp * (diff @ diff.T)
        return total_sum/(n-1)

    def get_TSM(self, features_train, reps_train, n):
        """Trace of the Within-Class and Between_Class Scatter Matrices.

        A measure of class discriminability.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the trace of within-class and between-class scatter matrices.
        """
        sw = self._get_withinClass_scatter(features_train,reps_train,n)
        sb = self._get_betweenClass_scatter(features_train,reps_train,n)
        sw_inv = np.linalg.pinv(sw)
        
        return np.trace(sw_inv @ sb)

    def get_DS(self, features_train, features_test, reps_train, reps_test, n):
        """Desirability Score.

        A function of the separability index, the mean semi-principal axes, and the repeatabilty index.

        Parameters
        ----------
        features_train: numpy.ndarray
            The training data is filtered, windowed and featured as desired and put into an array.
        features_test: numpy.ndarray
            The testing data is filtered, windowed and features as desired and put into an array.
        reps_train: dictionary
            A dictionary containing the indecies of each window of the training data. 
        reps_test: dictionary
            A dictionary containing the indecies of each window of the testing data.
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the desirability score.
        """
        si = self.get_SI(features_train,reps_train,n)
        ri = self.get_RI(features_train,features_test,reps_train,reps_test,n) 
        msa = self.get_MSA(features_train,reps_train,n) 
        return ((si)/((ri)*(msa)))
        
    def _get_discriminant_ratio(self,train_j, train_i):
        mu_j = np.mean(train_j, axis=0)
        mu_i = np.mean(train_i, axis=0)

        sigma_j = np.var(train_j,axis=0,ddof=1)
        sigma_i = np.var(train_i,axis=0,ddof=1)
        
        temp = mu_j - mu_i
        term1 = temp ** 2

        term2 = sigma_j + sigma_i
        
        ratio = term1 / term2
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0)
        return max(ratio)
    
    def get_FDR(self, features_train, features_test, reps_train, reps_test, n):
        """Fisher's Discriminant Ratio.

        A measure of the degree of feature discrimination between two motion classes.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        features_test: list
            A list of testing data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        reps_test: dictionary
            A dictionary containing the indices of each window of the testing data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the fisher's discriminant ratio.
        """
        total_sum = 0

        for x in range(1, n):
            train_j = self._find_training(features_train,reps_train,x)

            distances = []

            for i in range(1, n):
                if i != x:
                    train_i = self._find_training(features_train,reps_train,i)
                    
                    fdr = self._get_discriminant_ratio(train_j,train_i )

                    distances.append(fdr) 

            total_sum += min(distances)

        return total_sum / (n-1)

    def _get_partitioning(self,data, resolution):
        samples, features = data.shape

        min_j = np.min(data, axis=0)
        max_j = np.max(data, axis=0)

        R_j = max_j - min_j
        d_j = R_j / (resolution + 1)

        cells = {}

        for row_idx in range(samples):
            cell_index = []

            for j in range(features):
                if d_j[j] == 0:
                    interval = 0
                else:
                    interval = int((data[row_idx, j] - min_j[j]) / d_j[j])

                    # handles max value edge case
                    if interval == resolution + 1:
                        interval = resolution

                cell_index.append(interval)

            cell_index = tuple(cell_index)

            if cell_index not in cells:
                cells[cell_index] = []

            cells[cell_index].append(row_idx)

        return cells

    def _get_labels_from_reps(self,n_rows, reps):
        labels = np.full(n_rows, -1)

        for cls, ranges in reps.items():
            for start, end in ranges:
                labels[start:end] = cls

        return labels

    def _get_combine_reps(self, reps_train, offset):
        new_rep = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: []
        }
        for i in range(len(reps_train)):
            for key in reps_train[i]:
                for start,end in reps_train[i][key]:
                    new_rep[key].append((start+offset[i],end+offset[i]))
        return new_rep

    def get_PU(self, features_train, reps_train):
        """Purity.

        A measure derived from the PRISM framework that assesses the homogeneity of the training data.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'

        Returns
        ----------
        float
            Returns the purity.
        """
        train = np.vstack(features_train)
        N = train.shape[0]
        B = [0] * 32
        Sh = []
        x = []
        index = 31
        for k in range(32):  # B = 0 to 31
            cells = self._get_partitioning(train, k)
            if k!= 0 and len(B[k-1]) == len(cells):
                index = k-1
                break
            else:
                B[k] = cells
        offset = [0] + [features_train[k].shape[0] for k in range(len(reps_train) - 1)]
        labels = self._get_labels_from_reps(train.shape[0],self._get_combine_reps(reps_train,offset))
        for B in range(index+1):
            cells = self._get_partitioning(train,B)
            SH = 0
            for cell, rows in cells.items():
                # labels of the rows in this cell
                cell_labels = labels[rows]

                counts = Counter(cell_labels) 
                k = len(counts)

                if k == 1:
                    SH_l = 1
                else:
                    temp = 0

                    for cls, count in counts.items():
                        p_il = count / len(rows)
                        temp += (p_il - 1/k)**2

                    SH_l = np.sqrt((k/(k-1)) * temp)

                SH += SH_l * (len(rows) / N)

            w = 1 / (2 ** B)
            Sh.append(SH * w)
            x.append(B / index if index != 0 else 0)

        max_curve = [1 / (2 ** B) for B in range(index + 1)]
        max_area = np.trapz(max_curve, x)

        ASh = np.trapz(Sh, x) / max_area

        return ASh

    def _phi(self,items, labels, index):
        target_label = labels[index]

        distances = []

        for i in range(len(items)):
            if i != index:
                distances.append((math.dist(items[index], items[i]), labels[i]))

        distances.sort(key=lambda x: x[0]) #sort the list by distance

        max_k = min(len(distances), 11) #only consider first <11 neighbours

        if max_k == 0:
            return None

        p_values = []
        same = 0
        for k, (_, label) in enumerate(distances[:max_k], start=1):
            if label == target_label:
                same += 1

            p_k = same / k
            p_values.append(p_k)

        return np.trapz(p_values, dx=1) / max_k

    def get_NS(self, features_train, reps_train):
        """Neighborhood Separability.

        A measure derived from the PRISM framework that focuses on the class decision 
        boundaries by quantifying the relationship between nearest neighbors.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'

        Returns
        ----------
        float
            Returns the neighborhood separability.
        """
        train = np.vstack(features_train)
        N = train.shape[0]

        B_cells = [0] * 32
        index = 31

        for k in range(32):
            cells = self._get_partitioning(train, k)

            if k != 0 and len(B_cells[k-1]) == len(cells):
                index = k - 1
                break
            else:
                B_cells[k] = cells

        offset = [0] + [features_train[k].shape[0] for k in range(len(reps_train) - 1)]
        labels = self._get_labels_from_reps(train.shape[0],self._get_combine_reps(reps_train, offset))

        SNN_values = []
        x = []

        for B in range(index + 1):
            cells = self._get_partitioning(train, B)
            SNN = 0

            for cell, rows in cells.items():
                cell_data = train[rows]
                cell_labels = labels[rows]

                phi_values = []

                for j in range(len(rows)):
                    phi_j = self._phi(cell_data, cell_labels, j)

                    if phi_j is not None:
                        phi_values.append(phi_j)

                if len(phi_values) == 0:
                    continue

                p_l = np.mean(phi_values)
                SNN += p_l * (len(rows) / N)

            w = 1 / (2 ** B) #weight
            SNN_values.append(SNN * w)
            x.append(B / index if index != 0 else 0)

        max_curve = [1 / (2 ** B) for B in range(index + 1)]
        max_area = np.trapz(max_curve, x)

        ASNN = np.trapz(SNN_values, x) / max_area #can't divide by 0.702 because we're not using all 32 samples

        return ASNN

    def get_CE(self, features_train, reps_train):
        """Collective Entropy.

        A measure derived from the PRISM framework that represents the accumulates uncertainty in
        the data across different resolutions.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'

        Returns
        ----------
        float
            Returns the collective entropy.
        """
        train = np.vstack(features_train)
        N = train.shape[0]
        B = [0] * 32
        Ec = []
        x = []
        index = 31
        for k in range(32):  # B = 0 to 31
            cells = self._get_partitioning(train, k)
            if k!= 0 and len(B[k-1]) == len(cells):
                index = k-1
                break
            else:
                B[k] = cells
        
        offset = [0] + [features_train[k].shape[0] for k in range(len(reps_train) - 1)]
        labels = self._get_labels_from_reps(train.shape[0],self._get_combine_reps(reps_train,offset))
        for B in range(index+1):
            cells = self._get_partitioning(train,B)
            E = 0
            for cell, rows in cells.items():
                # labels of the rows in this cell
                cell_labels = labels[rows]

                counts = Counter(cell_labels) 
                E_l = 0
                for cls, count in counts.items():
                    p_il = count / len(rows)
                    E_l += -p_il * np.log(p_il)

            E += E_l * (len(rows) / N)
            
            EC = 1 - E
            w = 1 / (2 ** B)
            Ec.append(EC * w)
            x.append(B / index if index != 0 else 0)

        max_curve = [1 / (2 ** B) for B in range(index + 1)]
        max_area = np.trapz(max_curve, x)

        ASE = np.trapz(Ec, x) / max_area
        return ASE

    def get_C(self, features_train, reps_train):
        """Compactness.

        A measure derived from the PRISM framework that provides an estimate of the spread of the data.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'

        Returns
        ----------
        float
            Returns the compactness.
        """
        train = np.vstack(features_train)
        N = train.shape[0]
        B = [0] * 32
        Cp = []
        x = []
        index = 31
        for k in range(32):  # B = 0 to 31
            cells = self._get_partitioning(train, k)
            if k!= 0 and len(B[k-1]) == len(cells):
                index = k-1
                break
            else:
                B[k] = cells
        
        offset = [0] + [features_train[k].shape[0] for k in range(len(reps_train) - 1)]
        labels = self._get_labels_from_reps(train.shape[0],self._get_combine_reps(reps_train,offset))
        for B in range(index+1):
            cells = self._get_partitioning(train,B)
            NEC = len(cells)/N #non-empty cell
            w = 1-(1 / (2 ** B))
            Cp.append(NEC * w)
            x.append(B / index if index != 0 else 0)

        max_curve = [1-(1 / (2 **B)) for B in range(index + 1)]
        max_area = np.trapz(max_curve, x)

        C = np.trapz(Cp, x) / max_area

        return C

    def _get_stopping(self,features, labels, cell_indices):
        box_features = features[cell_indices]
        box_labels = labels[cell_indices]
        total_samples_N = len(labels) 

        unique_classes = np.unique(box_labels)
        if len(unique_classes) <= 1:
            return True
            
        if len(box_labels) < (total_samples_N ** 0.375):
            return True

        centroids = {}
        for c in unique_classes:
            centroids[c] = np.mean(box_features[box_labels == c], axis=0)
            
        for class_A, class_B in combinations(unique_classes, 2):
            line_vector = centroids[class_B] - centroids[class_A]
            norm = np.linalg.norm(line_vector)
            if norm == 0: 
                return False
            line_unit_vector = line_vector / norm
            
            projections = np.dot(box_features, line_unit_vector)

            class_ranges = {}
            for c in unique_classes:
                class_projs = projections[box_labels == c]
                class_ranges[c] = (np.min(class_projs), np.max(class_projs))
                
            overlap_found = False
            for c1, c2 in combinations(unique_classes, 2):
                min1, max1 = class_ranges[c1]
                min2, max2 = class_ranges[c2]
                if not (max1 < min2 or max2 < min1):
                    overlap_found = True
                    break 
                    
            if not overlap_found:
                return True
                
        return False

    def _get_split_cell(self,features, indices):
        cell_feats = features[indices]

        ranges = cell_feats.max(axis=0) - cell_feats.min(axis=0)
        split_feature = np.argmax(ranges)

        median_value = np.median(cell_feats[:, split_feature])

        left_indices = indices[cell_feats[:, split_feature] <= median_value]
        right_indices = indices[cell_feats[:, split_feature] > median_value]

        return left_indices, right_indices, split_feature, median_value

    def get_CDM(self, features_train, reps_train):
        """Class Discriminability Measure.

        A measure derived from the adaptive partitioning algorithm that provides information about 
        the relationship between clusters in feature space.

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'

        Returns
        ----------
        float
            Returns the class discriminability measure.
        """
        train = np.vstack(features_train)
        offset = [0] + [features_train[k].shape[0] for k in range(len(reps_train) - 1)]
        reps = self._get_combine_reps(reps_train,offset)
        
        labels = self._get_labels_from_reps(train.shape[0], reps)
        N = train.shape[0]

        indices = np.arange(N)
        cells_to_process = [indices]
        terminal_cells = []

        while len(cells_to_process) > 0:
            current = cells_to_process.pop()

            if self._get_stopping(train, labels, current):
                terminal_cells.append(current)

            else:
                left, right, split_feature, median_value = self._get_split_cell(train, current)

                cells_to_process.append(left)
                cells_to_process.append(right)

        total = 0

        for cell in terminal_cells:

            cell_labels = labels[cell]

            k_i = len(cell_labels)

            counts = np.bincount(cell_labels)

            majority = counts.max()

            total += k_i - majority

        return total / N

    def _find_label(self,t,rep,n):
        for key in range(n):
            for start,end in rep[key]:
                if start <= t and end > t:
                    return key

    def get_ICF(self, features_train, reps_train, n):
        """Inter-Class Fraction.

        A measure describing the ratio of the number of inter-class nearest neighbors to the total number of samples in the data set

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the inter-class fraction result.
        """
        total_sum = 0
        samples = np.sum(len(features_train[k]) for k in range(len(features_train)))
        data = np.vstack(features_train)
        offset = [0] + [features_train[k].shape[0] for k in range(len(reps_train) - 1)]
        reps = self._get_combine_reps(reps_train,offset)

        for t in range(1,samples):
            train_t = data[t]
            val = []
            rows = []
            for e in range(1,samples):
                if e != t:
                    temp = data[e]
                    val.append(temp)
                    rows.append(e)
                
            distances = []
            for i in range(len(val)):
                distances.append(distance.euclidean(train_t,val[i]))
            
            index_e = np.argmin(distances)
        
            if self._find_label(t,reps,n) != self._find_label(rows[index_e],reps,n):
                total_sum += 1
        return total_sum/samples

    def get_IIF(self, features_train, reps_train, n):
        """Intra-Inter Fraction.

        A measure describing the ratio of the average euclidean distance of intra-class
        nearest neighbors to the average euclidean distance of the inter-class nearest neighbors. 

        Parameters
        ----------
        features_train: list
            A list of training data of type np.array. Ensure the data is filtered, windowed and featured as desired.
        reps_train: dictionary
            A dictionary containing the indices of each window of the training data. Ensure the non-active class is 0, 
                if that is not the case please refer to function 'get_reps_dictionary_rest_zero'
        n: integer
            Number of classes.

        Returns
        ----------
        float
            Returns the intra-inter fraction result.
        """
        num_sum = 0
        dnum_sum = 0
        samples = np.sum(len(features_train[k]) for k in range(len(features_train)))
        data = np.vstack(features_train)
        offset = [0] + [features_train[k].shape[0] for k in range(len(reps_train) - 1)]
        reps = self._get_combine_reps(reps_train,offset)
        for t in range(1,samples):
            train_t = data[t]
            val = []
            rows = []
            for e in range(1,samples):
                if e != t:
                    temp = data[e]
                    val.append(temp)
                    rows.append(e)
                
            distances = []
            for i in range(len(val)):
                distances.append(distance.euclidean(train_t,val[i]))
            
            index_e = np.argmin(distances)
            term1 = min(distances)
            
            if self._find_label(t,reps,n) == self._find_label(rows[index_e],reps,n):
                num_sum += term1
                
            elif self._find_label(t,reps,n) != self._find_label(rows[index_e],reps,n):
                dnum_sum += term1
        return num_sum/dnum_sum

    def get_AER(self, y_true, y_predictions, null_class):
        """Active Error.

        Classification accuracy on active classes (i.e., all classes but no movement/rest). Rejected samples are ignored.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.
        null_class: int
            The null class that shouldn't be considered.

        Returns
        ----------
        float
            Returns the active error.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        nm_predictions = [i for i, x in enumerate(y_predictions) if x == null_class]
        return 1 - self.get_CA(np.delete(y_true, nm_predictions), np.delete(y_predictions, nm_predictions))

    def get_INS(self, y_true, y_predictions):
        """Instability.

        The number of subsequent predicitons that change normalized by the total number of predicitons.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        float
            Returns the instability.
        """
        num_gt_changes = np.count_nonzero(y_true[:-1] != y_true[1:])
        pred_changes = np.count_nonzero(y_predictions[:-1] != y_predictions[1:])
        ins = (pred_changes - num_gt_changes) / len(y_predictions)
        return ins if ins > 0 else 0.0

    def get_REJ_RATE(self, y_predictions):
        """Rejection Rate.

        The number of rejected predictions, normalized by the total number of predictions.

        Parameters
        ----------
        y_predictions: numpy.ndarray
            A list of predicted labels. -1 in the list correspond to rejected predictions.

        Returns
        ----------
        float
            Returns the rejection rate.
        """
        return sum(y_predictions == -1)/len(y_predictions)
    
    def get_CONF_MAT(self, y_true, y_predictions):
        """Confusion Matrix.

        A NxN matric where N is the number of classes. Each column represents the predicted class and 
        each row represents the true class. 

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns the confusion matrix.
        """
        classes = np.sort(np.unique(y_true))
        conf_mat = np.zeros(shape=(len(classes), len(classes)))
        for row in range(0, len(classes)):
            c_true = np.where(y_true == classes[row])
            for col in range(0, len(classes)):
                conf_mat[row,col] = len(np.where(y_predictions[c_true] == classes[col])[0])
        return conf_mat

    def get_RECALL(self, y_true, y_predictions):
        """Recall Score.

        The recall is simply: True Positives / (True Positives + False Negatives).
        This metric takes into account the corresponding weights of each class.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the recall for each class.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        recall, weights = self._get_RECALL_helper(y_true, y_predictions)
        return np.average(recall, weights=weights)

    def _get_RECALL_helper(self, y_true, y_predictions):
        classes = np.sort(np.unique(y_true))
        recall = np.zeros(shape=(len(classes)))
        weights = np.zeros(shape=(len(classes)))
        for c in range(0, len(classes)):
            c_true = np.where(y_true == classes[c])
            tp = len(np.where(y_predictions[c_true] == classes[c])[0])
            fn = len(np.where(y_predictions[c_true] != classes[c])[0])
            weights[c] = len(c_true[0])/len(y_true)
            recall[c] = tp / (tp + fn)
        return recall, weights

    def get_PREC(self, y_true, y_predictions):
        """Precision Score.

        The precision is simply: True Positives / (True Positives + False Positive).
        This metric takes into account the corresponding weights of each class.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the precision for each class.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        precision, weights = self._get_PREC_helper(y_true, y_predictions)
        return np.average(precision, weights=weights)
    
    def _get_PREC_helper(self, y_true, y_predictions):
        classes = np.sort(np.unique(y_true))
        precision = np.zeros(shape=(len(classes)))
        weights = np.zeros(shape=(len(classes)))
        for c in range(0, len(classes)):
            c_true = np.where(y_true == classes[c])
            c_false = np.where(y_true != classes[c])
            tp = len(np.where(y_predictions[c_true] == classes[c])[0])
            fp = len(np.where(y_predictions[c_false] == classes[c])[0])
            weights[c] = len(c_true[0])/len(y_true)
            precision[c] = tp / (tp + fp)
        return precision, weights

    def get_F1(self, y_true, y_predictions):
        """F1 Score.

        The f1 score is simply: 2 * (Precision * Recall)/(Precision + Recall).
        This metric takes into account the corresponding weights of each class.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the f1 score for each class.
        """
        y_predictions, y_true = self._ignore_rejected(y_predictions, y_true)
        prec, weights = self._get_PREC_helper(y_true, y_predictions)
        recall, _ = self._get_RECALL_helper(y_true, y_predictions)
        f1 = 2 * (prec * recall) / (prec + recall)
        return np.average(f1, weights=weights)  
    
    def get_R2(self, y_true, y_predictions):
        """R2 score.

        The R^2 score measures how well a regression model captures the variance in the predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the R2 score for each DOF.
        """
        ssr = np.sum((y_predictions - y_true) ** 2, axis=0)
        sst = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
        r2 = 1 - ssr/sst
        return r2
    
    def get_MSE(self, y_true, y_predictions):
        """Mean squared error.

        The MSE measures the averages squared errors between the true labels and predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the MSE score for each DOF.
        """
        values = (y_true - y_predictions) ** 2
        mse = np.sum(values, axis=0) / y_true.shape[0]
        return mse

    def get_MAPE(self, y_true, y_predictions):
        """Mean absolute percentage error.

        The MAPE measures the average error between the true labels and predictions as a percentage of the true value.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the MAPE score for each DOF.
        """
        values = np.abs((y_true - y_predictions) / np.maximum(np.abs(y_true), np.finfo(np.float64).eps))    # some values could be 0, so take epsilon if that's the case to avoid inf
        mape = np.sum(values, axis=0) / y_true.shape[0]
        return mape

    def get_RMSE(self, y_true, y_predictions):
        """Root mean square error.

        The RMSE measures the square root of the MSE.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the RMSE score for each DOF.
        """
        values = (y_true - y_predictions) ** 2
        mse = np.sum(values, axis=0) / y_true.shape[0]
        rmse = np.sqrt(mse)
        return rmse

    def get_NRMSE(self, y_true, y_predictions):
        """Normalized root mean square error.

        The NRMSE measures the RMSE normalized by the range of possible values.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the RMSE score for each DOF.
        """
        values = (y_true - y_predictions) ** 2
        mse = np.sum(values, axis=0) / y_true.shape[0]
        nrmse = np.sqrt(mse) / (y_true.max(axis=0) - y_true.min(axis=0))
        return nrmse

    def get_MAE(self, y_true, y_predictions):
        """Mean absolute error.

        The MAE measures the average L1 error between the true labels and predictions.

        Parameters
        ----------
        y_true: numpy.ndarray
            A list of ground truth labels.
        y_predictions: numpy.ndarray
            A list of predicted labels.

        Returns
        ----------
        list
            Returns a list consisting of the MAE score for each DOF.
        """
        residuals = np.abs(y_predictions - y_true)
        mae = np.mean(residuals, axis=0)
        return mae 
    

    def visualize(self, dic, y_axis=[0,1]):
        """Visualize the computed metrics in a bar chart.

        Parameters
        ----------
        dic: dict
            The output from the extract_offline_metrics function.
        y_axis: dict (optional), default=[0,1]
            A dictionary for lower and upper bounds of y axis. 
        """
        plt.style.use('ggplot')
        plt.title('Offline Metrics')
        plt.gca().set_ylim(y_axis)
        x = []
        y = []
        for key in dic:
            if key != 'CONF_MAT':
                x.append(key)
                y.append(dic[key])
        plt.bar(x,y)
        plt.show()
    
    def visualize_conf_matrix(self, mat, labels=None):
        """Visualize the 2D confusion matrix.

        Parameters
        ----------
        mat: list (2D)
            A NxN confusion matrix.
        """
        disp = ConfusionMatrixDisplay(confusion_matrix = mat, display_labels = labels)
        disp.plot()
        plt.show()