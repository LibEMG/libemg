from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Lock
import numpy as np
import pickle
import socket
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import time
import inspect
from scipy import stats
import csv
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re
from matplotlib.animation import FuncAnimation
from functools import partial

from libemg.feature_extractor import FeatureExtractor
from libemg.shared_memory_manager import SharedMemoryManager
from libemg.utils import get_windows
from libemg.environments.controllers import RegressorController, ClassifierController

class EMGPredictor:
    """Base class for EMG prediction. Parent class that shares common functionality between classifiers and regressors.

    Parameters
    ----------
    model: custom model (must have fit, predict and predict_proba functions)
        Object that will be used to fit and provide predictions.
    model_parameters: dictionary, default=None
        Mapping from parameter name to value based on the constructor of the specified model. Only used when a string is passed in for model.
    random_seed: int, default=0
        Constant value to control randomization seed.
    fix_feature_errors: bool (default=False)
        If True, the model will update any feature errors (INF, -INF, NAN) using the np.nan_to_num function.
    silent: bool (default=False)
        If True, the outputs from the fix_feature_errors parameter will be silenced. 
    """
    def __init__(self, 
                 model, 
                 model_parameters:   Optional[Dict[str, Any]] = None, 
                 random_seed:        int = 0, 
                 fix_feature_errors: bool = False, 
                 silent:             bool = False) -> None:
        
        self.model = model
        self.model_parameters = model_parameters
        self.feature_params = {}
        self.fix_feature_errors = fix_feature_errors
        self.silent = silent
        random.seed(random_seed)

    def fit(self, 
            feature_dictionary:    Optional[Dict[str, Any]] = None, 
            dataloader_dictionary: Optional[Dict[str, Any]] = None, 
            training_parameters:   Optional[Dict[str, Any]] = None) -> None:
        """The fit function for the EMG Prediction class. 

        This is the method called that actually optimizes model weights for the dataset. This method presents a fork for two 
        different kind of models being trained. The first we call "statistical" models (i.e., LDA, QDA, SVM, etc.)
        and these are interfaced with sklearn. The second we call "deep learning" models and these are designed to fit around
        the conventional programming style of pytorch. We distinguish which of these models are being trained by passing in a
        feature_dictionary for "statistical" models and a "dataloader_dictionary" for deep learning models.

        Parameters
        ----------
    
        feature_dictionary: dict
            A dictionary including the associated features and labels associated with a set of data. 
            Dictionary keys should include 'training_labels' and 'training_features'.
        dataloader_dictionary: dict
            A dictionary including the associated dataloader objects for the dataset you'd like to train with. 
            Dictionary keys should include 'training_dataloader', and 'validation_dataloader'.
        training_parameters: dict (optional)
            Training parameters passed to the fit() method of deep learning models (e.g., learning rate, num_epochs). Is not used
            for statistical models.
        """
        if training_parameters is None:
            # Convert to empty dictionary for compatibility with unpacking keywords
            training_parameters = {}
        if feature_dictionary is not None:
            self._fit_statistical_model(feature_dictionary)
        elif dataloader_dictionary is not None:
            self._fit_deeplearning_model(dataloader_dictionary, training_parameters)
        else:
            raise ValueError("Incorrect combination of values passed to fit method. A feature dictionary is needed for statistical models and a dataloader dictionary is needed for deep models.")

    @classmethod
    def from_file(self, 
                  filename: str) -> "EMGPredictor":
        """Loads a classifier - rather than creates a new one.

        After saving a statistical model, you can recreate it by running EMGClassifier.from_file(). By default 
        this function loads a previously saved and pickled classifier. 

        Parameters
        ----------
        filename: string
            The file path of the pickled model. 

        Returns
        ----------
        EMGClassifier
            Returns an EMGClassifier object.

        Examples
        -----------
        >>> classifier = EMGClassifier.from_file('lda.pickle')
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def _predict(self, 
                 data: Any) -> Any:
        """
        Predict using the model.

        Parameters
        ----------
        data: np.ndarray or torch.tensor
            The input to be processed 

        Returns
        ----------
        prediction: int
             the output prediction (categorical)
        """
        try:
            return self.model.predict(data)
        except AttributeError as e:
            raise AttributeError("Attempted to perform prediction when model doesn't have a predict() method. Please ensure model has a valid predict() method.") from e

    def _predict_proba(self, 
                       data: Any) -> Any:
        """
        Predict probabilities using the model.

        Parameters
        ----------
        data: np.ndarray or torch.tensor
            The input to be processed 

        Returns
        ----------
        probabilities: np.ndarray or torch.tensor
             the output probabilities (continuous valued)
        """
        try:
            return self.model.predict_proba(data)
        except AttributeError as e:
            raise AttributeError("Attempted to perform prediction when model doesn't have a predict_proba() method. Please ensure model has a valid predict_proba() method.") from e

    def save(self, 
             filename: str) -> None:
        """Saves (pickles) the EMGClassifier object to a file.

        Use this save function if you want to load the object later using the from_file function. Note that 
        this currently only support statistical models (i.e., not deep learning).

        Parameters
        ----------
        filename: string
            The path of the outputted pickled file. 
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def install_feature_parameters(self, 
                                   feature_params: Dict[str, Any]) -> None:
        """Installs the feature parameters for the classifier.

        This function is used to install the feature parameters for the classifier. This is necessary for the classifier
        to know how to extract features from the raw data. This is used primarily by the OnlineEMGClassifier class.

        Parameters
        ----------
        feature_params: dict
            A dictionary containing the feature parameters. 
        """
        self.feature_params = feature_params

    @staticmethod
    def _validate_model_parameters(model, 
                                   model_parameters: Optional[Dict[str, Any]], 
                                   model_config:     Dict[str, Any]) -> Any:
        """
        Provide a string representing a sklearn model and this function will validate if the model parameter dictionary is valid
        by checking the sklearn model constructor arguments.
        """
        if not isinstance(model, str):
            # Custom model
            return model
        valid_models = list(model_config.keys())
        assert model in valid_models, f"Please pass in one of the approved models: {valid_models}."
        
        model_reference, default_parameters = model_config[model]
        valid_parameters = default_parameters

        if model_parameters is not None:
            signature = list(inspect.signature(model_reference).parameters.keys())
            for p in model_parameters:
                if p in signature:
                    valid_parameters[p] = model_parameters[p]
                else:
                    print(str(p) + "is an invalid parameter.")

        valid_model = model_reference(**valid_parameters)
        return valid_model

    def _format_data(self, 
                     feature_dictionary: Union[Dict[str, Any], Any]) -> Any:
        """
        Format dictionary format of features into a single np.ndarray.
        """
        if isinstance(feature_dictionary, dict):
            # Loop through each element and stack
            arr = None
            for feat in feature_dictionary:
                if arr is None:
                    arr = feature_dictionary[feat]
                else:
                    arr = np.hstack((arr, feature_dictionary[feat]))
        else:
            arr = feature_dictionary

        if self.fix_feature_errors:
            if FeatureExtractor().check_features(arr, self.silent):
                arr = np.nan_to_num(arr, neginf=0, nan=0, posinf=0) 
        return arr

    def _fit_statistical_model(self, 
                               feature_dictionary: Dict[str, Any]) -> None:
        """
        Fit the model using a feature dictionary.
        """
        assert 'training_features' in feature_dictionary.keys()
        assert 'training_labels'   in feature_dictionary.keys()
        # convert dictionary of features format to np.ndarray for test/train set (NwindowxNfeature)
        feature_dictionary["training_features"] = self._format_data(feature_dictionary['training_features'])
        # self._set_up_classifier(model, feature_dictionary, parameters)
        self.model.fit(feature_dictionary['training_features'], feature_dictionary['training_labels'])
        
    def _fit_deeplearning_model(self, 
                                dataloader_dictionary: Dict[str, Any],
                                parameters:            Dict[str, Any]) -> None:
        """
        Fit a deep learning model using a dataloader dictionary.
        """
        assert 'training_dataloader' in dataloader_dictionary.keys()
        assert 'validation_dataloader'  in dataloader_dictionary.keys()
        self.model.fit(dataloader_dictionary, **parameters)

class EMGClassifier(EMGPredictor):
    """The Offline EMG Classifier. 

    This is the base class for any offline EMG classification. 

    Parameters
    ----------
    model: string or custom classifier (must have fit, predict and predic_proba functions)
        The type of machine learning model. Valid options include: 'LDA', 'QDA', 'SVM', 'KNN', 'RF' (Random Forest),  
        'NB' (Naive Bayes), 'GB' (Gradient Boost), 'MLP' (Multilayer Perceptron). Note, these models are all default sklearn 
        models with no hyperparameter tuning and may not be optimal. Pass in custom classifiers or parameters for more control.
    model_parameters: dictionary, default=None
        Mapping from parameter name to value based on the constructor of the specified model. Only used when a string is passed in for model.
    random_seed: int, default=0
        Constant value to control randomization seed.
    fix_feature_errors: bool (default=False)
        If True, the model will update any feature errors (INF, -INF, NAN) using the np.nan_to_num function.
    silent: bool (default=False)
        If True, the outputs from the fix_feature_errors parameter will be silenced. 
    """
    def __init__(self, 
                 model:              Union[str, Any], 
                 model_parameters:   Optional[Dict[str, Any]] = None, 
                 random_seed:        int = 0, 
                 fix_feature_errors: bool = False, 
                 silent:             bool = False):
        model_config = {
            'LDA': (LinearDiscriminantAnalysis, {}),
            'KNN': (KNeighborsClassifier, {"n_neighbors": 5}),
            'SVM': (SVC, {"kernel": "linear", "probability": True, "random_state": 0}),
            'QDA': (QuadraticDiscriminantAnalysis, {}),
            'RF': (RandomForestClassifier, {"random_state": 0}),
            'NB': (GaussianNB, {}),
            'GB': (GradientBoostingClassifier, {"random_state": 0}),
            'MLP': (MLPClassifier, {"random_state": 0, "hidden_layer_sizes": 126})
        }
        model = self._validate_model_parameters(model, model_parameters, model_config)
        super().__init__(model, model_parameters, random_seed=random_seed, fix_feature_errors=fix_feature_errors, silent=silent)

        self.velocity = False
        self.majority_vote = None
        self.rejection = False

        # For velocity control
        self.th_min_dic = None 
        self.th_max_dic = None 


        
    def run(self, 
            test_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the classifier on a pre-defined set of training data.

        Parameters
        ----------
        test_data: list
            A dictionary, np.ndarray of inputs appropriate for the model of the EMGClassifier.

        Returns
        ----------
        list
            A list of predictions, based on the passed in testing features.
        list
            A list of the probabilities (for each prediction), based on the passed in testing features.
        """
        test_data = self._format_data(test_data)
        prob_predictions = self._predict_proba(test_data)
            
        # Default
        predictions, probabilities = self._prediction_helper(prob_predictions)

        # Rejection
        if self.rejection:
            predictions = np.array([self._rejection_helper(predictions[i], probabilities[i]) for i in range(0,len(predictions))])
            rejected = np.where(predictions == -1)[0]
            predictions[rejected] = -1

        # Majority Vote
        if self.majority_vote:
            predictions = self._majority_vote_helper(predictions)

        # Accumulate Metrics
        return predictions, probabilities

    def add_rejection(self, 
                      threshold: float=0.9) -> None:
        """Adds the rejection post-processing block onto a classifier.

        Parameters
        ----------
        threshold: float (optional), default=0.9
            The confidence threshold (0-1). All predictions with confidence under the threshold will be rejected.
        """
        self.rejection = True
        self.rejection_threshold = threshold

    def add_majority_vote(self, 
                          num_samples: int=5) -> None:
        """Adds the majority voting post-processing block onto a classifier.

        Parameters
        ----------
        threshold: int (optional), default=5
            The number of samples that will be included in the majority vote.
        """
        self.majority_vote = num_samples

    def add_velocity(self, 
                     train_windows:           np.ndarray, 
                     train_labels:            np.ndarray,
                     velocity_metric_handle:  Optional[Callable[[Any], Any]] = None,
                     velocity_mapping_handle: Optional[Callable[[Any], Any]] = None):
        """Adds velocity (i.e., proportional) control where a multiplier is generated for the level of contraction intensity.

        Note, that when using this optional, ramp contractions should be captured for training. 

        Parameters
        -----------
        """
        self.velocity_metric_handle = velocity_metric_handle
        self.velocity_mapping_handle = velocity_mapping_handle
        self.velocity = True
        self.th_min_dic, self.th_max_dic = self._set_up_velocity_control(train_windows, train_labels)


    
    '''
    ---------------------- Private Helper Functions ----------------------
    '''
    def _prediction_helper(self, 
                           predictions: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to extract prediction and probability.
        
        Parameters
        ----------
        predictions : Any
            Raw predictions.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of predicted classes and probabilities.
        """
        probabilities = [] 
        prediction_vals = []
        for i in range(0, len(predictions)):
            pred_list = list(predictions[i])
            prediction_vals.append(pred_list.index(max(pred_list)))
            probabilities.append(pred_list[pred_list.index(max(pred_list))])
        return np.array(prediction_vals), np.array(probabilities)
        
    def _rejection_helper(self, 
                          prediction: Any, 
                          prob:       Any) -> Any:
        """
        Helper function for rejection.
        """
        if self.rejection:
            if prob > self.rejection_threshold:
                return prediction
            else:
                return -1
        return prediction
    
    def _majority_vote_helper(self, 
                              predictions: np.ndarray) -> np.ndarray:
        """
        Helper function for majority voting.
        """
        updated_predictions = []
        for i in range(0, len(predictions)):
            idxs = np.array(range(i-self.majority_vote+1, i+1))
            idxs = idxs[idxs >= 0]
            group = predictions[idxs]
            updated_predictions.append(stats.mode(group, keepdims=False)[0])
        return np.array(updated_predictions)
    
    def _get_velocity(self, 
                      window: Dict[str, Any], 
                      c:      Any) -> str:
        """
        Compute velocity output based on window data.
        
        Parameters
        ----------
        window : dict
            Window data.
        c : Any
            Class or index.
        
        Returns
        -------
        str
            Formatted velocity.
        """
        mod = "emg" # todo: specify another way to do this is needed
        
        if self.th_max_dic and self.th_min_dic:
            if self.velocity_metric_handle is None:
                velocity_metric = np.sum(np.mean(np.abs(window[mod]),2)[0], axis=0)
            else:
                velocity_metric = self.velocity_metric_handle(window[mod])
            
            velocity_output = (velocity_metric - self.th_min_dic[c])/(self.th_max_dic[c] - self.th_min_dic[c])
            if self.velocity_mapping_handle:
                velocity_output = self.velocity_mapping_handle(velocity_output)
            return '{0:.2f}'.format(min([1, max([velocity_output, 0])]))

    def _set_up_velocity_control(self, 
                                 train_windows: np.ndarray, 
                                 train_labels:  np.ndarray) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        Sets up velocity control thresholds.
        
        Returns
        -------
        Tuple[dict, dict]
            Dictionaries for min and max thresholds.
        """
        # Extract classes 
        th_min_dic = {}
        th_max_dic = {}
        classes = np.unique(train_labels)
        for c in classes:
            indices = np.where(train_labels == c)[0]
            c_windows = train_windows[indices]
            if self.velocity_metric_handle is None:
                velocity_metric = np.sum(np.mean(np.abs(c_windows),2), axis=1)
            else:
                velocity_metric = self.velocity_metric_handle(c_windows)
            # mav_tr = np.sum(np.mean(np.abs(c_windows),2), axis=1)
            tr_max = np.max(velocity_metric)
            tr_min = np.min(velocity_metric)
            # Calculate THmin 
            th_min = ((1-(10/100)) * tr_min) + (0.1 * tr_max)
            th_min_dic[c] = th_min 
            # Calculate THmax
            th_max = ((1-(70/100)) * tr_min) + (0.7 * tr_max)
            th_max_dic[c] = th_max
        return th_min_dic, th_max_dic

    def visualize(self, 
                  test_labels:   np.ndarray, 
                  predictions:   np.ndarray, 
                  probabilities: np.ndarray) -> None:
        """Visualize the decision stream of the classifier on the testing data. 

        You can call this visualize function to get a visual output of what the decision stream of what 
        the particular classifier looks like. 
        
        Parameters
        ----------
        test_labels: list
            A np.ndarray containing the labels for the test data.
        predictions: list
            A np.ndarray containing the preditions for the test data.
        probabilities: list
            A np.ndarray containing the probabilities from the classifier for the test data. This should be
            N samples x C classes.
        """
        assert len(predictions) > 0
        
        plt.style.use('ggplot')
        colors = {}

        plt.gca().set_ylim([0, 1.05])
        plt.gca().xaxis.grid(False)

        # Plot true class labels
        changed_locations = [0] + list(np.where((test_labels[:-1] != test_labels[1:]) == True)[0]) + [len(test_labels)-1]

        for i in range(1, len(changed_locations)):
            class_label = test_labels[changed_locations[i]]
            if class_label in colors.keys():
                plt.fill_betweenx([0,1.02], changed_locations[i-1], changed_locations[i], color=colors[class_label])
            else:
                val = plt.fill_betweenx([0,1.02], changed_locations[i-1], changed_locations[i], alpha=.2)
                colors[class_label] = val.get_facecolors().tolist()
            
        # Plot decision stream
        plt.title("Decision Stream")
        plt.xlabel("Class Output")
        plt.ylabel("Probability")
        for g in np.unique(predictions):
            i = np.where(predictions == g)[0]
            if g == -1:
                plt.scatter(i, probabilities[i], label=g, alpha=1, color='black')
            else:
                plt.scatter(i, probabilities[i], label=g, alpha=1, color=colors[g])
        
        plt.legend(loc='lower right')
        plt.show()
    
class EMGRegressor(EMGPredictor):
    """The Offline EMG Regressor. 

    This is the base class for any offline EMG regression. 

    Parameters
    ----------
    model: string or custom regressor (must have fit and predict functions)
        The type of machine learning model. Valid options include: 'LR' (Linear Regression), 'SVM' (Support Vector Machine), 'RF' (Random Forest),  
        'GB' (Gradient Boost), 'MLP' (Multilayer Perceptron). Note, these models are all default sklearn 
        models with no hyperparameter tuning and may not be optimal. Pass in custom regressors or parameters for more control.
    model_parameters: dictionary, default=None
        Mapping from parameter name to value based on the constructor of the specified model. Only used when a string is passed in for model.
    random_seed: int, default=0
        Constant value to control randomization seed.
    fix_feature_errors: bool (default=False)
        If True, the model will update any feature errors (INF, -INF, NAN) using the np.nan_to_num function.
    silent: bool (default=False)
        If True, the outputs from the fix_feature_errors parameter will be silenced. 
    deadband_threshold: float, default=0.0
        Threshold that controls deadband around 0 for output predictions. Values within this deadband will be output as 0 instead of their original prediction.
    """
    def __init__(self, 
                 model:              Union[str, Any],
                 model_parameters:   Optional[Dict[str, Any]] = None, 
                 random_seed:        int = 0, 
                 fix_feature_errors: bool = False, 
                 silent:             bool = False, 
                 deadband_threshold: float = 0.):
        model_config = {
            'LR': (LinearRegression, {}),
            'SVM': (SVR, {"kernel": "linear"}),
            'RF': (RandomForestRegressor, {"random_state": 0}),
            'GB': (GradientBoostingRegressor, {"random_state": 0}),
            'MLP': (MLPRegressor, {"random_state": 0, "hidden_layer_sizes": 126})
        }
        convert_to_multioutput = isinstance(model, str)
        model = self._validate_model_parameters(model, model_parameters, model_config)
        if convert_to_multioutput:
            model = MultiOutputRegressor(model)
        self.deadband_threshold = deadband_threshold
        super().__init__(model, model_parameters, random_seed=random_seed, fix_feature_errors=fix_feature_errors, silent=silent)

    
    def run(self, 
            test_data: Any) -> np.ndarray:
        """Runs the regressor on a pre-defined set of training data.

        Parameters
        ----------
        test_data: list
            A dictionary, np.ndarray of inputs appropriate for the model of the EMGRegressor.
    
        Returns
        ----------
        np.ndarray
            A list of predictions, based on the passed in testing features.
        """
        test_data = self._format_data(test_data)
        predictions = self._predict(test_data)

        # Set values within deadband to 0
        deadband_mask = np.abs(predictions) < self.deadband_threshold
        predictions[deadband_mask] = 0.

        return predictions

    def visualize(self, 
                  test_labels: np.ndarray, 
                  predictions: np.ndarray) -> None:
        """Visualize the decision stream of the regressor on test data.

        You can call this visualize function to get a visual output of what the decision stream looks like.

        Parameters
        ----------
        test_labels: np.ndarray
            N x M array, where N = # samples and M = # DOFs, containing the labels for the test data.
        predictions: np.ndarray
            N x M array, where N = # samples and M = # DOFs, containing the predictions for the test data.

        """
        assert len(predictions) > 0, 'Empty list passed in for predictions to visualize.'

        # Formatting
        plt.style.use('ggplot')
        fig, axs = plt.subplots(nrows=test_labels.shape[1], ncols=1, sharex=True, layout='constrained')
        fig.suptitle('Decision Stream')
        fig.supxlabel('Prediction Index')
        fig.supylabel('Model Output')

        marker_size = 5
        pred_color = 'black'
        label_color = 'blue'
        x = np.arange(test_labels.shape[0])
        handles = [mpatches.Patch(color=label_color, label='Labels'), mlines.Line2D([], [], color=pred_color, marker='o', markersize=marker_size, linestyle='None', label='Predictions')]
        for dof_idx, ax in enumerate(axs):
            ax.set_title(f"DOF {dof_idx}")
            ax.set_ylim((-1.05, 1.05))
            ax.xaxis.grid(False)
            ax.fill_between(x, test_labels[:, dof_idx], alpha=0.5, color=label_color)
            ax.scatter(x, predictions[:, dof_idx], color=pred_color, s=marker_size)

        fig.legend(handles=handles, loc='upper right')
        plt.show()
        

    def add_deadband(self, 
                     threshold: float) -> None:
        """Add a deadband around regressor predictions that will instead be output as 0.

        Parameters
        ----------
        threshold: float
            Deadband threshold. All output predictions from -threshold to +threshold will instead output 0.
        """
        self.deadband_threshold = threshold

class OnlineStreamer(ABC):
    """OnlineStreamer.

    This is a base class for using algorithms (classifiers/regressors/other) in conjunction with online streamers.


    Parameters
    ----------
    offline_predictor: EMGPredictor
        An EMGPredictor object. 
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    online_data_handler: OnlineDataHandler
        An online data handler object.
    features: list or None
        A list of features that will be extracted during real-time classification. These should be the 
        same list used to train the model. Pass in None if using the raw data (this is primarily for CNNs).
    file_path: str (optional)
        A location that the inputs and output of the classifier will be saved to.
    file: bool (optional)
        A toggle for activating the saving of inputs and outputs of the classifier.
    smm: bool (optional)
        A toggle for activating the storing of inputs and outputs of the classifier in the shared memory manager.
    smm_items: list (optional)
        A list of lists containing the tag, size, and multiprocessing locks for shared memory.
    parameters: dict (optional)
        A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
        in the sklearn docs for the given model.
    velocity: bool (optional), default = False
        If True, the classifier will output an associated velocity (used for velocity/proportional based control).
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    """

    def __init__(self, 
                 offline_predictor:   EMGPredictor, 
                 window_size:         int, 
                 window_increment:    int, 
                 online_data_handler: Any, 
                 file_path:           str, 
                 file:                bool, 
                 smm:                 bool, 
                 smm_items:           List[List[Any]], 
                 features:            Optional[List[Any]],
                 std_out:             bool):

        # setting arguments as class attributes
        self.window_size = window_size
        self.window_increment = window_increment
        self.odh = online_data_handler
        self.features = features
        self.predictor = offline_predictor
        self.file = file
        self.file_path = file_path
        self.std_out = std_out
        self.scaler = None

        required_smm_items = [
            ["adapt_flag", (1,1), np.int32],
            ["active_flag", (1,1), np.int8]
        ]
        current_smm_tags = [item[0] for item in smm_items]
        for smm_item in required_smm_items:
            if smm_item[0] not in current_smm_tags:
                smm_items.append(smm_item)
        self.smm = smm
        self.smm_items = smm_items

        self.smm_manager = None
        self.model_smm_writes = 0

        self.process = Process(target=self._run_helper, daemon=True,)
    
    def start_stream(self, 
                     block: bool =True) -> None:
        """
        Start the streaming process.
        
        Parameters
        ----------
        block : bool, default=True
            Whether to run in blocking mode.
        """
        if block:
            self._run_helper()
        else:
            self.process.start()
                    
    def prepare_smm(self) -> None:
        """
        Prepare shared memory by creating required variables.
        """
        for i in self.smm_items:
            if len(i) == 3:
                i.append(Lock())
        smm = SharedMemoryManager()
        for item in self.smm_items:
            smm.create_variable(*item)
        self.smm_manager = smm
        self.model_smm_writes = 0

    def analyze_predictor(self, 
                          ip:           str="127.0.0.1", 
                          port:         int=12346,
                          analyze_time: int=10) -> None:
        """Analyzes the latency of the designed predictor. 
        
        (1) Time Between Prediction (Average): The average time between subsequent predictions.
        (2) STD Between Predictions (Standard Deviation): The standard deviation between predictions. 
        (3) Total Number of Predictions: The number of predictions that were made. Sometimes if the increment is too small, samples will get dropped and this may be less than expected.  
        
        Parameters
        ----------
        ip: str (optional), default=localhost
            The ip address to listen to for model outputs.
        port: int (optional),  default=12346
            The port to listen to for model outputs.
        analyze_time: int (optional), default=10 (seconds)
            The time in seconds that you want to analyze the model for.
        """
        print("Starting analysis of predictor " + "(" + str(analyze_time) + "s)...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.bind((ip, port))
        st = time.time()
        times = []
        while(time.time() - st < analyze_time):
            data, _ = sock.recvfrom(1024)
            if data:
                times.append(time.time())
        sock.close()
        times = np.diff(times)
        print("Time Between Predictions (Average): " + str(np.mean(times)) + 's')
        print("Time Between Predictions (STD): " + str(np.std(times)) + 's')
        print("Total Number of Predictions: " + str(len(times) + 1))
        self.stop_running()

    def _format_data_sample(self, 
                            data: Dict[str, Any]) -> np.ndarray:
        """
        Stack data from a dictionary into one array. In this case 'data' is the feature dictionary.
        """
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr

    def _get_data_helper(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Retrieve and reverse data.
        """
        data, counts = self.odh.get_data(N=self.window_size)
        # TODO: this probably adds latency and isn't really needed
        # ODH has index 0=most recent sample
        # if you trained using typical csv formats, the most recent sample is at the end of the list
        # This inversion makes them the same, but is unnecessary like 99% of the time.
        for key in data.keys():
            data[key] = data[key][::-1]
        return data, counts
    
    def get_interaction_items(self) -> List[List[Any]]:
        """
        Return the shared memory items.
        """
        return self.smm_items
    
    def load_emg_predictor(self, 
                           number: int) -> None:
        """
        Load a predictor from a file.
        
        Parameters
        ----------
        number : int
            Model number.
        """
        with open(self.file_path + 'mdl' + str(number) + '.pkl', 'rb') as handle:
            self.predictor = pickle.load(handle)
            print(f"Loaded model #{number}.")
    

    # ----- Default functions for the streaming pipeline -----
    def default_startup(self) -> None:
        """
        Default startup: prepare shared memory and reset online data handler.
        """
        if self.smm:
            self.prepare_smm()
            self.smm_manager.modify_variable("active_flag", lambda x: 1)
            self.smm_manager.modify_variable("adapt_flag", lambda x: -1)
        self.odh.prepare_smm()
        self.expected_count = {mod: self.window_size for mod in self.odh.modalities}
        self.odh.reset()

    def default_model_flag_handler(self) -> bool:
        """
        Checks and handles the shared memory flags: if the active flag is not set,
        returns False immediately. Also checks if the adapt flag is set and if so,
        loads a new predictor.

        Returns
        -------
        bool
            True if flags are acceptable to run model; False otherwise.
        """
        if self.smm:
            if not self.smm_manager.get_variable("active_flag")[0, 0]:
                return False
            if self.smm_manager.get_variable("adapt_flag")[0][0] != -1:
                self.load_emg_predictor(self.smm_manager.get_variable("adapt_flag")[0][0])
                self.smm_manager.modify_variable("adapt_flag", lambda x: -1)
        return True


    def default_window_trigger(self) -> bool:
        """
        Check whether enough data samples are collected.
        
        Returns
        -------
        bool
            True if window is ready.
        """
        val, count = self.odh.get_data(N=self.window_size)
        modality_ready = [count[mod] > self.expected_count[mod] for mod in self.odh.modalities]
        return all(modality_ready)
        
    def default_on_window(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Extract window and prepare model input. This is the same for OnlineEMGRegressors or OnlineEMGClassifiers.
        
        Returns
        -------
        Tuple[Any, dict]
            The model input (processed single window ready for model, optionally scaled) and the raw window (raw samples pre scaling).
        """
        data, count = self._get_data_helper()
        window = {mod: get_windows(data[mod], self.window_size, self.window_increment) for mod in self.odh.modalities}
        fe = FeatureExtractor()
        if self.features is not None:
            model_input = None
            for mod in self.odh.modalities:
                mod_features = fe.extract_features(self.features, window[mod], feature_dic=self.predictor.feature_params, array=True)
                if model_input is None:
                    model_input = mod_features
                else:
                    model_input = np.hstack((model_input, mod_features))
            if self.scaler is not None:
                model_input = self.scaler.transform(model_input)
        else:
            model_input = window[list(window.keys())[0]]
        # TODO: This should be adding a per modality increment since they don't typically have the same Fs
        for mod in self.odh.modalities:
            self.expected_count[mod] += self.window_increment
        return model_input, window

    def run(self, 
            block: bool=True):
        """Runs the streamer.

        Parameters
        ----------
        block: bool (optional), default = True
            If True, the run function blocks the main thread. Otherwise it runs in a 
            seperate process.
        """
        self.start_stream(block)
    
    def _run_helper(self) -> None:
        """
        Main loop for online streaming.
        """
        # Startup stage
        self.on_startup_function_handle()

        while True:
            # Check flags
            if not self.model_flag_handle():
                continue
            # Window trigger stage
            if not self.window_trigger_function_handle():
                continue

            # Window processing stage
            model_input, window = self.on_window_function_handle()
            if model_input is None:
                continue

            # Prediction/Postprocessing stage
            self.on_prediction_function_handle(model_input, window)
    
    def install_standardization(self, 
                                standardization: np.ndarray | StandardScaler) -> None:
        """Install standardization to online model. Standardizes each feature based on training data (i.e., standardizes across windows).
        Standardization is only applied when features are extracted and is applied before feature queueing (i.e., features are standardized then queued)
        if relevant. To standardize data, use the standardize Filter.

        standardization : np.ndarray or StandardScaler
            Data or pre-fit scaler for standardization.
        """        
        scaler = standardization

        if not isinstance(scaler, StandardScaler):
            # Fit scaler to provided data
            scaler = StandardScaler().fit(np.array(standardization))

        self.scaler = scaler

    def stop_running(self) -> None:
        """Kills the process streaming decisions.
        """
        self.process.terminate()

    # ----- All of these are unique to each online streamer ----------
    
    @abstractmethod
    def default_on_prediction(self, 
                              model_input: Any, 
                              window:      Dict[str, Any]) -> None:
        """
        Default prediction routine.
        """
        pass

class OnlineEMGClassifier(OnlineStreamer):
    """OnlineEMGClassifier.

    Given a EMGClassifier and additional information, this class will stream class predictions over UDP in real-time.

    Parameters
    ----------
    offline_classifier: EMGClassifier
        An EMGClassifier object. 
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    online_data_handler: OnlineDataHandler
        An online data handler object.
    features: list or None
        A list of features that will be extracted during real-time classification. 
    file_path: str, default = '.'
        Location to store model outputs. Only used if file=True.
    file: bool, default = False
        True if model outputs should be stored in a file, otherwise False.
    smm: bool, default = False
        True if shared memory items should be tracked while running, otherwise False. If True, 'model_input' and 'model_output' are expected to be passed in as smm_items.
    smm_items: list, default = None
        List of shared memory items. Each shared memory item should be a list of the format: [name: str, buffer size: tuple, dtype: dtype]. 
        When modifying this variable, items with the name 'classifier_output' and 'classifier_input' are expected to be passed in to track classifier inputs and outputs.
        The 'classifier_input' item should be of the format ['classifier_input', (100, 1 + num_features), np.double]
        The 'classifier_output' item should be of the format ['classifier_output', (100, 1 + num_dofs), np.double].
        If None, defaults to:
        [
            ["classifier_output", (100,4), np.double], #timestamp, class prediction, confidence, velocity
            ['classifier_input', (100, 1 + 32), np.double], # timestamp <- features ->
        ]
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    output_format: str (optional), default=predictions
        If predictions, it will broadcast an integer of the prediction, if probabilities it broacasts the posterior probabilities
    output_writers: OutputWriter, default = None
        A list of OutputWriters. This defines what is typically done with the output of the OnlineStreamer.
    """
    def __init__(self, 
                 offline_classifier:  EMGClassifier, 
                 window_size:         int, 
                 window_increment:    int, 
                 online_data_handler: Any, 
                 features:            Optional[List[Any]], 
                 file_path:           str = '.', 
                 file:                bool=False,
                 smm:                 bool=False, 
                 smm_items:           Optional[List[List[Any]]]= None,
                 std_out:             bool=False,
                 output_format:       str="predictions",
                 output_writers:      Optional[List[Any]]=None) -> None:
        
        # SMM logic:
        if smm_items is None:
            smm_items = [
                ["model_output", (100,4), np.double], #timestamp, class prediction, confidence, velocity
                ["model_input", (100,1+32), np.double], # timestamp, <- features ->
            ]
        assert 'model_input' in [item[0] for item in smm_items], f"'model_input' tag not found in smm_items. Got: {smm_items}."
        assert 'model_output' in [item[0] for item in smm_items], f"'model_output' tag not found in smm_items. Got: {smm_items}."
        super(OnlineEMGClassifier, self).__init__(offline_classifier, window_size, window_increment, online_data_handler,
                                                  file_path, file, smm, smm_items, features, std_out)
        self.previous_predictions = deque(maxlen=self.predictor.majority_vote)
        self.smi = smm_items

        # OutputWriter logic:
        self.output_writers = output_writers if output_writers is not None else []
        # TODO: remove output_format. it doesn't make much sense to me that we have this and output_writers 
        self.output_format = output_format

        # Set the streaming pipeline function handles in the classifier subclass.
        self.on_startup_function_handle     = self.default_startup
        self.window_trigger_function_handle = self.default_window_trigger
        self.model_flag_handle              = self.default_model_flag_handler
        self.on_window_function_handle      = self.default_on_window
        self.prediction_function_handle     = self.default_prediction_function
        self.postprocessing_function_handle = self.default_postprocessing_function
        self.on_prediction_function_handle  = self.default_on_prediction
        
    def default_prediction_function(self, 
                                    model_input: Any, 
                                    window: Dict[str, Any]) -> Tuple[Any, Any]:
        probabilities = self.predictor.model.predict_proba(model_input)
        prediction, probability = self.predictor._prediction_helper(probabilities)
        return (prediction[0], probability[0])

    def default_postprocessing_function(self, 
                                        pred_tuple:  Tuple[Any, Any],
                                        model_input: Any, 
                                        window:      Dict[str, Any]):
        prediction, probability = pred_tuple
        if self.predictor.rejection:
            prediction = self.predictor._rejection_helper(prediction, probability)
        self.previous_predictions.append(prediction)
        if self.predictor.majority_vote:
            values, counts = np.unique(list(self.previous_predictions), return_counts=True)
            prediction = values[np.argmax(counts)]
        calculated_velocity = ""
        if self.predictor.velocity:
            calculated_velocity = " 0"
            if prediction >= 0:
                calculated_velocity = " " + str(self.predictor._get_velocity(window, prediction))
        return (prediction, probability, calculated_velocity)

    def format_output_info(self, 
                           processed:   Tuple[Any, Any, Any],
                           model_input: Any, 
                           window:      Dict[str, Any]) -> Dict[str, Any]:
        # Compose a dictionary with all information you wish to send.
        prediction, probability, calculated_velocity = processed
        info = {
            "timestamp": time.time(),
            "model_output": prediction,
            "probability": probability,
            "velocity": calculated_velocity,
            "model_input": model_input,
            "window":window
        }
        return info


    def default_on_prediction(self, 
                              model_input: Any, 
                              window:      Dict[str, Any]) -> None:
        raw = self.prediction_function_handle(model_input, window)
        processed = self.postprocessing_function_handle(raw, model_input, window)
        info = self.format_output_info(processed, model_input, window)
        for writer in self.output_writers:
            writer.write(info)

    def visualize(self, 
                  ip: str="127.0.0.1", 
                  port: int=12346, 
                  max_len: int=50, 
                  legend: Optional[List[str]]=None):
        """Produces a live plot of classifier decisions -- Note this consumes the decisions.
        Do not use this alongside the actual control operation of libemg. Online classifier has to
        be running in "probabilties" output mode for this plot.

        Parameters
        ----------
        ip: (str) (optional), default=localhost
            The ip address the classifier outputs decisions to.
        port: (int) (optional), default=12346
            The port the classifier outputs decisions to.
        max_len: (int) (optional) 
            number of decisions to visualize
        legend: (list) (optional)
            Labels used to populate legend. Must be passed in order of output classes.
        """
        if self.output_format != 'probabilities':
            raise ValueError(f"OnlineEMGClassifier output_format must be 'probabailities' for visualize() method to work, but current value is {self.output_format}.")
        plt.style.use("ggplot")
        figure, ax = plt.subplots()
        figure.suptitle("Live Classifier Output", fontsize=16)
        plot_handle = ax.scatter([],[],c=[])
        num_classes = len(self.predictor.model.classes_)    # assumes that user is using an sklearn model
        cmap = cm.get_cmap('turbo', num_classes)

        controller = ClassifierController(output_format=self.output_format, num_classes=num_classes, ip=ip, port=port)
        controller.start()

        if legend is not None:
            for i in range(num_classes):
                plt.plot(i, label=legend[i], color=cmap.colors[i])
            ax.legend()
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        decision_horizon_classes = []
        decision_horizon_probabilities = []
        timestamps = []
        start_time = time.time()
        while True:
            data = controller.get_data(['probabilities', 'timestamp'])
            if data is None:
                continue
            probabilities, timestamp = data
            max_prob = np.max(probabilities)
            prediction = np.argmax(probabilities)
            decision_horizon_classes.append(prediction)
            decision_horizon_probabilities.append(max_prob)
            timestamps.append(timestamp - start_time)

            decision_horizon_classes = decision_horizon_classes[-max_len:]
            decision_horizon_probabilities = decision_horizon_probabilities[-max_len:]
            timestamps = timestamps[-max_len:]

            if plt.fignum_exists(figure.number):
                plt.cla()
                ax.scatter(timestamps, decision_horizon_probabilities,c=cmap.colors[decision_horizon_classes])
                plt.ylim([0,1.5])
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Probability")
                ax.set_ylim([0, 1.5])
                if legend is not None:
                    ax.legend(handles=legend_handles, labels=legend_labels)
                plt.draw()
                plt.pause(0.01)
            else:
                return
            
    def _get_data_helper(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data, counts = self.odh.get_data(N=self.window_size)
        for key in data.keys():
            data[key] = data[key][::-1]
        return data, counts
    
class OnlineEMGRegressor(OnlineStreamer):
    """OnlineEMGRegressor.

    Given a EMGRegressor and additional information, this class will stream regression predictions over UDP or TCP in real-time.

    Parameters
    ----------
    offline_regressor: EMGRegressor
        An EMGRegressor object. 
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    online_data_handler: OnlineDataHandler
        An online data handler object.
    features: list
        A list of features that will be extracted during real-time regression. 
    file_path: str, default = '.'
        Location to store model outputs. Only used if file=True.
    file: bool, default = False
        True if model outputs should be stored in a file, otherwise False.
    smm: bool, default = False
        True if shared memory items should be tracked while running, otherwise False. If True, 'model_input' and 'model_output' are expected to be passed in as smm_items.
    smm_items: list, default = None
        List of shared memory items. Each shared memory item should be a list of the format: [name: str, buffer size: tuple, dtype: dtype]. 
        When modifying this variable, items with the name 'model_output' and 'model_input' are expected to be passed in to track model inputs and outputs.
        The 'model_input' item should be of the format ['model_input', (100, 1 + num_features), np.double]
        The 'model_output' item should be of the format ['model_output', (100, 1 + num_dofs), np.double].
        If None, defaults to:
        [
            ['model_output', (100, 3), np.double],  # timestamp, prediction 1, prediction 2... (assumes 2 DOFs)
            ['model_input', (100, 1 + 32), np.double], # timestamp <- features ->
        ]
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    output_writers: OutputWriter, default = None
        A list of OutputWriters. This defines what is typically done with the output of the OnlineStreamer.
    """
    def __init__(self, 
                 offline_regressor:   EMGRegressor, 
                 window_size:         int, 
                 window_increment:    int,
                 online_data_handler: Any,
                 features:            Optional[List[Any]], 
                 file_path:           str = '.', 
                 file:                bool = False, 
                 smm:                 bool = False, 
                 smm_items:           Optional[List[Any]] = None,
                 std_out:             bool = False,
                 output_writers:      Optional[List[Any]]=None) -> None:
        if smm_items is None:
            # I think probably just have smm_items default to None and remove the smm flag. Then if the user wants to track stuff, they can pass in smm_items and a function to handle them?
            smm_items = [
                ['model_input', (100, 1 + 32), np.double], # timestamp <- features ->
                ['model_output', (100, 3), np.double]  # timestamp, prediction 1, prediction 2... (assumes 2 DOFs)
            ]
        assert 'model_input' in [item[0] for item in smm_items], f"'model_input' tag not found in smm_items. Got: {smm_items}."
        assert 'model_output' in [item[0] for item in smm_items], f"'model_output' tag not found in smm_items. Got: {smm_items}."
        super(OnlineEMGRegressor, self).__init__(offline_regressor, window_size, window_increment, online_data_handler, file_path,
                                                 file, smm, smm_items, features, std_out)
        self.smi = smm_items

        # OutputWriter logic:
        self.output_writers = output_writers if output_writers is not None else []

        # Set the common function handles using the parent's defaults.
        self.on_startup_function_handle     = self.default_startup
        self.window_trigger_function_handle = self.default_window_trigger
        self.model_flag_handle              = self.default_model_flag_handler
        self.on_window_function_handle      = self.default_on_window
        
        # Now set the regressor-specific prediction pipeline function handles.
        self.on_prediction_function_handle  = self.regressor_on_prediction

        # These get called by on_prediction_function_handle
        self.prediction_function_handle     = self.regressor_prediction_function
        self.postprocessing_function_handle = self.regressor_postprocessing_function
        
    
    def regressor_prediction_function(self, 
                                      model_input: Any, 
                                      window:      Dict[str, Any]) -> Any:
        """
        Raw prediction: use the predictor's run() method and squeeze the output.
        """
        predictions = self.predictor.run(model_input).squeeze()
        return predictions

    def regressor_postprocessing_function(self, 
                                          predictions: Any,
                                          model_input: Any,
                                          window:      Dict[str, Any]) -> Any:
        """
        Postprocessing: apply additional processing if needed (e.g., deadband).
        In this simple example, we return the predictions unmodified (currently a pass-through).
        """
        return predictions

    def regressor_on_prediction(self, 
                                model_input: Any, 
                                window:      Dict[str, Any]) -> None:
        raw = self.prediction_function_handle(model_input, window)
        processed = self.postprocessing_function_handle(raw, model_input, window)
        info = self.format_output_info(processed, model_input, window)
        for writer in self.output_writers:
            writer.write(info)

    def format_output_info(self, 
                           processed:   Any, 
                           model_input: Any, 
                           window:      Dict[str, Any]) -> Dict[str, Any]:
        predictions = processed
        info = {
            "timestamp": time.time(),
            "model_output": predictions,
            "model_input": model_input,
            "window": window
        }
        return info

    def visualize(self, 
                  ip: str="127.0.0.1", 
                  port: int=12346, 
                  max_len: int = 50, 
                  legend: bool = False):
        """Plot a live visualization of the online regressor's predictions. Please note that the animation updates every 5 milliseconds,
        so keep this in mind when choosing window size and increment. For example, a window increment that's too small may cause delay in the plotting
        if the regressor is making predictions faster than the plot can be updated.

        Parameters
        ----------
        ip: str (optional), default="localhost"
            The ip to monitor for regressor outputs.
        port: int (optional), default=12346
            The port to monitor for regressor outputs.
        max_len: int (optional), default = 50
            Maximum number of predictions to plot at a time. Defaults to 50.
        legend: bool (optional), default = False
            True if a legend should be shown, otherwise False. Defaults to False.
        """

        plt.style.use('ggplot')
        fig, ax = plt.subplots(layout='constrained')
        fig.suptitle('Live Regressor Output', fontsize=16)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Prediction')

        controller = RegressorController(ip=ip, port=port)
        controller.start()

        # Wait for controller to start receiving data
        predictions = None
        while predictions is None:
            predictions = controller.get_data('predictions')
        cmap = cm.get_cmap('turbo', len(predictions))

        plots = [ax.plot([], [], '.', color=cmap.colors[dof_idx], alpha=0.8)[0] for dof_idx in range(len(predictions))]

        if legend:
            handles = [mpatches.Patch(color=cmap.colors[dof_idx], label=f"DOF {dof_idx}") for dof_idx in range(len(predictions))]

        start_time = time.time()

        def update(frame, decision_horizon_predictions, timestamps):
            data = controller.get_data(['predictions', 'timestamp'])
            if data is None:
                return
            predictions, timestamp = data

            timestamps.append(timestamp - start_time)
            decision_horizon_predictions.append(predictions)

            timestamps = timestamps[-max_len:]
            decision_horizon_predictions = decision_horizon_predictions[-max_len:]

            for dof_idx in range(len(predictions)):
                plots[dof_idx].set_xdata(timestamps)
                plots[dof_idx].set_ydata(np.array(decision_horizon_predictions)[:, dof_idx])

            if legend:
                ax.legend(handles=handles, loc='upper right')

            ax.relim()
            ax.autoscale_view()
            return plots
        
        _ = FuncAnimation(fig, partial(update, decision_horizon_predictions=[], timestamps=[]), interval=5, blit=False)  # must return value or animation won't work
        plt.show()

