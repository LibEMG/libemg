from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
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

from libemg.utils import get_windows

class EMGPredictor:
    def __init__(self, model, model_parameters = None, random_seed = 0, fix_feature_errors = False, silent = False) -> None:
        """Base class for EMG prediction.

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
        self.model = model
        self.model_parameters = model_parameters
        # default for feature parameters
        self.feature_params = {}
        self.fix_feature_errors = fix_feature_errors
        self.silent = silent
        random.seed(random_seed)

    def fit(self, feature_dictionary = None, dataloader_dictionary = None, training_parameters = None):
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
    def from_file(self, filename):
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

    def _predict(self, data):
        try:
            return self.model.predict(data)
        except AttributeError as e:
            raise AttributeError("Attempted to perform prediction when model doesn't have a predict() method. Please ensure model has a valid predict() method.") from e

    def _predict_proba(self, data):
        try:
            return self.model.predict_proba(data)
        except AttributeError as e:
            raise AttributeError("Attempted to perform prediction when model doesn't have a predict_proba() method. Please ensure model has a valid predict_proba() method.") from e

    def save(self, filename):
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

    def install_feature_parameters(self, feature_params):
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
    def _validate_model_parameters(model, model_parameters, model_config):
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

    def _format_data(self, feature_dictionary):
        if not isinstance(feature_dictionary, np.ndarray):
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

    def _fit_statistical_model(self, feature_dictionary):
        assert 'training_features' in feature_dictionary.keys()
        assert 'training_labels'   in feature_dictionary.keys()
        # convert dictionary of features format to np.ndarray for test/train set (NwindowxNfeature)
        feature_dictionary["training_features"] = self._format_data(feature_dictionary['training_features'])
        # self._set_up_classifier(model, feature_dictionary, parameters)
        self.model.fit(feature_dictionary['training_features'], feature_dictionary['training_labels'])
        
    def _fit_deeplearning_model(self, dataloader_dictionary, parameters):
        assert 'training_dataloader' in dataloader_dictionary.keys()
        assert 'validation_dataloader'  in dataloader_dictionary.keys()
        self.model.fit(dataloader_dictionary, **parameters)


class EMGClassifier(EMGPredictor):
    def __init__(self, model, model_parameters = None, random_seed = 0, fix_feature_errors = False, silent = False):
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


        
    def run(self, test_data):
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

    def add_rejection(self, threshold=0.9):
        """Adds the rejection post-processing block onto a classifier.

        Parameters
        ----------
        threshold: float (optional), default=0.9
            The confidence threshold (0-1). All predictions with confidence under the threshold will be rejected.
        """
        self.rejection = True
        self.rejection_threshold = threshold

    def add_majority_vote(self, num_samples=5):
        """Adds the majority voting post-processing block onto a classifier.

        Parameters
        ----------
        threshold: int (optional), default=5
            The number of samples that will be included in the majority vote.
        """
        self.majority_vote = num_samples

    def add_velocity(self, train_windows, train_labels):
        """Adds velocity (i.e., proportional) control where a multiplier is generated for the level of contraction intensity.

        Note, that when using this optional, ramp contractions should be captured for training. 

        Parameters:
        -----------
        """
        self.velocity = True
        self.th_min_dic, self.th_max_dic = self._set_up_velocity_control(train_windows, train_labels)


    
    '''
    ---------------------- Private Helper Functions ----------------------
    '''
    def _prediction_helper(self, predictions):
        probabilities = [] 
        prediction_vals = []
        for i in range(0, len(predictions)):
            pred_list = list(predictions[i])
            prediction_vals.append(pred_list.index(max(pred_list)))
            probabilities.append(pred_list[pred_list.index(max(pred_list))])
        return np.array(prediction_vals), np.array(probabilities)
        
    def _rejection_helper(self, prediction, prob):
        if self.rejection:
            if prob > self.rejection_threshold:
                return prediction
            else:
                return -1
        return prediction
    
    def _majority_vote_helper(self, predictions):
        updated_predictions = []
        for i in range(0, len(predictions)):
            idxs = np.array(range(i-self.majority_vote+1, i+1))
            idxs = idxs[idxs >= 0]
            group = predictions[idxs]
            updated_predictions.append(stats.mode(group, keepdims=False)[0])
        return np.array(updated_predictions)
    
    def _get_velocity(self, window, c):
        if self.th_max_dic and self.th_min_dic:
            velocity_output = (np.sum(np.mean(np.abs(window),2)[0], axis=0) - self.th_min_dic[c])/(self.th_max_dic[c] - self.th_min_dic[c])
            return '{0:.2f}'.format(min([1, max([velocity_output, 0])]))

    def _set_up_velocity_control(self, train_windows, train_labels):
        # Extract classes 
        th_min_dic = {}
        th_max_dic = {}
        classes = np.unique(train_labels)
        for c in classes:
            indices = np.where(train_labels == c)[0]
            c_windows = train_windows[indices]
            mav_tr = np.sum(np.mean(np.abs(c_windows),2), axis=1)
            mav_tr_max = np.max(mav_tr)
            mav_tr_min = np.min(mav_tr)
            # Calculate THmin 
            th_min = ((1-(10/100)) * mav_tr_min) + (0.1 * mav_tr_max)
            th_min_dic[c] = th_min 
            # Calculate THmax
            th_max = ((1-(70/100)) * mav_tr_min) + (0.7 * mav_tr_max)
            th_max_dic[c] = th_max
        return th_min_dic, th_max_dic

    def visualize(self, test_labels, predictions, probabilities):
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

    """
    def __init__(self, model, model_parameters = None, random_seed = 0, fix_feature_errors = False, silent = False, deadband_threshold = 0.):
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
        model_config = {
            'LR': (LinearRegression, {}),
            'SVM': (SVR, {"kernel": "linear"}),
            'RF': (RandomForestRegressor, {"random_state": 0}),
            'GB': (GradientBoostingRegressor, {"random_state": 0}),
            'MLP': (MLPRegressor, {"random_state": 0, "hidden_layer_sizes": 126})
        }
        model = self._validate_model_parameters(model, model_parameters, model_config)
        self.deadband_threshold = deadband_threshold
        super().__init__(model, model_parameters, random_seed=random_seed, fix_feature_errors=fix_feature_errors, silent=silent)

    
    def run(self, test_data):
        """Runs the regressor on a pre-defined set of training data.

        Parameters
        ----------
        test_data: list
            A dictionary, np.ndarray of inputs appropriate for the model of the EMGRegressor.
        Returns
        ----------
        list
            A list of predictions, based on the passed in testing features.
        """
        test_data = self._format_data(test_data)
        predictions = self._predict(test_data)

        # Set values within deadband to 0
        deadband_mask = np.abs(predictions) < self.deadband_threshold
        predictions[deadband_mask] = 0.

        return predictions

    def visualize(self, test_labels, predictions):
        """Visualize the decision stream of the regressor on test data.

        You can call this visualize function to get a visual output of what the decision stream looks like.

        :param test_labels: np.ndarray
        :type test_labels: N x M array, where N = # samples and M = # DOFs, containing the labels for the test data.
        :param predictions: np.ndarray
        :type predictions: N x M array, where N = # samples and M = # DOFs, containing the predictions for the test data.
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
        

    def add_deadband(self, threshold):
        """Add a deadband around regressor predictions that will instead be output as 0.

        Parameters
        ----------
        threshold: float
            Deadband threshold. All output predictions from -threshold to +threshold will instead output 0.
        """
        self.deadband_threshold = threshold



class OnlineStreamer:
    def __init__(self, offline_classifier, window_size, window_increment, online_data_handler, features, port=12346, ip='127.0.0.1', std_out=False, tcp=False):
        self.window_size = window_size
        self.window_increment = window_increment
        self.raw_data = online_data_handler.raw_data
        self.filters = online_data_handler.fi
        self.features = features
        self.port = port
        self.ip = ip
        self.classifier = offline_classifier
        self.fe = FeatureExtractor()

        self.tcp = tcp
        if not tcp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            print("Waiting for TCP connection...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.bind((ip, port))
            self.sock.listen()
            self.conn, addr = self.sock.accept()
            print(f"Connected by {addr}")

        self.process = Process(target=self._run_helper, daemon=True,)
        self.std_out = std_out
    
    def start_stream(self, block=True):
        if block:
            self._run_helper()
        else:
            self.process.start()

    def get_data(self, data):
        # Extract window and predict sample
        window = get_windows(data, self.window_size, self.window_size)
        # Dealing with the case for CNNs when no features are used
        if self.features:
            features = self.fe.extract_features(self.features, window, self.classifier.feature_params)
            classifier_input = self._format_data_sample(features)
        else:
            classifier_input = window
        self.raw_data.adjust_increment(self.window_size, self.window_increment)
        return window, classifier_input
    
    def write_output(self, prediction, calculated_velocity):
        # Write classifier output:
        if not self.tcp:
            self.sock.sendto(bytes(str(str(prediction) + calculated_velocity), "utf-8"), (self.ip, self.port))
        else:
            self.conn.sendall(str.encode(str(prediction) + calculated_velocity + '\n'))
        if self.std_out:
            print(f"{str(prediction)} {calculated_velocity} {time.time()}")

    def _format_data_sample(self, data):
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr

    def _get_data_helper(self):
        data = np.array(self.raw_data.get_emg())
        if self.filters is not None:
            try:
                data = self.filters.filter(data)
            except:
                pass
        return data
    
    # ----- All of these are unique to each online streamer ----------
    def run(self):
        pass 

    def stop_running(self):
        pass

    def _run_helper(self):
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
        A list of features that will be extracted during real-time classification. These should be the 
        same list used to train the model. Pass in None if using the raw data (this is primarily for CNNs).
    parameters: dict (optional)
        A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
        in the sklearn docs for the given model.
    port: int (optional), default = 12346
        The port used for streaming predictions over UDP.
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    velocity: bool (optional), default = False
        If True, the classifier will output an associated velocity (used for velocity/proportional based control).
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    tcp: bool (optional), default = False
        If True, will stream predictions over TCP instead of UDP.
    output_format: str (optional), default=predictions
        If predictions, it will broadcast an integer of the prediction, if probabilities it broacasts the posterior probabilities
    """
    def __init__(self, offline_classifier, window_size, window_increment, online_data_handler, features, port=12346, ip='127.0.0.1', std_out=False, tcp=False, output_format="predictions"):
        super(OnlineEMGClassifier, self).__init__(offline_classifier, window_size, window_increment, online_data_handler, features, port, ip, std_out, tcp)
        self.previous_predictions = deque(maxlen=self.classifier.majority_vote)
        self.output_format = output_format

    def run(self, block=True):
        """Runs the classifier - continuously streams predictions over UDP.

        Parameters
        ----------
        block: bool (optional), default = True
            If True, the run function blocks the main thread. Otherwise it runs in a 
            seperate process.
        """
        self.start_stream(block)

    def stop_running(self):
        """Kills the process streaming classification decisions.
        """
        self.process.terminate()

    def analyze_classifier(self, analyze_time=10, port=12346, ip='127.0.0.1'):
        """Analyzes the latency of the designed classifier. 

        Parameters
        ----------
        analyze_time: int (optional), default=10 (seconds)
            The time in seconds that you want to analyze the device for. 
        port: int (optional), default = 12346
            The port used for streaming predictions over UDP.
        ip: string (optional), default = '127.0.0.1'
            The ip used for streaming predictions over UDP.
        
        (1) Time Between Prediction (Average): The average time between subsequent predictions.
        (2) STD Between Predictions (Standard Deviation): The standard deviation between predictions. 
        (3) Total Number of Predictions: The number of predictions that were made. Sometimes if the increment is too small, samples will get dropped and this may be less than expected.  
        """
        print("Starting analysis of classifier " + "(" + str(analyze_time) + "s)...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.bind((ip, port))
        st = time.time()
        times = []
        while(time.time() - st < analyze_time):
            data, _ = sock.recvfrom(1024)
            if data:
                times.append(time.time())
        times = np.diff(times)
        print("Time Between Predictions (Average): " + str(np.mean(times)) + 's')
        print("Time Between Predictions (STD): " + str(np.std(times)) + 's')
        print("Total Number of Predictions: " + str(len(times) + 1))
        self.stop_running()
    
    def visualize(self, max_len=50, legend=None):
        """Produces a live plot of classifier decisions -- Note this consumes the decisions.
        Do not use this alongside the actual control operation of libemg. Online classifier has to
        be running in "probabilties" output mode for this plot.

        Parameters
        ----------
        max_len: (int) (optional) 
            number of decisions to visualize
        legend: (list) (optional)
            The legend to display on the plot
        """
        plt.style.use("ggplot")
        figure, ax = plt.subplots()
        figure.suptitle("Live Classifier Output", fontsize=16)
        plot_handle = ax.scatter([],[],c=[])
        

        # make a new socket that subscribes to the libemg events
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.bind(('127.0.0.1', 12346))
        num_classes = len(self.classifier.classifier.classes_)
        cmap = cm.get_cmap('turbo', num_classes)

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
            data, _ = sock.recvfrom(1024)
            data = str(data.decode("utf-8"))
            probabilities = np.array([float(i) for i in data.split(" ")[:num_classes]])
            max_prob = np.max(probabilities)
            prediction = np.argmax(probabilities)
            decision_horizon_classes.append(prediction)
            decision_horizon_probabilities.append(max_prob)
            timestamps.append(float(data.split(" ")[-1]) - start_time)

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
                plt.pause(0.0001)
            else:
                return


    def _run_helper(self):
        self.raw_data.reset_emg()
        while True:
            data = self._get_data_helper()
            if len(data) >= self.window_size:
                # Extract window and predict sample
                window, classifier_input = self.get_data(data)
                self.raw_data.adjust_increment(self.window_size, self.window_increment)

                # Make prediction
                probabilities = self.classifier.classifier.predict_proba(classifier_input)
                prediction, probability = self.classifier._prediction_helper(probabilities)
                
                prediction = prediction[0]
                probability = probability[0]

                # Check for rejection
                if self.classifier.rejection:
                    #TODO: Right now this will default to -1
                    prediction = self.classifier._rejection_helper(prediction, probability)
                self.previous_predictions.append(prediction)
                
                # Check for majority vote
                if self.classifier.majority_vote:
                    values, counts = np.unique(list(self.previous_predictions), return_counts=True)
                    prediction = values[np.argmax(counts)]
                
                # Check for velocity based control
                calculated_velocity = ""
                if self.classifier.velocity:
                    calculated_velocity = " 0"
                    # Dont check if rejected 
                    if prediction >= 0:
                        calculated_velocity = " " + str(self.classifier._get_velocity(window, prediction))
                
                #self.write_output(prediction, calculated_velocity, self.output_format)
                time_stamp = time.time()
                if not self.tcp:
                    if self.output_format == "predictions":
                        message = str(str(prediction) + calculated_velocity + " " + str(time_stamp))
                    elif self.output_format == "probabilities":
                        message = ' '.join([f'{i:.2f}' for i in probabilities[0]]) + calculated_velocity + " " + str(time_stamp)
                    self.sock.sendto(bytes(message, "utf-8"), (self.ip, self.port))
                else:
                    if self.output_format == "predictions":
                        message = str(prediction) + calculated_velocity + '\n'
                    elif self.output_format == "probabilities":
                        message = ' '.join([f'{i:.2f}' for i in probabilities[0]]) + calculated_velocity + " " + str(time_stamp)
                    self.conn.sendall(str.encode(message))
                
                if self.std_out:
                    print(message)
    


    def _format_data_sample(self, data):
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr

    def _get_data_helper(self):
        data = np.array(self.raw_data.get_emg())
        if self.filters is not None:
            try:
                data = self.filters.filter(data)
            except:
                pass
        return data
    
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
    parameters: dict (optional)
        A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
        in the sklearn docs for the given model.
    port: int (optional), default = 12346
        The port used for streaming predictions over UDP.
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    tcp: bool (optional), default = False
        If True, will stream predictions over TCP instead of UDP.
    """
    def __init__(self, offline_regressor, window_size, window_increment, online_data_handler, features, port=12346, ip='127.0.0.1', std_out=False, tcp=False):
        super(OnlineEMGRegressor, self).__init__(offline_regressor, window_size, window_increment, online_data_handler, features, port, ip, std_out, tcp)
        
    def run(self, block=True):
        """Runs the regressor - continuously streams predictions over UDP or TCP.

        Parameters
        ----------
        block: bool (optional), default = True
            If True, the run function blocks the main thread. Otherwise it runs in a 
            seperate process.
        """
        self.start_stream(block)

    def stop_running(self):
        """Kills the process streaming classification decisions.
        """
        self.process.terminate()

    def _run_helper(self):
        self.raw_data.reset_emg()
        while True:
            data = self._get_data_helper()
            if len(data) >= self.window_size:
                window, classifier_input = self.get_data(data)
                prediction = np.array(self.classifier.regressor.predict(classifier_input)).squeeze()
                self.write_output(prediction, "")

    def analyze_regressor(self, analyze_time):
        # Analyze latency of regressor
        raise NotImplementedError('The OnlineEMGRegressor.analyze_regressor() method has not been implemented yet.')

    def visualize(self, max_len = 50):
        # Make a line plot showing the current point on the DOF
        # Waiting until shared memory changes are implemented before implementing this
        raise NotImplementedError('The OnlineEMGRegressor.visualize() method has not been implemented yet.')

