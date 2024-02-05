from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
import numpy as np
import pickle
import socket
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import inspect
from scipy import stats

from libemg.utils import get_windows

class EMGClassifier:
    """The Offline EMG Classifier. 

    This is the base class for any offline EMG classification. 

    Parameters
    ----------
    velocity: bool (optional), default=False
        If True, the classifier will output an associated velocity (used for velocity/proportional based control).
    """
    def __init__(self, random_seed=0):
        random.seed(random_seed)
        
        self.classifier = None
        self.velocity = False
        self.majority_vote = None
        self.rejection = False

        # For velocity control
        self.th_min_dic = None 
        self.th_max_dic = None 

        # default for feature parameters
        self.feature_params = {}

        

    def fit(self, model, feature_dictionary=None, dataloader_dictionary=None, parameters=None):
        """The fit function for the EMG Classification class. 

        This is the method called that actually optimizes model weights for the dataset. This method presents a fork for two 
        different kind of models being trained. The first we call "statistical" models (i.e., LDA, QDA, SVM, etc.)
        and these are interfaced with sklearn. The second we call "deep learning" models and these are designed to fit around
        the conventional programming style of pytorch. We distinguish which of these models are being trained by passing in a
        feature_dictionary for "statistical" models and a "dataloader_dictionary" for deep learning models.

        Parameters
        ----------
    
        model: string or custom classifier (must have fit, predict and predic_proba functions)
            The type of machine learning model. Valid options include: 'LDA', 'QDA', 'SVM', 'KNN', 'RF' (Random Forest),  
            'NB' (Naive Bayes), 'GB' (Gradient Boost), 'MLP' (Multilayer Perceptron). Note, these models are all default sklearn 
            models with no hyperparameter tuning and may not be optimal. Pass in custom classifiers or parameters for more control.
        feature_dictionary: dict
            A dictionary including the associated features and labels associated with a set of data. 
            Dictionary keys should include 'training_labels' and 'training_features'.
        dataloader_dictionary: dict
            A dictionary including the associated dataloader objects for the dataset you'd like to train with. 
            Dictionary keys should include 'training_dataloader', and 'validation_dataloader'.
        parameters: dict (optional)
            A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
            in the sklearn docs for the given model. Alternatively, these can be custom parameters in the case of custom 
            statistical models or deep learning models.
        """
        # determine what sort of model we are fitting:
        if feature_dictionary is not None:
            if "training_features" in feature_dictionary.keys():
                self._fit_statistical_model(model, feature_dictionary, parameters)
        if dataloader_dictionary is not None:
            self._fit_deeplearning_model(model, dataloader_dictionary, parameters)

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
            classifier = pickle.load(f)
        return classifier

 
    def run(self, test_data, fix_feature_errors=False, silent=False):
        """Runs the classifier on a pre-defined set of training data.

        Parameters
        ----------
        test_data: list
            A dictionary, np.ndarray of inputs appropriate for the model of the EMGClassifier.
        fix_feature_errors: bool (default=False)
            If True, the classifier will update any feature erros (INF, -INF, NAN) using the np.nan_to_num function.
        silent: bool (default=False)
            If True, the outputs from the fix_feature_errors parameter will be silenced. 

        Returns
        ----------
        list
            A list of predictions, based on the passed in testing features.
        list
            A list of the probabilities (for each prediction), based on the passed in testing features.
        """
        if type(test_data) == dict:
            test_data = self._format_data(test_data)
        
        # Remove any faulty values from test_data (these may have occured from feature extraction e.g., NANs)
        if fix_feature_errors:
            if FeatureExtractor().check_features(test_data, silent):
                test_data = np.nan_to_num(test_data, neginf=0, nan=0, posinf=0) 
        
        prob_predictions = self.classifier.predict_proba(test_data)
            
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

        Parameters
        -----------
        """
        self.velocity = True
        self.th_min_dic, self.th_max_dic = self._set_up_velocity_control(train_windows, train_labels)


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
    
    '''
    ---------------------- Private Helper Functions ----------------------
    '''

    def _fit_statistical_model(self, model, feature_dictionary, parameters):
        assert 'training_features' in feature_dictionary.keys()
        assert 'training_labels'   in feature_dictionary.keys()
        # convert dictionary of features format to np.ndarray for test/train set (NwindowxNfeature)
        feature_dictionary["training_features"] = self._format_data(feature_dictionary['training_features'])
        self._set_up_classifier(model, feature_dictionary, parameters)
        
    def _fit_deeplearning_model(self, model, dataloader_dictionary, parameters):
        assert 'training_dataloader' in dataloader_dictionary.keys()
        assert 'validation_dataloader'  in dataloader_dictionary.keys()
        self.classifier = model
        self.classifier.fit(dataloader_dictionary, **parameters)
        pass

    def _format_data(self, feature_dictionary):
        arr = None
        for feat in feature_dictionary:
            if arr is None:
                arr = feature_dictionary[feat]
            else:
                arr = np.hstack((arr, feature_dictionary[feat]))
        return arr

    def _set_up_classifier(self, model, dataset_dictionary, parameters):
        valid_models = ["LDA", "KNN", "SVM", "QDA", "RF", "NB", "GB", "MLP"]
        if isinstance(model, str):
            assert model in valid_models
            valid_parameters = self._validate_parameters(model, parameters)

        # Set up classifier based on input
        if model == "LDA":
            self.classifier = LinearDiscriminantAnalysis(**valid_parameters)
        elif model == "KNN":
            self.classifier = KNeighborsClassifier(**valid_parameters)
        elif model == "SVM":
            self.classifier = SVC(**valid_parameters)
        elif model == "QDA":
            self.classifier = QuadraticDiscriminantAnalysis(**valid_parameters)
        elif model == "RF":
            self.classifier = RandomForestClassifier(**valid_parameters)
        elif model == "NB":
            self.classifier = GaussianNB(**valid_parameters)
        elif model == "GB":
            self.classifier = GradientBoostingClassifier(**valid_parameters)
        elif model == "MLP":
            self.classifier = MLPClassifier(**valid_parameters)
        elif not model is None:
            # Assume a custom classifier has been passed in
            self.classifier = model
        # Fit the model to the data set
        self.classifier.fit(dataset_dictionary['training_features'],dataset_dictionary['training_labels'])

    def _validate_parameters(self, model, parameters):
        default_parameters = {
            'LDA': {}, 
            'KNN': {"n_neighbors": 5}, 
            'SVM': {"kernel": "linear", "probability": True, "random_state": 0},
            'QDA': {},
            'RF': {"random_state": 0},
            'NB': {},
            'GB': {"random_state": 0},
            'MLP': {"random_state": 0, "hidden_layer_sizes": 126}
        }

        valid_parameters = default_parameters[model]

        if parameters is None:
            return valid_parameters

        dic = {'LDA': LinearDiscriminantAnalysis, 
               'KNN': KNeighborsClassifier, 
               'SVM': SVC, 
               'QDA': QuadraticDiscriminantAnalysis,
               'RF': RandomForestClassifier,
               'NB': GaussianNB,
               'GB': GradientBoostingClassifier,
               'MLP': MLPClassifier
        }

        signature = list(inspect.signature(dic[model]).parameters.keys())
        for p in parameters:
            if p in signature:
                valid_parameters[p] = parameters[p]
            else:
                print(str(p) + "is an invalid parameter.")
        return valid_parameters
            
    
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
    
class OnlineEMGClassifier:
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
    channels: list (optional), default=None 
        If not none, the list of channels that will be extracted. Used if you only want to use a subset of channels during classification. 
    """
    def __init__(self, offline_classifier, window_size, window_increment, online_data_handler, features, port=12346, ip='127.0.0.1', std_out=False, tcp=False, output_format="predictions", channels=None):
        self.window_size = window_size
        self.window_increment = window_increment
        self.raw_data = online_data_handler.raw_data
        self.filters = online_data_handler.fi
        self.features = features
        self.port = port
        self.ip = ip
        self.classifier = offline_classifier
        self.output_format = output_format
        self.channels = channels

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
        self.previous_predictions = deque(maxlen=self.classifier.majority_vote)
        
    def run(self, block=True):
        """Runs the classifier - continuously streams predictions over UDP.

        Parameters
        ----------
        block: bool (optional), default = True
            If True, the run function blocks the main thread. Otherwise it runs in a 
            seperate process.
        """
        if block:
            self._run_helper()
        else:
            self.process.start()

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
        fe = FeatureExtractor()
        self.raw_data.reset_emg()
        while True:
            if len(self.raw_data.get_emg()) >= self.window_size:
                data = self._get_data_helper()
                if self.channels is not None:
                    data = data[:,self.channels]

                # Extract window and predict sample
                window = get_windows(data[-self.window_size:][:], self.window_size, self.window_size)

                # Dealing with the case for CNNs when no features are used
                if self.features:
                    features = fe.extract_features(self.features, window, self.classifier.feature_params)
                    # If extracted features has an error - give error message
                    if (fe.check_features(features) != 0):
                        self.raw_data.adjust_increment(self.window_size, self.window_increment)
                        continue
                    classifier_input = self._format_data_sample(features)
                else:
                    classifier_input = window
                self.raw_data.adjust_increment(self.window_size, self.window_increment)
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
                
                time_stamp = time.time()
                # Write classifier output:
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
    
    