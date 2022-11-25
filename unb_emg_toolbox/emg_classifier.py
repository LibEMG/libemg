from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from multiprocessing import Process
import numpy as np
import pickle
import socket
import random
import matplotlib.pyplot as plt
import time
import inspect

from unb_emg_toolbox.utils import get_windows

class EMGClassifier:
    """Base EMG Classification class. 

    This class is the base class for offline EMG classification. Trains an sklearn ml model given a set
    of training and testing data and evalutes the results. 

    Parameters
    ----------
    model: string or custom classifier (must have fit, predict and predic_proba functions)
        The type of machine learning model. Valid options include: 'LDA', 'QDA', 'SVM', 'KNN', 'RF' (Random Forest),  
        'NB' (Naive Bayes), 'GB' (Gradient Boost), 'MLP' (Multilayer Perceptron). Note, these models are all default sklearn 
        models with no hyperparameter tuning and may not be optimal. Pass in custom classifiers or parameters for more control.
    data_set: dictionary
        A dictionary including the associated features and labels associated with a set of data. 
        Dictionary keys should include 'training_labels', 'training_features', 'testing_labels', and 
        'testing_features'.
    parameters: dictionary (optional)
        A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
        in the sklearn docs for the given model.
    continuous: bool (optional), default = False
        Specifies whether the testing data is continuous (no rest between contractions) or non-continuous. If False,
        majority vote is only applied to individual reps, not between them.
    rejection_type: string (optional)
        Used to specify the type of rejection used by the classifier. The only currently supported option
        is 'CONFIDENCE'.
    rejection_threshold: int (optional), default=0.9
        Used to specify the threshold used for rejection.
    majority_vote: int (optional) 
        Used to specify the number of predictions included in the majority vote.
    velocity: bool (optional), default=False
        If True, the classifier will output an associated velocity (used for velocity/proportional based control).
    """
    def __init__(self, model, data_set, parameters=None, continuous=False, rejection_type=None, rejection_threshold=0.9, majority_vote=None, velocity=False):
        random.seed(0)
        #TODO: Need some way to specify if its continuous testing data or not 
        self.data_set = data_set
        self.classifier = None
        self.rejection_type = rejection_type
        self.rejection_threshold = rejection_threshold
        self.majority_vote = majority_vote
        self.velocity = velocity
        self.predictions = []
        self.probabilities = []
        self.continuous = continuous

        # For velocity control
        self.th_min_dic = None 
        self.th_max_dic = None 

        # Functions to run:
        self._format_data('training_features')
        if 'testing_features' in self.data_set.keys():
            self._format_data('testing_features')
        self._set_up_classifier(model, parameters)
        if self.velocity:
            self.th_min_dic, self.th_max_dic = self._set_up_velocity_control()

    @classmethod
    def from_file(self, filename):
        """Loads a classifier - rather than creates a new one.

        After saving a model, you can recreate it by running EMGClassifier.from_file(). By default 
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

 
    def run(self):
        """Runs the classifier on a pre-defined set of training data.

        Returns
        ----------
        list
            Returns a list of predictions, based on the passed in testing features.
        """
        '''
        returns a list of typical offline evaluation metrics
        '''
        testing_labels = self.data_set['testing_labels'].copy()
        prob_predictions = self.classifier.predict_proba(self.data_set['testing_features'])
        
        # Default
        self.predictions, self.probabilities = self._prediction_helper(prob_predictions)

        # Rejection
        if self.rejection_type:
            self.predictions = np.array([self._rejection_helper(self.predictions[i], self.probabilities[i]) for i in range(0,len(self.predictions))])
            rejected = np.where(self.predictions == -1)[0]
            testing_labels[rejected] = -1

        # Majority Vote
        if self.majority_vote:
            self.predictions = self._majority_vote_helper(self.predictions)

        # Accumulate Metrics
        return self.predictions

    def save(self, filename):
        """Saves (pickles) the EMGClassifier object to a file.

        Use this save function to load the object later using the from_file function.

        Parameters
        ----------
        filename: string
            The path of the outputted pickled file. 
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    '''
    ---------------------- Private Helper Functions ----------------------
    '''
    def _format_data(self, i_key):
        arr = None
        for feat in self.data_set[i_key]:
            if arr is None:
                arr = self.data_set[i_key][feat]
            else:
                arr = np.hstack((arr, self.data_set[i_key][feat]))
        self.data_set[i_key] = arr

    def _set_up_classifier(self, model, parameters):
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
        self.classifier.fit(self.data_set['training_features'],self.data_set['training_labels'])

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
        if self.rejection_type == "CONFIDENCE":
            if prob > self.rejection_threshold:
                return prediction
            else:
                return -1
        return prediction
    
    def _majority_vote_helper(self, predictions):
        updated_predictions = []
        y_true = self.data_set['testing_labels']
        for i in range(0, len(predictions)):
            idxs = np.array(range(i-self.majority_vote+1, i+1))
            idxs = idxs[idxs >= 0]
            if not self.continuous:
                correct_group = y_true[i]
                idxs = idxs[np.where(y_true[idxs] == correct_group)]
            group = predictions[idxs]
            updated_predictions.append(np.argmax(np.bincount(group)))
        return np.array(updated_predictions)
    
    def _get_velocity(self, window, c):
        if self.th_max_dic and self.th_min_dic:
            return '{0:.2f}'.format((np.sum(np.mean(np.abs(window),2)[0], axis=0) - self.th_min_dic[c])/(self.th_max_dic[c] - self.th_min_dic[c]))

    def _set_up_velocity_control(self):
        # Extract classes 
        th_min_dic = {}
        th_max_dic = {}
        classes = np.unique(self.data_set['training_labels'])
        windows = self.data_set['training_windows']
        for c in classes:
            indices = np.where(self.data_set['training_labels'] == c)[0]
            c_windows = windows[indices]
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

    def visualize(self):
        """Visualize the decision stream of the classifier on the testing data. 

        After running the .run method, you can call this visualize function to get a visual 
        output of what the decision stream of what the particular classifier looks like. 
        """
        assert len(self.predictions) > 0
        
        plt.style.use('ggplot')
        colors = {}

        plt.gca().set_ylim([0, 1.05])
        plt.gca().xaxis.grid(False)

        # Plot true class labels
        changed_locations = [0] + list(np.where((self.data_set['testing_labels'][:-1] != self.data_set['testing_labels'][1:]) == True)[0]) + [len(self.data_set['testing_labels'])-1]

        for i in range(1, len(changed_locations)):
            class_label = self.data_set['testing_labels'][changed_locations[i]]
            if class_label in colors.keys():
                plt.fill_betweenx([0,1.02], changed_locations[i-1], changed_locations[i], color=colors[class_label])
            else:
                val = plt.fill_betweenx([0,1.02], changed_locations[i-1], changed_locations[i], alpha=.2)
                colors[class_label] = val.get_facecolors().tolist()
            
        # Plot decision stream
        plt.title("Decision Stream")
        plt.xlabel("Class Output")
        plt.ylabel("Probability")
        for g in np.unique(self.predictions):
            i = np.where(self.predictions == g)
            if g == -1:
                plt.scatter(i, self.probabilities[i], label=g, alpha=1, color='black')
            else:
                plt.scatter(i, self.probabilities[i], label=g, alpha=1, color=colors[g])
        
        plt.legend(loc='lower right')
        plt.show()
    
class OnlineEMGClassifier(EMGClassifier):
    """OnlineEMGClassifier (inherits from EMGClassifier) used for real-time classification.

    Given a set of training data and labels, this class will stream class predictions over UDP.

    Parameters
    ----------
    model: string
        The type of machine learning model. Valid options include: 'LDA', 'QDA', 'SVM' and 'KNN'. 
    data_set: dictionary
        A dictionary including the associated features and labels associated with a set of data. 
        Dictionary keys should include 'training_labels' and 'training_features'.
    num_channels: int
        The number of EMG channels.
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    online_data_handler: OnlineDataHandler
        An online data handler object.
    features: array_like
        A list of features that will be extracted during real-time classification. These should be the 
        same list used to train the model.
    parameters: dictionary (optional)
        A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
        in the sklearn docs for the given model.
    port: int (optional), default = 12346
        The port used for streaming predictions over UDP.
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    rejection_type: string (optional)
        Used to specify the type of rejection used by the classifier. The only currently supported option
        is 'CONFIDENCE'.
    rejection_threshold: int (optional), default = 0.9
        Used to specify the threshold used for rejection.
    majority_vote: int (optional)
        Used to specify the number of predictions included in the majority vote.
    velocity: bool (optional), default = False
        If True, the classifier will output an associated velocity (used for velocity/proportional based control).
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    """
    def __init__(self, model, data_set, num_channels, window_size, window_increment, online_data_handler, features, parameters=None, port=12346, ip='127.0.0.1', rejection_type=None, rejection_threshold=0.9, majority_vote=None, velocity=False, std_out=False):
        super().__init__(model, data_set, parameters=parameters, velocity=velocity)
        self.num_channels = num_channels
        self.window_size = window_size
        self.window_increment = window_increment
        self.raw_data = online_data_handler.raw_data
        self.features = features
        self.port = port
        self.ip = ip
        self.rejection_type = rejection_type
        self.rejection_threshold = rejection_threshold
        self.majority_vote = majority_vote
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.process = Process(target=self._run_helper, daemon=True,)
        self.std_out = std_out
        self.previous_predictions = deque(maxlen=self.majority_vote)

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

    def _run_helper(self):
        fe = FeatureExtractor(num_channels=self.num_channels)
        self.raw_data.reset_emg()
        while True:
            data = np.array(self.raw_data.get_emg())
            if len(data) >= self.window_size:
                # Extract window and predict sample
                window = get_windows(data, self.window_size, self.window_size)
                features = fe.extract_features(self.features, window)
                formatted_data = self._format_data_sample(features)
                self.raw_data.adjust_increment(self.window_size, self.window_increment)
                prediction, probability = self._prediction_helper(self.classifier.predict_proba(formatted_data))
                prediction = prediction[0]
                probability = probability[0]

                # Check for rejection
                if self.rejection_type:
                    #TODO: Right now this will default to -1
                    prediction = self._rejection_helper(prediction, probability)
                self.previous_predictions.append(prediction)
                
                # Check for majority vote
                if self.majority_vote:
                    values, counts = np.unique(list(self.previous_predictions), return_counts=True)
                    prediction = values[np.argmax(counts)]
                
                # Check for velocity based control
                calculated_velocity = ""
                if self.velocity:
                    calculated_velocity = " 0"
                    # Dont check if rejected 
                    if prediction >= 0:
                        calculated_velocity = " " + str(self._get_velocity(window, prediction))
                
                # Write classifier output:
                self.sock.sendto(bytes(str(str(prediction) + calculated_velocity), "utf-8"), (self.ip, self.port))
                if self.std_out:
                    print(f"{prediction} {calculated_velocity} {time.time()}")
    
    def _format_data_sample(self, data):
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr