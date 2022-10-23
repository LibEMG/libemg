from collections import deque
from site import venv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from multiprocessing import Process
import numpy as np
import pickle
import socket
import random
from unb_emg_toolbox.offline_metrics import *

random.seed(0)

from unb_emg_toolbox.utils import get_windows

class EMGClassifier:
    """Base EMG Classification class. 

    This class is the base class for offline EMG classification. Trains an sklearn ml model given a set
    of training and testing data and evalutes the results. 

    Parameters
    ----------
    model: string or custom classifier (must have fit, predict and predic_proba functions)
        The type of machine learning model. Valid options include: 'LDA', 'QDA', 'SVM' and 'KNN'. 
    data_set: dictionary
        A dictionary including the associated features and labels associated with a set of data. 
        Dictionary keys should include 'training_labels', 'training_features', 'testing_labels', 
        'testing_features' and 'null_label' (optional).
    arguments: dictionary (optional)
        Used for additional arguments to the classifiers. Currently the only option is for the n_neighbors 
        option for the KNN classifier.
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
    def __init__(self, model, data_set, arguments=None, rejection_type=None, rejection_threshold=0.9, majority_vote=None, velocity=False):
        #TODO: Need some way to specify if its continuous testing data or not 
        self.data_set = data_set
        self.arguments = arguments
        self.classifier = None
        self.rejection_type = rejection_type
        self.rejection_threshold = rejection_threshold
        self.majority_vote = majority_vote
        self.velocity = velocity
        self.predictions = []
        self.probabilities = []

        # For velocity control
        self.th_min_dic = None 
        self.th_max_dic = None 

        # Functions to run:
        self._format_data('training_features')
        if 'testing_features' in self.data_set.keys():
            self._format_data('testing_features')
        self._set_up_classifier(model)
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
        """
        with open(filename, 'rb') as f:
            classifier = pickle.load(f)
        return classifier

 
    def run(self):
        """Runs the classifier on a pre-defined set of training data.

        Returns
        ----------
        dictionary
            Returns a dictionary consisting of a variety of offline metrics including: 
            Classification Accuracy ('CA'), Active Error Rate ('AER'), Instability ('INS'), 
            and Rejection Rate ('REJ_RATE').
        """
        '''
        returns a list of typical offline evaluation metrics
        '''
        dic = {}
        testing_labels = self.data_set['testing_labels'].copy()
        prob_predictions = self.classifier.predict_proba(self.data_set['testing_features'])
        
        # Default
        self.predictions, self.probabilities = self._prediction_helper(prob_predictions)

        # Rejection
        if self.rejection_type:
            # self.predictions = 
            dic['REJ_RATE'] = get_REJ_RATE(self.predictions)
            # rejected = np.where(predictions == -1)[0]
            # Update Predictions and Testing Labels Array
            # predictions = np.delete(predictions, rejected)
            # testing_labels = np.delete(testing_labels, rejected)
        # Majority Vote
        if self.majority_vote:
            self.predictions = self._majority_vote_helper(self.predictions)
        
        # Accumulate Metrics
        dic['CA'] = get_CA(testing_labels, self.predictions)
        if 'null_label' in self.data_set.keys():
            dic['AER'] = get_AER(testing_labels, self.predictions, self.data_set['null_label'])
        dic['INST'] = get_INS(testing_labels, self.predictions)
        return dic

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

    def _set_up_classifier(self, model):
        if model == "LDA":
            self.classifier = LinearDiscriminantAnalysis()
        elif model == "KNN":
            num_neighbors = 5
            if self.arguments and 'n_neighbors' in self.arguments:
                num_neighbors = self.arguments['n_neighbors']
            self.classifier = KNeighborsClassifier(n_neighbors=num_neighbors)
        elif model == "SVM":
            self.classifier = SVC(kernel='linear', probability=True)
        elif model == "QDA":
            self.classifier = QuadraticDiscriminantAnalysis()
        elif not model is None:
            # Assume a custom classifier has been passed in
            self.classifier = model
        # Fit the model to the data set
        self.classifier.fit(self.data_set['training_features'],self.data_set['training_labels'])
    
    def _prediction_helper(self, predictions):
        probabilities = [] 
        prediction_vals = []
        for i in range(0, len(predictions)):
            pred_list = list(predictions[i])
            prediction_vals.append(pred_list.index(max(pred_list)))
            probabilities.append(pred_list[pred_list.index(max(pred_list))])
        return np.array(prediction_vals), np.array(probabilities)
        
    def _rejection_helper(self, predictions):
        # TODO: Do we just want to do nothing? Or default to null_class? 
        if self.rejection_type == "CONFIDENCE":
            for i in range(0, len(predictions)):
                pred_list = list(predictions[i])
                if pred_list[pred_list.index(max(pred_list))] > self.rejection_threshold:
                    predictions[i] = pred_list.index(max(pred_list))
        return np.array(predictions)
    
    def _majority_vote_helper(self, predictions):
        updated_predictions = []
        # TODO: Decide what we want to do here - talk to Evan 
        # Right now we are just majority voting the whole prediction stream
        for i in range(self.majority_vote, len(predictions)):
            values, counts = np.unique(predictions[(i-self.majority_vote):i], return_counts=True)
            updated_predictions.append(values[np.argmax(counts)])
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
            th_min = (1-(10/100)) * mav_tr_min + 0.1 * mav_tr_max
            th_min_dic[c] = th_min 
            # Calculate THmax
            th_max = (1-(70/100)) * mav_tr_min + 0.7 * mav_tr_max
            th_max_dic[c] = th_max
        return th_min_dic, th_max_dic
    
class OnlineEMGClassifier(EMGClassifier):
    """OnlineEMGClassifier (inherits from EMGClassifier) used for real-time classification.

    Given a set of training data and labels, this class will stream class predictions over TCP.

    Parameters
    ----------
    model: string
        The type of machine learning model. Valid options include: 'LDA', 'QDA', 'SVM' and 'KNN'. 
    data_set: dictionary
        A dictionary including the associated features and labels associated with a set of data. 
        Dictionary keys should include 'training_labels', 'training_features' and 'null_label' (optional).
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    online_data_handler: OnlineDataHandler
        An online data handler object.
    feature_extractor: FeatureExtractor
        A feature extractor object with the features desired passed into the init.
    port: int (optional), default = 12346
        The port used for streaming predictions over TCP.
    ip: string (option), default = '127.0.0.1'
        The ip used for streaming predictions over TCP.
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
    def __init__(self, model, data_set, window_size, window_increment, online_data_handler, feature_extractor, port=12346, ip='127.0.0.1', rejection_type=None, rejection_threshold=0.9, majority_vote=None, velocity=False, std_out=False):
        super().__init__(model, data_set, velocity=velocity)
        self.window_size = window_size
        self.window_increment = window_increment
        self.odh = online_data_handler
        self.fe = feature_extractor
        self.port = port
        self.ip = ip
        self.rejection_type = rejection_type
        self.rejection_threshold = rejection_threshold
        self.majority_vote = majority_vote
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.process = Process(target=self._stream_emg, daemon=True,)
        self.std_out = std_out
        self.previous_predictions = deque(maxlen=self.majority_vote)

    def run(self):
        """Runs the classifier - continuously streams predictions over TCP.

        Currently this function locks the main thread.
        """
        self.odh.raw_data.reset_emg()
        while True:
            data = np.array(self.odh.raw_data.get_emg())
            if len(data) >= self.window_size:
                # Extract window and predict sample
                window = get_windows(data, self.window_size, self.window_size)
                features = self.fe.extract_predefined_features(window)
                formatted_data = self._format_data_sample(features)
                self.odh.raw_data.adjust_increment(self.window_size, self.window_increment)
                prediction = self.classifier.predict(formatted_data)[0]
                
                # Check for rejection
                if self.rejection_type:
                    #TODO: Right now this will default to -1
                    prediction = self._rejection_helper(self.classifier.predict_proba(formatted_data)[0])
                self.previous_predictions.append(prediction)
                
                # Check for majority vote
                if self.majority_vote:
                    values, counts = np.unique(list(self.previous_predictions), return_counts=True)
                    prediction = values[np.argmax(counts)]
                
                # Check for velocity vased control
                calculated_velocity = ""
                if self.velocity:
                    calculated_velocity = " " + str(self._get_velocity(window, prediction))
                
                # Write classifier output:
                self.sock.sendto(bytes(str(str(prediction) + calculated_velocity), "utf-8"), (self.ip, self.port))
                if self.std_out:
                    print(str(prediction) + calculated_velocity)
    
    def _format_data_sample(self, data):
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr