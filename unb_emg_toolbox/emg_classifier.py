from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
from multiprocessing import Process
import numpy as np
import pickle
import socket

from unb_emg_toolbox.utils import get_windows

class EMGClassifier:
    '''
    EMG classification class - used to train and test models.
    Each EMGClassifier corresponds to an individual person.
    '''
    def __init__(self, model, data_set, arguments=None, rejection_type=None, rejection_threshold=0.9, majority_vote=None):
        '''
        model - the model that you want to train - options: ["LDA", "KNN", "SVM", "QDA"]
        data_set - the dataset acquired from the data loader class 
        arguments - a dictionary of arguments
        '''
        #TODO: Need some way to specify if its continuous testing data or not 
        self.data_set = data_set
        self.arguments = arguments
        self.classifier = None
        self.rejection_type = rejection_type
        self.rejection_threshold = rejection_threshold
        self.majority_vote = majority_vote
        # Functions to run:
        self._format_data('training_features')
        if 'testing_features' in self.data_set.keys():
            self._format_data('testing_features')
        self._set_up_classifier(model)
    
    @classmethod
    def from_file(self, filename):
        '''
        Loads a classifier - rather than creates one.
        filename - is the location that you want to load the classifier from
        '''
        with open(filename, 'rb') as f:
            classifier = pickle.load(f)
        return classifier
    '''
    ---------------------- Public Functions ----------------------
    '''
    def run(self):
        '''
        returns a list of typical offline evaluation metrics
        '''
        dic = {}
        predictions = None
        testing_labels = self.data_set['testing_labels'].copy()
        # Default
        predictions = self.classifier.predict(self.data_set['testing_features'])
        # Rejection
        if self.rejection_type:
            prediction_probs = self.classifier.predict_proba(self.data_set['testing_features'])
            predictions = np.array([self._check_for_rejection(pred) for pred in prediction_probs])
            dic['REJ_RATE'] = self._get_REJ_RATE(predictions)
            rejected = np.where(predictions == -1)[0]
            # Update Predictions and Testing Labels Array
            predictions = np.delete(predictions, rejected)
            testing_labels = np.delete(testing_labels, rejected)
        # Majority Vote
        if self.majority_vote:
            predictions = self._majority_vote_helper(predictions, testing_labels)
        
        # Accumulate Metrics
        dic['CA'] = self._get_CA(testing_labels, predictions)
        if 'null_label' in self.data_set.keys():
            dic['AER'] = self._get_AER(testing_labels, predictions, self.data_set['null_label'])
        dic['INST'] = self._get_INS(testing_labels, predictions)
        return dic

    def save(self, filename):
        '''
        filename - is the location that the classifier gets saved to
        '''
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
        # Fit the model to the data set
        self.classifier.fit(self.data_set['training_features'],self.data_set['training_labels'])
        
    def _check_for_rejection(self, prediction):
        # TODO: Do we just want to do nothing? Or default to null_class? 
        pred_list = list(prediction)
        if self.rejection_type == "CONFIDENCE":
            if pred_list[pred_list.index(max(pred_list))] > self.rejection_threshold:
                return pred_list.index(max(pred_list))
            else:
                return -1
        return pred_list.index(max(pred_list))
    
    def _majority_vote_helper(self, predictions, testing):
        # TODO: Decide what we want to do here - talk to Evan 
        # Right now we are just majority voting the whole prediction stream
        for i in range(self.majority_vote, len(predictions)):
            values, counts = np.unique(list(predictions[i:i+self.majority_vote]), return_counts=True)
            predictions[i] = values[np.argmax(counts)]
        return predictions
      
    # Offline Metrics:
    # TODO: Evan Review
    def _get_CA(self, y_true, y_predictions):
        return sum(y_predictions == y_true)/len(y_true)
    
    def _get_AER(self, y_true, y_predictions, null_class):
        nm_predictions = [i for i, x in enumerate(y_predictions) if x == null_class]
        return self._get_CA(np.delete(y_true, nm_predictions), np.delete(y_predictions, nm_predictions))

    def _get_INS(self, y_true, y_predictions):
        num_gt_changes = np.count_nonzero(y_true[:-1] != y_true[1:])
        pred_changes = np.count_nonzero(y_predictions[:-1] != y_predictions[1:])
        ins = (pred_changes - num_gt_changes) / len(y_predictions)
        return ins if ins > 0 else 0.0

    def _get_REJ_RATE(self, y_predictions):
        return sum(y_predictions == -1)/len(y_predictions)

    #TODO: Add additional metrics
    
class OnlineEMGClassifier(EMGClassifier):
    def __init__(self, dictionary, std_out=False):
        super().__init__(dictionary['model'], dictionary['data_set'])
        self.port = dictionary['port'] 
        self.ip = dictionary['ip'] 
        self.window_length = dictionary['window_length'] 
        self.window_increment = dictionary['window_increment']
        self.odh = dictionary['online_data_handler']
        self.fe = dictionary['feature_extractor']
        if 'rejection_type' in dictionary.keys():
            self.rejection_type = dictionary['rejection_type']
        if 'rejection_threshold' in dictionary.keys():
            self.rejection_threshold = dictionary['rejection_threshold']
        if 'majority_vote' in dictionary.keys():
            self.majority_vote = dictionary['majority_vote']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.process = Process(target=self._stream_emg, daemon=True,)
        self.std_out = std_out
        self.previous_predictions = deque(maxlen=self.majority_vote)

    def run(self):
        self.odh.raw_data.reset_emg()
        while True:
            data = np.array(self.odh.raw_data.get_emg())
            if len(data) >= self.window_length:
                window = get_windows(data, self.window_length, self.window_length)
                features = self.fe.extract_predefined_features(window)
                formatted_data = self._format_data_sample(features)
                self.odh.raw_data.adjust_increment(self.window_length, self.window_increment)
                prediction = self.classifier.predict(formatted_data)
                if self.rejection_type:
                    #TODO: Right now this will default to -1
                    prediction = self._check_for_rejection(self.classifier.predict_proba(formatted_data)[0])
                self.previous_predictions.append(prediction)
                if self.majority_vote:
                    values, counts = np.unique(list(self.previous_predictions), return_counts=True)
                    prediction = values[np.argmax(counts)]
                if self.std_out:
                    print(prediction)
                self.sock.sendto(bytes(str(prediction), "utf-8"), (self.ip, self.port))
    
    def _format_data_sample(self, data):
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr
                
