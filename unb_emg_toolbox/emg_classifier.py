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
    def __init__(self, model, data_set, arguments=None):
        '''
        model - the model that you want to train - options: ["LDA", "KNN", "SVM", "QDA"]
        data_set - the dataset acquired from the data loader class 
        arguments - a dictionary of arguments
        '''
        self.data_set = data_set
        self.arguments = arguments
        self.classifier = None
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
        predictions = self.classifier.predict(self.data_set['testing_features'])
        dic['CA'] = self._get_CA(self.data_set['testing_labels'], predictions)
        if 'null_label' in self.data_set.keys():
            dic['AER'] = self._get_AER(self.data_set['testing_labels'], predictions, self.data_set['null_label'])
        dic['INST'] = self._get_INS(self.data_set['testing_labels'], predictions)
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
        return (pred_changes - num_gt_changes) / len(y_predictions)

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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.process = Process(target=self._stream_emg, daemon=True,)
        self.std_out = std_out

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
                
