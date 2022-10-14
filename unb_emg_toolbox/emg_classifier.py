from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import pickle

class EMGClassifier:
    '''
    EMG classification class - used to train and test models.
    Each EMGClassifier corresponds to an individual person.
    '''
    def __init__(self, 
                 model, 
                 data_set, 
                 arguments=None):
        '''
        model - the model that you want to train - options: ["LDA", "KNN", "SVM", "QDA"]
        data_set - the dataset acquired from the data loader class 
        arguments - a dictionary of arguments
        '''
        self.data_set = data_set
        self.arguments = arguments
        self.classifier = None
        self._format_data('training_features')
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
    def offline_evaluation(self):
        '''
        returns a list of typical offline evaluation metrics
        '''
        dic = {}
        predictions = self.classifier.predict_proba(self.data_set['testing_features'])
        dic['TER'] = self._get_ca(predictions, self.data_set['testing_labels'])
        dic['AER'] = self._get_ca(predictions, self.data_set['testing_labels'], include_nm=False, null_label=self.data_set['null_label'])
        return dic

    def save(self, filename):
        '''
        filename - is the location that the classifier gets saved to
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

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
    def offline_evaluation(self):
        '''
        returns a list of typical offline evaluation metrics
        '''
        dic = {}
        predictions = self.classifier.predict_proba(self.data_set['testing_features'])
        dic['TER'] = self._get_ca(predictions, self.data_set['testing_labels'])
        dic['AER'] = self._get_ca(predictions, self.data_set['testing_labels'], include_nm=False, null_label=self.data_set['null_label'])
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
    def _get_ca(self, predictons, gt_labels, include_nm=True, null_label=None):
        correct = 0
        num_predictions = 0
        for i, prediction in enumerate(predictons):
            predicted_val = np.argmax(prediction)
            probability = prediction[predicted_val]
            if not include_nm and predicted_val == null_label:
                continue
            if predicted_val == gt_labels[i]:
                correct += 1
            num_predictions += 1
        return correct / num_predictions

    def _get_instability(self, predictons, gt_labels):
        print("0%")
    
    # Offline Metrics:
    def _get_ca(self, predictons, gt_labels, include_nm=True, null_label=None):
        correct = 0
        num_predictions = 0
        for i, prediction in enumerate(predictons):
            predicted_val = np.argmax(prediction)
            probability = prediction[predicted_val]
            if not include_nm and predicted_val == null_label:
                continue
            if predicted_val == gt_labels[i]:
                correct += 1
            num_predictions += 1
        return correct / num_predictions

    def _get_instability(self, predictons, gt_labels):
        print("0%")