from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

class GRABMyo(Dataset):
    """
    By default this just uses the 16 forearm electrodes. 
    """
    def __init__(self, dataset_folder='GRABMyo/', baseline=False, cross_user=False):
        if not cross_user:
            split = '7 Train, 14 Test (2 Seperate Days x 7 Reps) --> Cross Day Split'
            if baseline:
                split = '5 Train, 2 Test --> Baseline Split'
        else:
            split = '30 Subjects x 3 Sessions (Train) - 14 Subjects x 3 Sessions (Test) --> Cross User Split'
        Dataset.__init__(self, 
                         2048, 
                         16, 
                         'EMGUSB2+ device (OT Bioelletronica, Italy)', 
                         43, 
                         {0: 'Lateral Prehension', 1: 'Thumb Adduction', 2: 'Thumb and Little Finger Opposition', 3: 'Thumb and Index Finger Opposition', 4: 'Thumb and Index Finger Extension', 5: 'Thumb and Little Finger Extension', 6: 'Index and Middle Finger Extension',
                            7: 'Little Finger Extension', 8: 'Index Finger Extension', 9: 'Thumb Finger Extension', 10: 'Wrist Extension', 11: 'Wrist Flexion', 12: 'Forearm Supination', 13: 'Forearm Pronation', 14: 'Hand Open', 15: 'Hand Close', 16: 'Rest'},
                         split, 
                         "GrabMyo: A large cross session dataset including 17 gestures elicited across 3 seperate sessions.", 
                         'https://www.nature.com/articles/s41597-022-01836-y')
        self.dataset_folder = dataset_folder

    def get_odh(self, subjects = None):
        self.check_if_exist()

        sessions = ["1", "2", "3"]
        subject_list = np.array(list(range(1,44)))
        if subjects:
            subject_list = subject_list[subjects]
        subjects_values = [str(s) for s in subject_list]
        classes_values = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"]
        reps_values = ["1","2","3","4","5","6","7"]

        regex_filters = [
            RegexFilter(left_bound = "session", right_bound="_", values = sessions, description='sessions'),
            RegexFilter(left_bound = "_gesture", right_bound="_", values = classes_values, description='classes'),
            RegexFilter(left_bound = "trial", right_bound=".hea", values = reps_values, description='reps'),
            RegexFilter(left_bound="participant", right_bound="_",values=subjects_values, description='subjects')
        ]

        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")

        return odh.isolate_channels(list(range(0,16)))

    def check_if_exist(self):
        if (not self.check_exists(self.dataset_folder)):
            print("Please download the GRABMyo dataset from: https://physionet.org/content/grabmyo/1.0.2/") 
            return 
        print('\nPlease cite: ' + self.citation+'\n')

class GrabMyoCrossUser(GRABMyo):
    def __init__(self, dataset_folder="GRABMyo"):
        GRABMyo.__init__(self, dataset_folder=dataset_folder, baseline=False)

    def prepare_data(self, split = True):
        forearm_data = self.get_odh()
       
        train_data = forearm_data.isolate_data('subjects', list(range(0,30)), fast=True)
        test_data = forearm_data.isolate_data('subjects', list(range(30,43)), fast=True)

        data = forearm_data
        if split:
            data = {'All': forearm_data, 'Train': train_data, 'Test': test_data}

        return data

class GRABMyoCrossDay(GRABMyo):
    def __init__(self, dataset_folder="GRABMyo"):
        GRABMyo.__init__(self, dataset_folder=dataset_folder, baseline=False)
        
    def prepare_data(self, split = True, subjects = None):
        forearm_data = self.get_odh(subjects)
        train_data = forearm_data.isolate_data('sessions', [0], fast=True)
        test_data = forearm_data.isolate_data('sessions', [1,2], fast=True)

        data = forearm_data
        if split:
            data = {'All': forearm_data, 'Train': train_data, 'Test': test_data}

        return data

class GRABMyoBaseline(GRABMyo):
    def __init__(self, dataset_folder="GRABMyo"):
        GRABMyo.__init__(self, dataset_folder=dataset_folder, baseline=True)
        
    def prepare_data(self, split = True, subjects = None):
        forearm_data = self.get_odh(subjects)
        forearm_data = forearm_data.isolate_data('sessions', [0])

        train_data = forearm_data.isolate_data('reps', [0,1,2,3,4], fast=True)
        test_data = forearm_data.isolate_data('reps', [5,6], fast=True)

        data = forearm_data
        if split:
            data = {'All': forearm_data, 'Train': train_data, 'Test': test_data}

        return data