from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os

class GRABMyo(Dataset):
    """
    By default this just uses the 16 forearm electrodes. 
    """
    def __init__(self, save_dir='.', version='1.0.2', dataset_name="grabmyo", baseline=False):
        split = '7 Train, 14 Test (2 Seperate Days x 7 Reps)'
        if baseline:
            split = '5 Train, 2 Test (Basline)'
        Dataset.__init__(self, 
                         2048, 
                         16, 
                         'EMGUSB2+ device (OT Bioelletronica, Italy)', 
                         19, 
                         {0: 'Lateral Prehension', 1: 'Thumb Adduction', 2: 'Thumb and Little Finger Opposition', 3: 'Thumb and Index Finger Opposition', 4: 'Thumb and Index Finger Extension', 5: 'Thumb and Little Finger Extension', 6: 'Index and Middle Finger Extension',
                            7: 'Little Finger Extension', 8: 'Index Finger Extension', 9: 'Thumb Finger Extension', 10: 'Wrist Extension', 11: 'Wrist Flexion', 12: 'Forearm Supination', 13: 'Forearm Pronation', 14: 'Hand Open', 15: 'Hand Close', 16: 'Rest'},
                         split, 
                         "GrabMyo: A large cross session dataset including 17 gestures elicited across 3 seperate sessions.", 
                         'https://www.nature.com/articles/s41597-022-01836-y')
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(save_dir , self.dataset_name, version)

    def check_if_exist(self):
        if (not self.check_exists(self.dataset_folder)):
            print("Please download the GRABMyo dataset from: https://physionet.org/content/grabmyo/1.0.2/") 
            return 
        print('\nPlease cite: ' + self.citation+'\n')


class GRABMyoCrossDay(GRABMyo):
    def __init__(self, save_dir='.', version='1.0.2', dataset_name="grabmyo"):
        GRABMyo.__init__(self, save_dir=save_dir, version=version, dataset_name=dataset_name, baseline=False)
        
    def prepare_data(self):
        self.check_if_exist()

        sessions = ["1", "2", "3"]
        subjects = [str(i) for i in range(1,44)]
        classes_values = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"]
        reps_values = ["1","2","3","4","5","6","7"]

        regex_filters = [
            RegexFilter(left_bound = "session", right_bound="_", values = sessions, description='session'),
            RegexFilter(left_bound = "_gesture", right_bound="_", values = classes_values, description='classes'),
            RegexFilter(left_bound = "trial", right_bound=".hea", values = reps_values, description='reps'),
            RegexFilter(left_bound="participant", right_bound="_",values=subjects, description='subjects')
        ]

        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")

        forearm_data = odh.isolate_channels(list(range(0,16)))
        train_data = forearm_data.isolate_data('sets', [0])
        test_data = forearm_data.isolate_data('sets', [1])
        return {'All': forearm_data, 'Train': train_data, 'Test': test_data}

class GRABMyoBaseline(GRABMyo):
    def __init__(self, save_dir='.', version='1.0.2', dataset_name="grabmyo"):
        GRABMyo.__init__(self, save_dir=save_dir, version=version, dataset_name=dataset_name, baseline=True)
        
    def prepare_data(self):
        self.check_if_exist()

        sessions = ["1"]
        subjects = [str(i) for i in range(1,44)]
        classes_values = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"]
        reps_values = ["1","2","3","4","5","6","7"]

        regex_filters = [
            RegexFilter(left_bound = "session", right_bound="_", values = sessions, description='session'),
            RegexFilter(left_bound = "_gesture", right_bound="_", values = classes_values, description='classes'),
            RegexFilter(left_bound = "trial", right_bound=".hea", values = reps_values, description='reps'),
            RegexFilter(left_bound="participant", right_bound="_",values=subjects, description='subjects')
        ]

        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")

        forearm_data = odh.isolate_channels(list(range(0,16)))
        train_data = forearm_data.isolate_data('reps', [0,1,2,3,4])
        test_data = forearm_data.isolate_data('reps', [5,6])
        return {'All': forearm_data, 'Train': train_data, 'Test': test_data}