from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os

class GRABMyo(Dataset):
    def __init__(self):
        pass

    def __init__(self, save_dir='.', version='1.0.2', redownload=False, subjects=list(range(1,44)), sessions=list(range(1,4)), dataset_name="grabmyo"):
        Dataset.__init__(self, 
                        0, 
                        0, 
                        '', 
                        3, 
                        [], 
                        '',
                        "",
                        "",
                        save_dir, redownload)
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name, version)
        
    def download_data(self):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            print("Please download the GRABMyo dataset from: https://physionet.org/content/grabmyo/1.0.2/") #TODO: Fill this in
            return 

    def prepare_data(self, subjects=[str(i) for i in range(1,44)], sessions=["1"]):
        print('\nPlease cite: ' + self.citation+'\n')
        sessions = ["1"] # ["1", "2", "3"]
        subjects = ["1", "2", "3", "4", "5"] #[str(i) for i in range(1,44)]
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

#     def print_info(self):
#         print('Reference: https://www.physionet.org/content/grabmyo/1.0.2/') 
#         print('Name: ' + self.dataset_name)
#         print('Gestures: 17')
#         print('Trials: 7')
#         print('Time Per Rep: 5s')
#         print('Subjects: 43')
#         print("Forearm EMG (16): Columns 0-15\nWrist EMG (12): 18-23 and 26-31\nUnused (4): 16,23,24,31")
