from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

class OneSubjectMyoDataset(Dataset):
    def __init__(self, dataset_folder="OneSubjectMyoDataset/"):
        Dataset.__init__(self, 
                         200, 
                         8, 
                         'Myo Armband', 
                         1, 
                         {0: 'Close', 1: 'Open', 2: 'Rest', 3: 'Flexion', 4: 'Extension'}, 
                         '6 (4 Train, 2 Test)',
                         "A simple Myo dataset that is used for some of the LibEMG offline demos.", 
                         'N/A')
        self.url = "https://github.com/libemg/OneSubjectMyoDataset"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects=None):
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        sets_values = ["1","2","3","4","5","6"]
        classes_values = ["0","1","2","3","4"]
        reps_values = ["0","1"]
        regex_filters = [
            RegexFilter(left_bound = "/trial_", right_bound="/", values = sets_values, description='sets'),
            RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes'),
            RegexFilter(left_bound = "R_", right_bound="_", values = reps_values, description='reps')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        odh.subjects = []
        odh.subjects = [np.zeros((len(d), 1)) for d in odh.data]
        odh.extra_attributes.append('subjects')
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("sets", [0,1,2,3,4], fast=True), 'Test': odh.isolate_data("sets", [5,6], fast=True)}

        return data
