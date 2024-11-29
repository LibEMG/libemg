from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

class FougnerLP(Dataset):
    def __init__(self, dataset_folder="LimbPosition/"):
        Dataset.__init__(self, 
                        1000, 
                        8, 
                        'BE328 by Liberating Technologies, Inc.', 
                        12, 
                        {0: 'Wrist Flexion', 1: 'Wrist Extension', 2: 'Pronation', 3: 'Supination', 4: 'Hand Open', 5: 'Power Grip', 6: 'Pinch Grip', 7: 'Rest'}, 
                        '10 Reps (Train), 10 Reps x 4 Positions',
                        "A limb position dataset (with 5 static limb positions).",
                        "https://ieeexplore.ieee.org/document/5985538")
        self.url = "https://github.com/libemg/LimbPosition"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects = None):
        subject_list = np.array(list(range(1,13)))
        if subjects:
            subject_list = subject_list[subjects]
        subjects_values = [str(s) for s in subject_list]

        position_values = ["1", "2", "3", "4", "5"]
        classes_values = ["1", "2", "3", "4", "5", "8", "9", "12"]
        reps_values = ["1","2","3","4","5","6","7","8","9","10"]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
    
        regex_filters = [
            RegexFilter(left_bound="S", right_bound="_C",values=subjects_values, description='subjects'),
            RegexFilter(left_bound = "_P", right_bound="_R", values = position_values, description='positions'),
            RegexFilter(left_bound = "_C", right_bound="_P", values = classes_values, description='classes'),
            RegexFilter(left_bound = "_R", right_bound=".txt", values = reps_values, description='reps'),
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder + 'FougnerLimbPosition/', regex_filters=regex_filters, delimiter=",")
        odh = odh.isolate_channels(list(range(0,8)))
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("positions", [0], fast=True), 'Test': odh.isolate_data("positions", list(range(1, len(position_values))), fast=True)}

        return data