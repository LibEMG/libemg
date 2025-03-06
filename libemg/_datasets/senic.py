from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np
from os import walk

"""
Participants h0 to h5 have 10 sessions, h6 to h13 have 3 and h14 to h35 have 1. 
I cant figure out limb position or fatigue.
"""

class SeNic(Dataset):
    def __init__(self, dataset_folder="SeNic/"):
        Dataset.__init__(self, 
                         200, 
                         8, 
                         'Myo Armband', 
                         36, 
                         {'TODO': 'TODO'}, 
                         'TODO',
                         "A Myo Armband dataset with different confounding factors: electrode shift, limb position, fatigue, and cross-day.", 
                         'N/A')
        self.url = "https://github.com/bozhubo/SeNic"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects=None):
        # Download from GitHub
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        # Extract rar files 
        filenames = next(walk(self.dataset_folder), (None, None, []))[2]
        filenames.sort()
        for f in filenames:
            if '.rar' in f:
                print("Unpack all rars manually!")
                # TODO: Amir - figure out how to unpack rars automatically. 

        sets_values = [str(i) for i in range(0, 10)]
        subjects_values = [str(i) for i in range(0,6)]
        positions_values = [str(i) for i in range(0, 10)]
        classes_values = ['eversion', 'fist', 'open_hand', 'pinch_forefinger', 'pinch_middlefinger', 'two', 'varus']
        reps_values = ['0','1','2']

        regex_filters = [
            RegexFilter(left_bound = "h", right_bound="/", values = subjects_values, description='subjects'),
            RegexFilter(left_bound = "/", right_bound="/emg_", values = sets_values, description='days'),
            RegexFilter(left_bound = "_p", right_bound="_r", values = positions_values, description='positions'),
            RegexFilter(left_bound = "_", right_bound=".csv", values = classes_values, description='classes'),
            RegexFilter(left_bound = "_r", right_bound="_", values = reps_values, description='reps')
        ]

        # TODO: We need to active threshold because there is not an explicit NM class 

        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        return odh
