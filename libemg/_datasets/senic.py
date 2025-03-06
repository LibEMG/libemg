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
                # print(self.dataset_folder + f)
                # with rarfile.RarFile(self.dataset_folder + f) as rf:
                #     rf.extractall(self.dataset_folder)
                # # os.remove(self.dataset_folder + f)
        return None 

        # sets_values = ["1","2","3","4","5","6"]
        # classes_values = ["0","1","2","3","4"]
        # reps_values = ["0","1"]
        # regex_filters = [
        #     RegexFilter(left_bound = "/trial_", right_bound="/", values = sets_values, description='sets'),
        #     RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes'),
        #     RegexFilter(left_bound = "R_", right_bound="_", values = reps_values, description='reps')
        # ]
        # odh = OfflineDataHandler()
        # odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        # odh.subjects = []
        # odh.subjects = [np.zeros((len(d), 1)) for d in odh.data]
        # odh.extra_attributes.append('subjects')
        # data = odh
        # if split:
        #     data = {'All': odh, 'Train': odh.isolate_data("sets", [0,1,2,3,4], fast=True), 'Test': odh.isolate_data("sets", [5,6], fast=True)}

        # return data
