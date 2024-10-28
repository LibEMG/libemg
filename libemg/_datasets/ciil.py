from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter, FilePackager
from pathlib import Path
class CIIL_MinimalData(Dataset):
    def __init__(self, dataset_folder='CIILData/'):
        Dataset.__init__(self, 
                         200, 
                         8, 
                         'Myo Armband', 
                         11, 
                         {0: 'Close', 1: 'Open', 2: 'Rest', 3: 'Flexion', 4: 'Extension'}, 
                         '1 Train (1s), 15 Test',
                         "The goal of this Myo dataset is to explore how well models perform when they have a limited amount of training data (1s per class).", 
                         'https://ieeexplore.ieee.org/abstract/document/10394393')
        self.url = "https://github.com/LibEMG/CIILData"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = False):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        subfolder = 'MinimalTrainingData'
        subjects = [str(i) for i in range(0, 11)]
        classes_values = [str(i) for i in range(0,5)]
        reps_values = ["0","1","2"]
        sets = ["train", "test"]
        regex_filters = [
            RegexFilter(left_bound = "/", right_bound="/", values = sets, description='sets'),
            RegexFilter(left_bound = "/subject", right_bound="/", values = subjects, description='subjects'),
            RegexFilter(left_bound = "R_", right_bound="_", values = reps_values, description='reps'),
            RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder + '/' + subfolder, regex_filters=regex_filters, delimiter=",")

        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("sets", [0], fast=True), 'Test': odh.isolate_data("sets", [1], fast=True)}

        return data
    
class CIIL_ElectrodeShift(Dataset):
    def __init__(self, dataset_folder='CIILData/'):
        Dataset.__init__(self, 
                         200, 
                         8, 
                         'Myo Armband', 
                         21, 
                         {0: 'Close', 1: 'Open', 2: 'Rest', 3: 'Flexion', 4: 'Extension'}, 
                         '5 Train (Before Shift), 8 Test (After Shift)',
                         "An electrode shift confounding factors dataset.", 
                         'https://link.springer.com/article/10.1186/s12984-024-01355-4')
        self.url = "https://github.com/LibEMG/CIILData"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = False):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        subfolder = 'ElectrodeShift'
        subjects = [str(i) for i in range(0, 21)]
        classes_values = [str(i) for i in range(0,5)]
        reps_values = ["0","1","2","3","4"]
        sets = ["training", "trial_1", "trial_2", "trial_3", "trial_4"]
        regex_filters = [
            RegexFilter(left_bound = "/", right_bound="/", values = sets, description='sets'),
            RegexFilter(left_bound = "/subject", right_bound="/", values = subjects, description='subjects'),
            RegexFilter(left_bound = "R_", right_bound="_", values = reps_values, description='reps'),
            RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder + '/' + subfolder, regex_filters=regex_filters, delimiter=",")

        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("sets", [0], fast=True), 'Test': odh.isolate_data("sets", [1,2,3,4], fast=True)}

        return data


class CIIL_WeaklySupervised(Dataset):
    def __init__(self, dataset_folder='WS_CIILData/'):
        Dataset.__init__(self, 
                         1000, 
                         8, 
                         'OyMotion gForcePro+ EMG Armband', 
                         16, 
                         {0: 'Close', 1: 'Open', 2: 'Rest', 3: 'Flexion', 4: 'Extension'}, 
                         '30 min weakly supervised, 1 rep calibration, 14 reps test',
                         "A weakly supervised environment with sparse supervised calibration.", 
                         'In Submission')
        self.url = "https://github.com/ECEEvanCampbell/ZSCIIL_Dataset"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = False):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        # supervised odh loading
        subjects = [str(i) for i in range(0, 16)]
        classes_values = [str(i) for i in range(0,5)]
        reps_values = [str(i) for i in range(0,15)]
        setting_values     = [".csv", ""] # this is arbitrary to get a field that separates WS from S
        regex_filters = [
            RegexFilter(left_bound = "", right_bound="", values = setting_values, description='settings'),
            RegexFilter(left_bound = "/subject", right_bound="/", values = subjects, description='subjects'),
            RegexFilter(left_bound = "R_", right_bound="_", values = reps_values, description='reps'),
            RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes')
        ]
        odh_s = OfflineDataHandler()
        odh_s.get_data(folder_location=self.dataset_folder,
                       regex_filters=regex_filters,
                       delimiter=",")

        # weakly supervised odh loading
        subjects = [str(i) for i in range(0, 16)]
        rep      = [str(i) for i in range(3)]
        setting_values     = ["", ".csv"] # this is arbitrary to get a field that separates WS from S
        regex_filters = [
            RegexFilter(left_bound = "", right_bound="", values = setting_values, description='settings'),
            RegexFilter(left_bound = "/subject", right_bound="/", values = subjects, description='subjects'),
            RegexFilter(left_bound = "WS", right_bound=".csv", values = reps_values, description='reps'),
        ]
        metadata_fetchers = [
            FilePackager(regex_filter=[
                RegexFilter(left_bound="", right_bound="targets.csv", values=["_"], description="classses")
            ],
            package_function=lambda x, y: (x[2] == y[2]) and (Path(x).parent == Path(y).parent)
            )
        ]
        odh_ws = OfflineDataHandler()
        odh_ws.get_data(folder_location=self.dataset_folder, 
                     regex_filters=regex_filters, 
                     metadata_fetchers=metadata_fetchers,
                     delimiter=",")

        data = odh_s + odh_ws
        if split:
            data = {'All': data, 'Pretrain': odh_ws,
                    'Train': odh_s.isolate_data("reps", [0], fast=True), 
                    'Test': odh_s.isolate_data("reps", list(range(1,15)), fast=True)}

        return data
