from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os

class OneSubjectMyoDataset(Dataset):
    def __init__(self, save_dir='.', redownload=False, dataset_name="OneSubjectMyoDataset"):
        Dataset.__init__(self, 
                         200, 
                         8, 
                         'Myo Armband', 
                         1, 
                         ['Close', 'Open', 'Rest', 'Flexion', 'Extension'], 
                         '6 (4 Train, 2 Test)',
                         "A simple Myo dataset that is used for some of the LibEMG offline demos.", 
                         'N/A', save_dir, redownload)
        self.url = "https://github.com/libemg/OneSubjectMyoDataset"
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)

    def prepare_data(self, format=OfflineDataHandler):
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
        elif (self.redownload):
            self.remove_dataset(self.dataset_folder)
            self.download(self.url, self.dataset_folder)

        if format == OfflineDataHandler:
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
            return {'All': odh, 'Train': odh.isolate_data("sets", [0,1,2,3,4]), 'Test': odh.isolate_data("sets", [5,6])}