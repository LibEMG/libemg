from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

class RadmandLP(Dataset):
    def __init__(self, dataset_folder="LimbPosition/"):
        Dataset.__init__(self, 
                        1000, 
                        6, 
                        'DelsysTrigno', 
                        10, 
                        {'N/A': 'Uncertain'}, 
                        '4 Reps (Train), 4 Reps x 15 Positions',
                        "A large limb position dataset (with 16 static limb positions).",
                        "https://pubmed.ncbi.nlm.nih.gov/25570046/")
        self.url = "https://github.com/libemg/LimbPosition"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects = None):
        subject_list = np.array(list(range(1,11)))
        if subjects:
            subject_list = subject_list[subjects]
        subjects_values = [str(s) for s in subject_list]
        position_values = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
        classes_values = [str(i) for i in range(1,9)]
        reps_values = ["1","2","3","4"]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
    
        regex_filters = [
            RegexFilter(left_bound="/S", right_bound="/",values=subjects_values, description='subjects'),
            RegexFilter(left_bound = "_", right_bound="_R", values = position_values, description='positions'),
            RegexFilter(left_bound = "_C", right_bound="_P", values = classes_values, description='classes'),
            RegexFilter(left_bound = "_R", right_bound=".csv", values = reps_values, description='reps'),
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder + 'RadmandLimbPosition/', regex_filters=regex_filters, delimiter=",")
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("positions", [0], fast=True), 'Test': odh.isolate_data("positions", list(range(1, len(position_values))), fast=True)}

        return data