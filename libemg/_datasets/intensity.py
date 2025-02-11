from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

class ContractionIntensity(Dataset):
    def __init__(self, dataset_folder="ContractionIntensity/"):
        Dataset.__init__(self, 
                        1000, 
                        8, 
                        'BE328 by Liberating Technologies, Inc', 
                        10, 
                        {0: "No Motion", 1: "Wrist Flexion", 2: "Wrist Extension", 3: "Wrist Pronation", 4: "Wrist Supination", 5: "Chuck Grip", 6: "Hand Open"}, 
                        '4 Ramp Reps (Train), 4 Reps x 20%, 30%, 40%, 50%, 60%, 70%, 80%, MVC (Test)',
                        "A contraction intensity dataset.",
                        "https://pubmed.ncbi.nlm.nih.gov/23894224/")
        self.url = "https://github.com/libemg/ContractionIntensity"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects = None):
        subject_list = np.array(list(range(1,11)))
        if subjects:
            subject_list = subject_list[subjects]
        subjects_values = [str(s) for s in subject_list]
        intensity_values = ["Ramp", "20P", "30P", "40P", "50P", "60P", "70P", "80P", "MVC"]
        classes_values = [str(i) for i in range(1,8)]
        reps_values = ["1","2","3","4"]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
    
        regex_filters = [
            RegexFilter(left_bound="/S", right_bound="/",values=subjects_values, description='subjects'),
            RegexFilter(left_bound = "_", right_bound="_C", values = intensity_values, description='intensities'),
            RegexFilter(left_bound = "_C", right_bound="_R", values = classes_values, description='classes'),
            RegexFilter(left_bound = "_R", right_bound=".csv", values = reps_values, description='reps'),
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("intensities", [0], fast=True), 'Test': odh.isolate_data("intensities", list(range(1, len(intensity_values))), fast=True)}

        return data