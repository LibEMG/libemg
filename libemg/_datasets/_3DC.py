from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

class _3DCDataset(Dataset):
    def __init__(self, dataset_folder="_3DCDataset/"):
        Dataset.__init__(self, 
                        1000, 
                        10, 
                        '3DC Armband (Prototype)', 
                        22, 
                        {0: "Neutral", 1: "Radial Deviation", 2: "Wrist Flexion", 3: "Ulnar Deviation", 4: "Wrist Extension", 5: "Supination", 6: "Pronation", 7: "Power Grip", 8: "Open Hand", 9: "Chuck Grip", 10: "Pinch Grip"}, 
                        '8 (4 Train, 4 Test)',
                        "The 3DC dataset including 11 classes.",
                        "https://doi.org/10.3389/fbioe.2020.00158")
        self.url = "https://github.com/libemg/3DCDataset"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects = None):
        subject_list = np.array(list(range(1,23)))
        if subjects:
            subject_list = subject_list[subjects]
        subjects_values = [str(s) for s in subject_list]


        sets_values = ["train", "test"]
        reps_values = ["0","1","2","3"]
        classes_values = [str(i) for i in range(11)]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
    
        regex_filters = [
            RegexFilter(left_bound = "/", right_bound="/EMG", values = sets_values, description='sets'),
            RegexFilter(left_bound = "_", right_bound=".txt", values = classes_values, description='classes'),
            RegexFilter(left_bound = "EMG_gesture_", right_bound="_", values = reps_values, description='reps'),
            RegexFilter(left_bound="Participant", right_bound="/",values=subjects_values, description='subjects')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("sets", [0], fast=True), 'Test': odh.isolate_data("sets", [1], fast=True)}

        return data