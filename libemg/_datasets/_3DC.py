from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os

class _3DCDataset(Dataset):
    def __init__(self, save_dir='.', redownload=False, dataset_name="_3DCDataset"):
        Dataset.__init__(self, 
                        1000, 
                        10, 
                        '3DC Armband (Prototype)', 
                        22, 
                        {0: "Neutral", 1: "Radial Deviation", 2: "Wrist Flexion", 3: "Ulnar Deviation", 4: "Wrist Extension", 5: "Supination", 6: "Pronation", 7: "Power Grip", 8: "Open Hand", 9: "Chuck Grip", 10: "Pinch Grip"}, 
                        '8 (4 Train, 4 Test)',
                        "The 3DC dataset including 11 classes.",
                        "https://ieeexplore.ieee.org/document/8630679",
                        save_dir, redownload)
        self.url = "https://github.com/libemg/3DCDataset"
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)

    def prepare_data(self, format=OfflineDataHandler, subjects_values = [str(i) for i in range(1,23)],
                                                      sets_values = ["train", "test"],
                                                      reps_values = ["0","1","2","3"],
                                                      classes_values = [str(i) for i in range(11)]):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
        elif (self.redownload):
            self.remove_dataset(self.dataset_folder)
            self.download(self.url, self.dataset_folder)
    
        if format == OfflineDataHandler:
            regex_filters = [
                RegexFilter(left_bound = "/", right_bound="/EMG", values = sets_values, description='sets'),
                RegexFilter(left_bound = "_", right_bound=".txt", values = classes_values, description='classes'),
                RegexFilter(left_bound = "EMG_gesture_", right_bound="_", values = reps_values, description='reps'),
                RegexFilter(left_bound="Participant", right_bound="/",values=subjects_values, description='subjects')
            ]
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
            return {'All': odh, 'Train': odh.isolate_data("sets", [0]), 'Test': odh.isolate_data("sets", [1])}