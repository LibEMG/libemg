from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.utils import *
from libemg.feature_extractor import FeatureExtractor
import os

class MyoDisCo(Dataset):
    def __init__(self, dataset_folder="MyoDisCo/", cross_day=False):
        self.cross_day = cross_day
        desc = 'The MyoDisCo dataset which includes both the across day and limb position confounds. (Limb Position Version)'
        if self.cross_day:
            desc = 'The MyoDisCo dataset which includes both the across day and limb position confounds. (Cross Day Version)'
        Dataset.__init__(self, 
                        200, 
                        8, 
                        'Myo Armband', 
                        14, 
                        {0: "Wrist Extension", 1: "Finger Gun", 2: "Wrist Flexion", 3: "Hand Close", 4: "Hand Open", 5: "Thumbs Up", 6: "Rest"}, 
                        '20 (Train) and 20 (Test) - Each gesture ~0.5s',
                        desc,
                        "https://iopscience.iop.org/article/10.1088/1741-2552/ad4915/meta")
        self.url = "https://github.com/libemg/MyoDisCo"
        self.dataset_folder = dataset_folder

    def prepare_data(self):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
        elif (self.redownload):
            self.remove_dataset(self.dataset_folder)
            self.download(self.url, self.dataset_folder)
    
        
        sets_values = ['day1', 'day2', 'positions']
        subjects_value = [str(i) for i in range(1,15)]
        classes_values = ["1","2","3","4","5","8","9"]
        reps_values = [str(i) for i in range(0,20)]
        regex_filters = [
            RegexFilter(left_bound = "/", right_bound="/", values = sets_values, description='sets'),
            RegexFilter(left_bound = "C_", right_bound="_EMG", values = classes_values, description='classes'),
            RegexFilter(left_bound = "R_", right_bound="_C", values = reps_values, description='reps'),
            RegexFilter(left_bound="S", right_bound="/",values=subjects_value, description='subjects')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        
        fe = FeatureExtractor()
        # We need to parse each item to remove no motion 
        for i, d in enumerate(odh.data):
            w = get_windows(d, 20, 10)
            mav = fe.extract_features(['MAV'], w, array=True)
            max_idx = np.argmax(np.mean(mav, axis=1)) * 10 + 20
            if odh.classes[i][0][0] == 6:
                odh.data[i] = d[100:200]
            else:
                low = max_idx-50
                high = max_idx+50
                if low < 0:
                    high += np.abs(low)
                    low = 0
                elif high >= len(odh.data[i]):
                    low -= np.abs(len(odh.data[i])-high)
                    high = len(odh.data[i]) 
                odh.data[i] = d[low:high]

            odh.sets[i] = np.ones((len(odh.data[i]), 1)) * odh.sets[i][0][0]
            odh.classes[i] = np.ones((len(odh.data[i]), 1)) * odh.classes[i][0][0]
            odh.reps[i] = np.ones((len(odh.data[i]), 1)) * odh.reps[i][0][0]
            odh.subjects[i] = np.ones((len(odh.data[i]), 1)) * odh.subjects[i][0][0]

        if self.cross_day:
            return {'All': odh, 'Train': odh.isolate_data("sets", [0]), 'Test': odh.isolate_data("sets", [1])}
        
        return {'All': odh, 'Train': odh.isolate_data("sets", [1]), 'Test': odh.isolate_data("sets", [2])}
