
from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import *

class OneSiteBiopoint(Dataset):
    def __init__(self, dataset_folder='CIIL_WeaklySupervised/'):
        Dataset.__init__(self, 
                         2000, 
                         1, 
                         'SiFi-Labs BioPoint', 
                         8, 
                         {0: 'Close', 1: 'Open', 2: 'Rest', 3: 'Flexion', 4: 'Extension'}, 
                         'Six reps',
                         "A single site, multimodal sensor for gesture recognition", 
                         'EMBC 2024 - Not Yet Published')
        self.url = "https://unbcloud-my.sharepoint.com/:u:/g/personal/ecampbe2_unb_ca/EZG9zfWg_hdJl4De1Clnl34ByTjYqStTB90Nj6EaHkGSnA?e=JQLU7z"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = False):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download_via_onedrive(self.url, self.dataset_folder)

        subjects = [str(i) for i in range(0, 8)]
        classes_values = [str(i) for i in range(0,17)]
        reps_values = [str(i) for i in range(0,6)]
        regex_filters = [
            RegexFilter(left_bound = "/S", right_bound="/", values = subjects, description='subjects'),
            RegexFilter(left_bound = "R_", right_bound="EMG-bio.csv", values = reps_values, description='reps'),
            RegexFilter(left_bound = "C_", right_bound="_R", values = classes_values, description='classes')
        ]
        odh_s = OfflineDataHandler()
        odh_s.get_data(folder_location=self.dataset_folder+"OneSiteBioPoint/",
                       regex_filters=regex_filters,
                       delimiter=",")

        
        if split:
            data = {'All': data, 
                    'Train': odh_s.isolate_data("reps", list(range(0,3)), fast=True), 
                    'Test': odh_s.isolate_data("reps", list(range(3,6)), fast=True)}

        return data



