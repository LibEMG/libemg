from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

    # This is an audio class, not a myoelectric signal dataset!

class SpeakerDistortionDataset(Dataset):
    def __init__(self, dataset_folder="tests/audio_tests/SpeakerDistortionDataset"):
        Dataset.__init__(self,
                         96000,  
                         1,
                         'Focusrite Scarlet 2i2 (3rd Generation)',
                         1,
                         {1: 'No Added Resistance', 2: '1.5 ohm Added', 3: '3 ohm Added'},
                         '2 (1 Train, 1 Test)',
                         "A dataset of square wave responses to use with LibEMG offline demos.",
                         'N/A'
                         )
        self.url = "https://github.com/G-ODonnell/SpeakerDistortionDataset"
        self.dataset_folder = dataset_folder
    
    def prepare_data(self, split = True, subjects=None):
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        sets_values = ["1","2"]
        load_values = ["1","2","3"]
        distance_values = ["1","2","3","4","5"]  
        regex_filters = [
            RegexFilter(left_bound = "/trial_", right_bound="/", values = sets_values, description='sets'),
            RegexFilter(left_bound = "L_", right_bound=".csv", values = load_values, description='loads'),
            RegexFilter(left_bound = "D_", right_bound="_", values = distance_values, description='distance')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("sets", [0], fast=True), 'Test': odh.isolate_data("sets", [1], fast=True)}

        return data