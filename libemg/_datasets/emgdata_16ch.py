from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter

class EMGData16ch(Dataset):
    def __init__(self, dataset_folder='emgdata-16ch/'):
        Dataset.__init__(self, 
                        sampling=1000, 
                        num_channels=16, 
                        recording_device='GS26 Pre-gelled electrodes (bio-medical.com)', 
                        num_subjects={"No_Limb_Difference":12, 
                                      "Transradial_Limb_Loss":5}, 
                        gestures = {"CG": "Chuck Grip", 
                                    "FP": "Fine Pinch", 
                                    "HO": "Hand Open", 
                                    "KG": "Key Grip", 
                                    "NM": "No Motion", 
                                    "PG": "Power Grip", 
                                    "WP": "Wrist Pronation",
                                    "WS": "Wrist Supination"}, 
                        num_reps={'train':'15-24', 
                                  'test':'4'},
                        description="8-class forearm sEMG from participants with (n=5) and without (n=12) transradial limb loss",
                        citation='https://doi.org/10.1038/s41598-026-68446-1')
        self.url = "https://github.com/bella-montanaro/emgdata-16ch"
        self.dataset_folder = dataset_folder
        
    def prepare_data(self, split=True):
        # Cite original publication for which these data were collected
        print('\nPlease cite: ' + self.citation+'\n')

        # Create dataset folder
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        # Define dataset parameters
        populations = list(self.num_subjects.keys())
        participants = [str(i) for i in range(1, 13)] # 12 subjects without limb difference, 5 with limb loss
        datatypes = ["train", "test"]
        classes = list(self.gestures.keys())
        reps = [str(i) for i in range(1, 25)] # A maximum of 24 reps

        # Define filters
        regex_filters = [
            RegexFilter(left_bound="/", right_bound="/Participant", values=populations, description='population'),
            RegexFilter(left_bound="Participant", right_bound="/", values=participants, description='participant'),
            RegexFilter(left_bound="/", right_bound="/", values=datatypes, description='datatype'),
            RegexFilter(left_bound="/", right_bound="_", values=classes, description='class'),
            RegexFilter(left_bound="_", right_bound=".csv", values=reps, description='rep'),
        ]

        # Access and structure data
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        data = odh
        if split:
            data = {'All': odh, 
                    'Train': odh.isolate_data("datatype", [0], fast=True), 
                    'Test': odh.isolate_data("datatype", [1], fast=True)}

        return data