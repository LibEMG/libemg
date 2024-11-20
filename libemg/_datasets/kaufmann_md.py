from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter

class KaufmannMD(Dataset):
    def __init__(self, dataset_folder="MultiDay/"):
        Dataset.__init__(self, 
                        2048, 
                        4, 
                        'MindMedia', 
                        1, 
                        {0: "No Motion", 1:"Wrist Extension", 2:"Wrist Flexion", 3:"Wrist Adduction",
                         4:"Wrist Abduction", 5:"Wrist Supination", 6:"Wrist Pronation", 7:"Hand Open",
                         8:"Hand Closed", 9:"Key Grip", 10:"Index Point"}, 
                        '1 rep per day, 120 days total. 60/60 train-test split',
                        "A single subject, multi-day (120) collection.",
                        "https://ieeexplore.ieee.org/document/5627288")
        self.url = "https://github.com/LibEMG/MultiDay"
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects = None):
        subjects_values = ["0"]
        day_values = [str(i) for i in range(1,122)]
        classes_values = [str(i) for i in range(11)]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
    
        regex_filters = [
            RegexFilter(left_bound="/S", right_bound="_D",values=subjects_values, description='subjects'),
            RegexFilter(left_bound = "_D", right_bound="_C", values = day_values, description='days'),
            RegexFilter(left_bound = "_C", right_bound=".csv", values = classes_values, description='classes'),
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=" ")
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("days", list(range(60)), fast=True), 'Test': odh.isolate_data("days", list(range(60,121)), fast=True)}

        return data