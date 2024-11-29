from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import numpy as np

class TMRShirleyRyanAbilityLab(Dataset):
    def __init__(self, dataset_folder="TMR/", desc=''):
        Dataset.__init__(self, 
                        1000, 
                        32, 
                        'Ag/AgCl', 
                        6, 
                        {0:"HandOpen",
                         1:"KeyGrip",
                         2:"PowerGrip",
                         3:"FinePinchOpened",
                         4:"FinePinchClosed",
                         5:"TripodOpened",
                         6:"TripodClosed",
                         7:"Tool",
                         8:"Hook",
                         9:"IndexPoint",
                         10:"ThumbFlexion",
                         11:"ThumbExtension",
                         12:"ThumbAbduction",
                         13:"ThumbAdduction",
                         14:"IndexFlexion",
                         15:"RingFlexion",
                         16:"PinkyFlexion",
                         17:"WristSupination",
                         18:"WristPronation",
                         19:"WristFlexion",
                         20:"WristExtension",
                         21:"RadialDeviation",
                         22:"UlnarDeviation",
                         23:"NoMotion"}, 
                         8,
                        desc,
                        "https://pmc.ncbi.nlm.nih.gov/articles/PMC9879512/")
        self.url = "https://github.com/LibEMG/TMR_ShirleyRyanAbilityLab"
        self.dataset_folder = dataset_folder

    def get_odh(self, subjects = None):
        subject_list = np.array([1,2,3,4,7,10])
        if subjects:
            subject_list = subject_list[subjects]
        subjects_values = [str(s) for s in subject_list]

        reps_values     = [str(i) for i in range(8)]
        classes_values  = [str(i) for i in range(24)]
        intervention_values = ["preTMR","postTMR"]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
    
        regex_filters = [
            RegexFilter(left_bound="/S", right_bound="/",values=subjects_values, description='subjects'),
            RegexFilter(left_bound = "_R", right_bound=".txt", values = reps_values, description='reps'),
            RegexFilter(left_bound = "/C", right_bound="_R", values = classes_values, description='classes'),
            RegexFilter(left_bound = "/", right_bound="/C", values = intervention_values, description='intervention')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        return odh

class TMR_Pre(TMRShirleyRyanAbilityLab):
    """
    Data from participants pre TMR surgery. 
    """
    def __init__(self, dataset_folder="TMR/"):
        TMRShirleyRyanAbilityLab.__init__(self, dataset_folder=dataset_folder, desc='TMR Dataset: 6 subjects, 8 reps, 24 motions, pre intervention')

    def prepare_data(self, split=True, subjects=None):
        odh = self.get_odh(subjects)
        odh = odh.isolate_data('intervention', [0])
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("reps", list(range(6)), fast=True), 'Test': odh.isolate_data("reps", list(range(6,8)), fast=True)}
        return data 

class TMR_Post(TMRShirleyRyanAbilityLab):
    """
    Data from participants post TMR surgery. 
    """
    def __init__(self, dataset_folder="TMR/"):
        TMRShirleyRyanAbilityLab.__init__(self, dataset_folder=dataset_folder, desc='TMR Dataset: 6 subjects, 8 reps, 24 motions, post intervention')

    def prepare_data(self, split=True, subjects=None):
        odh = self.get_odh(subjects)
        odh = odh.isolate_data('intervention', [1])
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data("reps", list(range(6)), fast=True), 'Test': odh.isolate_data("reps", list(range(6,8)), fast=True)}
        return data 