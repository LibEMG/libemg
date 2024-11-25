import numpy as np 
from pathlib import Path
from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter, FilePackager

class UserComplianceDataset(Dataset):
    def __init__(self, dataset_folder = 'UserComplianceDataset/', analysis = 'baseline'):
        super().__init__(
            sampling=1010,
            num_channels=64,
            recording_device='EMaGer',
            num_subjects=6,
            gestures={0: 'Hand Close (-) / Hand Open (+)', 1: 'Pronation (-) / Supination (+)'},
            num_reps=5,
            description='Regression dataset used for investigation into user compliance during mimic training.',
            citation='https://conferences.lib.unb.ca/index.php/mec/article/view/2507'
        )
        self.url = 'https://github.com/LibEMG/UserComplianceDataset'
        self.dataset_folder = dataset_folder
        self.analysis = analysis
        self.subject_list = np.array(['subject-001', 'subject-002', 'subject-003', 'subject-006', 'subject-007', 'subject-008'])

    def prepare_data(self, split = True, subjects = None):
        subject_list = self.subject_list
        if subjects:
            subject_list = subject_list[subjects]
            
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)

        regex_filters = [
            RegexFilter(left_bound='/', right_bound='/', values=['open-close', 'pro-sup'], description='movements'),
            RegexFilter(left_bound='_R_', right_bound='.csv', values=[str(idx) for idx in range(self.num_reps)], description='reps'),
            RegexFilter(left_bound='/', right_bound='/', values=['anticipation', 'all-or-nothing', 'baseline'], description='behaviours'),
            RegexFilter(left_bound='/', right_bound='/', values=list(subject_list), description='subjects')
        ]
        package_function = lambda x, y: Path(x).parent.absolute() == Path(y).parent.absolute()
        metadata_fetchers = [FilePackager(RegexFilter(left_bound='/', right_bound='.txt', values=['labels'], description='labels'), package_function)]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        data = odh
        if split:
            if self.analysis == 'baseline':
                data = {
                    'All': odh, 
                    'Train': odh.isolate_data('behaviours', [0, 1], fast=True), 
                    'Test': odh.isolate_data('behaviours', [2], fast=True)
                }
            elif self.analysis == 'all-or-nothing':
                data = {'All': odh, 'Train': odh.isolate_data('behaviours', [1], fast=True), 'Test': odh.isolate_data('behaviours', [2], fast=True)}
            elif self.analysis == 'anticipation':
                data = {'All': odh, 'Train': odh.isolate_data('behaviours', [0], fast=True), 'Test': odh.isolate_data('behaviours', [2], fast=True)}
            else:
                raise ValueError(f"Unexpected value for analysis. Got: {self.analysis}.")

        return data
