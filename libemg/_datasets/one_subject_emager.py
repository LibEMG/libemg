from pathlib import Path

import numpy as np 
from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter, FilePackager


class OneSubjectEMaGerDataset(Dataset):
    def __init__(self, dataset_folder = 'OneSubjectEMaGerDataset/'):
        super().__init__(
            sampling=1010,
            num_channels=64,
            recording_device='EMaGer',
            num_subjects=1,
            gestures={0: 'Hand Close (-) / Hand Open (+)', 1: 'Pronation (-) / Supination (+)'},
            num_reps=5,
            description='A simple EMaGer dataset used for regression examples in LibEMG demos.',
            citation='N/A'
        )
        self.url = 'https://github.com/LibEMG/OneSubjectEMaGerDataset'
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects = None):
        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
        regex_filters = [
            RegexFilter(left_bound='/', right_bound='/', values=['open-close', 'pro-sup'], description='movements'),
            RegexFilter(left_bound='_R_', right_bound='_emg.csv', values=[str(idx) for idx in range(self.num_reps)], description='reps')
        ]
        package_function = lambda x, y: Path(x).parent.absolute() == Path(y).parent.absolute()
        metadata_fetchers = [FilePackager(RegexFilter(left_bound='/', right_bound='.txt', values=['labels'], description='labels'), package_function)]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        odh.subjects = []
        odh.subjects = [np.zeros((len(d), 1)) for d in odh.data]
        odh.extra_attributes.append('subjects')
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data('reps', [0, 1, 2, 3], fast=True), 'Test': odh.isolate_data('reps', [4], fast=True)}

        return data
