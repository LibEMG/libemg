from abc import ABC, abstractmethod
from copy import deepcopy

from libemg.data_handler import RegexFilter, FilePackager, OfflineDataHandler
from libemg._datasets.dataset import Dataset

class Hyser(Dataset, ABC):
    def __init__(self, gestures, num_reps, description, dataset_folder, analysis = 'baseline'):
        super().__init__(
            sampling=2048,
            num_channels=256,
            recording_device='OT Bioelettronica Quattrocento',
            num_subjects=20,
            gestures=gestures,
            num_reps=num_reps,
            description=description,
            citation='https://doi.org/10.13026/ym7v-bh53'
        )

        self.url = 'https://www.physionet.org/content/hd-semg/1.0.0/'
        self.dataset_folder = dataset_folder
        self.analysis = analysis

    def prepare_data(self, split = False):
        if (not self.check_exists(self.dataset_folder)):
            raise FileNotFoundError(f"Didn't find Hyser data in {self.dataset_folder} directory. Please download the dataset and \
                                    store it in the appropriate directory before running prepare_data(). See {self.url} for download details.")
        return self._prepare_data_helper(split=split)

    @abstractmethod
    def _prepare_data_helper(self, split = False):
        ...


class Hyser1DOF(Hyser):
    def __init__(self, dataset_folder = 'Hyser1DOF', analysis = 'baseline'):
        gestures = {1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little Finger'}
        definition = 'Hyser 1 DOF dataset. Includes within-DOF finger movements. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures=gestures, num_reps=3, description=definition, dataset_folder=dataset_folder, analysis=analysis)

    def _prepare_data_helper(self, split = False):
        sessions_values = ['1', '2'] if self.analysis == 'session' else ['1']   # only grab first session unless both are desired
        common_filters = [
            RegexFilter(left_bound='subject', right_bound='_session', values=[str(idx).zfill(2) for idx in range(self.num_subjects + 1)], description='subjects'),   # +1 due to Python indexing
            RegexFilter(left_bound='_session', right_bound='/1dof_', values=sessions_values, description='session'),
            RegexFilter(left_bound='_finger', right_bound='_sample', values=['1', '2', '3', '4', '5'], description='finger'),
            RegexFilter(left_bound='_sample', right_bound='.hea', values=['1', '2', '3'], description='rep')
        ]

        regex_filters = deepcopy(common_filters)
        regex_filters.append(RegexFilter(left_bound='1dof_', right_bound='_finger', values=['raw'], description='data_type'))

        metadata_fetchers = [
            FilePackager(RegexFilter(left_bound='/1dof_', right_bound='_finger', values=['force'], description='labels'),
                          package_function=common_filters, load='p_signal')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        data = odh
        if split:
            # Can likely move this to parent class...
            if self.analysis == 'session':
                data = {'All': odh, 'Train': odh.isolate_data('session', [0], fast=True), 'Test': odh.isolate_data('session', [1], fast=True)}
            elif self.analysis == 'baseline':
                data = {'All': odh, 'Train': odh.isolate_data('rep', [0, 1], fast=True), 'Test': odh.isolate_data('rep', [2], fast=True)}
            else:
                raise ValueError(f"Unexpected value for analysis. Suported values are session, baseline. Got: {self.analysis}.")
        return data
