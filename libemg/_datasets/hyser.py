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
        
        sessions_values = ['1', '2'] if self.analysis == 'sessions' else ['1']   # only grab first session unless both are desired
        self.common_regex_filters = [
            RegexFilter(left_bound='subject', right_bound='_session', values=[str(idx + 1).zfill(2) for idx in range(self.num_subjects)], description='subjects'),   # +1 due to Python indexing
            RegexFilter(left_bound='_session', right_bound='/', values=sessions_values, description='sessions'),
            RegexFilter(left_bound='_sample', right_bound='.hea', values=[str(idx + 1) for idx in range(self.num_reps)], description='reps') 
        ]

    def prepare_data(self, split = False):
        if (not self.check_exists(self.dataset_folder)):
            raise FileNotFoundError(f"Didn't find Hyser data in {self.dataset_folder} directory. Please download the dataset and \
                                    store it in the appropriate directory before running prepare_data(). See {self.url} for download details.")
        return self._prepare_data_helper(split=split)
        
    @abstractmethod
    def _prepare_data_helper(self, split = False) -> dict | OfflineDataHandler:
        ...


class Hyser1DOF(Hyser):
    def __init__(self, dataset_folder = 'Hyser1DOF', analysis = 'baseline'):
        gestures = {1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little'}
        definition = 'Hyser 1 DOF dataset. Includes within-DOF finger movements. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures=gestures, num_reps=3, description=definition, dataset_folder=dataset_folder, analysis=analysis)

    def _prepare_data_helper(self, split = False):
        filename_filters = deepcopy(self.common_regex_filters)
        filename_filters.append(RegexFilter(left_bound='_finger', right_bound='_sample', values=['1', '2', '3', '4', '5'], description='finger'))

        regex_filters = deepcopy(filename_filters)
        regex_filters.append(RegexFilter(left_bound='1dof_', right_bound='_finger', values=['raw'], description='data_type'))

        metadata_fetchers = [
            FilePackager(RegexFilter(left_bound='/1dof_', right_bound='_finger', values=['force'], description='labels'),
                          package_function=filename_filters, load='p_signal')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        data = odh
        if split:
            if self.analysis == 'sessions':
                data = {'All': odh, 'Train': odh.isolate_data('sessions', [0], fast=True), 'Test': odh.isolate_data('sessions', [1], fast=True)}
            elif self.analysis == 'baseline':
                data = {'All': odh, 'Train': odh.isolate_data('reps', [0, 1], fast=True), 'Test': odh.isolate_data('reps', [2], fast=True)}
            else:
                raise ValueError(f"Unexpected value for analysis. Suported values are session, baseline. Got: {self.analysis}.")
        return data

        
class HyserNDOF(Hyser):
    def __init__(self, dataset_folder = 'HyserNDOF', analysis = 'baseline'):
        # TODO: Add a 'regression' flag... maybe add a 'DOFs' parameter instead of just gestures?
        gestures = {1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little'}
        definition = 'Hyser N DOF dataset. Includes combined finger movements. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures=gestures, num_reps=2, description=definition, dataset_folder=dataset_folder, analysis=analysis) 
        self.finger_combinations = {
            1: 'Thumb + Index',
            2: 'Thumb + Middle',
            3: 'Thumg + Ring',
            4: 'Thumb + Little',
            5: 'Index + Middle',
            6: 'Thumb + Index + Middle',
            7: 'Index + Middle + Ring',
            8: 'Middle + Ring + Little',
            9: 'Index + Middle + Ring + Little',
            10: 'All Fingers',
            11: 'Thumb + Index (Opposing)',
            12: 'Thumb + Middle (Opposing)',
            13: 'Thumg + Ring (Opposing)',
            14: 'Thumb + Little (Opposing)',
            15: 'Index + Middle (Opposing)'
        }

    def _prepare_data_helper(self, split = False) -> dict | OfflineDataHandler:
        filename_filters = deepcopy(self.common_regex_filters)
        filename_filters.append(RegexFilter(left_bound='_combination', right_bound='_sample', values=[str(idx + 1) for idx in range(len(self.finger_combinations))], description='finger_combinations'))

        regex_filters = deepcopy(filename_filters)
        regex_filters.append(RegexFilter(left_bound='/ndof_', right_bound='_combination', values=['raw'], description='data_type'))

        metadata_fetchers = [
            FilePackager(RegexFilter(left_bound='/ndof_', right_bound='_combination', values=['force'], description='labels'),
                          package_function=filename_filters, load='p_signal')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        data = odh
        if split:
            if self.analysis == 'sessions':
                data = {'All': odh, 'Train': odh.isolate_data('sessions', [0], fast=True), 'Test': odh.isolate_data('sessions', [1], fast=True)}
            elif self.analysis == 'baseline':
                data = {'All': odh, 'Train': odh.isolate_data('reps', [0], fast=True), 'Test': odh.isolate_data('reps', [1], fast=True)}
            else:
                raise ValueError(f"Unexpected value for analysis. Suported values are session, baseline. Got: {self.analysis}.")
            
        return data
        

class HyserRandom(Hyser):
    def __init__(self, dataset_folder = 'HyserRandom', analysis = 'baseline'):
        gestures = {1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little'}
        definition = 'Hyser random dataset. Includes random motions performed by users. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures=gestures, num_reps=5, description=definition, dataset_folder=dataset_folder, analysis=analysis) 

    def _prepare_data_helper(self, split = False) -> dict | OfflineDataHandler:
        filename_filters = deepcopy(self.common_regex_filters)

        regex_filters = deepcopy(filename_filters)
        regex_filters.append(RegexFilter(left_bound='/random_', right_bound='_sample', values=['raw'], description='data_type'))

        metadata_fetchers = [
            FilePackager(RegexFilter(left_bound='/random_', right_bound='_sample', values=['force'], description='labels'),
                          package_function=filename_filters, load='p_signal')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        data = odh
        if split:
            if self.analysis == 'sessions':
                data = {'All': odh, 'Train': odh.isolate_data('sessions', [0], fast=True), 'Test': odh.isolate_data('sessions', [1], fast=True)}
            elif self.analysis == 'baseline':
                data = {'All': odh, 'Train': odh.isolate_data('reps', [0, 1, 2], fast=True), 'Test': odh.isolate_data('reps', [3, 4], fast=True)}
            else:
                raise ValueError(f"Unexpected value for analysis. Suported values are session, baseline. Got: {self.analysis}.")
            
        return data
