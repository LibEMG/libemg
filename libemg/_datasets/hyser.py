from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np

from libemg.data_handler import RegexFilter, FilePackager, OfflineDataHandler, MetadataFetcher
from libemg._datasets.dataset import Dataset


class _Hyser(Dataset, ABC):
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
        subjects = [str(idx + 1).zfill(2) for idx in range(self.num_subjects)]   # +1 due to Python indexing

        self.url = 'https://www.physionet.org/content/hd-semg/1.0.0/'
        self.dataset_folder = dataset_folder
        self.analysis = analysis
        self.subjects = subjects
        
    @property
    def common_regex_filters(self):
        sessions_values = ['1', '2'] if self.analysis == 'sessions' else ['1']   # only grab first session unless both are desired
        filters = [
            RegexFilter(left_bound='subject', right_bound='_session', values=self.subjects, description='subjects'),
            RegexFilter(left_bound='_session', right_bound='/', values=sessions_values, description='sessions')
        ]
        return filters

    def prepare_data(self, split = True, subjects = None):
        if (not self.check_exists(self.dataset_folder)):
            raise FileNotFoundError(f"Didn't find Hyser data in {self.dataset_folder} directory. Please download the dataset and \
                                    store it in the appropriate directory before running prepare_data(). See {self.url} for download details.")
        return self._prepare_data_helper(split=split, subjects = subjects)
        
    @abstractmethod
    def _prepare_data_helper(self, split = True, subjects = None) -> dict | OfflineDataHandler:
        ...


class Hyser1DOF(_Hyser):
    def __init__(self, dataset_folder: str = 'Hyser1DOF', analysis: str = 'baseline'):
        """1 degree of freedom (DOF) Hyser dataset.

        Parameters
        ----------
        dataset_folder: str, default='Hyser1DOF'
            Directory that contains Hyser 1 DOF dataset.
        analysis: str, default='baseline'
            Determines which type of data will be extracted and considered train/test splits. If 'baseline', only grabs data from the first session and splits based on
            reps. If 'sessions', grabs data from both sessions and return the first session as train and the second session as test.
        """
        gestures = {1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little'}
        description = 'Hyser 1 DOF dataset. Includes within-DOF finger movements. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures=gestures, num_reps=3, description=description, dataset_folder=dataset_folder, analysis=analysis)

    def _prepare_data_helper(self, split = True, subjects = None):
        subject_list = np.array(list(range(1,21)))
        if subjects:
            subject_list = subject_list[subjects]
        self.subjects = [f'{s:02d}' for s in subject_list]

        filename_filters = deepcopy(self.common_regex_filters)
        filename_filters.append(RegexFilter(left_bound='_sample', right_bound='.hea', values=[str(idx + 1) for idx in range(self.num_reps)], description='reps'))
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
                raise ValueError(f"Unexpected value for analysis. Suported values are sessions, baseline. Got: {self.analysis}.")
        return data

        
class HyserNDOF(_Hyser):
    def __init__(self, dataset_folder: str = 'HyserNDOF', analysis: str = 'baseline'):
        """N degree of freedom (DOF) Hyser dataset.

        Parameters
        ----------
        dataset_folder: str, default='HyserNDOF'
            Directory that contains Hyser N DOF dataset.
        analysis: str, default='baseline'
            Determines which type of data will be extracted and considered train/test splits. If 'baseline', only grabs data from the first session and splits based on
            reps. If 'sessions', grabs data from both sessions and return the first session as train and the second session as test.
        """
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
        description = 'Hyser N DOF dataset. Includes combined finger movements. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures=self.finger_combinations, num_reps=2, description=description, dataset_folder=dataset_folder, analysis=analysis) 

    def _prepare_data_helper(self, split = True, subjects = None) -> dict | OfflineDataHandler:
        subject_list = np.array(list(range(1,21)))
        if subjects:
            subject_list = subject_list[subjects]
        self.subjects = [f'{s:02d}' for s in subject_list]

        filename_filters = deepcopy(self.common_regex_filters)
        filename_filters.append(RegexFilter(left_bound='_sample', right_bound='.hea', values=[str(idx + 1) for idx in range(self.num_reps)], description='reps'))
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
                raise ValueError(f"Unexpected value for analysis. Suported values are sessions, baseline. Got: {self.analysis}.")
            
        return data
        

class HyserRandom(_Hyser):
    def __init__(self, dataset_folder: str = 'HyserRandom', analysis: str = 'baseline'):
        """Random task (DOF) Hyser dataset.

        Parameters
        ----------
        dataset_folder: str, default='HyserRandom'
            Directory that contains Hyser random task dataset.
        analysis: str, default='baseline'
            Determines which type of data will be extracted and considered train/test splits. If 'baseline', only grabs data from the first session and splits based on
            reps. If 'sessions', grabs data from both sessions and return the first session as train and the second session as test.
        """
        description = 'Hyser random dataset. Includes random motions performed by users. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures={}, num_reps=5, description=description, dataset_folder=dataset_folder, analysis=analysis)
        self.num_subjects = 19


    def _prepare_data_helper(self, split = True, subjects = None) -> dict | OfflineDataHandler:
        subject_list = np.delete(np.array(list(range(1,21))), 9)
        if subjects:
            subject_list = subject_list[subjects]
        self.subjects = [f'{s:02d}' for s in subject_list]

        filename_filters = deepcopy(self.common_regex_filters)
        filename_filters.append(RegexFilter(left_bound='_sample', right_bound='.hea', values=[str(idx + 1) for idx in range(self.num_reps)], description='reps'))

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
                raise ValueError(f"Unexpected value for analysis. Suported values are sessions, baseline. Got: {self.analysis}.")
            
        return data

        
class _PRLabelsFetcher(MetadataFetcher):
    def __init__(self):
        super().__init__(description='classes')
        self.sample_regex = RegexFilter(left_bound='_sample', right_bound='.hea', values=[str(idx + 1) for idx in range(204)], description='samples')

    def _get_labels(self, filename):
        label_filename_map = {
            'dynamic': 'label_dynamic.txt',
            'maintenance': 'label_maintenance.txt'
        }
        matches = []
        for task_type, labels_file in label_filename_map.items():
            if task_type in filename:
                matches.append(labels_file)

        assert len(matches) == 1, f"Expected a single label file for this file, but got {len(matches)}. Got filename: {filename}. Filename should contain either 'dynamic' or 'maintenance'."
        
        labels_file = matches[0]
        parent = Path(filename).absolute().parent
        labels_file = Path(parent, labels_file).as_posix()
        return np.loadtxt(labels_file, delimiter=',', dtype=int)

    def __call__(self, filename, file_data, all_files):
        labels = self._get_labels(filename)       
        sample_idx = self.sample_regex.get_metadata(filename)
        return labels[sample_idx] - 1   # -1 to produce 0-indexed labels
        

class _PRRepFetcher(_PRLabelsFetcher):
    def __init__(self):
        super().__init__()
        self.description = 'reps'

    def __call__(self, filename, file_data, all_files):
        label = super().__call__(filename, file_data, all_files) + 1    # +1 b/c this returns 0-indexed labels, but the files are 1-indexed
        labels = self._get_labels(filename)
        same_label_mask = np.where(labels == label)[0]
        sample_idx = self.sample_regex.get_metadata(filename)
        rep_idx = list(same_label_mask).index(sample_idx)
        if 'dynamic' in filename:
            # Each trial is 3 dynamic reps, 1 maintenance rep
            rep_idx = rep_idx // 3

        assert rep_idx <= 1, f"Rep values should be 0 or 1 (2 total reps). Got: {rep_idx}."
        return np.array(rep_idx)

        
class HyserPR(_Hyser):
    def __init__(self, dataset_folder: str = 'HyserPR', analysis: str = 'baseline'):
        """Pattern recognition (PR) Hyser dataset.

        Parameters
        ----------
        dataset_folder: str, default='HyserPR'
            Directory that contains Hyser PR dataset.
        analysis: str, default='baseline'
            Determines which type of data will be extracted and considered train/test splits. If 'baseline', only grabs data from the first session and splits based on
            reps. If 'sessions', grabs data from both sessions and return the first session as train and the second session as test.
        """
        gestures = {
            1: 'Thumb Extension',
            2: 'Index Finger Extension',
            3: 'Middle Finger Extension',
            4: 'Ring Finger Extension',
            5: 'Little Finger Extension',
            6: 'Wrist Flexion',
            7: 'Wrist Extension',
            8: 'Wrist Radial',
            9: 'Wrist Ulnar',
            10: 'Wrist Pronation',
            11: 'Wrist Supination',
            12: 'Extension of Thumb and Index Fingers',
            13: 'Extension of Index and Middle Fingers',
            14: 'Wrist Flexion Combined with Hand Close',
            15: 'Wrist Extension Combined with Hand Close',
            16: 'Wrist Radial Combined with Hand Close',
            17: 'Wrist Ulnar Combined with Hand Close',
            18: 'Wrist Pronation Combined with Hand Close',
            19: 'Wrist Supination Combined with Hand Close',
            20: 'Wrist Flexion Combined with Hand Open',
            21: 'Wrist Extension Combined with Hand Open',
            22: 'Wrist Radial Combined with Hand Open',
            23: 'Wrist Ulnar Combined with Hand Open',
            24: 'Wrist Pronation Combined with Hand Open',
            25: 'Wrist Supination Combined with Hand Open',
            26: 'Extension of Thumb, Index and Middle Fingers',
            27: 'Extension of Index, Middle and Ring Fingers',
            28: 'Extension of Middle, Ring and Little Fingers',
            29: 'Extension of Index, Middle, Ring and Little Fingers',
            30: 'Hand Close',
            31: 'Hand Open',
            32: 'Thumb and Index Fingers Pinch',
            33: 'Thumb, Index and Middle Fingers Pinch',
            34: 'Thumb and Middle Fingers Pinch'
        }
        description = 'Hyser pattern recognition (PR) dataset. Includes dynamic and maintenance tasks for 34 hand gestures.'
        super().__init__(gestures=gestures, num_reps=2, description=description, dataset_folder=dataset_folder, analysis=analysis)  # num_reps=2 b/c 2 trials
        self.num_subjects = 18 # Removed 2 subjects because they're missing classes

    def _prepare_data_helper(self, split = True, subjects = None) -> dict | OfflineDataHandler:
        # Need to remove subjects 3 and 11 b/c they're missing classes
        subject_list = np.delete(np.array(list(range(1,21))), [2,10])
        if subjects:
            subject_list = subject_list[subjects]

        self.subjects = [f'{s:02d}' for s in subject_list]

        filename_filters = deepcopy(self.common_regex_filters)
        filename_filters.append(RegexFilter(left_bound='_sample', right_bound='.hea', values=[str(idx + 1) for idx in range(204)], description='samples')) # max # of dynamic tasks
        filename_filters.append(RegexFilter(left_bound='/', right_bound='_', values=['dynamic', 'maintenance'], description='tasks'))

        regex_filters = deepcopy(filename_filters)
        regex_filters.append(RegexFilter(left_bound='_', right_bound='_sample', values=['raw'], description='data_type'))

        metadata_fetchers = [
            _PRLabelsFetcher(),
            _PRRepFetcher()
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
                raise ValueError(f"Unexpected value for analysis. Suported values are sessions, baseline. Got: {self.analysis}.")
            
        return data


class HyserMVC(_Hyser):
    def __init__(self, dataset_folder: str = 'HyserMVC'):
        """Maximum voluntary contraction (MVC) Hyser dataset.

        Parameters
        ----------
        dataset_folder: str, default='HyserMVC'
            Directory that contains the Hyser MVC dataset.
        """
        gestures = {1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little'}
        description = 'Hyser maximum voluntary contraction (MVC) dataset. Includes MVC for flexion and extension of each finger. Typically used for normalization of other Hyser datasets.'
        super().__init__(gestures=gestures, num_reps=5, description=description, dataset_folder=dataset_folder, analysis='sessions')
        
    def _prepare_data_helper(self, split=True, subjects=None):
        subject_list = np.array(list(range(1,21)))
        if subjects:
            subject_list = subject_list[subjects]
        self.subjects = [f'{s:02d}' for s in subject_list]

        filename_filters = deepcopy(self.common_regex_filters)
        filename_filters.append(RegexFilter(left_bound='_', right_bound='.hea', values=['flexion', 'extension'], description='movement'))
        filename_filters.append(RegexFilter(left_bound='_finger', right_bound='_', values=['1', '2', '3', '4', '5'], description='finger'))

        regex_filters = deepcopy(filename_filters)
        regex_filters.append(RegexFilter(left_bound='mvc_', right_bound='_finger', values=['raw'], description='data_type'))

        metadata_fetchers = [
            FilePackager(RegexFilter(left_bound='/mvc_', right_bound='_finger', values=['force'], description='labels'),
                          package_function=filename_filters, load='p_signal')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        data = odh
        if split:
            # Split on different sessions (no split for within-session)
            data = {'All': odh, 'Train': odh.isolate_data('sessions', [0], fast=True), 'Test': odh.isolate_data('sessions', [1], fast=True)}
        return data
