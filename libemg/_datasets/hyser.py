from pathlib import Path
from abc import ABC, abstractmethod

from libemg.data_handler import RegexFilter, FilePackager, OfflineDataHandler
from libemg._datasets.dataset import Dataset

class Hyser(Dataset, ABC):
    def __init__(self, gestures, num_reps, description, dataset_folder):
        # super().__init__(
        #     sampling=1010,
        #     num_channels=64,
        #     recording_device='EMaGer',
        #     num_subjects=1,
        #     gestures={0: 'Hand Close (-) / Hand Open (+)', 1: 'Pronation (-) / Supination (+)'},
        #     num_reps=5,
        #     description='A simple EMaGer dataset used for regression examples in LibEMG demos.',
        #     citation='N/A'
        # )
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

        self.url = 'https://www.physionet.org/content/hd-semg/1.0.0/#files-panel'
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = False):
        if (not self.check_exists(self.dataset_folder)):
            raise FileNotFoundError(f"Didn't find Hyser data in {self.dataset_folder}. Please download the dataset and store it in the appropriate directory before running
                             prepare_data(). See {self.url} for download details.")
        return self._prepare_data_helper(split=split)
        # regex_filters = [
        #     RegexFilter(left_bound='/', right_bound='/', values=['open-close', 'pro-sup'], description='movements'),
        #     RegexFilter(left_bound='_R_', right_bound='_emg.csv', values=[str(idx) for idx in range(self.num_reps)], description='reps')
        # ]
        # package_function = lambda x, y: Path(x).parent.absolute() == Path(y).parent.absolute()
        # metadata_fetchers = [FilePackager(RegexFilter(left_bound='/', right_bound='.txt', values=['labels'], description='labels'), package_function)]
        # odh = OfflineDataHandler()
        # odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        # data = odh
        # if split:
        #     data = {'All': odh, 'Train': odh.isolate_data('reps', [0, 1, 2, 3], fast=True), 'Test': odh.isolate_data('reps', [4], fast=True)}

        # return data

    @abstractmethod
    def _prepare_data_helper(self, split = False):
        ...


class Hyser1DOF(Hyser):
    def __init__(self, dataset_folder = 'Hyser1DOF'):
        gestures = {1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little Finger'}
        definition = 'Hyser 1 DOF dataset. Includes within-DOF finger movements. Ground truth finger forces are recorded for use in finger force regression.'
        super().__init__(gestures=gestures, num_reps=3, description=definition, dataset_folder=dataset_folder)

    def _prepare_data_helper(self, split = False):
        def package_function(metadata_file, data_file):
            same_parent_directory = Path(metadata_file).parent.absolute() == Path(data_file).parent.absolute()

            finger_filter = RegexFilter(left_bound='_finger', right_bound='_sample', values=['1', '2', '3', '4', '5'], description='finger')
            same_finger = finger_filter.get_metadata(metadata_file) == finger_filter.get_metadata(data_file)
            return same_parent_directory and same_finger
            
        
        regex_filters = [
            RegexFilter(left_bound='subject', right_bound='_session', values=[str(idx) for idx in range(self.num_subjects + 1)], description='subjects'),   # +1 due to Python indexing
            RegexFilter(left_bound='_session', right_bound='/1dof_', values=['1', '2'], description='session'),
            RegexFilter(left_bound='1dof_', right_bound='_finger', values=['raw'], description='data_type'),
            RegexFilter(left_bound='_finger', right_bound='_sample', values=['1', '2', '3', '4', '5'], description='finger'),
            RegexFilter(left_bound='_sample', right_bound='.hea', values=['1', '2', '3'], description='rep')
        ]
        metadata_fetchers = [
            FilePackager(RegexFilter(left_bound='/1dof_', right_bound='_finger', values=['force'], description='data_type'),
                          package_function=package_function, load='p_signal')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers)
        data = odh
        if split:
            data = {}
        return data
