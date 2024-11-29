from pathlib import Path

from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter, ColumnFetcher
import os
import scipy.io as sio
import zipfile
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

def find_all_files_of_type_recursively(dir, terminator):
    files = os.listdir(dir)
    file_list = []
    for file in files:
        if file.endswith(terminator):
            file_list.append(dir+file)
        else:
            if os.path.isdir(dir+file):
                file_list += find_all_files_of_type_recursively(dir+file+'/',terminator)
    return file_list

class Ninapro(Dataset):
    def __init__(self, 
                 sampling, num_channels, recording_device, num_subjects, gestures, num_reps, description, citation,
                 dataset_folder="Ninapro"):
        # downloading the Ninapro dataset is not supported (no permission given from the authors)'
        # however, you can download it from http://ninapro.hevs.ch/DB8
        # the subject zip files should be placed at: <save_dir>/NinaproDB8/DB8_s#.zip
        Dataset.__init__(self, sampling, num_channels, recording_device, num_subjects, gestures, num_reps, description, citation)
        self.dataset_folder = dataset_folder
        self.exercise_step = []
    
    def convert_to_compatible(self):
        # get the zip files (original format they're downloaded in)
        zip_files = find_all_files_of_type_recursively(self.dataset_folder,".zip")
        # unzip the files -- if any are there (successive runs skip this)
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(zip_file[:-4]+'/')
            os.remove(zip_file)
        # get the mat files (the files we want to convert to csv)
        mat_files = find_all_files_of_type_recursively(self.dataset_folder,".mat")
        for mat_file in mat_files:
            self.convert_to_csv(mat_file)
    
    def convert_to_csv(self, mat_file):
        # read the mat file
        mat_file = mat_file.replace("\\", "/")
        mat_dir = mat_file.split('/')
        mat_dir = os.path.join(*mat_dir[:-1],"")
        mat = sio.loadmat(mat_file)
        # get the data
        exercise = int(mat_file.split('_')[-1][1])
        exercise_offset = self.exercise_step[exercise-1] # 0 reps already included
        data = mat['emg']
        restimulus = mat['restimulus']
        rerepetition = mat['rerepetition']
        try:
            cyberglove_data = mat['glove']
            cyberglove_directory = 'cyberglove'
        except KeyError:
            # No cyberglove data
            cyberglove_data = None
            cyberglove_directory = ''
        if data.shape[0] != restimulus.shape[0]: # this happens in some cases
            min_shape = min([data.shape[0], restimulus.shape[0]])
            data = data[:min_shape,:]
            restimulus = restimulus[:min_shape,]
            rerepetition = rerepetition[:min_shape,]
            if cyberglove_data is not None:
                cyberglove_data = cyberglove_data[:min_shape,]
        # remove 0 repetition - collection buffer
        remove_mask = (rerepetition != 0).squeeze()
        data = data[remove_mask,:]
        restimulus = restimulus[remove_mask]
        rerepetition = rerepetition[remove_mask]
        if cyberglove_data is not None:
            cyberglove_data = cyberglove_data[remove_mask, :]
        # important little not here: 
        # the "rest" really is only the rest between motions, not a dedicated rest class.
        # there will be many more rest repetitions (as it is between every class)
        # so usually we really care about classifying rest as its important (most of the time we do nothing)
        # but for this dataset it doesn't make sense to include (and not its just an offline showcase of the library)
        # I encourage you to plot the restimulus to see what I mean. -> plt.plot(restimulus)
        # so we remove the rest class too
        remove_mask = (restimulus != 0).squeeze()
        data = data[remove_mask,:]
        restimulus = restimulus[remove_mask]
        rerepetition = rerepetition[remove_mask]
        if cyberglove_data is not None:
            cyberglove_data = cyberglove_data[remove_mask, :]
        tail = 0
        while tail < data.shape[0]-1:
            rep = rerepetition[tail][0] # remove the 1 offset (0 was the collection buffer)
            motion = restimulus[tail][0] # remove the 1 offset (0 was between motions "rest")
            # find head
            head = np.where(rerepetition[tail:] != rep)[0]
            if head.shape == (0,): # last segment of data
                head = data.shape[0] -1
            else:
                head = head[0] + tail
            if cyberglove_data is not None:
                # Combine cyberglove and EMG data
                data_for_file = np.concatenate((data[tail:head, :], cyberglove_data[tail:head, :]), axis=1)
            else:
                data_for_file = data[tail:head,:]

            # downsample to 1kHz from 2kHz using decimation
            data_for_file = data_for_file[::2, :]
            # write to csv
            csv_file = Path(mat_dir, cyberglove_directory, f"C{motion - 1}R{rep - 1 + exercise_offset}.csv")
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(csv_file, data_for_file, delimiter=',')
            tail = head
        os.remove(mat_file)


class NinaproDB2(Ninapro):
    def __init__(self, dataset_folder="NinaProDB2/", use_cyberglove: bool = False):
        Ninapro.__init__(self, 
                         2000, 
                         12, 
                         'Delsys', 
                         40, 
                         {0: 'See Exercises B and C from: https://ninapro.hevs.ch/instructions/DB2.html'},
                         '4 Train, 2 Test',
                         "NinaProb DB2.", 
                         'https://ninapro.hevs.ch/',
                         dataset_folder = dataset_folder)
        self.exercise_step = [0,0,0]
        self.num_cyberglove_dofs = 22
        self.use_cyberglove = use_cyberglove    # needed b/c some files have EMG but no cyberglove

    def prepare_data(self, split = True, subjects = None):
        subject_list = np.array(list(range(1,41)))
        if subjects:
            subject_list = subject_list[subjects]
        subjects_values = [str(s) for s in subject_list]
        reps_values = [str(i) for i in range(6)]
        classes_values = [str(i) for i in range(50)]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            raise FileNotFoundError("Please download the NinaProDB2 dataset from: https://ninapro.hevs.ch/instructions/DB2.html") 
        self.convert_to_compatible()
        regex_filters = [
            RegexFilter(left_bound = "/C", right_bound="R", values = classes_values, description='classes'),
            RegexFilter(left_bound="R", right_bound=".csv", values=reps_values, description='reps'),
            RegexFilter(left_bound="DB2_s", right_bound="/",values=subjects_values, description='subjects')
        ]

        if self.use_cyberglove:
            # Only want cyberglove files
            regex_filters.append(RegexFilter(left_bound="/", right_bound="/C", values=['cyberglove'], description=''))
            metadata_fetchers = [
                ColumnFetcher('cyberglove', column_mask=[idx for idx in range(self.num_channels, self.num_channels + self.num_cyberglove_dofs)])
            ]
        else:
            metadata_fetchers = None

        emg_column_mask = [idx for idx in range(self.num_channels)] # first columns should be EMG
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",", data_column=emg_column_mask)
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data('reps', [0,1,2,3], fast=True), 'Test': odh.isolate_data('reps', [4,5], fast=True)}

        return data

class NinaproDB8(Ninapro):
    def __init__(self, dataset_folder="NinaProDB8/", map_to_finger_dofs = True, normalize_labels = True):
        # NOTE: This expects each subject's data to be in its own zip file, so the data files for one subject end up in a single directory once we unzip them (e.g., DB8_s1)
        gestures = {
            0: "rest",
            1: "thumb flexion/extension",
            2: "thumb abduction/adduction",
            3: "index finger flexion/extension",
            4: "middle finger flexion/extension",
            5: "combined ring and little fingers flexion/extension",
            6: "index pointer",
            7: "cylindrical grip",
            8: "lateral grip",
            9: "tripod grip"
        }

        super().__init__(
            sampling=1111,
            num_channels=16,
            recording_device='Delsys Trigno',
            num_subjects=12,
            gestures=gestures,
            num_reps=22,
            description='Ninapro DB8 - designed for regression of finger kinematics. Ground truth labels are provided via cyberglove data.',
            citation='https://ninapro.hevs.ch/',
            dataset_folder=dataset_folder
        )
        self.exercise_step = [0,10,20]
        self.num_cyberglove_dofs = 18
        self.map_to_finger_dofs = map_to_finger_dofs
        self.normalize_labels = normalize_labels

    def _remap_labels(self, odh):
        # Linear mapping matrix pulled from original paper: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00891/full
        finger_map_matrix = np.array([
            [0.639, 0, 0, 0, 0],
            [0.383, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [-0.639, 0, 0, 0, 0],
            [0, 0, 0.4, 0, 0],
            [0, 0, 0.6, 0, 0],
            [0, 0, 0, 0.4, 0],
            [0, 0, 0, 0.6, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.1667],
            [0, 0, 0, 0, 0.3333],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.1667],
            [0, 0, 0, 0, 0.3333],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-0.19, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])

        remapped_labels = []
        for labels in odh.labels:
            finger_labels = np.copy(labels)
            if self.map_to_finger_dofs:
                finger_labels = labels @ finger_map_matrix
            if self.normalize_labels:
                finger_labels = MinMaxScaler().fit_transform(finger_labels)
            remapped_labels.append(finger_labels)
        odh.labels = remapped_labels
        return odh

    def prepare_data(self, split = True, subjects = None):
        subjects_values = np.array([str(i) for i in range(1,self.num_subjects + 1)])
        if subjects:
            subjects_values = subjects_values[subjects]
        reps_values = [str(i) for i in range(self.num_reps)]
        classes_values = [str(i) for i in range(9)]

        self.convert_to_compatible()

        regex_filters = [
            RegexFilter(left_bound = "/C", right_bound="R", values = classes_values, description='classes'),
            RegexFilter(left_bound = "R", right_bound=".csv", values = reps_values, description='reps'),
            RegexFilter(left_bound="DB8_s", right_bound="/",values=list(subjects_values), description='subjects')
        ]
        metadata_fetchers = [
            ColumnFetcher('labels', column_mask=[idx for idx in range(self.num_channels, self.num_channels + self.num_cyberglove_dofs)])
        ]
        emg_column_mask = [idx for idx in range(self.num_channels)] # first columns should be EMG
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",", data_column=emg_column_mask)
        odh = self._remap_labels(odh)
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data('reps', [0, 1, 2, 3], fast=True), 'Test': odh.isolate_data('reps', [4, 5], fast=True)}
        return data
