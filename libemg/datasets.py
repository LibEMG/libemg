import os
import numpy as np
import zipfile
import scipy.io as sio
from libemg.data_handler import _ColumnFetcher, _MetadataFetcher, OfflineDataHandler, RegexFilter
from libemg.utils import make_regex
from glob import glob
from os import walk
from pathlib import Path
from datetime import datetime
# this assumes you have git downloaded (not pygit, but the command line program git)

class Dataset:
    def __init__(self, save_dir='.', redownload=False):
        self.save_dir = save_dir
        self.redownload=redownload

    def download(self, url, dataset_name):
        clone_command = "git clone " + url + " " + dataset_name
        os.system(clone_command)
    
    def remove_dataset(self, dataset_folder):
        remove_command = "rm -rf " + dataset_folder
        os.system(remove_command)

    def check_exists(self, dataset_folder):
        return os.path.exists(dataset_folder)

    def prepare_data(self, format=OfflineDataHandler):
        pass


class _3DCDataset(Dataset):
    def __init__(self, save_dir='.', redownload=False, dataset_name="_3DCDataset"):
        Dataset.__init__(self, save_dir, redownload)
        self.url = "https://github.com/libemg/3DCDataset"
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)
        self.class_list = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
        elif (self.redownload):
            self.remove_dataset(self.dataset_folder)
            self.download(self.url, self.dataset_folder)



    def prepare_data(self, format=OfflineDataHandler, subjects_values = [str(i) for i in range(1,23)],
                                                      sets_values = ["train", "test"],
                                                      reps_values = ["0","1","2","3"],
                                                      classes_values = [str(i) for i in range(11)]):
        if format == OfflineDataHandler:
            regex_filters = [
                RegexFilter(left_bound = "/", right_bound="/EMG", values = sets_values, description='sets'),
                RegexFilter(left_bound = "_", right_bound=".txt", values = classes_values, description='classes'),
                RegexFilter(left_bound = "EMG_gesture_", right_bound="_", values = reps_values, description='reps'),
                RegexFilter(left_bound="Participant", right_bound="/",values=subjects_values, description='subjects')
            ]
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
            return odh

class Ninapro(Dataset):
    def __init__(self, save_dir='.', dataset_name="Ninapro"):
        # downloading the Ninapro dataset is not supported (no permission given from the authors)'
        # however, you can download it from http://ninapro.hevs.ch/DB8
        # the subject zip files should be placed at: <save_dir>/NinaproDB8/DB8_s#.zip
        Dataset.__init__(self, save_dir)
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name, "")
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
        exercise = int(mat_file.split('_')[3][1])
        exercise_offset = self.exercise_step[exercise-1] # 0 reps already included
        data = mat['emg']
        restimulus = mat['restimulus']
        rerepetition = mat['rerepetition']
        if data.shape[0] != restimulus.shape[0]: # this happens in some cases
            min_shape = min([data.shape[0], restimulus.shape[0]])
            data = data[:min_shape,:]
            restimulus = restimulus[:min_shape,]
            rerepetition = rerepetition[:min_shape,]
        # remove 0 repetition - collection buffer
        remove_mask = (rerepetition != 0).squeeze()
        data = data[remove_mask,:]
        restimulus = restimulus[remove_mask]
        rerepetition = rerepetition[remove_mask]
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
            # downsample to 1kHz from 2kHz using decimation
            data_for_file = data[tail:head,:]
            data_for_file = data_for_file[::2, :]
            # write to csv
            csv_file = mat_dir + 'C' + str(motion-1) + 'R' + str(rep-1 + exercise_offset) + '.csv'
            np.savetxt(csv_file, data_for_file, delimiter=',')
            tail = head
        os.remove(mat_file)

class NinaproDB8(Ninapro):
    def __init__(self, save_dir='.', dataset_name="NinaproDB8"):
        Ninapro.__init__(self, save_dir, dataset_name)
        self.class_list = ["Thumb Flexion/Extension", "Thumb Abduction/Adduction", "Index Finger Flexion/Extension", "Middle Finger Flexion/Extension", "Combined Ring and Little Fingers Flexion/Extension",
         "Index Pointer", "Cylindrical Grip", "Lateral Grip", "Tripod Grip"]
        self.exercise_step = [0,10,20]

    def prepare_data(self, format=OfflineDataHandler, subjects_values = [str(i) for i in range(1,13)],
                                                      reps_values = [str(i) for i in range(22)],
                                                      classes_values = [str(i) for i in range(9)]):
        
        if format == OfflineDataHandler:
            regex_filters = [
                RegexFilter(left_bound = "/C", right_bound="R", values = classes_values, description='classes'),
                RegexFilter(left_bound = "R", right_bound=".csv", values = reps_values, description='reps'),
                RegexFilter(left_bound="DB8_s", right_bound="/",values=subjects_values, description='subjects')
            ]
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
            return odh

class NinaproDB2(Ninapro):
    def __init__(self, save_dir='.', dataset_name="NinaproDB2"):
        Ninapro.__init__(self, save_dir, dataset_name)
        self.class_list = ["TODO"]
        self.exercise_step = [0,0,0]

    def prepare_data(self, format=OfflineDataHandler, subjects_values = [str(i) for i in range(1,41)],
                                                      reps_values = [str(i) for i in range(6)],
                                                      classes_values = [str(i) for i in range(50)]):
        
        if format == OfflineDataHandler:
            regex_filters = [
                RegexFilter(left_bound = "/C", right_bound="R", values = classes_values, description='classes'),
                RegexFilter(left_bound = "R", right_bound=".csv", values = reps_values, description='reps'),
                RegexFilter(left_bound="DB2_s", right_bound="/",values=subjects_values, description='subjects')
            ]
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
            return odh

# given a directory, return a list of files in that directory matching a format
# can be nested
# this is just a handly utility
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


class OneSubjectMyoDataset(Dataset):
    def __init__(self, save_dir='.', redownload=False, dataset_name="OneSubjectMyoDataset"):
        Dataset.__init__(self, save_dir, redownload)
        self.url = "https://github.com/libemg/OneSubjectMyoDataset"
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)

        if (not self.check_exists(self.dataset_folder)):
            self.download(self.url, self.dataset_folder)
        elif (self.redownload):
            self.remove_dataset(self.dataset_folder)
            self.download(self.url, self.dataset_folder)

    def prepare_data(self, format=OfflineDataHandler):
        if format == OfflineDataHandler:
            sets_values = ["1","2","3","4","5","6"]
            classes_values = ["0","1","2","3","4"]
            reps_values = ["0","1"]
            regex_filters = [
                RegexFilter(left_bound = "/trial_", right_bound="/", values = sets_values, description='sets'),
                RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes'),
                RegexFilter(left_bound = "R_", right_bound="_", values = reps_values, description='reps')
            ]
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
            return odh


class _SessionFetcher(_MetadataFetcher):
    def __init__(self):
        super().__init__('sessions')

    def __call__(self, filename, file_data, all_files):
        def split_filename(f):
            # Split date and name into separate variables
            date_idx = f.find('2018')
            date = datetime.strptime(Path(f[date_idx:]).stem, '%Y-%m-%d-%H-%M-%S-%f')
            description = f[:date_idx]
            return date, description

        data_file_date, data_file_description = split_filename(filename)

        # Grab the other file of a different date. Return the index of which session it is
        same_subject_files = [f for f in all_files if data_file_description in f]
        file_dates = [split_filename(subject_filename)[0] for subject_filename in same_subject_files]
        file_dates.sort()
        session_idx = file_dates.index(data_file_date)
        return session_idx * np.ones((file_data.shape[0], 1), dtype=int)


class _RepFetcher(_ColumnFetcher):
    def __call__(self, filename, file_data, all_files):
        column_data = super().__call__(filename, file_data, all_files)
        
        # Get rep transitions
        diff = np.diff(column_data, axis=0)
        rep_end_row_mask, rep_end_col_mask = np.nonzero((diff < 0) & (column_data[1:] == 0))
        unique_rep_end_row_mask = np.unique(rep_end_row_mask)  # remove duplicate start indices (for combined movements)
        # rest_end_row_mask = np.nonzero(np.diff(np.nonzero(column_data == 0)[0]) > 1)[0]
        # rest_end_row_mask = np.nonzero(np.diff(np.nonzero(np.all(column_data == 0, axis=1))[0]) > 1)[0]
        # unique_rep_end_row_mask = np.concatenate((unique_rep_end_row_mask, rest_end_row_mask))
        # unique_rep_end_row_mask = np.sort(unique_rep_end_row_mask)


        # Populate metadata array
        metadata = np.empty((column_data.shape[0], 1), dtype=np.int16)
        rep_counters = [0 for _ in range(5)]    # 5 different press types
        previous_rep_start = 0
        for idx, rep_start in enumerate(unique_rep_end_row_mask):
            movement_idx = 4 if np.sum(rep_end_row_mask == rep_start) > 1 else rep_end_col_mask[idx]    # if multiple columns are nonzero then it's a combined movement
            rep = rep_counters[movement_idx]
            metadata[previous_rep_start:rep_start] = rep
            rep_counters[movement_idx] += 1
            previous_rep_start = rep_start

        # Fill in final samples
        metadata[rep_start:] = rep

        return metadata


class PutEMGForceDataset(Dataset):
    def __init__(self, save_dir = '.', dataset_name = 'PutEMGForceDataset', data_filetype = None):
        """Dataset wrapper for putEMG-Force dataset. Used for regression of finger forces.

        Parameters
        ----------
        save_dir : str, default='.'
            Base data directory.
        dataset_name : str, default='PutEMGForceDataset'
            Name of dataset. Looks for dataset in filepath created by appending save_dir and dataset_name.
        data_filetype : list or None, default=None
            Type of data file to use. Accepted values are 'repeats_long', 'repeats_short', 'sequential', or any combination of those. If None is passed, all will be used.
        """
        # TODO: Implement downloading dataset using .sh or .py file
        super().__init__(save_dir)
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir, self.dataset_name)
        if data_filetype is None:
            data_filetype = ['repeats_short', 'repeats_long', 'sequential']
        elif not isinstance(data_filetype, list):
            data_filetype = [data_filetype]
        self.data_filetype = data_filetype

    def prepare_data(self, format=OfflineDataHandler, subjects = None, sessions = None, reps = None, labels = 'forces', label_dof_mask = None):
        if subjects is None:
            subjects = [str(idx).zfill(2) for idx in range(60)] 

        if labels == 'forces':
            column_mask = np.arange(25, 35)
        elif labels == 'trajectories':
            column_mask = np.arange(36, 40)
        else:
            raise ValueError(f"Expected either 'forces' or trajectories' for labels parameter, but received {labels}.")

        if label_dof_mask is not None:
            column_mask = column_mask[label_dof_mask]

        if format == OfflineDataHandler:
            regex_filters = [
                RegexFilter(left_bound='/emg_force-', right_bound='-', values=subjects, description='subjects'),
                RegexFilter(left_bound='-', right_bound='-', values=self.data_filetype, description='data_filetype'),
            ]
            metadata_fetchers = [
                _SessionFetcher(),
                _ColumnFetcher('labels', column_mask),
                _RepFetcher('reps', list(range(36, 40)))
            ]
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, metadata_fetchers=metadata_fetchers, delimiter=',', skiprows=1, data_column=list(range(1, 25)))
            if sessions is not None:
                odh = odh.isolate_data('sessions', sessions)
            if reps is not None:
                odh = odh.isolate_data('reps', reps)
            return odh
            

# class GRABMyo(Dataset):
#     def __init__(self, save_dir='.', redownload=False, subjects=list(range(1,44)), sessions=list(range(1,4)), dataset_name="GRABMyo"):
#         Dataset.__init__(self, save_dir, redownload)
#         self.url = "https://physionet.org/files/grabmyo/1.0.2/"
#         self.dataset_name = dataset_name
#         self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)
#         self.subjects = subjects
#         self.sessions = sessions

#         if (not self.check_exists(self.dataset_folder)):
#             self.download_data()
#         elif (self.redownload):
#             self.remove_dataset(self.dataset_folder)
#             self.download_data()
#         else:
#             print("Data Already Downloaded.")
    
#     def download_data(self):
#         curl_command = "curl --create-dirs" + " -O --output-dir " + str(self.dataset_folder) + "/ "
#         # Download files
#         print("Starting download...")
#         files = ['readme.txt', 'subject-info.csv', 'MotionSequence.txt']
#         for f in files:
#             os.system(curl_command + self.url + f)
#         for session in self.sessions:
#             curl_command = "curl --create-dirs" + " -O --output-dir " + str(self.dataset_folder) + "/" + "Session" + str(session) + "/ "
#             for p in self.subjects:
#                 for t in range(1,8):
#                     for g in range(1,18):
#                         endpoint = self.url + "Session" + str(session) + "/session" + str(session) + "_participant" + str(p) + "/session" + str(session) + "_participant" + str(p) + "_gesture" + str(g) + "_trial" + str(t)
#                         os.system(curl_command + endpoint + '.hea')
#                         os.system(curl_command + endpoint + '.dat')
#         print("Download complete.")

#     def prepare_data(self, format=OfflineDataHandler, subjects=[str(i) for i in range(1,44)], sessions=["1","2","3"]):
#         if format == OfflineDataHandler:
#             sets_regex = make_regex(left_bound = "session", right_bound="_", values = sessions)
#             classes_values = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"]
#             classes_regex = make_regex(left_bound = "_gesture", right_bound="_", values = classes_values)
#             reps_values = ["1","2","3","4","5","6","7"]
#             reps_regex = make_regex(left_bound = "trial", right_bound=".hea", values = reps_values)
#             subjects_regex = make_regex(left_bound="participant", right_bound="_",values=subjects)
#             dic = {
#                 "sessions": sessions,
#                 "sessions_regex": sets_regex,
#                 "reps": reps_values,
#                 "reps_regex": reps_regex,
#                 "classes": classes_values,
#                 "classes_regex": classes_regex,
#                 "subjects": subjects,
#                 "subjects_regex": subjects_regex
#             }
#             odh = OfflineDataHandler()
#             odh.get_data(folder_location=self.dataset_folder, filename_dic=dic, delimiter=",")
#             return odh

#     def print_info(self):
#         print('Reference: https://www.physionet.org/content/grabmyo/1.0.2/') 
#         print('Name: ' + self.dataset_name)
#         print('Gestures: 17')
#         print('Trials: 7')
#         print('Time Per Rep: 5s')
#         print('Subjects: 43')
#         print("Forearm EMG (16): Columns 0-15\nWrist EMG (12): 18-23 and 26-31\nUnused (4): 16,23,24,31")


# class NinaDB1(Dataset):
#     def __init__(self, dataset_dir, subjects):
#         Dataset.__init__(self, dataset_dir)
#         self.dataset_folder = dataset_dir
#         self.subjects = subjects

#         if (not self.check_exists(self.dataset_folder)):
#             print("The dataset does not currently exist... Please download it from: http://ninaweb.hevs.ch/data1") 
#             exit(1)      
#         else:
#             filenames = next(walk(self.dataset_folder), (None, None, []))[2]
#             if not any("csv" in f for f in filenames):
#                 self.setup(filenames)
#                 print("Extracted and set up repo.")
#             self.prepare_data()
    
#     def setup(self, filenames):
#         for f in filenames:
#             if "zip" in f:
#                 file_path = os.path.join(self.dataset_folder, f)
#                 with zipfile.ZipFile(file_path, 'r') as zip_ref:
#                     zip_ref.extractall(self.dataset_folder)
#         self.convert_data()

#     def convert_data(self):
#         mat_files = [y for x in os.walk(self.dataset_folder) for y in glob(os.path.join(x[0], '*.mat'))]
#         for f in mat_files:
#             mat_dict = sio.loadmat(f)
#             output_ = np.concatenate((mat_dict['emg'], mat_dict['restimulus'], mat_dict['rerepetition']), axis=1)
#             mask_ids = output_[:,11] != 0
#             output_ = output_[mask_ids,:]
#             np.savetxt(f[:-4]+'.csv', output_,delimiter=',')
    
#     def cleanup_data(self):
#         mat_files = [y for x in os.walk(self.dataset_folder) for y in glob(os.path.join(x[0], '*.mat'))]
#         zip_files = [y for x in os.walk(self.dataset_folder) for y in glob(os.path.join(x[0], '*.zip'))]
#         files = mat_files + zip_files
#         for f in files:
#             os.remove(f)
    
#     def prepare_data(self, format=OfflineDataHandler):
#         if format == OfflineDataHandler:
#             classes_values = list(range(1,24))
#             classes_column = [10]
#             classset_values = [str(i) for i in list(range(1,4))]
#             classset_regex  = make_regex(left_bound="_E", right_bound=".csv", values=classset_values)
#             reps_values = list(range(1,11))

#             reps_column = [11]
#             subjects_values = [str(s) for s in self.subjects]
#             subjects_regex = make_regex(left_bound="S", right_bound="_A", values=subjects_values)
#             data_column = list(range(0,10))
#             dic = {
#                 "reps": reps_values,
#                 "reps_column": reps_column,
#                 "classes": classes_values,
#                 "classes_column": classes_column,
#                 "subjects": subjects_values,
#                 "subjects_regex": subjects_regex,
#                 "classset": classset_values,
#                 "classset_regex": classset_regex,
#                 "data_column": data_column
#             }
#             odh = OfflineDataHandler()
#             odh.get_data(folder_location=self.dataset_folder, filename_dic=dic, delimiter=",")
#             return odh
