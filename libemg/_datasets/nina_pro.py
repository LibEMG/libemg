from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os
import scipy.io as sio
import zipfile
import numpy as np 

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

# class NinaproDB8(Ninapro):
#     def __init__(self, save_dir='.', dataset_name="NinaProDB8"):
#         Ninapro.__init__(self, save_dir, dataset_name)
#         self.class_list = ["Thumb Flexion/Extension", "Thumb Abduction/Adduction", "Index Finger Flexion/Extension", "Middle Finger Flexion/Extension", "Combined Ring and Little Fingers Flexion/Extension",
#          "Index Pointer", "Cylindrical Grip", "Lateral Grip", "Tripod Grip"]
#         self.exercise_step = [0,10,20]

#     def prepare_data(self, format=OfflineDataHandler, subjects_values = [str(i) for i in range(1,13)],
#                                                       reps_values = [str(i) for i in range(22)],
#                                                       classes_values = [str(i) for i in range(9)]):
        
#         if format == OfflineDataHandler:
#             regex_filters = [
#                 RegexFilter(left_bound = "/C", right_bound="R", values = classes_values, description='classes'),
#                 RegexFilter(left_bound = "R", right_bound=".csv", values = reps_values, description='reps'),
#                 RegexFilter(left_bound="DB8_s", right_bound="/",values=subjects_values, description='subjects')
#             ]
#             odh = OfflineDataHandler()
#             odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
#             return odh

class NinaproDB2(Ninapro):
    def __init__(self, dataset_folder="NinaProDB2/"):
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

    def prepare_data(self, split = False, subjects_values = None, reps_values = None, classes_values = None):
        if subjects_values is None:
            subjects_values = [str(i) for i in range(1,41)]
        if reps_values is None:
            reps_values = [str(i) for i in range(6)]
        if classes_values is None:
            classes_values = [str(i) for i in range(50)]

        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            print("Please download the NinaProDB2 dataset from: https://ninapro.hevs.ch/instructions/DB2.html") 
            return 
        self.convert_to_compatible()
        regex_filters = [
            RegexFilter(left_bound = "/C", right_bound="R", values = classes_values, description='classes'),
            RegexFilter(left_bound = "R", right_bound=".csv", values = reps_values, description='reps'),
            RegexFilter(left_bound="DB2_s", right_bound="/",values=subjects_values, description='subjects')
        ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",")
        data = odh
        if split:
            data = {'All': odh, 'Train': odh.isolate_data('reps', [0,1,2,3]), 'Test': odh.isolate_data('reps', [4,5])}

        return data