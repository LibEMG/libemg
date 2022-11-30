import os
import numpy as np
import zipfile
import scipy.io as sio
from libemg.data_handler import OfflineDataHandler
from libemg.utils import make_regex
from glob import glob
# this assumes you have git downloaded (not pygit, but the command line program git)

class Dataset:
    def __init__(self, save_dir='.', redownload=True):
        self.save_dir = save_dir
        self.redownload=redownload

    def github_download(self, url, dataset_name):
        clone_command = "git clone " + url + " " + dataset_name
        os.system(clone_command)
    
    def remove_dataset(self, dataset_folder):
        remove_command = "rm -rf " + dataset_folder
        os.system(remove_command)

    def check_exists(self, dataset_folder):
        if os.path.exists(dataset_folder):
            return True
        else:
            return False

    def prepare_data(self, format=OfflineDataHandler):
        pass

class _3DCDataset(Dataset):
    def __init__(self, save_dir='.', redownload=True, dataset_name="_3DCDataset"):
        Dataset.__init__(self, save_dir, redownload)
        self.url = "https://github.com/ECEEvanCampbell/3DCDataset"
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)

        if (not self.check_exists(self.dataset_folder)):
            self.github_download(self.url, self.dataset_folder)
        elif (self.redownload):
            self.remove_dataset(self.dataset_folder)
            self.github_download(self.url, self.dataset_folder)


    def prepare_data(self, format=OfflineDataHandler):
        if format == OfflineDataHandler:
            sets_values = ["train", "test"]
            sets_regex = make_regex(left_bound = "/", right_bound="/EMG", values = sets_values)
            classes_values = ["0","1","2","3","4","5","6","7","8","9","10"]
            classes_regex = make_regex(left_bound = "_", right_bound=".txt", values = classes_values)
            reps_values = ["0","1","2","3"]
            reps_regex = make_regex(left_bound = "EMG_gesture_", right_bound="_", values = reps_values)
            subjects_values = ["1","2","3","4","5","6","7","8","9","10","11", "12","13","14","15","16","17","18","19","20","21","22"]
            subjects_regex = make_regex(left_bound="Participant", right_bound="/",values=subjects_values)
            dic = {
                "sets": sets_values,
                "sets_regex": sets_regex,
                "reps": reps_values,
                "reps_regex": reps_regex,
                "classes": classes_values,
                "classes_regex": classes_regex,
                "subjects": subjects_values,
                "subjects_regex": subjects_regex
            }
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, filename_dic=dic, delimiter=",")
            return odh
        
class OneSubjectMyoDataset(Dataset):
    def __init__(self, save_dir='.', redownload=True, dataset_name="OneSubjectMyoDataset"):
        Dataset.__init__(self, save_dir, redownload)
        self.url = "https://github.com/eeddy/OneSubjectMyoDataset"
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)

        if (not self.check_exists(self.dataset_folder)):
            self.github_download(self.url, self.dataset_folder)
        elif (self.redownload):
            self.remove_dataset(self.dataset_folder)
            self.github_download(self.url, self.dataset_folder)

    def prepare_data(self, format=OfflineDataHandler):
        if format == OfflineDataHandler:
            sets_values = ["training", "testing"]
            sets_regex = make_regex(left_bound = "/", right_bound="/", values = sets_values)
            classes_values = ["0","1","2","3","4"]
            classes_regex = make_regex(left_bound = "C_", right_bound=".csv", values = classes_values)
            reps_values = ["0","1","2","3","4"]
            reps_regex = make_regex(left_bound = "R_", right_bound="_", values = reps_values)
            dic = {
                "sets": sets_values,
                "sets_regex": sets_regex,
                "reps": reps_values,
                "reps_regex": reps_regex,
                "classes": classes_values,
                "classes_regex": classes_regex,
            }
            odh = OfflineDataHandler()
            odh.get_data(folder_location=self.dataset_folder, filename_dic=dic, delimiter=",")
            return odh

# Commenting this out for now - may revisit in the future.
# class NinaDB1(Dataset):
#     def __init__(self, save_dir='.', redownload=True, subjects=list(range(1,28)), dataset_name="NinaDB1", remove_rest=True):
#         Dataset.__init__(self, save_dir, redownload)
#         self.dataset_folder = os.path.join(self.save_dir, dataset_name)
#         self.subjects = subjects
        
#         self.file_dic = {
#             1: '212',
#             2: '239',
#             3: '240',
#             4: '242',
#             5: '243',
#             6: '244',
#             7: '245',
#             8: '246',
#             9: '247',
#             10: '248',
#             11: '249',
#             12: '250',
#             13: '251',
#             14: '252',
#             15: '253',
#             16: '254',
#             17: '255',
#             18: '256',
#             19: '257',
#             20: '259',
#             21: '258',
#             22: '260',
#             23: '261',
#             24: '262',
#             25: '263',
#             26: '264',
#             27: '265',
#         }

#         if (redownload):
#             self.remove_dataset(self.dataset_folder)
#         if (not self.check_exists(self.dataset_folder)):
#             os.mkdir(self.dataset_folder)
#             self.download_data()
#             self.convert_data(remove_rest)
#             self.cleanup_data()
        
    
#     def download_data(self):
#         for sub in self.subjects:
#             print("Downloading Subject " + str(sub) + "...")
#             path = os.path.join(self.dataset_folder, str(sub) + '.zip')
#             curl_command = "curl --header \"Cookie: SESSce2427b5b9e40907ca31bbb47df5316b=lhVhjbcKiWu-SF0GqZNnKxo8WIdykhfYgHOAdrmzbNc\" http://ninaweb.hevs.ch/download/file/fid/" + self.file_dic[sub] + " --output " + str(path)
#             os.system(curl_command)
#             with zipfile.ZipFile(path, 'r') as zip_ref:
#                 zip_ref.extractall(self.dataset_folder)
            
#     def convert_data(self, remove_rest=True):
#         mat_files = [y for x in os.walk(self.dataset_folder) for y in glob(os.path.join(x[0], '*.mat'))]
#         for f in mat_files:
#             mat_dict = sio.loadmat(f)
#             output_ = np.concatenate((mat_dict['emg'], mat_dict['restimulus'], mat_dict['rerepetition']), axis=1)
#             if remove_rest:
#                 # repetition=0 and class=0 refers to rest - let's remove that to minimize data size
#                 mask_ids = output_[:,11] != 0
#                 output_ = output_[mask_ids,:]
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
#             classset_regex  = make_regex(left_bound="_E", right_bound=".csv",values=classset_values)
#             reps_values = list(range(1,11))

#             reps_column = [11]
#             subjects_values = [str(i) for i in list(range(1,28))]
#             subjects_regex = make_regex(left_bound="NinaDB1/S", right_bound="_A",values=subjects_values)
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