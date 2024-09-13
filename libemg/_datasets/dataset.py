import os
from libemg.data_handler import OfflineDataHandler
# this assumes you have git downloaded (not pygit, but the command line program git)

class Dataset:
    def __init__(self, sampling, num_channels, recording_device, num_subjects, gestures, num_reps, description, citation, save_dir='.', redownload=False, ):
        self.save_dir = save_dir
        self.redownload=redownload

        # Every class should have this 
        self.sampling=sampling
        self.num_channels=num_channels 
        self.recording_device=recording_device
        self.num_subjects=num_subjects
        self.gestures=gestures
        self.num_reps=num_reps
        self.description=description
        self.citation=citation

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

    def get_info(self):
        print(self.description + '\n' + 'Sampling Rate: ' + str(self.sampling) + '\nNumber of Channels: ' + str(self.num_channels) + 
              '\nDevice: ' + self.recording_device + '\nGestures: ' + str(self.gestures) + '\nNumber of Reps: ' + str(self.num_reps) + '\nNumber of Subjects: ' + str(self.num_subjects) +
              '\nCitation: ' + str(self.citation))

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
