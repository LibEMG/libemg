import os
from libemg.data_handler import OfflineDataHandler
from onedrivedownloader import download as onedrive_download
# this assumes you have git downloaded (not pygit, but the command line program git)

class Dataset:
    def __init__(self, sampling, num_channels, recording_device, num_subjects, gestures, num_reps, description, citation):
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
    
    def download_via_onedrive(self, url, dataset_name, unzip=True, clean=True):
        onedrive_download(url=url,
                          filename = dataset_name,
                          unzip=unzip,
                          clean=clean)
    
    def remove_dataset(self, dataset_folder):
        remove_command = "rm -rf " + dataset_folder
        os.system(remove_command)

    def check_exists(self, dataset_folder):
        return os.path.exists(dataset_folder)

    def prepare_data(self, split = True):
        pass

    def get_info(self):
        print(str(self.description) + '\n' + 'Sampling Rate: ' + str(self.sampling) + '\nNumber of Channels: ' + str(self.num_channels) + 
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