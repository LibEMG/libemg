import os
from unb_emg_toolbox.data_handler import OfflineDataHandler
from unb_emg_toolbox.utils import make_regex
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

    def prepare_dataset(self, format=OfflineDataHandler):
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
        