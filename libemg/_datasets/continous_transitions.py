from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler
import h5py
import numpy as np

class ContinuousTransitions(Dataset):
    def __init__(self, dataset_folder="ContinuousTransitions/"):
        Dataset.__init__(self, 
                        2000, 
                        6, 
                        'Delsys', 
                        43, 
                        {0: 'No Motion', 1: 'Wrist Flexion', 2: 'Wrist Extension', 3: 'Wrist Pronation', 4: 'Wrist Supination', 5: 'Hand Close', 6: 'Hand Open'}, 
                        '6 Training (Ramp), 42 Transitions (All combinations of Transitions) x 6 Reps',
                        "The testing set in this dataset has continuous transitions between classes which is a more realistic offline evaluation standard for myoelectric control.",
                        "https://ieeexplore.ieee.org/document/10254242")
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = True, subjects=None):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            print("Please download the dataset from: https://unbcloud-my.sharepoint.com/:f:/g/personal/ecampbe2_unb_ca/EjgjhM9ZHJxOglKoAf062ngBf4wFj2Mn2bORKY1-aMYGRw?e=WkZNwI") 
            return 
        
        # Training ODH
        odh_tr = OfflineDataHandler()
        odh_tr.subjects = []
        odh_tr.classes = []
        odh_tr.extra_attributes = ['subjects', 'classes']

        # Testing ODH
        odh_te = OfflineDataHandler()
        odh_te.subjects = []
        odh_te.classes = []
        odh_te.extra_attributes = ['subjects', 'classes']

        subject_list = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
        if subjects:
            subject_list = subject_list[subjects]

        for s_i, s in enumerate(subject_list):
            data = h5py.File(self.dataset_folder + '/P' + f"{s:02}" + '.hdf5', "r")
            cont_labels = data['continuous']['emg']['prompt'][()]
            cont_labels = np.hstack([np.ones((1000)) * cont_labels[0], cont_labels[0:len(cont_labels)-1000]]) # Rolling about 0.5s as per Shri's suggestion
            cont_emg = data['continuous']['emg']['signal'][()]
            cont_chg_idxs = np.insert(np.where(cont_labels[:-1] != cont_labels[1:])[0], 0, -1)
            cont_chg_idxs = np.insert(cont_chg_idxs, len(cont_chg_idxs), len(cont_emg))
            for i in range(0, len(cont_chg_idxs)-1):
                odh_te.data.append(cont_emg[cont_chg_idxs[i]+1:cont_chg_idxs[i+1]])
                odh_te.classes.append(np.expand_dims(cont_labels[cont_chg_idxs[i]+1:cont_chg_idxs[i+1]]-1, axis=1))
                odh_te.subjects.append(np.ones((len(odh_te.data[-1]), 1)) * s_i) 
            
            ramp_emg = data['ramp']['emg']['signal'][()]
            ramp_labels = data['ramp']['emg']['prompt'][()]
            r_chg_idxs = np.insert(np.where(ramp_labels[:-1] != ramp_labels[1:])[0], 0, -1)
            r_chg_idxs = np.insert(r_chg_idxs, len(r_chg_idxs), len(ramp_emg))
            for i in range(0, len(r_chg_idxs)-1):
                odh_tr.data.append(ramp_emg[r_chg_idxs[i]+1:r_chg_idxs[i+1]])
                odh_tr.classes.append(np.expand_dims(ramp_labels[r_chg_idxs[i]+1:r_chg_idxs[i+1]]-1, axis=1))
                odh_tr.subjects.append(np.ones((len(odh_tr.data[-1]), 1)) * s_i) 
            
        odh_all = odh_tr + odh_te
        data = odh_all
        if split:
            data = {'All': odh_all, 'Train': odh_tr, 'Test': odh_te}

        return data
