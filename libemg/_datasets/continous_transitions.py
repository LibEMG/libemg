from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os
import pandas as pd
import h5py
import numpy as np

class ContinuousTransitions(Dataset):
    def __init__(self, save_dir='.', redownload=False, dataset_name="ContinuousTransitions"):
        Dataset.__init__(self, 
                        2000, 
                        6, 
                        'Delsys', 
                        2, 
                        [], 
                        '',
                        "",
                        "https://doi.org/10.57922/mec.2503",
                        save_dir, redownload)
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)

    def prepare_data(self):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            print("Please download the dataset from: ") #TODO: Update
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

        for s in [2,3]:
            data = h5py.File('ContinuousTransitions/P' + f"{s:02}" + '.hdf5', "r")
            cont_labels = data['continuous']['emg']['prompt'][()]
            cont_labels = np.hstack([np.ones((1000)) * cont_labels[0], cont_labels[0:len(cont_labels)-1000]]) # Rolling about 0.5s as per Shri's suggestion
            cont_emg = data['continuous']['emg']['signal'][()]
            cont_chg_idxs = np.insert(np.where(cont_labels[:-1] != cont_labels[1:])[0], 0, -1)
            cont_chg_idxs = np.insert(cont_chg_idxs, len(cont_chg_idxs), len(cont_emg))
            for i in range(0, len(cont_chg_idxs)-1):
                odh_te.data.append(cont_emg[cont_chg_idxs[i]+1:cont_chg_idxs[i+1]])
                odh_te.classes.append(np.expand_dims(cont_labels[cont_chg_idxs[i]+1:cont_chg_idxs[i+1]]-1, axis=1))
                odh_te.subjects.append(np.ones((len(odh_te.data[-1]), 1)) * s-2)  #TODO: Update
            
            ramp_emg = data['ramp']['emg']['signal'][()]
            ramp_labels = data['ramp']['emg']['prompt'][()]
            r_chg_idxs = np.insert(np.where(ramp_labels[:-1] != ramp_labels[1:])[0], 0, -1)
            r_chg_idxs = np.insert(r_chg_idxs, len(r_chg_idxs), len(ramp_emg))
            for i in range(0, len(r_chg_idxs)-1):
                odh_tr.data.append(ramp_emg[r_chg_idxs[i]+1:r_chg_idxs[i+1]])
                odh_tr.classes.append(np.expand_dims(ramp_labels[r_chg_idxs[i]+1:r_chg_idxs[i+1]]-1, axis=1))
                odh_tr.subjects.append(np.ones((len(odh_tr.data[-1]), 1)) * s-2) 
            
        return {'All': odh_tr+odh_te, 'Train': odh_tr, 'Test': odh_te}