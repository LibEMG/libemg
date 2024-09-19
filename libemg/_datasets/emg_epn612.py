from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import pickle
import random
import numpy as np

class EMGEPN612(Dataset):
    def __init__(self, dataset_file='EMGEPN612.pkl'):
        Dataset.__init__(self, 
                         200, 
                         8, 
                         'Myo Armband', 
                         612, 
                         {0: 'Close', 1: 'Open', 2: 'Rest', 3: 'Flexion', 4: 'Extension'}, 
                         '50 Reps x 306 Users (Train), 25 Reps x 306 Users (Test)',
                         "A large 612 user dataset for developing cross user models.", 
                         'https://doi.org/10.5281/zenodo.4421500')
        self.url = "https://github.com/libemg/OneSubjectMyoDataset"
        self.dataset_name = dataset_file

    def prepare_data(self, split = False):
        random.seed(1)
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_name)):
            print("Please download the pickled dataset from: https://unbcloud-my.sharepoint.com/:u:/g/personal/ecampbe2_unb_ca/EWf3sEvRxg9HuAmGoBG2vYkBDXh4xNst3FAXV0lNoodrAA?e=t6HPaR") 
            return 
        
        file = open(self.dataset_name, 'rb')
        data = pickle.load(file)

        emg = data[0]
        labels = data[2]

        odh_tr = OfflineDataHandler()
        odh_tr.subjects = []
        odh_tr.classes = []
        odh_tr.reps = []
        tr_reps = [0,0,0,0,0,0]
        odh_tr.extra_attributes = ['subjects', 'classes', 'reps']
        for i, e in enumerate(emg['training']):
            odh_tr.data.append(e)
            odh_tr.classes.append(np.ones((len(e), 1)) * labels['training'][i])
            odh_tr.subjects.append(np.ones((len(e), 1)) * i//300)
            odh_tr.reps.append(np.ones((len(e), 1)) * tr_reps[labels['training'][i]])
            tr_reps[labels['training'][i]] += 1
            if i % 300 == 0:
                tr_reps = [0,0,0,0,0,0]
        odh_te = OfflineDataHandler()
        odh_te.subjects = []
        odh_te.classes = []
        odh_te.reps = []
        te_reps = [0,0,0,0,0,0]
        odh_te.extra_attributes = ['subjects', 'classes', 'reps']
        for i, e in enumerate(emg['testing']):
            odh_te.data.append(e)
            odh_te.classes.append(np.ones((len(e), 1)) * labels['testing'][i])
            odh_te.subjects.append(np.ones((len(e), 1)) * (i//150 + 306))
            odh_te.reps.append(np.ones((len(e), 1)) * te_reps[labels['training'][i]])
            te_reps[labels['training'][i]] += 1
            if i % 150 == 0:
                te_reps = [0,0,0,0,0,0]

        odh_all = odh_tr + odh_te # Has no cropping 
        odh_tr = self._update_odh(odh_tr)
        odh_te = self._update_odh(odh_te)

        data = odh_all
        if split:
            data = {'All': odh_all, 'Train': odh_tr, 'Test': odh_te}

        return data
    
    def _update_odh(self, odh):
        active = [c[0][0] != 0 for c in odh.classes]
        lens = [len(e) for e in np.array(odh.data, dtype='object')[active]]
        for i_e, e in enumerate(odh.data):
            if odh.classes[i_e][0][0] == 0: 
                # It is no motion and we need to crop it (make datset even)
                odh.data[i_e] = e[100:100+random.randint(min(lens), max(lens))]
                odh.subjects[i_e] = odh.subjects[i_e][100:100+random.randint(min(lens), max(lens))]
                odh.classes[i_e] = odh.classes[i_e][100:100+random.randint(min(lens), max(lens))]
                odh.reps[i_e] = odh.reps[i_e][100:100+random.randint(min(lens), max(lens))]
        return odh 

        