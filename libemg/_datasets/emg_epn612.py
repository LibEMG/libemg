from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler
import pickle
import numpy as np
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import *

class EMGEPN612(Dataset):
    def __init__(self, dataset_file='EMGEPN612.pkl', cross_user=True):
        split = '50 Reps x 306 Users (Train), 25 Reps x 306 Users (Test) --> Cross User Split'
        if not cross_user:
            split = '20 Reps (Train), 5 Reps (Test) from the 612 Test Users --> User Dependent Split'

        Dataset.__init__(self, 
                         200, 
                         8, 
                         'Myo Armband', 
                         612, 
                         {0: 'No Movement', 1: 'Hand Close', 2: 'Flexion', 3: 'Extension', 4: 'Hand Open', 5: 'Pinch'}, 
                         split,
                         "A large 612 user dataset for developing cross user models.", 
                         'https://doi.org/10.5281/zenodo.4421500')
        self.url = "https://unbcloud-my.sharepoint.com/:u:/g/personal/ecampbe2_unb_ca/EWf3sEvRxg9HuAmGoBG2vYkBLyFv6UrPYGwAISPDW9dBXw?e=vjCA14"
        self.dataset_name = dataset_file

    def get_odh(self, subjects=None, feature_list = None, window_size = None, window_inc = None, feature_dic = None):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_name)):
            self.download_via_onedrive(self.url, self.dataset_name, unzip=False, clean=False)

        if feature_list or window_size or window_inc:
            assert feature_list
            assert window_size
            assert window_inc
            fe = FeatureExtractor()

        subject_list = np.array(list(range(0,612)))
        if subjects:
            subject_list = np.array(subjects)

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
            if i // 300 not in subject_list:
                continue
            if feature_list:
                odh_tr.data.append(fe.extract_features(feature_list, get_windows(e, window_size, window_inc), feature_dic=feature_dic, array=True))                
                odh_tr.classes.append(np.ones((len(odh_tr.data[-1]), 1)) * labels['training'][i])
                odh_tr.subjects.append(np.ones((len(odh_tr.data[-1]), 1)) * i//300)
                odh_tr.reps.append(np.ones((len(odh_tr.data[-1]), 1)) * tr_reps[labels['training'][i]])
            else:
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
            if (i // 150 + 306) not in subject_list:
                continue
            if feature_list:
                odh_te.data.append(fe.extract_features(feature_list, get_windows(e, window_size, window_inc), feature_dic=feature_dic, array=True))
                odh_te.classes.append(np.ones((len(odh_te.data[-1]), 1)) * labels['testing'][i])                
                odh_te.subjects.append(np.ones((len(odh_te.data[-1]), 1)) * (i//150 + 306))
                odh_te.reps.append(np.ones((len(odh_te.data[-1]), 1)) * te_reps[labels['testing'][i]])
            else:
                odh_te.data.append(e)
                odh_te.classes.append(np.ones((len(e), 1)) * labels['testing'][i])
                odh_te.subjects.append(np.ones((len(e), 1)) * (i//150 + 306))
                odh_te.reps.append(np.ones((len(e), 1)) * te_reps[labels['testing'][i]])
            te_reps[labels['testing'][i]] += 1
            if i % 150 == 0:
                te_reps = [0,0,0,0,0,0]

        return odh_tr + odh_te
    
class EMGEPN_UserDependent(EMGEPN612):
    def __init__(self, dataset_file='EMGEPN612.pkl'):
        EMGEPN612.__init__(self, dataset_file=dataset_file, cross_user=False)
    
    def prepare_data(self, split = True, subjects = None):
        odh = self.get_odh(subjects)
        odh_tr = odh.isolate_data('reps', list(range(0,20)))
        odh_te = odh.isolate_data('reps', list(range(20,25)))

        if split:
            data = {'All': odh, 'Train': odh_tr, 'Test': odh_te}
        return data
    
class EMGEPN_UserIndependent(EMGEPN612):
    def __init__(self, dataset_file='EMGEPN612.pkl'):
        EMGEPN612.__init__(self, dataset_file=dataset_file, cross_user=True)
    
    def prepare_data(self, split = True, subjects=None, feature_list = None, window_size = None, window_inc = None, feature_dic = None):
        odh = self.get_odh(subjects, feature_list, window_size, window_inc, feature_dic)
        odh_tr = odh.isolate_data('subjects', values=list(range(0,306)))
        odh_te = odh.isolate_data('subjects', values=list(range(306,612)))
        if split:
            data = {'All': odh_tr + odh_te, 'Train': odh_tr, 'Test': odh_te}
        return data

        