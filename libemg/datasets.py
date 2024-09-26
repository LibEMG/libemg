from libemg._datasets._3DC import _3DCDataset
from libemg._datasets.one_subject_myo import OneSubjectMyoDataset
from libemg._datasets.emg_epn612 import EMGEPN612
from libemg._datasets.ciil import CIIL_MinimalData, CIIL_ElectrodeShift
from libemg._datasets.grab_myo import GRABMyoBaseline, GRABMyoCrossDay
from libemg._datasets.continous_transitions import ContinuousTransitions
from libemg._datasets.nina_pro import NinaproDB2
from libemg._datasets.myodisco import MyoDisCo
from libemg._datasets.fors_emg import FORSEMG
from libemg._datasets.intensity import ContractionIntensity
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGClassifier
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter
import numpy as np

def get_dataset_list():
    """Gets a list of all available datasets.
    
    Returns
    ----------
    dictionary
        A dictionary with the all available datasets and their respective classes.
    """
    return {
        'OneSubjectMyo': OneSubjectMyoDataset,
        '3DC': _3DCDataset,
        'CIIL_MinimalData': CIIL_MinimalData,
        'CIIL_ElectrodeShift': CIIL_ElectrodeShift,
        'GRABMyoBaseline': GRABMyoBaseline,
        'GRABMyoCrossDay': GRABMyoCrossDay,
        'ContinuousTransitions': ContinuousTransitions,
        'NinaProDB2': NinaproDB2,
        'MyoDisCo': MyoDisCo,
        'FORS-EMG': FORSEMG,
        'EMGEPN612': EMGEPN612,
        'ContractionIntensity': ContractionIntensity,
    }
    
def get_dataset_info(dataset):
    """Prints out the information about a certain dataset. 
    
    Parameters
    ----------
    dataset: string
        The name of the dataset you want the information of.
    """
    if dataset in get_dataset_list():
        get_dataset_list()[dataset]().get_info()
    else:
        print("ERROR: Invalid dataset name")

#TODO: Update docs
def evaluate(model, window_size, window_inc, feature_list=['MAV'], included_datasets=['OneSubjectMyo', '3DC', 'CIIL_ElectrodeShift', 'GRABMyoCrossDay'], feature_dic={}):
    """Evaluates an algorithm against all included datasets.
    
    Parameters
    ----------
    window_size: int
        The window size (**in ms**). 
    window_inc: int
        The window increment (**in ms**). 
    """
    for d in included_datasets:
        print('Evaluating ' + d + ' dataset...')
        dataset = get_dataset_list()[d]()
        data = dataset.prepare_data(split=True)
        
        train_data = data['Train']
        test_data = data['Test']

        filter = Filter(dataset.sampling)
        filter.install_common_filters()
        filter.filter(train_data)
        filter.filter(test_data)
        
        for s in range(0, dataset.num_subjects):
            print(str(s) + '/' + str(dataset.num_subjects) + ' completed.')
            s_train_dh = train_data.isolate_data('subjects', [s])
            s_test_dh = test_data.isolate_data('subjects', [s])
            train_windows, train_meta = s_train_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc))
            test_windows, test_meta = s_test_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc))

            fe = FeatureExtractor()
            train_feats = fe.extract_features(feature_list, train_windows, feature_dic=feature_dic)
            test_feats = fe.extract_features(feature_list, test_windows, feature_dic=feature_dic)

            clf = EMGClassifier(model)
            ds = {
                'training_features': train_feats,
                'training_labels': train_meta['classes']
            }
            clf.fit(ds)
        
            preds, _ = clf.run(test_feats)
            om = OfflineMetrics()
            print(om.get_CA(test_meta['classes'], preds))    