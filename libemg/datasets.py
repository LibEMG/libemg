from libemg._datasets._3DC import _3DCDataset
from libemg._datasets.one_subject_myo import OneSubjectMyoDataset
from libemg._datasets.emg_epn612 import EMGEPN612
from libemg._datasets.ciil import CIIL_MinimalData, CIIL_ElectrodeShift
from libemg._datasets.grab_myo import GRABMyoBaseline, GRABMyoCrossDay
from libemg._datasets.continous_transitions import ContinuousTransitions
from libemg._datasets.nina_pro import NinaproDB2
from libemg._datasets.myodisco import MyoDisCo
from libemg._datasets.fors_emg import FORSEMG

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
    }
    
def get_dataset_info(dataset):
    if dataset in get_dataset_list():
        get_dataset_list()[dataset]().get_info()
    else:
        print("ERROR: Invalid dataset name")

    