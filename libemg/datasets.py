from libemg._datasets._3DC import _3DCDataset
from libemg._datasets.one_subject_myo import OneSubjectMyoDataset
from libemg._datasets.emg_epn612 import EMGEPN612
from libemg._datasets.ciil import CIIL_MinimalData, CIIL_ElectrodeShift
from libemg._datasets.grab_myo import GRABMyo
from libemg._datasets.continous_transitions import ContinuousTransitions

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
        'EMGEPN612': EMGEPN612,
        'CIIL_MinimalData': CIIL_MinimalData,
        'CIIL_ElectrodeShift': CIIL_ElectrodeShift,
        'GRABMyo': GRABMyo,
        'ContinuousTransitions': ContinuousTransitions,
    }
    
def get_dataset_info(dataset):
    if dataset in get_dataset_list():
        get_dataset_list()[dataset]().get_info()
    else:
        print("ERROR: Invalid dataset name")

    