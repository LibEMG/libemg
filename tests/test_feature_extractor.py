import pytest
import numpy as np
from emg_feature_extraction.feature_extractor import FeatureExtractor as featureextractor

@pytest.fixture(scope='session') 
def fe():
    fe = featureextractor(num_channels=10)
    return fe

def test_get_windows(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    assert len(windows) == 18
    assert len(windows[0]) == 8
    assert len(windows[0][0]) == 50

def test_get_feature_groups(fe):
    assert fe.get_feature_groups() == ['HTD',
                                       'TD4',
                                       'TD9']

def test_get_feature_list(fe):
    assert fe.get_feature_list() == ['MAV',
                                     'ZC',
                                     'SSC',
                                     'WL',
                                     'LS',
                                     'MFL',
                                     'MSR',
                                     'WAMP',
                                     'RMS',
                                     'IAV',
                                     'DASDV',
                                     'VAR',
                                     'M0',
                                     'M2',
                                     'M4',
                                     'SPARSI',
                                     'IRF',
                                     'WLF',
                                     'AR', 
                                     'CC',
                                     'LD',
                                     'MAVFD',
                                     'MAVSLP',
                                     'MDF',
                                     'MNF',
                                     'MNP',
                                     'MPK',
                                     'SAMPEN',
                                     'SKEW',
                                     'KURT']
