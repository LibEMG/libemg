import pytest
import numpy as np
from emg_feature_extraction.feature_extractor import FeatureExtractor as featureextractor

@pytest.fixture(scope='session') 
def fe():
    fe = featureextractor(num_channels=8)
    return fe

def test_extract_features_valid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    features = fe.extract_features(['MAV','ZC','WAMP'], windows)
    assert(len(features) == 3)
    assert('MAV' in features)
    assert('ZC' in features)
    assert('WAMP' in features)

def test_extract_features_invalid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    features = fe.extract_features(['MAV','RANDOM'], windows)
    assert(len(features) == 1)
    assert('MAV' in features)
    assert('RANDOM' not in features)

def test_extract_feature_group_valid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    features = fe.extract_feature_group('TD4', windows)
    assert(len(features) == 4)
    assert('LS' in features)
    assert('MFL' in features)
    assert('MSR' in features)
    assert('WAMP' in features)

def test_extract_feature_group_invalid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    features = fe.extract_feature_group('RANDOM', windows)
    assert(len(features) == 0)

def test_get_windows(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    assert len(windows) == 18
    assert len(windows[0]) == 8
    assert len(windows[0][0]) == 50

def test_get_feature_groups(fe):
    assert fe.get_feature_groups() == {'HTD': ['MAV', 'ZC', 'SSC', 'WL'],
                                       'TD4': ['LS', 'MFL', 'MSR', 'WAMP'],
                                       'TD9': ['LS', 'MFL', 'MSR', 'WAMP', 'ZC', 'RMS', 'IAV', 'DASDV', 'VAR']}

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

def test_all_features_normal(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    feature_list = ['MAV',
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
    features = fe.extract_features(feature_list,windows)
    assert len(features) == len(feature_list)

def test_all_features_zeros(fe):
    data = np.loadtxt('tests/data/emg_data_zeros.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    feature_list = ['MAV',
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
    features = fe.extract_features(feature_list,windows)                
    assert len(features) == len(feature_list)