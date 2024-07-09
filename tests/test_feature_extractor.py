import pytest
import numpy as np
from libemg.feature_extractor import FeatureExtractor as featureextractor
from libemg.utils import get_windows

@pytest.fixture(scope='session') 
def fe():
    fe = featureextractor()
    return fe

def test_extract_features_valid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    features = fe.extract_features(['MAV','ZC','WAMP'], windows)
    assert(len(features) == 3)
    assert('MAV' in features)
    assert('ZC' in features)
    assert('WAMP' in features)

def test_extract_features_invalid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    features = fe.extract_features(['MAV','RANDOM'], windows)
    assert(len(features) == 1)
    assert('MAV' in features)
    assert('RANDOM' not in features)

def test_extract_feature_group_valid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    features = fe.extract_feature_group('LS4', windows)
    assert(len(features) == 4)
    assert('LS' in features)
    assert('MFL' in features)
    assert('MSR' in features)
    assert('WAMP' in features)

def test_extract_TPSD(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    features = fe.extract_feature_group('TDPSD', windows)
    assert(len(features) == 6)
    assert('M0' in features)
    assert('M2' in features)
    assert('M4' in features)
    assert('SPARSI' in features)
    assert('IRF' in features)
    assert('WLF' in features)

def test_extract_feature_group_invalid(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    features = fe.extract_feature_group('RANDOM', windows)
    assert(len(features) == 0)

def test_get_windows(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    assert len(windows) == 19
    assert len(windows[0]) == 8
    assert len(windows[0][0]) == 50

def test_get_feature_groups(fe):
    assert fe.get_feature_groups() == {'HTD': ['MAV', 'ZC', 'SSC', 'WL'],
                                    'TSTD': ['MAVFD','DASDV','WAMP','ZC','MFL','SAMPEN','M0','M2','M4','SPARSI','IRF','WLF'],
                                    'DFTR': ['DFTR'],
                                    'ITD': ['ISD','COR','MDIFF','MLK'],
                                    'HJORTH': ['ACT','MOB','COMP'],
                                    'LS4': ['LS', 'MFL', 'MSR', 'WAMP'],
                                    'LS9': ['LS', 'MFL', 'MSR', 'WAMP', 'ZC', 'RMS', 'IAV', 'DASDV', 'VAR'],
                                    'TDPSD': ['M0','M2','M4','SPARSI','IRF','WLF'],
                                    'TDAR': ['MAV', 'ZC', 'SSC', 'WL', 'AR'],
                                    'COMB': ['WL', 'SSC', 'LD', 'AR9'],     
                                    'MSWT': ['WENG','WV','WWL','WENT']   }

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
                                    'SKEW',
                                    'KURT',
                                    "RMSPHASOR",
                                    "PAP",
                                    "WLPHASOR",
                                    "MZP",
                                    "TM",
                                    "SM",
                                    "SAMPEN",
                                    "FUZZYEN",
                                    "DFTR",
                                    "ISD",
                                    "COR",
                                    "MDIFF",
                                    "MLK",
                                    "ACT",
                                    "MOB",
                                    "COMP",
                                    "WENG",
                                    "WV",
                                    "WWL",
                                    "WENT",
                                    "MEAN"]

def test_all_features_normal(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    feature_list = fe.get_feature_list()
    features = fe.extract_features(feature_list,windows)
    assert len(features) == len(feature_list)

def test_all_features_zeros(fe):
    data = np.loadtxt('tests/data/emg_data_zeros.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    feature_list = fe.get_feature_list()
    features = fe.extract_features(feature_list,windows)                
    assert len(features) == len(feature_list)