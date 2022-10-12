import pytest
import numpy as np
from unb_emg_toolbox.feature_extractor import FeatureExtractor as featureextractor

@pytest.fixture(scope='session') 
def fe():
    fe = featureextractor(num_channels=8)
    return fe

def test_normal_case(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    windows = [data[0:5,:].transpose()]
    zc_feat = fe.getZCfeat(windows)
    assert len(zc_feat) == 1
    assert len(zc_feat[0]) == 8
    assert list(zc_feat[0]) == [2,3,4,3,2,1,1,0]

def test_zero_case(fe):
    data = np.loadtxt('tests/data/emg_data_zeros.csv', delimiter=',')
    windows = fe.get_windows(data, 50, 25)
    assert len(fe.getZCfeat(windows)) == len(windows)
    assert len(fe.getZCfeat(windows)[0]) == 8

def test_threshold_case(fe):
    pass