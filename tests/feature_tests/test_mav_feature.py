import pytest
import numpy as np
from libemg.feature_extractor import FeatureExtractor as featureextractor
from libemg.utils import get_windows

@pytest.fixture(scope='session') 
def fe():
    fe = featureextractor()
    return fe

def test_normal_case(fe):
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    print(data.shape)
    windows = [data[0:5,:].transpose()]
    mav_feat = fe.getMAVfeat(np.array(windows))
    assert len(mav_feat) == 1
    assert len(mav_feat[0]) == 8
    assert list(mav_feat[0]) == list(np.mean(np.abs(data[0:5, :]), axis=0))
    assert True

def test_zero_case(fe):
    data = np.loadtxt('tests/data/emg_data_zeros.csv', delimiter=',')
    windows = get_windows(data, 50, 25)
    assert len(fe.getMAVfeat(windows)) == len(windows)
    assert len(fe.getMAVfeat(windows)[0]) == 8
    assert True