import pytest
from emg_feature_extraction.feature_extractor import FeatureExtractor as featureextractor

@pytest.fixture(scope='session') 
def fe():
    fe = featureextractor(num_channels=10)
    return fe

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
