import pytest
from src.emg_feature_extraction_eeddy import feature_extractor

@pytest.fixture(scope='session') 
def fe():
    fe = feature_extractor.FeatureExtractor(num_channels=10)
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
