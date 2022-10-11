import pytest
from src.emg_feature_extraction_eeddy import feature_extractor

@pytest.fixture(scope='session') 
def fe():
    fe = feature_extractor.FeatureExtractor(num_channels=10)
    return fe

def test_get_feature_groups(fe):
    print(fe.get_feature_groups())
    assert fe.get_feature_groups() == ['HTD','TD4','TD9']
