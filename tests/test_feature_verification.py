import pytest
import numpy as np
from libemg.feature_extractor import FeatureExtractor as featureextractor
from libemg.utils import get_windows 

def test_all_features_versus_matlab():
    fe = featureextractor()
    # same data and window parameters as was used in get_feature_values.m
    data = np.loadtxt('tests/data/emg_data_myo.csv', delimiter=',')
    window_size = 200
    window_inc  = 25
    windows = get_windows(data,
                          window_size=window_size,
                          window_increment=window_inc)
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
                        'SKEW',
                        'KURT',
                        "RMSPHASOR",
                        "PAP",
                        "WLPHASOR",
                        "MZP",
                        "TM",
                        "SM",
                        "SAMPEN",
                        "FUZZYEN"]

    print("Highest percent difference between matlab and python:")
    print("-"*50)
    dic = {}
    for f in feature_list:
        python_feature = fe.extract_features([f], windows)[f]
        matlab_feature = np.loadtxt("tests/data/matlab_"+f+".csv", delimiter=",")
        feature_percent_diff = np.abs(2*((matlab_feature - python_feature)/(matlab_feature+python_feature)))
        dic[f] = [np.median(feature_percent_diff)*100, np.max(feature_percent_diff)*100]
        print(f"{f.ljust(6)}:: med: {np.median(feature_percent_diff)*100:.2f}%; max: {np.max(feature_percent_diff)*100:.2f}%")

    exclusions = ['VAR', 'AR', 'CC', 'SM','SAMPEN',"FUZZYEN"]
    for d in dic:
        if d in exclusions:
            continue
        assert dic[d][0] < 1
        assert dic[d][1] < 1
