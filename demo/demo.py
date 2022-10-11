from emg_feature_extraction.feature_extractor import FeatureExtractor as fe
import numpy as np


def main():
    print("Demo of feature extraction process")
    test_file = "emg_dummy_file.txt"
    data = np.loadtxt('demo/'+test_file, delimiter=',')
    num_channels = data.shape[1]

    feature_extractor = fe(num_channels=num_channels)
    windows = feature_extractor.get_windows(data, window_size=200, window_increment=100)

    mav = feature_extractor.getMAVfeat(windows)

    print("example for MAV!")




if __name__ == "__main__":
    main()