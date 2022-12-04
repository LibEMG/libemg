import numpy as np
import time
import socket
import pytest
from libemg.data_handler import OfflineDataHandler, OnlineDataHandler
from libemg.utils import make_regex, get_windows
from libemg.streamers import mock_emg_stream
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

"""
By default these tests are marked @slow - and they do not work in the CI
pipeline. They should still be tested locally.

To test - run "pytest --slow"
"""

@pytest.mark.slow
def test_emg_classifier():
    odh = OfflineDataHandler()
    dataset_folder = 'tests/data/myo_dataset/'
    classes_values = ["0","1","2","3","4"]
    classes_regex = make_regex(left_bound = "_C_", right_bound="_EMG", values = classes_values)
    reps_values = ["0","1","2","3"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }
    odh = OfflineDataHandler()
    odh.get_data(folder_location=dataset_folder, filename_dic = dic, delimiter=",")
    
    windows, metadata = odh.parse_windows(50,25)
    test_data = np.loadtxt("tests/data/stream_data_tester.csv", delimiter=",")
    test_windows = get_windows(test_data, 50,25)

    fe = FeatureExtractor()

    data_set = {}
    
    data_set['training_features'] = fe.extract_feature_group('HTD', windows)
    data_set['training_labels'] = metadata['classes']
    data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
    data_set['testing_labels'] = []

    off_class = EMGClassifier()
    off_class.fit("LDA", data_set.copy())
    offline_preds = off_class.run()

    online_data_handler = OnlineDataHandler(emg_arr=True)
    online_data_handler.start_listening()

    online_classifier = OnlineEMGClassifier("LDA",
                                            data_set.copy(),
                                            num_channels=8,
                                            window_size=50,
                                            window_increment=25,
                                            online_data_handler=online_data_handler,
                                            features=fe.get_feature_groups()["HTD"])
    online_classifier.run(block=False)
    online_preds = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
    sock.bind(('127.0.0.1', 12346))
    mock_emg_stream("tests/data/stream_data_tester.csv", num_channels=8, sampling_rate=200)
    st = time.time()
    while(len(online_preds) != len(offline_preds)):
        data, _ = sock.recvfrom(1024)
        data = str(data.decode("utf-8"))
        if data:
            online_preds.append(int(data))
        if time.time() - st > 15:
            break
    online_classifier.stop_running()
    online_data_handler.stop_listening() 
    assert len(online_preds) == len(offline_preds)
    assert online_preds == list(offline_preds)