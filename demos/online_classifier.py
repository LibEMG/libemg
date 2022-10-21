from genericpath import isdir
import os
import time
import sys
import socket
import multiprocessing
from pyomyo import Myo, emg_mode

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.data_handler import OnlineDataHandler, OfflineDataHandler
from unb_emg_toolbox.emg_classifier import EMGClassifier, OnlineEMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import make_regex

def worker():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def write_to_socket(emg, movement):
        sock.sendto(bytes(str(emg), "utf-8"), ('127.0.0.1', 12345))
    m.add_emg_handler(write_to_socket)
    
    while True:
        try:
            m.run()
        except:
            print("Worker Stopped")
            quit() 
    
if __name__ == "__main__" :
    # Online classification
    dataset_folder = 'demos/data/sgt_example'
    classes_values = ["0","1","2","3","4", "5", "6"]
    classes_regex = make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
    reps_values = ["0", "1", "2"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }

    odh = OfflineDataHandler()
    odh.get_data(dataset_folder=dataset_folder, dictionary = dic, delimiter=",")
    train_windows, train_metadata = odh.parse_windows(50,25)
    
    # # Extract features from data set
    fe = FeatureExtractor(num_channels=8, feature_group="HTD")
    training_features = fe.extract_predefined_features(train_windows)

    # # Create data set dictionary 
    data_set = {}
    data_set['training_features'] = training_features
    data_set['training_labels'] = train_metadata['classes']
    data_set['training_windows'] = train_windows

    # Create Stream Bindings
    p = multiprocessing.Process(target=worker, daemon=True)
    p.start()
    online_data_handler = OnlineDataHandler(emg_arr=True)
    online_data_handler.get_data()
    time.sleep(5)

    # Create Classifier and Run
    classifier = OnlineEMGClassifier(model="LDA", data_set=data_set, window_size=50, window_increment=25, 
            online_data_handler=online_data_handler, feature_extractor=fe, velocity=True, std_out=True)
    classifier.run()