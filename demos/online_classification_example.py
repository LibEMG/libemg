import os
import sys
import socket
import multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler
from libemg.emg_classifier import OnlineEMGClassifier, EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.streamers import mock_emg_stream

# def worker():
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     m = Myo(mode=emg_mode.FILTERED)
#     m.connect()

#     def write_to_socket(emg, movement):
#         sock.sendto(bytes(str(emg), "utf-8"), ('127.0.0.1', 12345))
#     m.add_emg_handler(write_to_socket)
    
#     while True:
#         try:
#             m.run()
#         except:
#             print("Worker Stopped")
#             quit() 
    
if __name__ == "__main__" :
    # Online classification
    dataset_folder = 'demos/data/myo_dataset/training/'
    classes_values = ["0","1","2","3","4"]
    classes_regex = make_regex(left_bound = "_C_", right_bound="_EMG", values = classes_values)
    reps_values = ["0", "1", "2", "3"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }

    odh = OfflineDataHandler()
    odh.get_data(folder_location=dataset_folder, filename_dic=dic, delimiter=",")
    train_windows, train_metadata = odh.parse_windows(50,25)
    
    # # Extract features from data set
    fe = FeatureExtractor()
    feature_list = fe.get_feature_groups()['HTD']
    training_features = fe.extract_features(feature_list, train_windows)

    # Create data set dictionary 
    data_set = {}
    data_set['training_features'] = training_features
    data_set['training_labels'] = train_metadata['classes']

    o_classifier = EMGClassifier()
    o_classifier.fit('SVM', feature_dictionary=data_set)

    # Create Stream Bindings
    mock_emg_stream(file_path="demos/data/stream_data.csv", sampling_rate=200, num_channels=8)
    online_data_handler = OnlineDataHandler(emg_arr=True)
    online_data_handler.start_listening()

    # Create Classifier and Run
    classifier = OnlineEMGClassifier(o_classifier, window_size=50, window_increment=25, 
            online_data_handler=online_data_handler, features=feature_list, std_out=True)
    classifier.run(block=True)