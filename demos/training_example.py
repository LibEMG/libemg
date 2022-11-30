import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.training_ui import TrainingUI 
from libemg.data_handler import OnlineDataHandler
from libemg.data_handler import OfflineDataHandler
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_classifier import OnlineEMGClassifier, EMGClassifier
from libemg.offline_metrics import OfflineMetrics
from libemg.utils import make_regex
from libemg.utils import  delsys_streamer#myo_streamer, sifi_streamer,

def offline_analysis():
    odh = OfflineDataHandler()
    dataset_folder = 'demos/data/delsys_sgt_example'
    classes_values = ["0","1","2"]
    classes_regex = make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
    reps_values = ["0","1","2"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }
    odh = OfflineDataHandler()
    odh.get_data(folder_location=dataset_folder, filename_dic = dic, delimiter=",")
    
    # values=[0] corresponds to training since the 0th element of sets_values is training
    train_odh = odh.isolate_data(key="reps", values=[0,1])
    train_windows, train_metadata = train_odh.parse_windows(200,50)
    test_odh = odh.isolate_data(key="reps", values=[2])
    test_windows, test_metadata = test_odh.parse_windows(200,50)

    fe = FeatureExtractor(num_channels=8)

    data_set = {}
    data_set['training_windows'] = train_windows # used for velocity control
    data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
    data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
    data_set['testing_labels'] = test_metadata['classes']
    data_set['training_labels'] = train_metadata['classes']

    om = OfflineMetrics()
    metrics = ['CA']
    classifier = EMGClassifier()
    classifier.fit("LDA", data_set.copy())
    preds = classifier.run()
    y_true = data_set['testing_labels']
    metrics = om.extract_offline_metrics(metrics, preds, y_true)
    print(metrics)

def online_analysis(online_data_handler):
    odh = OfflineDataHandler()
    dataset_folder = 'demos/data/delsys_sgt_example'
    classes_values = ["0","1","2"]
    classes_regex = make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
    reps_values = ["0","1","2"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }
    odh = OfflineDataHandler()
    odh.get_data(folder_location=dataset_folder, filename_dic = dic, delimiter=",")
    
    windows, metadata = odh.parse_windows(500,250)

    fe = FeatureExtractor(num_channels=8)

    data_set = {}
    
    data_set['training_features'] = fe.extract_feature_group('HTD', windows)
    data_set['training_labels'] = metadata['classes']

    classifier = OnlineEMGClassifier("LDA",
                                    data_set.copy(),
                                    num_channels=8,
                                    window_size=500,
                                    window_increment=500,
                                    online_data_handler=online_data_handler,
                                    features=fe.get_feature_groups()["HTD"],
                                    std_out=True)
    classifier.run(block=True)


if __name__ == "__main__" :
    # Training Piece:
    odh = OnlineDataHandler(emg_arr=True)
    odh.start_listening()

    delsys_streamer()
    odh.visualize_channels(channels=[0,1,2,3,4,5,6,7], y_axes=[-150,150])
    
    #train_ui = TrainingUI(3, 3, "demos/images/", "demos/data/delsys_sgt_example/", odh)

    # offline_analysis()
    online_analysis(odh)
    odh.stop_listenting()

