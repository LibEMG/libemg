import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler
from libemg.streamers import myo_streamer
from libemg.utils import make_regex
from libemg.feature_extractor import FeatureExtractor

if __name__ == "__main__" :
    # Training Piece:
    odh = OnlineDataHandler(emg_arr=True)
    odh.start_listening()
    myo_streamer()

    # Only need to do this once
    # train_ui = ScreenGuidedTraining()
    # train_ui.download_gestures(list(range(1,4)), "demos/images/pca_example/")
    # train_ui.launch_training(odh,output_folder="demos/data/pca_example/", rep_folder="demos/images/pca_example/")

    dataset_folder = "demos/data/pca_example/"
    classes_values = ["0","1","2"]
    classes_regex = make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
    reps_values = ["0", "1", "2"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }
    off_dh = OfflineDataHandler()
    off_dh.get_data(folder_location=dataset_folder, filename_dic=dic, delimiter=",")
    train_windows, train_metadata = off_dh.parse_windows(50,25)

    fe = FeatureExtractor()

    extracted_features = fe.extract_feature_group('HTD', train_windows)

    odh.visualize_feature_space(extracted_features, 50, 25, 200, classes=train_metadata['classes'])