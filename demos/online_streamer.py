import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.training_ui import TrainingUI 
from unb_emg_toolbox.data_handler import OnlineDataHandler
from unb_emg_toolbox.data_handler import OfflineDataHandler
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.emg_classifier import OnlineEMGClassifier, EMGClassifier
from unb_emg_toolbox.offline_metrics import OfflineMetrics
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.utils import  myo_streamer

if __name__ == '__main__':
    odh = OnlineDataHandler(emg_arr=True)
    odh.start_listening()

    myo_streamer()
    odh.visualize(num_channels=8,y_axes=[-150,150])
    # odh.visualize_channels(channels=[0,1,2,3,4,5,6,7], y_axes=[-150,150])