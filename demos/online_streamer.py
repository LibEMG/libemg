import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.screen_guided_training import TrainingUI 
from libemg.data_handler import OnlineDataHandler
from libemg.data_handler import OfflineDataHandler
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_classifier import OnlineEMGClassifier, EMGClassifier
from libemg.offline_metrics import OfflineMetrics
from libemg.utils import make_regex
from libemg.utils import  myo_streamer

if __name__ == '__main__':
    odh = OnlineDataHandler(emg_arr=True)
    odh.start_listening()

    myo_streamer()
    odh.visualize(num_channels=8,y_axes=[-150,150])
    # odh.visualize_channels(channels=[0,1,2,3,4,5,6,7], y_axes=[-150,150])