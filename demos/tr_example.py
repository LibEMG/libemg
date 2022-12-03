import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.data_handler import OnlineDataHandler
from libemg.streamers import myo_streamer

if __name__ == "__main__" :
    # Training Piece:
    odh = OnlineDataHandler(emg_arr=True)
    odh.start_listening()

    myo_streamer()
    
    # train_ui = ScreenGuidedTraining()
    # train_ui.download_gestures(list(range(1,10)), "demos/images/test/", download_gifs=True)
    # train_ui.launch_training(odh,output_folder="demos/data/sgt/", rep_folder="demos/images/test/", exclude_files=['Chuck_Grip.png', 'Hand_Close.png'])