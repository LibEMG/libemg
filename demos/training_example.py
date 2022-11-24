import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.training_ui import TrainingUI 
from unb_emg_toolbox.data_handler import OnlineDataHandler
from unb_emg_toolbox.utils import myo_streamer, sifi_streamer

       
if __name__ == "__main__" :
    # Training Piece:
    myo_streamer()
    odh = OnlineDataHandler(emg_arr=True)
    odh.start_listening()
    odh.visualize(num_channels=8, y_axes=[-150,150])
    # odh.visualize_channels(channels=[0,1,2,3,4,5,6,7], y_axes=[-150,150])
    odh.stop_listenting()
    # train_ui = TrainingUI(3, 3, "demos/images/", "demos/data/sgt_example/", odh)