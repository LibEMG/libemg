import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.training_ui import TrainingUI 
from unb_emg_toolbox.data_handler import OnlineDataHandler
from unb_emg_toolbox.utils import myo_streamer, sifi_streamer

       
if __name__ == "__main__" :
    # Training Piece:
    sifi_streamer()
    odh = OnlineDataHandler(emg_arr=True)
    odh.get_data()
    train_ui = TrainingUI(3, 3, "demos/images/", "demos/data/sgt_example/", odh)