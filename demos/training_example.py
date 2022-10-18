import os
import sys
import socket
import multiprocessing
from pyomyo import Myo, emg_mode

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.training_ui import TrainingUI 
from unb_emg_toolbox.data_handler import OnlineDataHandler
from unb_emg_toolbox.emg_classifier import EMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import create_folder_dictionary
from unb_emg_toolbox.data_handler import OfflineDataHandler

def worker():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def write_to_socket(emg, movement):
        sock.sendto(bytes(str(emg), "utf-8"), ('127.0.0.1', 12345))
    m.add_emg_handler(write_to_socket)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    m.set_leds([128, 0, 0], [128, 0, 0])
    m.vibrate(1)
    
    while True:
        try:
            m.run()
        except:
            print("Worker Stopped")
            quit() 
        
if __name__ == "__main__" :
    # Training Piece:
    p = multiprocessing.Process(target=worker, daemon=True)
    p.start()
    odh = OnlineDataHandler(emg_arr=True)
    odh.get_data()
    train_ui = TrainingUI(3, 3, "demos/images/", "demos/data/sgt_example/", odh, time_between_reps=2)