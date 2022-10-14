import os
import sys
import socket
import random
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from unb_emg_toolbox.data_collector import DataCollector

if __name__== "__main__" :
    # dc = DataCollector(file=True, std_out=False, emg_arr=True)
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # # Generating random mock EMG data
    # while True:
    #     simulated_emg = random.sample(range(-100, 100), 8)
    #     sock.sendto(bytes(str(simulated_emg), "utf-8"), ('127.0.0.1', 12345))
    #     time.sleep(0.01)