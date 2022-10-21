import os
import sys
import socket
import random
import time
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.emg_classifier import EMGClassifier
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import get_windows
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.data_handler import OfflineDataHandler, OnlineDataHandler
from unb_emg_toolbox.filtering import Filter






def offline_dataset_filtering_demo():
    ## Example for filtering an offline dataset.
    dataset_folder = 'demos/data/myo_dataset'
    sets_values = ["training", "testing"]
    sets_regex = make_regex(left_bound = "dataset\\\\", right_bound="\\\\", values = sets_values)
    classes_values = ["0","1","2","3","4"]
    classes_regex = make_regex(left_bound = "_C_", right_bound="_EMG.csv", values = classes_values)
    reps_values = ["0","1","2","3"]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "sets": sets_values,
        "sets_regex": sets_regex,
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }
    odh = OfflineDataHandler()
    odh.get_data(dataset_folder=dataset_folder, dictionary = dic, delimiter=",")
    

    filter = Filter(sampling_frequency=200)
    filter_dictionary={"name":"bandpass",
                        "cutoff": [20, 99],
                        "order": 4 }
    
    filter.install_filters(filter_dictionary=filter_dictionary)
    filter_dictionary={"name":"notch",
                        "cutoff": 60,
                        "bandwidth": 3 }
    filter.install_filters(filter_dictionary=filter_dictionary)
    filter.visualize_filters()
    # for offlinedatahandlers, the filtered data will overwrite the odh.data attribut
    filter.filter(odh)


def np_ndarray_filtering_demo():
    ## Example for filtering a single .npy file
    data = np.loadtxt('demos/data/myo_dataset/training/R_0_C_0_EMG.csv', delimiter=",")
    filter = Filter(sampling_frequency=200)
    filter_dictionary={"name":"bandpass",
                        "cutoff": [20, 99],
                        "order": 4 }
    
    filter.install_filters(filter_dictionary=filter_dictionary)
    filter_dictionary={"name":"notch",
                        "cutoff": 60,
                        "bandwidth": 3 }
    filter.install_filters(filter_dictionary=filter_dictionary)
    filter.visualize_filters()
    filtered_data = filter.filter(data)
    plt.plot(filtered_data)
    plt.show()



def worker():
    # used for the online demo -- generate random data and put into tcp/ip socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def write_to_socket(sock):
        simulated_emg = random.sample(range(-100, 100), 8)
        sock.sendto(bytes(str(simulated_emg), "utf-8"), ('127.0.0.1', 12345))
        time.sleep(0.05)
    while True:
        try:
            write_to_socket(sock)
        except:
            print("Worker Stopped")
            quit()


def online_filtering_demo():
    window_size = 100
    window_increment = 20
    sampling_frequency = 200
    
    # get the filters ready
    filter = Filter(sampling_frequency=sampling_frequency)
    filter_dictionary={"name":"bandpass",
                        "cutoff": [20, 99],
                        "order": 4 }
    filter.install_filters(filter_dictionary=filter_dictionary)
    filter_dictionary={"name":"notch",
                        "cutoff": 60,
                        "bandwidth": 3 }
    filter.install_filters(filter_dictionary=filter_dictionary)
    
    # Create A stream of random data
    p = multiprocessing.Process(target=worker, daemon=True)
    p.start()

    # variable for the window (windowsizex8 shape)
    window = np.zeros((window_size, 8))

    odh = OnlineDataHandler(file=True, std_out=False, emg_arr=True)
    # start the stream listener
    odh.get_data()
    odh.raw_data.reset_emg()
    while True:
        # get the new data
        data = np.array(odh.raw_data.get_emg())
        if data.shape[0] == 0:
            time.sleep(window_increment/sampling_frequency)
            continue
        window = np.concatenate((window, data))
        # correct the window size
        if len(data > window_size):
            window = window[-1*window_size:,:]

        # the data has been grabbed and is in np.ndarray format
        filtered_data = filter.filter(window)
        plt.plot(filtered_data)
        plt.draw()
        plt.pause(window_increment/sampling_frequency)
        plt.clf()
        # rest for window_increment amount of time
        time.sleep(window_increment/sampling_frequency)


# Currently this file is for only one individual
if __name__ == "__main__" :
    offline_dataset_filtering_demo()

    np_ndarray_filtering_demo()

    # this function has a sleep(5) in it, so wait a minute for it to start :)
    online_filtering_demo()


    

