import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.data_handler import OfflineDataHandler, OnlineDataHandler
from unb_emg_toolbox.filtering import Filter
from unb_emg_toolbox.utils import mock_emg_stream
def offline_dataset_filtering_demo():
    ## Example for filtering an offline dataset.
    dataset_folder = 'demos/data/data'
    classes_values = ["0","1","2","3","4","5","6","7","8","9","10"]
    classes_regex = make_regex(left_bound = "_", right_bound=".txt", values = classes_values)
    reps_values = ["0"]
    reps_regex = make_regex(left_bound = "gesture_", right_bound="_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }
    odh = OfflineDataHandler()
    odh.get_data(folder_location=dataset_folder, filename_dic = dic, delimiter=",")

    # make a filter object
    filter = Filter(sampling_frequency=1000)

    filter_dictionary={"name":"standardize",
                       "data": odh}
    filter.install_filters(filter_dictionary=filter_dictionary)
    
    filter_dictionary={"name":"bandpass",
                        "cutoff": [20, 450],
                        "order": 4 }
    
    filter.install_filters(filter_dictionary=filter_dictionary)
    filter_dictionary={"name":"notch",
                        "cutoff": 60,
                        "bandwidth": 3 }
    filter.install_filters(filter_dictionary=filter_dictionary)
    filter.visualize_filters()

    # get a file from the odh data handler
    sample_data = odh.data[0]
    # let's look what our planned filters actually end up doing to the data
    filter.visualize_affect(sample_data)

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
    plt.clf()
    plt.plot(filtered_data)
    plt.show()


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
    

    # variable for the window (windowsizex8 shape)
    window = np.zeros((window_size, 8))

    # Create A stream of emg data
    mock_emg_stream("demos/data/stream_data.csv", sampling_rate=sampling_frequency, num_channels=8)

    odh = OnlineDataHandler(file=False, std_out=False, emg_arr=True)
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
        plt.clf()
        plt.plot(filtered_data)
        plt.draw()
        plt.pause(window_increment/sampling_frequency)
        
        # rest for window_increment amount of time
        time.sleep(window_increment/sampling_frequency)


# Currently this file is for only one individual
if __name__ == "__main__" :
    offline_dataset_filtering_demo()

    np_ndarray_filtering_demo()

    # this function has a sleep(5) in it, so wait a minute for it to start :)
    online_filtering_demo()


    

