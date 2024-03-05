import libemg
import time
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Note, Never use all modalities at the same time. you don't get full bandwidth
    p, smi = libemg.streamers.sifibridge_streamer(version="1_1",
                                                  shared_memory_items=[["emg", (3000, 8), np.double],
                                                                       ["emg_count", (1,1), np.int32 ],
                                                                    #    ["imu", (3000, 7), np.double],
                                                                    #    ["imu_count", (1,1), np.int32 ],
                                                                    #    ["eda", (3000, 1), np.double],
                                                                    #    ["eda_count", (1,1), np.int32 ],
                                                                    #    ["ppg", (3000, 4), np.double],
                                                                    #    ["ppg_count", (1,1), np.int32 ]
                                                                       ],
                                                  emg=True,
                                                #   imu=True,
                                                #   ppg=True,
                                                #   eda=True
                                                  )
    time.sleep(1)


    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for i in smi:
        smm.find_variable(*i)

    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi,
                                                file=True,
                                                timestamps=True)
    odh.start_listening()
    # odh.visualize(4000)

    args = {
        "online_data_handler": odh,
        "streamer": p,
        "media_folder":"images/",
        "data_folder": "data/",
        "num_reps":5,
        "rep_time": 1,
        "rest_time":1,
        "auto_advance":True
    }

    gui = libemg.gui.GUI(args=args, debug=False)

    time.sleep(10)


def inspect():
    data = np.loadtxt("emg.csv", delimiter=" ")
    delta_time = data[-1,0] - data[0,0]
    num_samples = data.shape[0]
    sampling_freq = num_samples/delta_time
    print(f"{delta_time} s, {num_samples} samples, {sampling_freq} hz")

    plt.plot(data[:,0])
    plt.show()
    # plt.hist(np.diff(data[:,0]),bins=100)

    # fi = libemg.filtering.Filter(1500)
    # fi.install_common_filters()
    # fi.visualize_effect(data[:,1:])
    A  = 1


if __name__ == "__main__":
    # main()
    inspect()

