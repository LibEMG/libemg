import libemg
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Lock


def main():
    p, smi = libemg.streamers.sifibridge_streamer(version="1_1",
                                                  shared_memory_items=[["emg", (3000, 8), np.double],
                                                                       ["emg_count", (1,1), np.int32 ],
                                                                       ["imu", (100, 7), np.double],
                                                                       ["imu_count", (1,1), np.int32 ],
                                                                       ["eda", (100, 1), np.double],
                                                                       ["eda_count", (1,1), np.int32 ],
                                                                       ["ppg", (100, 4), np.double],
                                                                       ["ppg_count", (1,1), np.int32 ]
                                                                       ],
                                                  emg=True,
                                                  imu=True,
                                                  ppg=True,
                                                  eda=True
                                                  )
    time.sleep(3)


    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for i in smi:
        smm.find_variable(*i)

    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi,
                                                timestamps=True)
    

    # fi = libemg.filtering.Filter(1500)
    # fi.install_common_filters()
    # odh.install_filter(fi)

    odh.log_to_file()
    # odh.visualize_channels(channels=[0,1,4],num_samples=4000)
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
    odh.stop_all()
    time.sleep(5)


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


def offline_test():
    WINDOW_SIZE = 250
    WINDOW_INC  = 17
    odh = libemg.data_handler.OfflineDataHandler()
    class_values = ["0","1","2","3","4"]
    rep_values = ["0","1","2","3","4"]
    odh.get_data("data/",
                 {
                     "classes":class_values,
                     "classes_regex": libemg.utils.make_regex("data/C_", "_R_", class_values),
                     "reps":rep_values,
                     "reps_regex":libemg.utils.make_regex("_R_", "_emg.csv", rep_values)
                 })
    train_odh = odh.isolate_data("reps",[0,1,2,3])
    test_odh = odh.isolate_data("reps", [3,4])

    train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE, WINDOW_INC)
    test_windows, test_metadata   = test_odh.parse_windows(WINDOW_SIZE, WINDOW_INC)

    fe = libemg.feature_extractor.FeatureExtractor()
    train_features = fe.extract_feature_group("HTD", train_windows)
    test_features  = fe.extract_feature_group("HTD", test_windows)

    fe.visualize_feature_space(train_features, "PCA", classes=train_metadata["classes"])

    mdl = libemg.emg_classifier.EMGClassifier(0)
    mdl.fit("LDA", {"training_features":train_features,
                    "training_labels": train_metadata["classes"]})
    
    predictions = mdl.run(test_features)[0]

    om = libemg.offline_metrics.OfflineMetrics()
    mat = om.get_CONF_MAT(test_metadata["classes"], predictions)
    # om.visualize_conf_matrix(mat)
    A = 1
    return mdl

def online_test(mdl):
    p, smi = libemg.streamers.sifibridge_streamer(version="1_1",
                                                  shared_memory_items=[["emg", (3000, 8), np.double],
                                                                       ["emg_count", (1,1), np.int32 ],
                                                                    #    ["imu", (100, 7), np.double],
                                                                    #    ["imu_count", (1,1), np.int32 ],
                                                                    #    ["eda", (100, 1), np.double],
                                                                    #    ["eda_count", (1,1), np.int32 ],
                                                                    #    ["ppg", (100, 4), np.double],
                                                                    #    ["ppg_count", (1,1), np.int32 ]
                                                                       ],
                                                  emg=True,
                                                  imu=False,
                                                  ppg=False,
                                                  eda=False
                                                  )
    time.sleep(3)


    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for i in smi:
        smm.find_variable(*i)

    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi,
                                                timestamps=True)
    
    mdl_smm_items  = [["classifier_output", (100,3), np.double],#timestamp, class prediction, confidence
                      ["classifier_input", (100,1+32), np.double]] # timestamp, <- features ->
    for item in mdl_smm_items:
        item.append(Lock())
    online_mdl = libemg.emg_classifier.OnlineEMGClassifier(mdl,
                                                           window_size = 250,
                                                           window_increment = 17,
                                                           online_data_handler=odh,
                                                           file=True,
                                                           smm=True,
                                                           smm_items = mdl_smm_items,
                                                           features=["MAV","ZC","SSC","WL"],
                                                           std_out=True)
    online_mdl.run(block=False)
    while True:
        time.sleep(10)


def inspect_classifier():
    data = np.loadtxt("classifier_output.txt", delimiter=" ")
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
    # inspect()
    mdl = offline_test()
    online_test(mdl)


    