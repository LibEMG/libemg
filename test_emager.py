import libemg
import time
import matplotlib.pyplot as plt

def collect():
    p=libemg.streamers.emager_streamer()

    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()
    
    training_ui = libemg.screen_guided_training.ScreenGuidedTraining()
    training_ui.download_gestures([1,2,3,4,5,6,7,8,9,10], "images/")
    training_ui.launch_training(odh, 2, 3, "images/", "data/subject/", 1)
    # p.kill()

def visualize():
    # p=libemg.streamers.sifibridge_streamer(notch_on=False, notch_freq=50)
    p = libemg.streamers.emager_streamer()
    odh = libemg.data_handler.OnlineDataHandler(file=False)
    odh.start_listening()
    odh.visualize_channels([0,1,2,31,32,33], num_samples=5000)

    while True:
        time.sleep(10)

def plot_data():
    folder_location = 'data/'
    reps_values = ["0","1"]
    classes_values = ["0","1","2","3","4","5","6","7","8","9"]
    dic = {
        "reps": reps_values,
        "classes": classes_values,
        "reps_regex": libemg.utils.make_regex(left_bound="subject/R_", right_bound="_C_", values=reps_values),
        "classes_regex": libemg.utils.make_regex(left_bound="_C_", right_bound=".csv", values=classes_values)
    }
    odh = libemg.data_handler.OfflineDataHandler()
    odh.get_data(folder_location=folder_location,
                 filename_dic=dic)
    windows, metadata = odh.parse_windows(150, 25)

    fe = libemg.feature_extractor.FeatureExtractor()
    features = fe.extract_features(["RMS"], windows)
    
    fe.check_features(features)
    # import matplotlib.pyplot as plt
    # for i in range(64):
    #     plt.subplot(8,8,i+1)
    #     plt.plot(features["RMS"][:,i])
    # plt.show()
    fe.visualize_feature_space(features, "PCA", metadata["classes"],normalize=False)

    A = 1

    
if __name__ == "__main__":
    # plt.rc('lines', linewidth=1, color='r')
    # plt.rc('axes', labelsize=3)        # Controls Axes Labels
    # plt.rc('xtick', labelsize=0)       # Controls x Tick Labels
    # plt.rc('ytick', labelsize=0)
    # streamer_class = libemg.streamers.SiFiBridgeStreamer(ip='127.0.0.1', port=12345,
    #                                                      ecg=False,
    #                                                      emg=True,
    #                                                      eda=False,
    #                                                      imu=False,
    #                                                      ppg=False,
    #                                                      emgfir_on=True,
    #                                                      emg_fir=[20,450])
    # streamer_class.start_stream()

    visualize()

    # collect()
    # plot_data()