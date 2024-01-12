import libemg

if __name__ == "__main__":
    p = libemg.streamers.sifibridge_streamer(version="1_3",# 1_3
                                             ppg=True,
                                             eda=True,
                                             imu=True,
                                             notch_on=False,
                                             emg_fir_on=False,
                                             other=True,
                                             streaming=True)
    odh = libemg.data_handler.OnlineDataHandler(emg_arr=True,
                                                imu_arr=True,
                                                other_arr=True)
    odh.start_listening()

    # odh.visualize(num_samples=5000)
    # odh.analyze_hardware()
    args = {
        "online_data_handler"  : odh,
        "media_folder"         : "media/",
        "data_folder"          : "data/",
        "num_reps"             : 5,
        "rep_time"             : 17,
        "rest_time"            : 1,
        "auto_advance"         : True,
        "window_size"          : 250,
        "window_inc"           : 150,
        "features"             : libemg.feature_extractor.FeatureExtractor.get_feature_groups(None)["HTD"],
        "visualization_horizon": 5000,
        "visualization_rate"   : 24,
    }
    gui = libemg.gui.GUI(args = args, debug=False)