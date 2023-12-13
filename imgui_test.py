import libemg

if __name__ == "__main__":
    p = libemg.streamers.sifibridge_streamer(version="1_1",
                                             notch_on=False,
                                             emg_fir_on=False)
    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()

    # odh.visualize(num_samples=5000)
    # odh.analyze_hardware()
    args = {
        "online_data_handler"  : odh,
        "media_folder"         : "media2/",
        "data_folder"          : "data/",
        "num_reps"             : 5,
        "rep_time"             : 39,
        "rest_time"            : 1,
        "auto_advance"         : True,
        "window_size"          : 250,
        "window_inc"           : 150,
        "features"             : libemg.feature_extractor.FeatureExtractor.get_feature_groups(None)["HTD"],
        "visualization_horizon": 5000,
        "visualization_rate"   : 24,
    }
    gui = libemg.gui.GUI(args = args, debug=False)