import libemg

if __name__ == "__main__":
    p = libemg.streamers.sifibridge_streamer(version="1_1")
    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()
    # odh.analyze_hardware()
    args = {
        "odh"         : odh,
        "media_folder": "media/",
        "data_folder" : "data/",
        "num_reps"    : 5,
        "rep_time"    : 3,
        "rest_time"   : 1,
        "auto_advance":True,
    }
    gui = libemg.gui.GUI(args = args, debug=False)