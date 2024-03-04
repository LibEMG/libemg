import libemg
import time

def main():
    p = libemg.streamers.sifibridge_streamer(version="1_1")
    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()

    args = {
        "online_data_handler": odh,
        "streamer": p,
        "media_folder": "images/",
        "data_folder": "data/S" + str(0) + "/",
        "num_reps": 6,
        "rep_time": 5,
        "rest_time": 1,
        "auto_advance": True
    }
    gui = libemg.gui.GUI(args=args, debug=False)



if __name__ == "__main__":
    main()