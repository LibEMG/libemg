from libemg.data_handler import OnlineDataHandler
from libemg.streamers import myo_streamer
from libemg.screen_guided_training import ScreenGuidedTraining

if __name__ == '__main__':
    myo_streamer(imu=True)
    odh = OnlineDataHandler()
    odh.start_listening()

    training_ui = ScreenGuidedTraining()
    training_ui.download_gestures([1,2,3,4,5], "images_tmp/")
    training_ui.launch_training(odh, 2, 3, "images_tmp/", "data_tmp/", 1, wait_btwn_prompts=True, continuous=True)