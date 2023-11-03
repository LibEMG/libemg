from libemg.data_handler import OnlineDataHandler
from libemg.streamers import myo_streamer

if __name__ == '__main__':
    myo_streamer(imu=True)
    odh = OnlineDataHandler(std_out=True, file=True, timestamps=False)
    odh.start_listening()
    while(True):
        pass