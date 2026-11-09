import libemg
from datetime import date
import os

def make_directory() -> str:
    # make directory
    today = date.today()
    DATA_DIRECTORY = f"Data/{today}-"
    data_index = 1
    if os.path.exists(DATA_DIRECTORY + str(data_index)):
        while os.path.exists(DATA_DIRECTORY + str(data_index)):
            data_index += 1
    DATA_DIRECTORY = DATA_DIRECTORY + str(data_index)
    os.makedirs(DATA_DIRECTORY)
    return DATA_DIRECTORY

def device_setup() -> libemg.data_handler.OnlineDataHandler:
    # device setup
    # p, smi = libemg.streamers.sifi_bioarmband_streamer(ecg=True,
    #                                                    emg=True,
    #                                                    eda=False,
    #                                                    imu=True,
    #                                                    ppg=True,
    #                                                    temperature=True,
    #                                                    filtering=False,
    #                                                    streaming=False,
    #                                                    imu_fs=25,
    #                                                    emg_fs=1000,
    #                                                    ecg_fs=500,
    #                                                    ppg_sps=50,
    #                                                    ppg_avg=1)
    p, smi = libemg.streamers.sifi_biopoint_streamer(ecg=True,
                                                       emg=True,
                                                       eda=False,
                                                       imu=True,
                                                       ppg=True,
                                                       temperature=True,
                                                       filtering=False,
                                                       streaming=True,
                                                       emg_fs=2000,
                                                       ecg_fs=500,
                                                       ppg_sps=50,
                                                       ppg_avg=1,
                                                       temperature_fs=1
                                                       )
    odh = libemg.data_handler.OnlineDataHandler(smi)

    odh.visualize(num_samples = 3000, block = True)
    return odh, p

def main():
    DATA_DIRECTORY = make_directory()
    
    odh, p = device_setup()

    gui = libemg.gui.GUI(online_data_handler=odh,
                         args = {
                             'media_folder': 'heartrate_classes_speed/',
                             'data_directory': DATA_DIRECTORY,
                             'num_reps': 2,
                             'rep_time': 60*10,
                             'rest_time': 5,
                             'auto_advance': False,
                             "timestamps": True
                         })
    gui.start_gui()


    p.signal.set()



if __name__ == "__main__":
    main()