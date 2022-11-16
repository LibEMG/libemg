import socket
import pickle
from pyomyo import Myo, emg_mode

class MyoStreamer:
    def __init__(self, filtered, ip, port):
        self.filtered = filtered
        self.ip = ip 
        self.port = port

    def start_stream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mode = emg_mode.FILTERED
        if not self.filtered:
            mode = emg_mode.RAW
        m = Myo(mode=mode)
        m.connect()

        def write_emg(emg, _):
            data_arr = pickle.dumps(list(emg))
            sock.sendto(data_arr, (self.ip, self.port))
        m.add_emg_handler(write_emg)

        m.set_leds([128, 0, 0], [128, 0, 0])
        m.vibrate(1)

        # def write_imu(quat, acc, gyro):
        #     imu_arr = [*quat, *acc, *gyro]
        #     sock.sendto(bytes(str(imu_arr), "utf-8"), (ip, port))
        #     pass
        # m.add_imu_handler(write_imu)
        
        while True:
            try:
                m.run()
            except:
                print("Worker Stopped.")
                quit() 
