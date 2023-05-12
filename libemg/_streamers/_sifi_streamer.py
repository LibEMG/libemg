import socket
import struct
from tkinter import E
import numpy as np
import pickle

class SiFiLabServer:
    def __init__(self, stream_port, stream_ip, sifi_port, sifi_ip):
        self.stream_port = stream_port 
        self.stream_ip = stream_ip
        self.m_port = sifi_port
        self.m_host_addr = sifi_ip
        self.ECG_GAIN = 194
        self.EMG_GAIN_AFTER_DECIM = 7500000
        self.stream_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def manageEMGdata(self, p_samples):
        transpose = np.transpose(p_samples)
        for row in transpose:
            emg_arr = pickle.dumps(list(row))
            self.stream_sock.sendto(emg_arr, (self.stream_ip, self.stream_port))

    # 0 - Quat W; 1 - Quat X; 2 - Quat Y; 3 - Quat Z; 4 - Accel X; 5 - Accel Y; 6 - Accel z;
    def manageIMUdata(self, p_samples):
        pass
            
    def bioArmBandManageData(self, p_data, p_amount):
        for k in range(0, p_amount, 227):
            packet_type = p_data[k]
            if packet_type == 1: #EMG data received
                emg_channels_samples_list = []
                for i in range(0, 10, 1):
                    emg_channels_samples_list.append([])
                for j in range(0, 210, 30):
                    for i in range(0, 10, 1):
                        emg_sample = float(int.from_bytes([p_data[k + j + i*3 + 5], p_data[k + j + i*3 + 4], p_data[k + j + i*3 + 3]], byteorder='big', signed=True))
                        emg_channels_samples_list[i].append(emg_sample/self.EMG_GAIN_AFTER_DECIM)
                self.manageEMGdata(emg_channels_samples_list)
            # elif packet_type == 3: #IMU data received
            #     IMU_data_samples_list = []
            #     for i in range(0, 7, 1):
            #         IMU_data_samples_list.append([])
            #     for j in range(0, 224, 28):
            #         for i in range(0, 7, 1):
            #             imu_sample_tmp = bytearray([p_data[k + j + i*4 + 6], p_data[k + j + i*4 + 5], p_data[k + j + i*4 + 4], p_data[k + j + i*4 + 3]]) 
            #             imu_sample = struct.unpack('f', imu_sample_tmp)
            #             IMU_data_samples_list[i].append(imu_sample)
            #     self.manageIMUdata(IMU_data_samples_list)

    def start_stream(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.m_host_addr, self.m_port))
            s.listen()
            conn, _ = s.accept()
            if conn:
                print(f"Connected to Sifi cuff.")
                while True:
                    raw_data_amount = conn.recv(4)
                    if len(raw_data_amount) == 0:
                        break
                    nb_bytes_in_packet = struct.unpack("!i", raw_data_amount)[0]
                    raw_data = conn.recv(nb_bytes_in_packet)
                    if len(raw_data) == 0:
                        break
                    self.bioArmBandManageData(raw_data, nb_bytes_in_packet)
            return