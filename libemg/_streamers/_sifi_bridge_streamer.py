import time
import os
import requests
from libemg.shared_memory_manager import SharedMemoryManager
import platform
from multiprocessing import Process, Event
import subprocess
import socket
import pickle
import json
import numpy as np

class SiFiBridgeStreamer(Process):
    def __init__(self, ip, port, 
                 version='1_2',
                 ecg=False,
                 emg=True, 
                 eda=False,
                 imu=False,
                 ppg=False,
                 notch_on=True,
                 notch_freq = 60,
                 emgfir_on=True,
                 emg_fir = [20, 450],
                 eda_cfg = True,
                 fc_lp = 0, # low pass eda
                 fc_hp = 5, # high pass eda
                 freq = 250,# eda sampling frequency
                 other=False,
                 streaming=False):
        Process.__init__(self, daemon=True)
        self.ip = ip
        self.port = port
        self.connected = False
        self.signal = Event()
        self.other = other
        self.emg_handlers = []
        self.imu_handlers = []
        self.other_handlers = []
        self.prepare_config_message(ecg, emg, eda, imu, ppg, 
                                    notch_on, notch_freq, emgfir_on, emg_fir,
                                    eda_cfg, fc_lp, fc_hp, freq, streaming)
        self.prepare_connect_message(version)

    def prepare_config_message(self, ecg, emg, eda, imu, ppg, 
                                    notch_on, notch_freq, emgfir_on, emg_fir,
                                    eda_cfg, fc_lp, fc_hp, freq, streaming):
        self.config_message = "-s ch " +  str(int(ecg)) +","+str(int(emg))+","+str(int(eda))+","+str(int(imu))+","+str(int(ppg))
        if notch_on or emgfir_on:
            self.config_message += " enable_filters 1 "
            if notch_on:
                self.config_message += " emg_notch " + str(notch_freq)
            else:
                self.config_message += " emg_notch 0"
            if emgfir_on:
                self.config_message += " emg_fir " + str(emg_fir[0]) + "," + str(emg_fir[1]) + ""
        else:
            self.config_message += " enable_filters 0"

        if eda_cfg:
            self.config_message += " eda_cfg " + str(int(fc_lp)) + "," + str(int(fc_hp)) + "," + str(int(freq))

        if streaming:
            self.config_message += " data_mode 1"
        self.config_message += "\n"
        self.config_message = bytes(self.config_message,"UTF-8")

    def prepare_connect_message(self, version):
        self.connect_message = '-c BioPoint_v' + str(version) + '\n'
        self.connect_message = bytes(self.connect_message,"UTF-8")

    def start_pipe(self):
        # note, for linux you may need to use sudo chmod +x sifi_bridge_linux
        if platform.system() == 'Linux':
           self.proc = subprocess.Popen(['sifi_bridge_linux'],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        elif platform.system() == "Windows":  # need a way to get these without curling -- 
            self.proc = subprocess.Popen(['sifi_bridge_windows.exe'],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
            
    def connect(self):
        while not self.connected:
            self.proc.stdin.write(self.connect_message)
            self.proc.stdin.flush()

            ret = self.proc.stdout.readline().decode()

            dat = json.loads(ret)

            if dat["connected"] == 1:
                self.connected = True
                print("Connected to Sifi device.")
            else:
                print("Could not connect. Retrying.")
        # Setup channels
        self.proc.stdin.write(self.config_message)
        self.proc.stdin.flush()

        self.proc.stdin.write(b'-cmd 1\n')
        self.proc.stdin.flush()
        self.proc.stdin.write(b'-cmd 0\n')
        self.proc.stdin.flush()

    def add_emg_handler(self, closure):
        self.emg_handlers.append(closure)

    def add_imu_handler(self, closure):
        self.imu_handlers.append(closure)

    def add_other_handler(self, closure):
        self.other_handlers.append(closure)
    
    def process_packet(self, data):
        packet = np.zeros((14,8))
        if data == "" or data.startswith("sending cmd"):
            return
        data = json.loads(data)
        if "data" in list(data.keys()):
            if "emg0" in list(data["data"].keys()):
                for c in range(packet.shape[1]):
                    packet[:,c] = data['data']["emg"+str(c)]
                for s in range(packet.shape[0]):
                    for h in self.emg_handlers:
                        h(packet[s,:].tolist())
            elif "emg" in list(data["data"].keys()): # This is the biopoint emg 
                emg = data['data']["emg"]
                for e in emg:
                    if not self.other:
                        self.emg_handlers[0]([e])
                    else:
                        self.other_handlers[0]('EMG-bio', [e])
            if "acc_x" in list(data["data"].keys()):
                accel = np.transpose(np.vstack([data['data']['acc_x'], data['data']['acc_y'], data['data']['acc_z']]))
                quat = np.transpose(np.vstack([data['data']['w'], data['data']['x'], data['data']['y'], data['data']['z']]))
                imu = np.hstack((accel, quat))
                for i in imu:
                    if not self.other:
                        self.imu_handlers[0](i)
                    else:
                        self.other_handlers[0]('IMU-bio', i)
            if "eda" in list(data["data"].keys()):
                eda = data['data']['eda']
                for e in eda:
                    self.other_handlers[0]('EDA-bio', [e])
            if "b" in list(data["data"].keys()):
                sizes = [len(data['data']['b']), len(data['data']['g']), len(data['data']['r']), len(data['data']['ir'])]
                ppg = np.transpose(np.vstack([data['data']['b'][0:min(sizes)], data['data']['g'][0:min(sizes)], data['data']['r'][0:min(sizes)], data['data']['ir'][0:min(sizes)]]))
                for p in ppg:
                    self.other_handlers[0]('PPG-bio', p)

    def run(self):
        # process is started beyond this point!
        self.start_pipe()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        def write_emg(emg):
            data_arr = pickle.dumps(list(emg))
            sock.sendto(data_arr, (self.ip, self.port))
        self.add_emg_handler(write_emg)

        def write_imu(imu):
            imu_list = ['IMU', imu]
            data_arr = pickle.dumps(list(imu_list))
            sock.sendto(data_arr, (self.ip, self.port))
        self.add_imu_handler(write_imu)

        def write_other(other, data):
            other_list = [other, data]
            data_arr = pickle.dumps(list(other_list))
            sock.sendto(data_arr, (self.ip, self.port))
        self.add_other_handler(write_other)
        self.connect()

        
        while True:
            try:
                data_from_processess = self.proc.stdout.readline().decode()
                self.process_packet(data_from_processess)
            except Exception as e:
                print("Error Occured: " + str(e))
                continue
            if self.signal.is_set():
                self.cleanup()
                break
        print("Process Ended")

    def close(self):
        self.proc.stdin.write(b'-cmd 1\n')
        self.proc.stdin.flush()
        return

    def turnoff(self):
        self.proc.stdin.write(b'-cmd 13\n')
        self.proc.stdin.flush()
        return
    
    def disconnect(self):
        self.proc.stdin.write(b'-d\n')
        self.proc.stdin.flush()
        while self.connected:
            ret = self.proc.stdout.readline().decode()
            dat = json.loads(ret)
            if 'connected' in dat.keys():
                if dat["connected"] == 0:
                    self.connected = False
                    print("Device disconnected.")
        return self.connected

    def cleanup(self):
        self.close()
        time.sleep(3)
        self.disconnect()
    