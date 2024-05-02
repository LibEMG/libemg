import time
import os
import requests
from libemg.shared_memory_manager import SharedMemoryManager
import platform
from multiprocessing import Process, Event, Lock
import subprocess
import socket
import pickle
import json
import numpy as np


class SiFiBridgeStreamer(Process):
    def __init__(self, 
                 version='1_2',
                 shared_memory_items=[],
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
                 streaming=False):
        Process.__init__(self, daemon=True)

        self.connected=False
        self.signal = Event()
        self.shared_memory_items = shared_memory_items

        self.emg_handlers = []
        self.imu_handlers = []
        self.eda_handlers = []
        self.ecg_handlers = []
        self.ppg_handlers = []
        
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
        
        self.config_message += "  tx_power 2"
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

    def add_ppg_handler(self, closure):
        self.ppg_handlers.append(closure)
    
    def add_ecg_handler(self, closure):
        self.ecg_handlers.append(closure)
    
    def add_eda_handler(self, closure):
        self.eda_handlers.append(closure)

    def process_packet(self, data):
        packet = np.zeros((14,8))
        if data == "" or data.startswith("sending cmd"):
            return
        data = json.loads(data)
        
        if "data" in list(data.keys()):
            if "emg0" in list(data["data"].keys()): # this is multi-channel (armband) emg
                emg = np.stack((data["data"]["emg0"],
                                data["data"]["emg1"],
                                data["data"]["emg2"],
                                data["data"]["emg3"],
                                data["data"]["emg4"],
                                data["data"]["emg5"],
                                data["data"]["emg6"],
                                data["data"]["emg7"]
                                )).T
                for h in self.emg_handlers:
                    h(emg)
                # print(data['sample_rate'])
            if "emg" in list(data["data"].keys()): # This is the biopoint emg 
                emg = np.expand_dims(np.array(data['data']["emg"]),0).T
                for h in self.emg_handlers:
                    self.emg_handlers(emg) # check to see that this doesn't
            if "acc_x" in list(data["data"].keys()):
                imu = np.stack((data["data"]["acc_x"],
                                data["data"]["acc_y"],
                                data["data"]["acc_z"],
                                data["data"]["w"],
                                data["data"]["x"],
                                data["data"]["y"],
                                data["data"]["z"]
                                )).T
                for h in self.imu_handlers:
                    h(imu)
            if "eda" in list(data["data"].keys()):
                eda = np.expand_dims(np.array(data['data']['eda']),0).T
                for h in self.eda_handlers:
                    h(eda)
            if "ecg" in list(data["data"].keys()):
                ecg = np.stack((data["data"]["ecg"],
                                )).T
                for h in self.ecg_handlers:
                    h(ecg)
            if "b" in list(data["data"].keys()):
                if self.old_ppg_packet is None:
                    self.old_ppg_packet = data
                else:
                    ppg = np.stack((data["data"]["b"]  + self.old_ppg_packet["data"]["b"],
                                    data["data"]["g"]  + self.old_ppg_packet["data"]["g"],
                                    data["data"]["r"]  + self.old_ppg_packet["data"]["r"],
                                    data["data"]["ir"] + self.old_ppg_packet["data"]["ir"]
                                    )).T
                    self.old_ppg_packet = None
                    for h in self.ppg_handlers:
                        h(ppg)
                    
    def run(self):
        # process is started beyond this point!
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)
        self.start_pipe()
        def write_emg(emg):
            # update the samples in "emg"
            self.smm.modify_variable("emg", lambda x: np.vstack((np.flip(emg,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("emg_count", lambda x: x + emg.shape[0])
        self.add_emg_handler(write_emg)

        def write_imu(imu):
            # update the samples in "imu"
            self.smm.modify_variable("imu", lambda x: np.vstack((np.flip(imu,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("imu_count", lambda x: x + imu.shape[0])
            # sock.sendto(data_arr, (self.ip, self.port))
        self.add_imu_handler(write_imu)

        def write_eda(eda):
            # update the samples in "eda"
            self.smm.modify_variable("eda", lambda x: np.vstack((np.flip(eda,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("eda_count", lambda x: x + eda.shape[0])
        self.add_eda_handler(write_eda)

        def write_ppg(ppg):
            # update the samples in "ppg"
            self.smm.modify_variable("ppg", lambda x: np.vstack((np.flip(ppg,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("ppg_count", lambda x: x + ppg.shape[0])
        self.add_ppg_handler(write_ppg)

        def write_ecg(ecg):
            # update the samples in "ecg"
            self.smm.modify_variable("ecg", lambda x: np.vstack((np.flip(ecg,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("ecg_count", lambda x: x + ecg.shape[0])
        self.add_ecg_handler(write_ecg)

        self.connect()
        
        self.old_ppg_packet = None # required for now since ppg sends non-uniform packet length
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

    def stop_sampling(self):
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
        return self.connected

    def deep_sleep(self):
        self.proc.stdin.write(b'-cmd 14\n')
        self.proc.stdin.flush()

    def cleanup(self):
        
        self.stop_sampling()  # stop sampling
        print("Device sampling stopped.")
        time.sleep(1)
        self.deep_sleep() # stops status packets
        print("Device put to sleep.")
        time.sleep(1)
        self.disconnect() # disconnect
        print("Device disconnected.")
        time.sleep(1)
        self.proc.kill()
        print("SiFi bridge killed.")
        self.smm.cleanup()
        print("SiFi SMM cleanedup.")
    