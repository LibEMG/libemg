import os
import requests
from libemg.shared_memory_manager import SharedMemoryManager
from multiprocessing import Process, Event, Lock
import subprocess
import json
import numpy as np
import shutil
import json
from semantic_version import Version
from collections.abc import Callable
from platform import system


class SiFiBridgeStreamer(Process):
    """
    SiFi Labs Hardware Streamer.

    This streamer works with the SiFi Bioarmband and the SiFi Biopoint.
    It is capable of streaming from all modalities the device provides.

    Parameters
    ----------
    
    version : str
        The version of the devie ('1_1 for bioarmband, 1_2 or 1_3 for biopoint).
    shared_memory_items : list
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype, Lock()].
    ecg : bool
        Turn ECG modality on or off.
    emg : bool
        Turn EMG modality on or off.
    eda : bool
        Turn EDA modality on or off.
    imu : bool
        Turn IMU modality on or off.
    ppg : bool
        Turn PPG modality on or off
    notch_on : bool
        Turn on-system EMG notch filtering on or off.
    notch_freq : int
        Specify the frequency of the on-system notch filter.
    emgfir_on : bool
        Turn on-system EMG bandpass filter on or off.
    emg_fir : list
        The cutoff frequencies of the on-system bandpass filter.
    eda_cfg : bool
        Turn EDA into high sampling frequency mode (Bioimpedance at high frequency).
    fc_lp : int
        Bioimpedance bandpass low cutoff frequency.
    fc_hp : int
        Bioimpedance bandpass upper cutoff frequency.
    freq : int
        EDA/Bioimpedance sampling frequency.
    streaming : bool
        Reduce latency by joining packets of different modalities together.
    bridge_version : str
        Version of sifi bridge to use for PIPE.
    mac : str
        MAC address of the device to be connected with.
    
    """
    def __init__(self, 
                 version:              str  = '1_2',
                 shared_memory_items:  list = [],
                 ecg:                  bool = False,
                 emg:                  bool = True, 
                 eda:                  bool = False,
                 imu:                  bool = False,
                 ppg:                  bool = False,
                 notch_on:             bool = True,
                 notch_freq:           int  = 60,
                 emgfir_on:            bool = True,
                 emg_fir:              list = [20, 450],
                 eda_cfg:              bool = True,
                 fc_lp:                int  = 0, # low pass eda
                 fc_hp:                int  = 5, # high pass eda
                 freq:                 int  = 250,# eda sampling frequency
                 streaming:            bool = False,
                 bridge_version:       str | None = None,
                 mac:                  str | None = None):
        
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
        self.prepare_connect_message(version, mac)
        self.prepare_executable(bridge_version)
        


    def prepare_config_message(self, 
                               ecg:                  bool = False,
                               emg:                  bool = True, 
                               eda:                  bool = False,
                               imu:                  bool = False,
                               ppg:                  bool = False,
                               notch_on:             bool = True,
                               notch_freq:           int  = 60,
                               emgfir_on:            bool = True,
                               emg_fir:              list = [20, 450],
                               eda_cfg:              bool = True,
                               fc_lp:                int  = 0, # low pass eda
                               fc_hp:                int  = 5, # high pass eda
                               freq:                 int  = 250,# eda sampling frequency
                               streaming:            bool = False,):
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

    def prepare_connect_message(self, 
                                version: str,
                                mac : str):
        if mac is not None:
            self.connect_message = '-c ' + str(mac) + '\n'
        else:
            self.connect_message = '-c BioPoint_v' + str(version) + '\n'
        self.connect_message = bytes(self.connect_message,"UTF-8")
    
    def prepare_executable(self,
                           bridge_version: str):
        pltfm = system()
        self.executable = f"sifi_bridge%s-{pltfm.lower()}" + (
            ".exe" if pltfm == "Windows" else ""
        )
        if bridge_version is None:
            # Find the latest upstream version
            try:
                releases = requests.get(
                    "https://api.github.com/repos/sifilabs/sifi-bridge-pub/releases",
                    timeout=5,
                ).json()
                bridge_version = str(
                    max([Version(release["tag_name"]) for release in releases])
                )
            except Exception:
                # Probably some network error, so try to find an existing version
                # Expected to find sifi_bridge-V.V.V-platform in the current directory
                for file in os.listdir():
                    if not file.startswith("sifi_bridge"):
                        continue
                    bridge_version = file.split("-")[1].replace(".exe", "")
                if bridge_version is None:
                    raise ValueError(
                        "Could not fetch from upstream nor find a version of sifi_bridge to use in the current directory."
                    )

        self.executable = self.executable % ("-" + bridge_version)

        if self.executable not in os.listdir():
            ext = ".zip" if pltfm == "Windows" else ".tar.gz"
            arch = None
            if pltfm == "Linux":
                arch = "x86_64-unknown-linux-gnu"
                print(
                    "Please run <chmod +x sifi_bridge> in the terminal to indicate this is an executable file! You only need to do this once."
                )
            elif pltfm == "Darwin":
                arch = "aarch64-apple-darwin"
            elif pltfm == "Windows":
                arch = "x86_64-pc-windows-gnu"

            # Get Github releases
            releases = requests.get(
                "https://api.github.com/repos/sifilabs/sifi-bridge-pub/releases",
                timeout=5,
            ).json()

            # Extract the release matching the requested version
            release_idx = [release["tag_name"] for release in releases].index(
                bridge_version
            )
            assets = releases[release_idx]["assets"]

            # Find the asset that matches the architecture
            archive_url = None
            for asset in assets:
                asset_name = asset["name"]
                if arch not in asset_name:
                    continue
                archive_url = asset["browser_download_url"]
            if not archive_url:
                ValueError(f"No upstream version found for {self.executable}")
            print(f"Fetching sifi_bridge from {archive_url}")

            # Fetch and write to disk as a zip file
            r = requests.get(archive_url)
            zip_path = "sifi_bridge" + ext
            with open(zip_path, "wb") as file:
                file.write(r.content)

            # Unpack & delete the archive
            shutil.unpack_archive(zip_path, "./")
            os.remove(zip_path)
            extracted_path = f"sifi_bridge-{bridge_version}-{arch}/"
            for file in os.listdir(extracted_path):
                if not file.startswith("sifi_bridge"):
                    continue
                shutil.move(extracted_path + file, f"./{self.executable}")
            shutil.rmtree(extracted_path)

        

    def start_pipe(self):
        # note, for linux you may need to use sudo chmod +x sifi_bridge_linux
        self.proc = subprocess.Popen(
            ["./" + self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        assert self.proc is not None

            
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

    def add_emg_handler(self, 
                        closure: Callable):
        self.emg_handlers.append(closure)

    def add_imu_handler(self,
                        closure: Callable):
        self.imu_handlers.append(closure)

    def add_ppg_handler(self,
                        closure: Callable):
        self.ppg_handlers.append(closure)
    
    def add_ecg_handler(self,
                        closure: Callable):
        self.ecg_handlers.append(closure)
    
    def add_eda_handler(self,
                        closure: Callable):
        self.eda_handlers.append(closure)

    def process_packet(self,
                       data: str):
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
                    h(emg)
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
                print("Error Occurred: " + str(e))
                continue
            if self.signal.is_set():
                self.cleanup()
                break
        print("LibEMG -> SiFiBridgeStreamer (process ended).")

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
        print("LibEMG -> SiFiBridgeStreamer (sampling stopped).")
        self.deep_sleep() # stops status packets
        print("LibEMG -> SiFiBridgeStreamer (device sleeped).")
        self.disconnect() # disconnect
        print("LibEMG -> SiFiBridgeStreamer (device disconnected).")
        self.proc.kill()
        print("LibEMG -> SiFiBridgeStreamer (bridge killed).")
        self.smm.cleanup()
        print("LibEMG -> SiFiBridgeStreamer (SMM cleaned up).")

    def __del__(self):
        # self.proc.stdin.write(b"-d -q\n")
        # print("LibEMG -> SiFiBridgeStreamer (device disconnected).")
        # print("LibEMG -> SiFiBridgeStreamer (bridge killed).")
        # self.smm.cleanup()
        # print("LibEMG -> SiFiBridgeStreamer (SMM cleaned up).")
        pass
