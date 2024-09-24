import os
import requests
from multiprocessing import Process, Event, Lock
import subprocess
import numpy as np
import shutil
from semantic_version import Version
from collections.abc import Callable
from platform import system

import sifi_bridge_py as sbp

from libemg.shared_memory_manager import SharedMemoryManager

class SiFiBridgeStreamer(Process):
    """
    SiFi Labs Hardware Streamer.

    This streamer works with the SiFi Bioarmband and the SiFi Biopoint.
    It is capable of streaming from all modalities the device provides.

    Parameters
    ----------
    
    device : str
        The name of the devie (eg BioArmband, BioPoint_v1_2, BioPoint_v1_3, etc.).
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
    filtering, default = True
        Enable on-device filtering, including bandpass filters and notch filters.
    emg_notch_freq, default = 60
        EMG notch filter frequency, useful for eliminating Mains power interference. Can be {None, 50, 60} Hz.
    emg_bandpass: tuple
        The (lower, higher) cutoff frequencies of the on-system EMG bandpass filter.
    eda_bandpass: tuple
        The (lower, higher) cutoff frequencies of the on-system EDA/BIOZ bandpass filter.
    eda_freq : int
        EDA/Bioimpedance injected signal frequency. 0 for DC.
    streaming : bool
        Reduce latency by joining packets of different modalities together.
    bridge_version : str | None
        SiFi Bridge executable version to use (and fetch if not found).
    mac : str | None
        MAC address of the device to be connected with.
    
    """
    def __init__(
        self, 
        device:               str  = 'BioArmband',
        shared_memory_items:  list = [],
        ecg:                  bool = False,
        emg:                  bool = True, 
        eda:                  bool = False,
        imu:                  bool = False,
        ppg:                  bool = False,
        filtering:            bool = True,
        emg_notch_freq:       int  = 60,
        emg_bandpass:         tuple = (20, 450),
        eda_bandpass:         tuple = (0, 5),
        eda_freq:             int  = 250,
        streaming:            bool = False,
        bridge_version:       str | None = None,
        mac:                  str | None = None
    ):

        Process.__init__(self, daemon=True)

        self.connected=False
        self.signal = Event()
        self.shared_memory_items = shared_memory_items

        self.emg_handlers = []
        self.imu_handlers = []
        self.eda_handlers = []
        self.ecg_handlers = []
        self.ppg_handlers = []
        
        self.prepare_executable(bridge_version)
        self.sb = sbp.SifiBridge("./" + self.executable)
        
        self.configure(
            ecg, 
            emg, 
            eda, 
            imu, 
            ppg, 
            filtering, 
            emg_notch_freq,
            emg_bandpass,
            eda_bandpass, 
            eda_freq,
            streaming
        )
        self.handle = mac if mac is not None else device

    def configure(
        self, 
        ecg:                  bool = False,
        emg:                  bool = True, 
        eda:                  bool = False,
        imu:                  bool = False,
        ppg:                  bool = False,
        filtering:            bool = True,
        notch_freq:           int  = 60,
        emg_bandpass:         tuple = (20, 450),
        eda_bandpass:         tuple = (0, 5),
        eda_freq:             int  = 250,
        streaming:            bool = False,
    ):
        self.sb.set_channels(ecg, emg, eda, imu, ppg)
        self.sb.set_filters(filtering)
        
        self.sb.configure_emg(emg_bandpass, notch_freq)
        self.sb.configure_eda(eda_bandpass, eda_freq)
        
        self.sb.set_low_latency_mode(streaming)
        self.sb.set_ble_power(sbp.BleTxPower.HIGH)
        self.sb.set_memory_mode(sbp.MemoryMode.BOTH)
    
    def prepare_executable(self,
                           bridge_version: str):
        pltfm = system()
        self.executable = f"sifibridge%s-{pltfm.lower()}" + (
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
                # Expected to find sifibridge-V.V.V-platform in the current directory
                for file in os.listdir():
                    if not file.startswith("sifibridge"):
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
            print(f"Fetching sifibridge from {archive_url}")

            # Fetch and write to disk as a zip file
            r = requests.get(archive_url)
            zip_path = "sifibridge" + ext
            with open(zip_path, "wb") as file:
                file.write(r.content)

            # Unpack & delete the archive
            shutil.unpack_archive(zip_path, "./")
            os.remove(zip_path)
            extracted_path = f"sifibridge-{bridge_version}-{arch}/"
            for file in os.listdir(extracted_path):
                if not file.startswith("sifibridge"):
                    continue
                shutil.move(extracted_path + file, f"./{self.executable}")
            shutil.rmtree(extracted_path)

            
    def connect(self):
        while not self.sb.connect(self.handle):
                print(f"Could not connect to {self.handle}. Retrying.")
                
        self.connected = True
        print("Connected to Sifi device.")

        self.sb.stop()
        self.sb.start()

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
                       data: dict):
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
                imu = np.stack((data["data"]["ax"],
                                data["data"]["ay"],
                                data["data"]["az"],
                                data["data"]["qw"],
                                data["data"]["qx"],
                                data["data"]["qy"],
                                data["data"]["qz"]
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
                new_packet = self.sb.get_data()
                self.process_packet(new_packet)
            except Exception as e:
                print("Error Occurred: " + str(e))
                continue
            if self.signal.is_set():
                self.cleanup()
                break
        print("LibEMG -> SiFiBridgeStreamer (process ended).")

    def stop_sampling(self):
        self.sb.stop()
        return

    def turnoff(self):
        self.sb.send_command(sbp.DeviceCommand.POWER_OFF)
        return
    
    def disconnect(self):
        self.connected = self.sb.disconnect()["connected"]
        return self.connected

    def deep_sleep(self):
        self.sb.send_command(sbp.DeviceCommand.POWER_DEEP_SLEEP)

    def cleanup(self):
        self.stop_sampling()  # stop sampling
        print("LibEMG -> SiFiBridgeStreamer (sampling stopped).")
        self.deep_sleep() # stops status packets
        print("LibEMG -> SiFiBridgeStreamer (device sleeped).")
        self.disconnect() # disconnect
        print("LibEMG -> SiFiBridgeStreamer (device disconnected).")
        self.sb._bridge.kill()
        print("LibEMG -> SiFiBridgeStreamer (bridge killed).")
        self.smm.cleanup()
        print("LibEMG -> SiFiBridgeStreamer (SMM cleaned up).")
