from multiprocessing import Process, Event
import time
import numpy as np
from collections.abc import Callable

import sifi_bridge_py as sbp

from libemg.shared_memory_manager import SharedMemoryManager


class SiFiBridgeStreamer(Process):
    """
    SiFi Labs Hardware Streamer.

    This streamer works with the SiFi Bioarmband and the SiFi Biopoint.
    It is capable of streaming from all modalities the device provides.

    Parameters
    ----------
    name : str
        The name of the devie (eg BioArmband, BioPoint_v1_2, BioPoint_v1_3, etc.). None to auto-connect to any device.
    device_id : int | None
        If multiple devices are connected, this is the device ID to differentiate them. None by default (just one device).
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
    mac : str | None
        MAC address of the device to be connected with.

    """

    def __init__(
        self,
        name: str | None = None,
        device_id: int | None = None,
        shared_memory_items: list = [],
        ecg:            bool = False,
        emg:            bool = True,
        eda:            bool = False,
        imu:            bool = False,
        ppg:            bool = False,
        filtering:      bool = True,
        emg_notch_freq: int = 60,
        emg_bandpass:   tuple = (20, 450),
        eda_bandpass:   tuple = (0, 5),
        eda_freq:       int = 0,
        streaming:      bool = False,
        mac:            str | None = None,
        ble_power: sbp.BleTxPower = sbp.BleTxPower.HIGH,
        memory_mode: sbp.MemoryMode = sbp.MemoryMode.BOTH, 
        
    ):

        Process.__init__(self, daemon=True)

        self.connected = False
        self.signal = Event()
        self.shared_memory_items = shared_memory_items

        self.emg_handlers = []
        self.imu_handlers = []
        self.eda_handlers = []
        self.ecg_handlers = []
        self.ppg_handlers = []
        self.temp_handlers = []


        self.name = name
        self.device_id = device_id if device_id is not None else ""
        self.ecg = ecg
        self.emg = emg
        self.eda = eda
        self.imu = imu
        self.ppg = ppg
        self.filtering = filtering
        self.emg_notch_freq = emg_notch_freq
        self.emg_bandpass = emg_bandpass
        self.eda_bandpass = eda_bandpass
        self.eda_freq = eda_freq
        self.streaming = streaming
        self.mac = mac
        self.ble_power = ble_power
        self.memory_mode = memory_mode

    def configure(
        self,
        ecg: bool = False,
        emg: bool = True,
        eda: bool = False,
        imu: bool = False,
        ppg: bool = False,
        filtering: bool = True,
        notch_freq: int = 60,
        emg_bandpass: tuple = (20, 450),
        eda_bandpass: tuple = (0, 5),
        eda_freq: int = 0,
        streaming: bool = False,
        ble_power: sbp.BleTxPower = sbp.BleTxPower.MEDIUM,
        memory_mode: sbp.MemoryMode = sbp.MemoryMode.BOTH
    ):
        self.sb.set_channels(ecg, emg, eda, imu, ppg)
        self.sb.set_filters(filtering)
        
        if filtering:
            self.sb.configure_emg(emg_bandpass, notch_freq)
            self.sb.configure_eda(eda_bandpass, eda_freq)

        self.sb.set_low_latency_mode(streaming)
        self.sb.set_ble_power(ble_power)
        self.sb.set_memory_mode(memory_mode)

    def connect(self):
        while not self.sb.connect(self.handle):
            print(f"Could not connect to {self.handle}. Retrying.")

        self.connected = True
        print(f"Connected to Sifi device{self.device_id}.")

        self.sb.stop()
        self.sb.start()

    def add_emg_handler(self, closure: Callable):
        self.emg_handlers.append(closure)

    def add_imu_handler(self, closure: Callable):
        self.imu_handlers.append(closure)

    def add_ppg_handler(self, closure: Callable):
        self.ppg_handlers.append(closure)

    def add_ecg_handler(self, closure: Callable):
        self.ecg_handlers.append(closure)

    def add_eda_handler(self, closure: Callable):
        self.eda_handlers.append(closure)
        
    def add_temperature_handler(self, closure: Callable):
        self.temp_handlers.append(closure)

    def process_packet(self, data: dict):
        if "data" in list(data.keys()):
            if "emg0" in list(
                data["data"].keys()
            ):  # this is multi-channel (armband) emg
                emg = np.stack(
                    (
                        data["data"]["emg0"],
                        data["data"]["emg1"],
                        data["data"]["emg2"],
                        data["data"]["emg3"],
                        data["data"]["emg4"],
                        data["data"]["emg5"],
                        data["data"]["emg6"],
                        data["data"]["emg7"],
                    )
                ).T
                for h in self.emg_handlers:
                    h(emg)
                # print(data['sample_rate'])
            if "emg" in list(data["data"].keys()):  # This is the biopoint emg
                # print(data["data"]["emg"])
                emg = np.expand_dims(np.array(data["data"]["emg"]), 0).T
                for h in self.emg_handlers:
                    h(emg)
            if "ax" in list(data["data"].keys()):
                imu = np.stack(
                    (
                        data["data"]["ax"],
                        data["data"]["ay"],
                        data["data"]["az"],
                        data["data"]["qw"],
                        data["data"]["qx"],
                        data["data"]["qy"],
                        data["data"]["qz"],
                    )
                ).T
                for h in self.imu_handlers:
                    h(imu)
            if "eda" in list(data["data"].keys()):
                eda = np.expand_dims(np.array(data["data"]["eda"]), 0).T
                for h in self.eda_handlers:
                    h(eda)
            if "ecg" in list(data["data"].keys()):
                ecg = np.stack((data["data"]["ecg"],)).T
                for h in self.ecg_handlers:
                    h(ecg)
            if "b" in list(data["data"].keys()):
                if self.old_ppg_packet is None:
                    self.old_ppg_packet = data
                else:
                    ppg = np.stack(
                        (
                            data["data"]["b"] + self.old_ppg_packet["data"]["b"],
                            data["data"]["g"] + self.old_ppg_packet["data"]["g"],
                            data["data"]["r"] + self.old_ppg_packet["data"]["r"],
                            data["data"]["ir"] + self.old_ppg_packet["data"]["ir"],
                        )
                    ).T
                    self.old_ppg_packet = None
                    for h in self.ppg_handlers:
                        h(ppg)
            if "temperature" in list(data["data"].keys()):
                temperature = np.stack((data["data"]["temperature"],)).T
                for h in self.temp_handlers:
                    h(temperature)

    def run(self):
        # process is started beyond this point!
        self.sb = sbp.SifiBridge()

        self.configure(
            self.ecg,
            self.emg,
            self.eda,
            self.imu,
            self.ppg,
            self.filtering,
            self.emg_notch_freq,
            self.emg_bandpass,
            self.eda_bandpass,
            self.eda_freq,
            self.streaming,
            ble_power=self.ble_power,
            memory_mode=self.memory_mode
        )
        self.handle = self.mac if self.mac is not None else self.name

        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        def write_emg(emg):
            # update the samples in "emg"
            self.smm.modify_variable(
                f"emg{self.device_id}", lambda x: np.vstack((np.flip(emg, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable(f"emg{self.device_id}_count", lambda x: x + emg.shape[0])

        self.add_emg_handler(write_emg)

        def write_imu(imu):
            # update the samples in "imu"
            self.smm.modify_variable(
                f"imu{self.device_id}", lambda x: np.vstack((np.flip(imu, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable(f"imu{self.device_id}_count", lambda x: x + imu.shape[0])
            # sock.sendto(data_arr, (self.ip, self.port))

        self.add_imu_handler(write_imu)

        def write_eda(eda):
            # update the samples in "eda"
            self.smm.modify_variable(
                f"eda{self.device_id}", lambda x: np.vstack((np.flip(eda, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable(f"eda{self.device_id}_count", lambda x: x + eda.shape[0])

        self.add_eda_handler(write_eda)

        def write_ppg(ppg):
            # update the samples in "ppg"
            self.smm.modify_variable(
                f"ppg{self.device_id}", lambda x: np.vstack((np.flip(ppg, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable(f"ppg{self.device_id}_count", lambda x: x + ppg.shape[0])

        self.add_ppg_handler(write_ppg)

        def write_ecg(ecg):
            # update the samples in "ecg"
            self.smm.modify_variable(
                f"ecg{self.device_id}", lambda x: np.vstack((np.flip(ecg, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable(f"ecg{self.device_id}_count", lambda x: x + ecg.shape[0])

        self.add_ecg_handler(write_ecg)
        
        def write_temperature(temperature):
            # update the samples in "temperature"
            self.smm.modify_variable(
                f"temp{self.device_id}", lambda x: np.vstack((np.flip(temperature, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable(f"temp{self.device_id}_count", lambda x: x + temperature.shape[0])

        self.add_temperature_handler(write_temperature)
        
        self.connect()

        self.old_ppg_packet = (
            None  # required for now since ppg sends non-uniform packet length
        )
        
        print("LibEMG -> SiFiBridgeStreamer (process started).")
        while True:
            try:
                new_packet = self.sb.get_data()
                self.process_packet(new_packet)
            except Exception as e:
                print("Error Occurred maybe: ", e, "type : ", type(e), "representation : ", repr(e), " -> continuing.")
                import traceback
                traceback.print_exc()
                continue
            if self.signal.is_set():
                print("LibEMG -> SiFiBridgeStreamer (signal received).")
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
        try:
            result = self.sb.disconnect()
            # Handle both dictionary and boolean return types
            if isinstance(result, dict) and "connected" in result:
                self.connected = result["connected"]
            else:
                # If it returns a boolean directly, assume False means disconnected
                self.connected = not result if isinstance(result, bool) else False
        except Exception as e:
            print(f"Error during disconnect: {e}")
            self.connected = False
        return self.connected

    def deep_sleep(self):
        self.sb.send_command(sbp.DeviceCommand.POWER_DEEP_SLEEP)

    def cleanup(self):
        self.stop_sampling()  # stop sampling
        print("LibEMG -> SiFiBridgeStreamer (sampling stopped).")
        time.sleep(1)
        if self.mac == "CE:9B:59:A6:BD:EC" or self.mac == "DD:67:FD:19:06:03":
            self.turnoff()
            print("LibEMG -> SiFiBridgeStreamer (device turned off).")
        else:
            self.deep_sleep()  # stops status packets
            print("LibEMG -> SiFiBridgeStreamer (device sleeped).")
        
        time.sleep(1)
        self.disconnect()  # disconnect
        print("LibEMG -> SiFiBridgeStreamer (device disconnected).")
        self.sb._bridge.kill()
        print("LibEMG -> SiFiBridgeStreamer (bridge killed).")
        self.smm.cleanup()
        print("LibEMG -> SiFiBridgeStreamer (SMM cleaned up).")
