from multiprocessing import Process, Event
import numpy as np
from collections.abc import Callable

import sifi_bridge_py as sbp

from libemg.shared_memory_manager import SharedMemoryManager


# Sampling rates (Hz) accepted by the SiFi hardware for each modality.
ECG_SAMPLING_RATES = (250, 500, 1000, 2000)
EMG_SAMPLING_RATES = (500, 1000, 2000)
EMG_SAMPLING_RATES_BIOARMBAND = (500, 1000, 1600, 2000)  # 1600 Hz is BioArmband only
EDA_SAMPLING_RATES = (4, 8, 16, 32, 50)
IMU_SAMPLING_RATES = (25, 50, 100, 200)
PPG_SAMPLING_RATES = (50, 100, 200, 400, 800)
PPG_AVERAGING_FACTORS = (1, 2, 4, 8, 16, 32)
PPG_MAX_EFFECTIVE_RATE = 400  # sps / avg must not exceed this
TEMPERATURE_SAMPLING_RATES = (0.1, 1, 2, 10)

# Seconds without a single data packet before the link is presumed down. Every
# modality packetizes far faster than this even at its lowest supported rate, so
# a gap this long is a dropped connection rather than a quiet stream.
DATA_STALL_TIMEOUT = 1.0
# Consecutive stalled reads tolerated before escalating from a BLE-level
# reconnect to tearing down and rebuilding the whole sifibridge subprocess.
STALLS_BEFORE_BRIDGE_REBUILD = 5
# Cap on stderr lines read per drain, so a bridge flooding its stderr cannot
# starve the recovery it was supposed to explain.
MAX_STDERR_LINES_PER_DRAIN = 200


def _validate_setting(name: str, value, allowed):
    """Raise a ValueError if value is not one of the hardware-supported settings."""
    if value not in allowed:
        raise ValueError(
            f"Invalid {name} of {value}. Must be one of {list(allowed)}."
        )
    return value


def validate_sifi_sampling_rates(
    ecg_fs,
    emg_fs,
    eda_fs,
    imu_fs,
    ppg_sps,
    ppg_avg,
    temperature_fs,
    bioarmband: bool = False,
):
    """
    Check every SiFi sampling rate against the values the hardware supports.

    Parameters
    ----------
    bioarmband : bool
        Whether the target device is a BioArmband, which additionally supports a
        1600 Hz EMG sampling rate.

    Raises
    ------
    ValueError
        If any setting is unsupported.
    """
    _validate_setting("ECG sampling rate", ecg_fs, ECG_SAMPLING_RATES)
    _validate_setting(
        "EMG sampling rate",
        emg_fs,
        EMG_SAMPLING_RATES_BIOARMBAND if bioarmband else EMG_SAMPLING_RATES,
    )
    _validate_setting("EDA sampling rate", eda_fs, EDA_SAMPLING_RATES)
    _validate_setting("IMU sampling rate", imu_fs, IMU_SAMPLING_RATES)
    _validate_setting("PPG sampling rate", ppg_sps, PPG_SAMPLING_RATES)
    _validate_setting("PPG averaging factor", ppg_avg, PPG_AVERAGING_FACTORS)
    if ppg_sps / ppg_avg > PPG_MAX_EFFECTIVE_RATE:
        raise ValueError(
            f"Invalid PPG configuration: sps of {ppg_sps} with an averaging factor of "
            f"{ppg_avg} gives an effective sampling rate of {ppg_sps / ppg_avg} Hz, "
            f"which exceeds the maximum of {PPG_MAX_EFFECTIVE_RATE} Hz."
        )
    _validate_setting(
        "temperature sampling rate", temperature_fs, TEMPERATURE_SAMPLING_RATES
    )


class SiFiBridgeStreamer(Process):
    """
    SiFi Labs Hardware Streamer.

    This streamer works with the SiFi Bioarmband and the SiFi Biopoint.
    It is capable of streaming from all modalities the device provides.

    Parameters
    ----------
    name : sifi_bridge_py.DeviceType | None
        The name of the devie (eg BioArmband, BioPoint_v1_2, BioPoint_v1_3, etc.). None to auto-connect to any device.
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
    ecg_fs : int
        ECG sampling rate (Hz). Can be {250, 500, 1000, 2000}.
    emg_fs : int
        EMG sampling rate (Hz). Can be {500, 1000, 2000}, plus 1600 on the BioArmband.
    eda_fs : int
        EDA sampling rate (Hz). Can be {4, 8, 16, 32, 50}.
    imu_fs : int
        IMU sampling rate (Hz). Can be {25, 50, 100, 200}.
    ppg_sps : int
        PPG sampling rate (Hz). Can be {50, 100, 200, 400, 800}.
    ppg_avg : int
        PPG averaging factor. Can be {1, 2, 4, 8, 16, 32}. The effective sampling rate
        (ppg_sps / ppg_avg) must be <= 400 Hz.
    temperature_fs : float
        Temperature sampling rate (Hz). Can be {0.1, 1, 2, 10}.
    bioarmband : bool | None
        Whether the target device is a BioArmband, which additionally supports a
        1600 Hz EMG sampling rate. None infers this from the device name.

    """

    def __init__(
        self,
        name: str | None = None,
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
        ecg_fs:         int = 500,
        emg_fs:         int = 2000,
        eda_fs:         int = 50,
        imu_fs:         int = 50,
        ppg_sps:        int = 50,
        ppg_avg:        int = 1,
        temperature_fs: float = 1,
        bioarmband:     bool | None = None,
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

 
        self.device_name = name
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
        # 1600 Hz EMG is only available on the BioArmband. When the caller doesn't say
        # which device this is, fall back to identifying an armband by its name.
        self.bioarmband = (
            bool(name is not None and "armband" in str(name).lower())
            if bioarmband is None
            else bioarmband
        )
        validate_sifi_sampling_rates(
            ecg_fs,
            emg_fs,
            eda_fs,
            imu_fs,
            ppg_sps,
            ppg_avg,
            temperature_fs,
            self.bioarmband,
        )
        self.ecg_fs = ecg_fs
        self.emg_fs = emg_fs
        self.eda_fs = eda_fs
        self.imu_fs = imu_fs
        self.ppg_sps = ppg_sps
        self.ppg_avg = ppg_avg
        self.temperature_fs = temperature_fs
        # connecting to device can either have a string for the name, device class, or mac address. If None is provided, it autoconnects.
        self.handle = self.mac if self.mac is not None else self.device_name

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
    ):
        self.sb.configure_sensors(ecg, emg, eda, imu, ppg)

        if ecg:
            self.sb.configure_ecg(fs=self.ecg_fs,
                                  dc_notch=filtering,
                                  mains_notch=notch_freq,
                                  bandpass=filtering,
                                  flo=0,
                                  fhi=30)

        if emg:
            self.sb.configure_emg(fs=self.emg_fs,
                                dc_notch=filtering,
                                mains_notch=notch_freq,
                                bandpass=filtering,
                                flo=emg_bandpass[0],
                                fhi=emg_bandpass[1])



        if eda:
            self.sb.configure_eda(fs=self.eda_fs,
                                  dc_notch=filtering,
                                  mains_notch=notch_freq,
                                  bandpass=filtering,
                                  flo=eda_bandpass[0],
                                  fhi=eda_bandpass[1],)

        if imu:
            self.sb.configure_imu(fs=self.imu_fs)

        if ppg:
            self.sb.configure_ppg(sps=self.ppg_sps, avg=self.ppg_avg)

        self.sb.configure_temperature(fs=self.temperature_fs)

        self.sb.set_low_latency_mode(True)
        self.sb.set_ble_power(sbp.BleTxPower.HIGH)
        self.sb.set_memory_mode(sbp.MemoryMode.STREAMING)

    def connect(self):
        """Join the device, apply the configuration, and begin sampling.

        Returns
        -------
        bool
            True once connected and sampling. False if the shutdown signal was
            raised while retrying, so callers can abandon the attempt instead of
            retrying forever against a device that is off or out of range.
        """
        while not self.sb.connect(self.handle):
            print(f"Could not connect to {self.handle}. Retrying.")
            # Without this check a device that is off or out of range wedges the
            # process in a tight retry loop that ignores cleanup requests.
            if self.signal.is_set():
                print("LibEMG -> SiFiBridgeStreamer (connect abandoned, stopping).")
                return False

        self.connected = True
        print("Connected to Sifi device.")

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
        )

        self.sb.stop()
        self.sb.start()
        self.sb.clear_data_buffer()
        return True

    def _rebuild_bridge(self):
        """Replace the sifibridge subprocess, then reconnect.

        A BLE-level reconnect is not always enough. ``SifiBridge`` opens its data
        socket and starts the thread that reads from it once, in its constructor;
        if that socket closes (bridge subprocess died, host dropped the
        connection) the reader thread exits and no packet ever reaches the queue
        again. Reconnecting BLE alone would then appear to succeed while
        delivering nothing, so the bridge itself has to be rebuilt.

        Returns
        -------
        bool
            Whether the rebuilt bridge is connected and sampling.
        """
        print("LibEMG -> SiFiBridgeStreamer (rebuilding bridge).")
        try:
            self.sb.close()
        except Exception as e:
            # Already-dead bridges raise here; the replacement matters, not this.
            print(f"LibEMG -> SiFiBridgeStreamer (error closing old bridge: {e}).")
        self.sb = sbp.SifiBridge()
        self.sb._DEFAULT_REQUEST_TIMEOUT = 10.0
        return self.connect()

    def _drain_bridge_diagnostics(self):
        """Print anything sifibridge wrote to stderr, and return the lines.

        sifibridge explains link failures on its own stderr, which its Python
        wrapper buffers in a queue that is only ever drained inside ``connect()``.
        Nothing reads it while streaming, so the one message that says *why* a
        recording stopped is discarded. Draining it here turns a bare stall into
        an actionable reason, and keeps the unbounded queue from growing for the
        length of a long session.
        """
        lines = []
        stderr_queue = getattr(self.sb, "_stderr_queue", None)
        if stderr_queue is None:
            return lines
        for _ in range(MAX_STDERR_LINES_PER_DRAIN):
            try:
                lines.append(str(stderr_queue.get_nowait()).strip())
            except Exception:
                # Empty, or a bridge object that does not expose this queue.
                break
        for line in lines:
            if line:
                print(f"LibEMG -> SiFiBridgeStreamer (bridge said: {line})")
        return lines

    def _recover_stream(self, stalls):
        """Try to restore a stalled stream, escalating with the stall count.

        Parameters
        ----------
        stalls : int
            Number of consecutive stalled reads observed so far.

        Returns
        -------
        bool
            Whether data is expected to flow again.
        """
        self.connected = False
        try:
            if stalls < STALLS_BEFORE_BRIDGE_REBUILD:
                # Cheap path: the socket is still live and only the BLE link
                # dropped, which a reconnect plus reconfigure repairs.
                return self.connect()
            return self._rebuild_bridge()
        except Exception as e:
            print(f"LibEMG -> SiFiBridgeStreamer (recovery attempt failed: {e}).")
            return False

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
                if emg.dtype != 'float64':
                    # Remove None rows while preserving 2D structure
                    emg = emg.astype('float64')
                    emg = emg[[any(~np.isnan(row)) for row in emg]]
                    
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
            if "ir" in list(data["data"].keys()):
                ppg = np.array([data["data"]["ir"], data["data"]["r"], data["data"]["g"], data["data"]["b"]]).T
                for h in self.ppg_handlers:
                    h(ppg)

    def run(self):
        # process is started beyond this point!
        self.sb = sbp.SifiBridge()
        self.sb._DEFAULT_REQUEST_TIMEOUT = 10.0

        self.connect()

        

        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        def write_emg(emg):
            # update the samples in "emg"
            self.smm.modify_variable(
                "emg", lambda x: np.vstack((np.flip(emg, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable("emg_count", lambda x: x + emg.shape[0])

        self.add_emg_handler(write_emg)

        def write_imu(imu):
            # update the samples in "imu"
            self.smm.modify_variable(
                "imu", lambda x: np.vstack((np.flip(imu, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable("imu_count", lambda x: x + imu.shape[0])
            # sock.sendto(data_arr, (self.ip, self.port))

        self.add_imu_handler(write_imu)

        def write_eda(eda):
            # update the samples in "eda"
            self.smm.modify_variable(
                "eda", lambda x: np.vstack((np.flip(eda, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable("eda_count", lambda x: x + eda.shape[0])

        self.add_eda_handler(write_eda)

        def write_ppg(ppg):
            # update the samples in "ppg"
            self.smm.modify_variable(
                "ppg", lambda x: np.vstack((np.flip(ppg, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable("ppg_count", lambda x: x + ppg.shape[0])

        self.add_ppg_handler(write_ppg)

        def write_ecg(ecg):
            # update the samples in "ecg"
            self.smm.modify_variable(
                "ecg", lambda x: np.vstack((np.flip(ecg, 0), x))[: x.shape[0], :]
            )
            # update the number of samples retrieved
            self.smm.modify_variable("ecg_count", lambda x: x + ecg.shape[0])

        self.add_ecg_handler(write_ecg)

        self.old_ppg_packet = (
            None  # required for now since ppg sends non-uniform packet length
        )
        # Consecutive reads that returned no packet. Reset by any real packet.
        stalls = 0
        while True:
            # Checked first so a stalled or unrecoverable link still shuts down.
            if self.signal.is_set():
                self.cleanup()
                break
            try:
                # A bounded wait is what makes a dropped link observable at all.
                # get_data() defaults to blocking forever, and the thread feeding
                # its queue exits silently when the data socket closes, so an
                # unbounded read parks here for the rest of the session: no
                # exception to catch, no samples, and no way back out.
                new_packet = self.sb.get_data(timeout=DATA_STALL_TIMEOUT)
            except Exception as e:
                print("Error Occurred: " + str(e))
                stalls += 1
                self._recover_stream(stalls)
                continue

            if new_packet:
                if stalls:
                    print(
                        f"LibEMG -> SiFiBridgeStreamer (stream resumed after "
                        f"~{stalls * DATA_STALL_TIMEOUT:.1f}s gap; samples in "
                        f"that window are lost)."
                    )
                stalls = 0
                try:
                    self.process_packet(new_packet)
                except Exception as e:
                    print("Error Occurred: " + str(e))
                continue

            # No packet within the timeout: the link is down. Say so loudly, so a
            # truncated recording is not mistaken for a complete one, and try to
            # get it back rather than waiting on a queue nothing is filling.
            stalls += 1
            print(
                f"LibEMG -> SiFiBridgeStreamer (no data for "
                f"{stalls * DATA_STALL_TIMEOUT:.1f}s, attempting recovery)."
            )
            self._drain_bridge_diagnostics()
            self._recover_stream(stalls)
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
        # Each step is attempted independently: shutdown often runs with the link
        # already down, and letting an early failure propagate would skip the
        # shared-memory release and leak the segments past process exit.
        # Every callable is wrapped so attribute lookup happens inside the try
        # too: a bridge that never finished starting has no _bridge to resolve,
        # and that lookup failing here would skip the steps after it.
        steps = (
            (lambda: self.stop_sampling(), "sampling stopped"),
            (lambda: self.deep_sleep(), "device sleeped"),  # stops status packets
            (lambda: self.disconnect(), "device disconnected"),
            (lambda: self.sb._bridge.kill(), "bridge killed"),
            (lambda: self.smm.cleanup(), "SMM cleaned up"),
        )
        for step, message in steps:
            try:
                step()
                print(f"LibEMG -> SiFiBridgeStreamer ({message}).")
            except Exception as e:
                print(f"LibEMG -> SiFiBridgeStreamer ({message} failed: {e}).")
