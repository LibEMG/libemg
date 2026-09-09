from multiprocessing import Process, Event
import time
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
# Seconds to let a quiet link settle before trying to recover it. Windows goes
# on reporting the device connected for several seconds after it stops sending,
# while the GATT objects behind that connection are already closed; a connect
# inside that window hands service discovery those stale handles, which fails
# with E_ILLEGAL_METHOD_CALL and leaves a session that reports connected and
# delivers nothing. Recovery cannot succeed sooner than this, so waiting costs
# nothing and avoids poisoning the attempt that can.
STALL_GRACE_PERIOD = 8.0
# How long one recovery attempt waits for the device to start advertising again.
# Having dropped the link, this device stays dark for a minute or more; there is
# nothing to connect to until it is back.
ADVERTISING_WAIT_TIMEOUT = 180.0
# Seconds between scans while waiting for the device to reappear.
ADVERTISING_POLL_INTERVAL = 5.0
# Seconds between reports of samples the device could not deliver. Losses tend
# to be continuous rather than one-off, so this is summarised, not printed per
# packet.
LOSS_REPORT_INTERVAL = 10.0
# Fraction of the configured sampling rate a modality has to reach before its
# throughput is treated as healthy. Loose enough to ignore packetisation jitter
# and the ramp-up of the first seconds, tight enough to catch a modality
# arriving at a fraction of what was asked for.
STREAM_RATE_TOLERANCE = 0.9
# Highest 8-channel EMG rate a BioArmband BLE link was measured to carry without
# loss. Above this the device samples faster than the link can drain: sifibridge
# fills the shortfall with empty rows and counts them as samples_lost, so the
# configured rate is still reported while a growing fraction of the signal is
# simply absent. Measured on this device family at 500/1000/1600/2000 Hz --
# lossless at 1000 and below, ~35% short at 1600, ~40% at 2000. It is a property
# of the link, not a hardware limit, so it is a warning rather than a cap.
BIOARMBAND_SUSTAINABLE_EMG_FS = 1000
# IMU rate that was observed to deliver no data at all on BioArmband firmware
# v5, and to leave the IMU silent afterwards: once the device has been
# configured to it, no reconfiguration recovers the sensor -- not a plain
# rewrite of the rate, not disabling and re-enabling it, not stepping back down
# through the supported rates, not reconnecting. Only a power cycle does. It is
# a documented-supported value, so it stays accepted, but not silently.
IMU_RATE_KNOWN_TO_SILENCE_THE_SENSOR = 200
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
        Reduce latency by joining packets of different modalities together
        (sifibridge's low-latency mode).
    night_mode : bool
        Turn the device LEDs off during acquisition.
    high_gain : bool
        Use more of the ECG/EMG ADC's dynamic range, at the cost of saturating
        more easily.
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
        streaming:      bool = True,
        night_mode:     bool = False,
        high_gain:      bool = False,
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

        # What arrived and what the device said it could not transmit, per
        # packet type. Tracked so a link delivering less than was configured is
        # visible instead of silently shortening the recording (see
        # _track_stream).
        self._lost_samples = {}
        self._received_samples = {}
        self._stream_started = None
        self._last_loss_report = 0.0

        self.emg_handlers = []
        self.imu_handlers = []
        self.eda_handlers = []
        self.ecg_handlers = []
        self.ppg_handlers = []
        self.temperature_handlers = []

 
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
        self.night_mode = night_mode
        self.high_gain = high_gain
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
        if imu and imu_fs == IMU_RATE_KNOWN_TO_SILENCE_THE_SENSOR:
            print(
                f"LibEMG -> SiFiBridgeStreamer (an IMU rate of {imu_fs} Hz delivered no data "
                "at all on the firmware this was tested against, and left the IMU silent "
                "until the device was power cycled -- reconfiguring it does not bring the "
                "sensor back. Use 100 Hz or lower unless you have confirmed your firmware "
                "handles it.)"
            )
        if emg and self.bioarmband and emg_fs > BIOARMBAND_SUSTAINABLE_EMG_FS:
            print(
                f"LibEMG -> SiFiBridgeStreamer (an EMG rate of {emg_fs} Hz is above the "
                f"{BIOARMBAND_SUSTAINABLE_EMG_FS} Hz this link was measured to carry across 8 "
                "channels; the device will sample faster than it can send and the shortfall "
                "is dropped, not slowed. Watch for the samples-lost report below, and lower "
                "emg_fs if it appears.)"
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
        streaming: bool = True,
        night_mode: bool = False,
        high_gain: bool = False,
    ):
        """Write the whole device configuration, with the device silenced.

        Status updates are turned off for the duration. The device keeps
        streaming status packets across disconnects, and sifibridge syncs its
        view of the device from them; one landing mid-push snapshots whatever
        the device was still running and undoes settings just written. Quiet,
        configure, resume leaves nothing to race.
        """
        self.sb.set_status_updates(False)
        try:
            self._configure_sensors(ecg, emg, eda, imu, ppg, filtering, notch_freq,
                                    emg_bandpass, eda_bandpass, eda_freq, streaming,
                                    night_mode, high_gain)
        finally:
            # Resume even if a command failed, or the device is left mute and
            # nothing downstream can tell why.
            self.sb.set_status_updates(True)

    def _configure_sensors(
        self,
        ecg,
        emg,
        eda,
        imu,
        ppg,
        filtering,
        notch_freq,
        emg_bandpass,
        eda_bandpass,
        eda_freq,
        streaming,
        night_mode,
        high_gain,
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

        # Every device-level setting is written on every connect, including the
        # ones we are happy with the default of. The device retains all of
        # these across disconnects, so anything left unwritten is inherited
        # from whoever configured the device last -- another tool, or an
        # earlier run with different arguments.
        self.sb.set_night_mode(night_mode)
        self.sb.set_high_gain(high_gain)
        # 'streaming' is sifibridge's low-latency mode: the device packs several
        # sensors into each BLE packet instead of sending one packet per sensor.
        self.sb.set_low_latency_mode(streaming)
        self.sb.set_ble_power(sbp.BleTxPower.HIGH)
        self.sb.set_memory_mode(sbp.MemoryMode.STREAMING)

    def connect(self, max_attempts: int | None = None):
        """Join the device, apply the configuration, and begin sampling.

        Parameters
        ----------
        max_attempts : int | None
            Give up after this many failed attempts. None retries until the
            shutdown signal is raised, which is what the initial connection
            wants; recovery bounds it so it can escalate instead of looping.

        Returns
        -------
        bool
            True once connected and sampling. False if the attempt was
            abandoned, either because the shutdown signal was raised or because
            max_attempts was reached, so callers can give up instead of
            retrying forever against a device that is off or out of range.
        """
        attempts = 0
        while True:
            attempts += 1
            try:
                if self.sb.connect(self.handle):
                    break
                reason = "the device did not accept the connection"
            except sbp.SifiBridgeError as e:
                # sifibridge reports "could not find device" and "Already
                # connected" as errors rather than as a False return. Letting
                # those propagate kills the streamer process outright, which is
                # the opposite of what this retry loop exists for.
                reason = str(e)
            # Without this check a device that is off or out of range wedges the
            # process in a tight retry loop that ignores cleanup requests.
            if self.signal.is_set():
                print("LibEMG -> SiFiBridgeStreamer (connect abandoned, stopping).")
                return False
            if max_attempts is not None and attempts >= max_attempts:
                print(f"LibEMG -> SiFiBridgeStreamer (connect failed: {reason}).")
                return False
            print(f"Could not connect to {self.handle} ({reason}). Retrying.")

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
            self.night_mode,
            self.high_gain,
        )

        self.sb.stop()
        self.sb.start()
        self.sb.clear_data_buffer()
        self._reset_stream_tracking()
        return True

    def _rebuild_bridge(self):
        """Replace the sifibridge subprocess. Does not connect.

        A BLE-level reconnect is not enough. ``SifiBridge`` opens its data socket
        and starts the thread that reads from it once, in its constructor; if
        that socket closes (bridge subprocess died, host dropped the connection)
        the reader thread exits and no packet ever reaches the queue again. And
        on Windows the OS caches the GATT service objects of the dead
        connection, so rediscovery on the same bridge gets back handles that are
        already closed. A fresh process resolves the device from scratch, which
        is the only path measured to restore the stream.
        """
        print("LibEMG -> SiFiBridgeStreamer (rebuilding bridge).")
        try:
            self.sb.close()
        except Exception as e:
            # Already-dead bridges raise here; the replacement matters, not this.
            print(f"LibEMG -> SiFiBridgeStreamer (error closing old bridge: {e}).")
        self.sb = sbp.SifiBridge()
        self.sb._DEFAULT_REQUEST_TIMEOUT = 10.0

    def _wait_for_device(self):
        """Block until a SiFi device is advertising, or shutdown is requested.

        Connect attempts made while the device is dark all fail, each costing a
        scan and a round trip, and they bury the one message that matters in
        noise. Scan results are already filtered to SiFi hardware by the bridge,
        so anything in the list is worth a connect attempt.

        Returns
        -------
        bool
            True if a device is advertising, False on shutdown or timeout.
        """
        deadline = time.time() + ADVERTISING_WAIT_TIMEOUT
        waiting_announced = False
        while time.time() < deadline:
            if self.signal.is_set():
                return False
            try:
                devices = self.sb.list_devices("ble")
            except Exception as e:
                print(f"LibEMG -> SiFiBridgeStreamer (scan failed: {e}).")
                devices = []
            if devices:
                print(
                    "LibEMG -> SiFiBridgeStreamer (device is advertising again after "
                    f"{ADVERTISING_WAIT_TIMEOUT - (deadline - time.time()):.0f}s)."
                )
                return True
            if not waiting_announced:
                waiting_announced = True
                print(
                    "LibEMG -> SiFiBridgeStreamer (waiting for the device to advertise "
                    "again; it stays dark for a while after dropping the link)."
                )
            time.sleep(ADVERTISING_POLL_INTERVAL)
        print(
            "LibEMG -> SiFiBridgeStreamer (no device advertised within "
            f"{ADVERTISING_WAIT_TIMEOUT:.0f}s)."
        )
        return False

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

    def _recover_stream(self):
        """Rebuild the bridge, wait for the device to return, and reconnect.

        The cheap path -- disconnect and reconnect on the existing bridge -- is
        deliberately not attempted. When the link goes quiet without a
        disconnect event, sifibridge still counts the device as connected and
        answers every connect with "Already connected"; once it does notice, the
        OS hands rediscovery the closed GATT objects of the dead connection. In
        neither state can a reconnect on that bridge produce data. So: a fresh
        process, then wait for the device to advertise, then connect.

        Returns
        -------
        bool
            Whether the stream is expected to flow again.
        """
        self.connected = False
        try:
            self._rebuild_bridge()
            if not self._wait_for_device():
                return False
            # Several attempts: the scan only proves some SiFi device is back,
            # not that this one has finished settling.
            return self.connect(max_attempts=3)
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

    def add_temperature_handler(self, closure: Callable):
        self.temperature_handlers.append(closure)

    def _reset_stream_tracking(self):
        """Start the throughput accounting over for a newly opened acquisition.

        Rates are measured against the time the stream has been up. Carrying the
        counters across a reconnect would divide one connection's samples by a
        span that includes the outage before it, reporting a healthy link as
        badly short of its configured rate.
        """
        self._lost_samples = {}
        self._received_samples = {}
        # Left unset: the clock starts when data does, not when connect()
        # returns. The gap between the two would otherwise be charged to the
        # link as a rate shortfall on every reconnect.
        self._stream_started = None
        self._last_loss_report = 0.0

    def _configured_rate(self, packet_type: str):
        """The sampling rate we asked for, for a given packet type, or None."""
        return {
            "emg_armband": self.emg_fs,
            "emg": self.emg_fs,
            "ecg": self.ecg_fs,
            "eda": self.eda_fs,
            "imu": self.imu_fs,
            "ppg": self.ppg_sps / self.ppg_avg if self.ppg_avg else self.ppg_sps,
        }.get(packet_type)

    def _track_stream(self, packet_type: str, received: int, lost: int):
        """Accumulate what actually arrived, and periodically say how it compares.

        Two different things go wrong quietly here. sifibridge reports
        ``samples_lost`` when the link could not carry what the device produced,
        and leaves None-filled rows in their place; those rows are dropped
        below, which closes the gap, so the samples either side of it end up
        adjacent and the recording is shortened rather than marked. And a
        modality can simply arrive at a fraction of its configured rate with no
        loss reported at all, which nothing else here would notice. Comparing
        what arrived against what was asked for catches both.
        """
        if received:
            self._received_samples[packet_type] = (
                self._received_samples.get(packet_type, 0) + received
            )
        if lost > 0:
            self._lost_samples[packet_type] = self._lost_samples.get(packet_type, 0) + lost
        now = time.time()
        if self._stream_started is None:
            self._stream_started = now
            self._last_loss_report = now
            return
        if now - self._last_loss_report < LOSS_REPORT_INTERVAL:
            return
        elapsed = now - self._stream_started
        if elapsed < LOSS_REPORT_INTERVAL:
            # Too early to judge a rate; the first packets arrive in a burst.
            return
        problems = []
        for kind, total in sorted(self._received_samples.items()):
            configured = self._configured_rate(kind)
            if not configured:
                continue
            achieved = total / elapsed
            missing = self._lost_samples.get(kind, 0)
            # Judged on the achieved rate alone. Dropped samples already show up
            # as a lower rate, so triggering on the loss counter as well would
            # report a link running at 99% every ten seconds for a handful of
            # samples, and bury the shortfalls that matter.
            if achieved >= STREAM_RATE_TOLERANCE * configured:
                continue
            note = f"{missing} dropped in transit" if missing else "nothing reported lost"
            problems.append(
                f"{kind} {achieved:.0f}/{configured:g} Hz "
                f"({100 * achieved / configured:.0f}%, {note})"
            )
        if not problems:
            return
        self._last_loss_report = now
        print(
            "LibEMG -> SiFiBridgeStreamer (receiving less than configured: "
            + "; ".join(problems)
            + "). Samples that never arrive are absent from the recording rather "
            "than marked, so lower the sampling rate or disable modalities if "
            "every sample matters."
        )

    def process_packet(self, data: dict):
        if "data" in list(data.keys()):
            packet_type = str(data.get("packet_type", "unknown"))
            received = max(
                (len(v) for v in data["data"].values() if isinstance(v, list)),
                default=0,
            )
            self._track_stream(
                packet_type, received, int(data.get("samples_lost") or 0)
            )
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
                    # A non-float dtype means the packet carried None entries:
                    # the placeholders sifibridge leaves for samples lost in
                    # transit. Drop those rows (reported by _track_stream)
                    # while preserving 2D structure.
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
            if "temperature" in list(data["data"].keys()):
                # Skin temperature rides along in the device's status packet
                # (alongside battery and memory use) rather than in a stream of
                # its own, so it arrives here whatever modalities are enabled.
                temperature = np.expand_dims(
                    np.array(data["data"]["temperature"], dtype=np.double), 0
                ).T
                for h in self.temperature_handlers:
                    h(temperature)

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

        def write_temperature(temperature):
            # update the samples in "temperature"
            self.smm.modify_variable(
                "temperature",
                lambda x: np.vstack((np.flip(temperature, 0), x))[: x.shape[0], :],
            )
            # update the number of samples retrieved
            self.smm.modify_variable(
                "temperature_count", lambda x: x + temperature.shape[0]
            )

        # Unlike every other modality, status packets arrive whether or not the
        # caller asked for temperature, so the handler is only wired up when a
        # buffer was allocated for it -- otherwise the first status packet would
        # write to a variable that does not exist.
        if "temperature" in self.smm.variables:
            self.add_temperature_handler(write_temperature)

        self.old_ppg_packet = (
            None  # required for now since ppg sends non-uniform packet length
        )
        # When the stream first went quiet, or None while data is flowing.
        stalled_since = None
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
                new_packet = None

            if new_packet:
                if stalled_since is not None:
                    print(
                        f"LibEMG -> SiFiBridgeStreamer (stream resumed after "
                        f"~{time.time() - stalled_since:.1f}s gap; samples in "
                        f"that window are lost)."
                    )
                    stalled_since = None
                try:
                    self.process_packet(new_packet)
                except Exception as e:
                    print("Error Occurred: " + str(e))
                continue

            now = time.time()
            if stalled_since is None:
                stalled_since = now
                continue
            quiet_for = now - stalled_since
            if quiet_for < STALL_GRACE_PERIOD:
                # Still inside the window where the link looks alive but every
                # object behind it is dead. Keep reading: a brief hiccup
                # resolves itself here, and a real drop cannot be repaired yet.
                continue

            # Say so loudly, so a truncated recording is not mistaken for a
            # complete one, then go and get the stream back.
            print(
                f"LibEMG -> SiFiBridgeStreamer (no data for {quiet_for:.1f}s, "
                "attempting recovery)."
            )
            self._drain_bridge_diagnostics()
            if self._recover_stream():
                stalled_since = None
            else:
                # Restart the clock so the next attempt is a grace period away
                # rather than immediate.
                stalled_since = time.time()
        print("LibEMG -> SiFiBridgeStreamer (process ended).")

    def stop_sampling(self):
        self.sb.stop()
        return

    def turnoff(self):
        """Power the device off."""
        self.sb.power_off()
        return

    def disconnect(self):
        """Drop the BLE link and the sifibridge session that holds it."""
        # sifi_bridge_py returns the connection state directly here.
        self.connected = self.sb.disconnect()
        return self.connected

    def cleanup(self):
        # Each step is attempted independently: shutdown often runs with the link
        # already down, and letting an early failure propagate would skip the
        # shared-memory release and leak the segments past process exit.
        # Every callable is wrapped so attribute lookup happens inside the try
        # too: a bridge that never finished starting has no _bridge to resolve,
        # and that lookup failing here would skip the steps after it.
        # Disconnecting is what stops the device streaming, and it has to happen
        # before the bridge goes away: killing sifibridge while the link is up
        # leaves the device connected to a process that no longer exists, and
        # the next session then meets an "Already connected" it cannot clear.
        # close() sends 'quit' and waits, rather than killing outright.
        steps = (
            (lambda: self.stop_sampling(), "sampling stopped"),
            (lambda: self.disconnect(), "device disconnected"),
            (lambda: self.sb.close(), "bridge closed"),
            (lambda: self.smm.cleanup(), "SMM cleaned up"),
        )
        for step, message in steps:
            try:
                step()
                print(f"LibEMG -> SiFiBridgeStreamer ({message}).")
            except Exception as e:
                print(f"LibEMG -> SiFiBridgeStreamer ({message} failed: {e}).")
