import queue
import time
from multiprocessing import Event
from unittest.mock import Mock, call

import numpy as np
import pytest

from libemg._streamers import _sifi_bridge_streamer as sbs
from libemg._streamers._sifi_bridge_streamer import SiFiBridgeStreamer
from libemg.shared_memory_manager import assign_shared_memory_locks


class FakeBridge:
    """A sifibridge whose data stream dies after a fixed number of packets.

    Reproduces the real failure mode: once the thread feeding the packet queue
    has exited, ``get_data`` reports the outage by returning ``{}`` on timeout
    rather than by raising, so nothing is left for an ``except`` to catch.
    """

    def __init__(self, packets_before_death, packets_after_recovery=0):
        self.remaining = packets_before_death
        self.packets_after_recovery = packets_after_recovery
        self.reads = 0
        self.timeouts_seen = []
        self.recovered = False
        # A real queue, not the catch-all Mock from __getattr__: a Mock never
        # raises Empty, so a drain loop reading one would never terminate.
        self._stderr_queue = queue.Queue()

    def get_data(self, timeout=None):
        self.reads += 1
        self.timeouts_seen.append(timeout)
        if self.remaining > 0:
            self.remaining -= 1
            return {"data": {"emg": [0.0, 1.0]}}
        if self.recovered and self.packets_after_recovery > 0:
            self.packets_after_recovery -= 1
            return {"data": {"emg": [2.0, 3.0]}}
        return {}

    def connect(self, handle):
        return True

    def __getattr__(self, name):
        # configure_*/start/stop/clear_data_buffer/_bridge and friends are all
        # irrelevant to the read loop under test.
        mock = Mock()
        object.__setattr__(self, name, mock)
        return mock


class FakeSmm:
    """Real numpy buffers so the writers' prepend arithmetic actually runs."""

    def __init__(self):
        # Named to match SharedMemoryManager: the streamer asks whether a
        # modality was allocated by looking in .variables.
        self.variables = {}

    def create_variable(self, tag, shape, type, lock=None):
        self.variables[tag] = np.zeros(shape, dtype=type)

    def modify_variable(self, tag, fn):
        self.variables[tag][:] = fn(self.variables[tag])

    def commit(self, tag, samples, count_tag=None):
        """The prepend-and-count contract of SharedMemoryManager.commit."""
        samples = np.atleast_2d(samples)
        if count_tag is None:
            candidate = tag + "_count"
            count_tag = candidate if candidate in self.variables else None
        data = self.variables[tag]
        capacity, arriving = data.shape[0], samples.shape[0]
        incoming = np.flip(samples, 0)
        if arriving >= capacity:
            data[:] = incoming[:capacity]
        else:
            data[arriving:] = data[:capacity - arriving]
            data[:arriving] = incoming
        if count_tag is not None and count_tag in self.variables:
            self.variables[count_tag][:] += arriving

    def cleanup(self):
        pass

    def emg_samples(self):
        return int(self.variables["emg_count"][0, 0])


def build(monkeypatch, bridge):
    """Wire a streamer up to fakes so run() executes without hardware."""
    monkeypatch.setattr(sbs.sbp, "SifiBridge", Mock(return_value=bridge))
    smm = FakeSmm()
    monkeypatch.setattr(sbs, "SharedMemoryManager", Mock(return_value=smm))

    items = [["emg", (4000, 1), np.double], ["emg_count", (1, 1), np.int32]]
    assign_shared_memory_locks(items)

    streamer = SiFiBridgeStreamer(name="BioPoint_v1_3", shared_memory_items=items)
    streamer.signal = Event()
    return streamer, smm


def test_reads_are_bounded_so_a_dead_link_cannot_park_the_process(monkeypatch):
    bridge = FakeBridge(packets_before_death=1)
    streamer, smm = build(monkeypatch, bridge)
    monkeypatch.setattr(sbs, "STALL_GRACE_PERIOD", 0.0)
    streamer._recover_stream = Mock(side_effect=lambda: streamer.signal.set())

    streamer.run()

    assert bridge.timeouts_seen, "the loop never read from the bridge"
    assert all(t == sbs.DATA_STALL_TIMEOUT for t in bridge.timeouts_seen), (
        "an unbounded read blocks forever on a queue nothing is filling"
    )
    assert smm.emg_samples() == 2


def test_stalled_stream_recovers_and_keeps_collecting(monkeypatch):
    bridge = FakeBridge(packets_before_death=2, packets_after_recovery=1)
    streamer, smm = build(monkeypatch, bridge)
    monkeypatch.setattr(sbs, "STALL_GRACE_PERIOD", 0.0)

    attempts = []

    def recover():
        attempts.append(1)
        bridge.recovered = True
        if len(attempts) > 1:
            streamer.signal.set()
        return True

    streamer._recover_stream = recover
    streamer.run()

    assert attempts, "a stalled stream must attempt recovery"
    # 2 packets x 2 samples before the drop, plus 1 packet x 2 samples after.
    assert smm.emg_samples() == 6


def test_shutdown_signal_is_honoured_while_the_stream_is_stalled(monkeypatch):
    """The original loop checked the signal only after an unbounded read."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, smm = build(monkeypatch, bridge)
    streamer.cleanup = Mock()
    streamer.signal.set()

    streamer.run()

    streamer.cleanup.assert_called_once_with()
    assert bridge.reads == 0, "must not read once shutdown has been requested"


def test_connect_abandons_its_retry_loop_on_shutdown(monkeypatch):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    bridge.connect = Mock(return_value=False)
    streamer.signal.set()

    assert streamer.connect() is False


def test_recovery_rebuilds_then_waits_for_the_device_then_connects(monkeypatch):
    """A reconnect on the old bridge cannot work: sifibridge still counts the
    device as connected, and once it notices, Windows hands rediscovery the
    closed GATT objects of the dead connection."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    order = []
    streamer._rebuild_bridge = Mock(side_effect=lambda: order.append("rebuild"))
    streamer._wait_for_device = Mock(side_effect=lambda: order.append("wait") or True)
    streamer.connect = Mock(side_effect=lambda **_kw: order.append("connect") or True)

    assert streamer._recover_stream() is True
    assert order == ["rebuild", "wait", "connect"]


def test_recovery_does_not_connect_while_the_device_is_still_dark(monkeypatch):
    """Every connect made while it is not advertising fails, costing a scan and
    a round trip and burying the message that matters."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    streamer._rebuild_bridge = Mock()
    streamer._wait_for_device = Mock(return_value=False)
    streamer.connect = Mock(return_value=True)

    assert streamer._recover_stream() is False
    streamer.connect.assert_not_called()


def test_a_brief_hiccup_is_not_treated_as_a_dropped_link(monkeypatch):
    """Inside the grace window the link looks alive but its objects are dead, so
    recovering there poisons the attempt; data that comes back needs none."""
    bridge = FakeBridge(packets_before_death=1, packets_after_recovery=1)
    bridge.recovered = True  # data returns on its own, without any recovery
    streamer, smm = build(monkeypatch, bridge)
    monkeypatch.setattr(sbs, "STALL_GRACE_PERIOD", 30.0)
    streamer._recover_stream = Mock(side_effect=lambda: streamer.signal.set())

    def stop_after_resume(packet):
        SiFiBridgeStreamer.process_packet(streamer, packet)
        if smm.emg_samples() >= 4:
            streamer.signal.set()

    streamer.process_packet = stop_after_resume
    streamer.run()

    streamer._recover_stream.assert_not_called()
    assert smm.emg_samples() == 4


def test_cleanup_releases_shared_memory_even_when_the_bridge_is_dead(monkeypatch):
    bridge = FakeBridge(packets_before_death=0)
    streamer, smm = build(monkeypatch, bridge)
    streamer.sb = bridge
    streamer.smm = smm
    smm.cleanup = Mock()
    streamer.stop_sampling = Mock(side_effect=RuntimeError("bridge is gone"))
    streamer.disconnect = Mock(side_effect=RuntimeError("bridge is gone"))
    bridge.close = Mock(side_effect=RuntimeError("already dead"))

    streamer.cleanup()

    smm.cleanup.assert_called_once_with()


def test_bridge_stderr_is_surfaced_when_the_stream_stalls(monkeypatch, capsys):
    """The reason for a drop must not stay buried in an undrained queue."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    bridge._stderr_queue.put("BLE peripheral disconnected: ConnectionTimeout\n")

    monkeypatch.setattr(sbs, "STALL_GRACE_PERIOD", 0.0)
    streamer._recover_stream = Mock(side_effect=lambda: streamer.signal.set())
    streamer.run()

    out = capsys.readouterr().out
    assert "ConnectionTimeout" in out
    assert bridge._stderr_queue.empty(), "the queue must be drained, not just read"


def test_cleanup_survives_a_bridge_that_never_finished_starting(monkeypatch):
    """Attribute lookup on a half-built bridge must not skip later steps."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, smm = build(monkeypatch, bridge)

    class NoBridge:
        def __getattr__(self, name):
            raise AttributeError(name)

    streamer.sb = NoBridge()
    streamer.smm = smm
    smm.cleanup = Mock()

    streamer.cleanup()

    smm.cleanup.assert_called_once_with()


def test_cleanup_drops_the_link_before_the_bridge_goes_away(monkeypatch):
    """Killing sifibridge with the link up strands the device on a dead process,
    and the next session then meets an "Already connected" it cannot clear."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, smm = build(monkeypatch, bridge)
    streamer.sb = bridge
    streamer.smm = smm
    order = []
    bridge.stop = Mock(side_effect=lambda *a: order.append("stop"))
    bridge.disconnect = Mock(side_effect=lambda: order.append("disconnect") or False)
    bridge.close = Mock(side_effect=lambda: order.append("close"))

    streamer.cleanup()

    assert order == ["stop", "disconnect", "close"]


def test_disconnect_reads_the_boolean_the_bridge_returns(monkeypatch):
    """sifi_bridge_py returns the connection state directly; subscripting it raises."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    bridge.disconnect = Mock(return_value=False)

    assert streamer.disconnect() is False
    assert streamer.connected is False


def test_configure_applies_the_streaming_flag_as_low_latency_mode(monkeypatch):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge

    streamer.configure(emg=True, streaming=False)
    bridge.set_low_latency_mode.assert_called_once_with(False)

    bridge.set_low_latency_mode.reset_mock()
    streamer.configure(emg=True, streaming=True)
    bridge.set_low_latency_mode.assert_called_once_with(True)


def _stream(streamer, packet_type, packets, rows, lost=0):
    for _ in range(packets):
        streamer.process_packet({
            "packet_type": packet_type,
            "samples_lost": lost,
            "data": {"ch": [0.0] * rows},
        })


def test_a_modality_arriving_short_of_its_configured_rate_is_reported(monkeypatch, capsys):
    """Samples that never arrive are absent from the recording rather than
    marked, so nothing downstream can tell the rate was not met."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.imu_fs = 50
    streamer._stream_started = time.time() - 20.0

    _stream(streamer, "imu", packets=1, rows=500)  # 500 samples in 20s = 25 Hz

    out = capsys.readouterr().out
    assert "receiving less than configured" in out
    assert "imu 25/50 Hz (50%" in out
    assert "nothing reported lost" in out


def test_a_modality_meeting_its_rate_is_not_reported(monkeypatch, capsys):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.ecg_fs = 500
    streamer._stream_started = time.time() - 20.0

    _stream(streamer, "ecg", packets=1, rows=10000)  # 10000 in 20s = 500 Hz

    assert capsys.readouterr().out == ""


def test_samples_lost_in_transit_are_named_as_such(monkeypatch, capsys):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.emg_fs = 1600
    streamer._stream_started = time.time() - 20.0

    _stream(streamer, "emg_armband", packets=1, rows=20000, lost=8000)

    out = capsys.readouterr().out
    assert "8000 dropped in transit" in out
    assert streamer._lost_samples["emg_armband"] == 8000


def test_stream_health_is_summarised_rather_than_printed_per_packet(monkeypatch, capsys):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.emg_fs = 1600
    streamer._stream_started = time.time() - 20.0

    _stream(streamer, "emg_armband", packets=50, rows=10, lost=512)

    assert capsys.readouterr().out.count("receiving less than configured") == 1
    assert streamer._lost_samples["emg_armband"] == 50 * 512


def test_no_report_before_there_is_enough_data_to_judge_a_rate(monkeypatch, capsys):
    """The first packets arrive in a burst; judging a rate off them is noise."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.emg_fs = 1600
    streamer._stream_started = time.time()

    _stream(streamer, "emg_armband", packets=5, rows=1, lost=512)

    assert capsys.readouterr().out == ""


def test_connect_retries_when_sifibridge_reports_an_error(monkeypatch, capsys):
    """sifibridge raises for "could not find device" instead of returning False;
    letting that propagate kills the streamer the retry loop exists to protect."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    attempts = []

    def flaky(handle):
        attempts.append(handle)
        if len(attempts) < 3:
            raise sbs.sbp.SifiBridgeError("Failed to connect: could not find device")
        return True

    bridge.connect = flaky

    assert streamer.connect() is True
    assert len(attempts) == 3
    assert "could not find device" in capsys.readouterr().out


def test_connect_gives_up_after_max_attempts(monkeypatch):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    bridge.connect = Mock(side_effect=sbs.sbp.SifiBridgeError("Already connected"))

    assert streamer.connect(max_attempts=1) is False
    assert bridge.connect.call_count == 1


def test_device_is_silenced_while_it_is_configured(monkeypatch):
    """sifibridge syncs its view of the device from status packets; one landing
    mid-push undoes settings just written, so configuration runs with the device
    quiet and status updates come back after."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    order = []
    bridge.set_status_updates = Mock(
        side_effect=lambda on: order.append(f"status={on}"))
    bridge.configure_sensors = Mock(side_effect=lambda *a: order.append("sensors"))
    bridge.set_low_latency_mode = Mock(side_effect=lambda v: order.append("low_latency"))

    streamer.configure(emg=True)

    assert order[0] == "status=False", "configured before silencing the device"
    assert order[-1] == "status=True", "left the device silenced"
    assert "sensors" in order and "low_latency" in order


def test_status_updates_resume_even_if_configuration_fails(monkeypatch):
    """A device left mute reports nothing about why the next step went wrong."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    bridge.set_status_updates = Mock()
    bridge.configure_sensors = Mock(side_effect=RuntimeError("bridge said no"))

    with pytest.raises(RuntimeError):
        streamer.configure(emg=True)

    assert bridge.set_status_updates.call_args_list[-1] == call(True)


def test_every_device_level_setting_is_written_on_connect(monkeypatch):
    """The device retains these across disconnects, so anything left unwritten
    is inherited from whoever configured it last."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge

    streamer.configure(emg=True, streaming=True, night_mode=True, high_gain=True)

    bridge.set_night_mode.assert_called_once_with(True)
    bridge.set_high_gain.assert_called_once_with(True)
    bridge.set_low_latency_mode.assert_called_once_with(True)
    bridge.set_ble_power.assert_called_once()
    bridge.set_memory_mode.assert_called_once()


def test_disabling_the_mains_notch_is_spelled_out_rather_than_left_unset(monkeypatch):
    """sifi_bridge_py 2.0.1 reassigned None on every configure_* parameter to
    mean "leave the device as it is". Our notch_freq of None means "no mains
    notch", so passing it straight through would inherit the previous session's
    filter instead of turning it off."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge

    streamer.configure(ecg=True, emg=True, eda=True, notch_freq=None)

    for configure in (bridge.configure_ecg, bridge.configure_emg,
                      bridge.configure_eda):
        assert configure.call_args.kwargs["mains_notch"] == "off"


def test_a_mains_notch_frequency_is_passed_through_unchanged(monkeypatch):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge

    streamer.configure(emg=True, notch_freq=50)

    assert bridge.configure_emg.call_args.kwargs["mains_notch"] == 50


def test_a_ppg_rate_the_link_cannot_carry_is_rejected(monkeypatch):
    """sifi_bridge_py 2.0.1 refuses an effective PPG rate above 200 Hz inside
    configure_ppg, which would surface in the streamer subprocess after
    connecting. Rejecting it here keeps the failure where the caller is."""
    monkeypatch.setattr(sbs.sbp, "SifiBridge", Mock())
    with pytest.raises(ValueError, match="exceeds the maximum"):
        SiFiBridgeStreamer(name="BioPoint_v1_3", shared_memory_items=[],
                           ppg=True, ppg_sps=800, ppg_avg=2)

    SiFiBridgeStreamer(name="BioPoint_v1_3", shared_memory_items=[],
                       ppg=True, ppg_sps=800, ppg_avg=4)


def test_an_emg_rate_the_link_cannot_carry_is_flagged(monkeypatch, capsys):
    """Above the link's ceiling the device samples faster than it can send and
    the shortfall is dropped, while the configured rate is still reported."""
    monkeypatch.setattr(sbs.sbp, "SifiBridge", Mock())
    SiFiBridgeStreamer(name="BioArmband", shared_memory_items=[], emg=True,
                       emg_fs=1600, bioarmband=True)
    assert "above the" in capsys.readouterr().out

    SiFiBridgeStreamer(name="BioArmband", shared_memory_items=[], emg=True,
                       emg_fs=sbs.BIOARMBAND_SUSTAINABLE_EMG_FS, bioarmband=True)
    assert capsys.readouterr().out == ""


def test_a_biopoint_is_not_warned_about_the_armband_ceiling(monkeypatch, capsys):
    """The ceiling is for 8 armband channels; a single-channel BioPoint is a
    fraction of that throughput."""
    monkeypatch.setattr(sbs.sbp, "SifiBridge", Mock())
    SiFiBridgeStreamer(name="BioPoint_v1_3", shared_memory_items=[], emg=True,
                       emg_fs=2000, bioarmband=False)
    assert capsys.readouterr().out == ""


def test_the_imu_rate_that_silences_the_sensor_is_flagged(monkeypatch, capsys):
    """It delivers nothing and no reconfiguration brings the sensor back, so
    walking into it unwarned costs a power cycle to notice and undo."""
    monkeypatch.setattr(sbs.sbp, "SifiBridge", Mock())
    SiFiBridgeStreamer(name="BioArmband", shared_memory_items=[], emg=False, imu=True,
                       imu_fs=sbs.IMU_RATE_KNOWN_TO_SILENCE_THE_SENSOR, bioarmband=True)
    assert "left the IMU silent" in capsys.readouterr().out

    SiFiBridgeStreamer(name="BioArmband", shared_memory_items=[], emg=False, imu=True,
                       imu_fs=100, bioarmband=True)
    assert capsys.readouterr().out == ""


def test_the_imu_rate_is_not_flagged_when_the_imu_is_off(monkeypatch, capsys):
    monkeypatch.setattr(sbs.sbp, "SifiBridge", Mock())
    SiFiBridgeStreamer(name="BioArmband", shared_memory_items=[], emg=False, imu=False,
                       imu_fs=sbs.IMU_RATE_KNOWN_TO_SILENCE_THE_SENSOR, bioarmband=True)
    assert capsys.readouterr().out == ""


def test_wait_for_device_returns_once_something_advertises(monkeypatch):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    monkeypatch.setattr(sbs, "ADVERTISING_POLL_INTERVAL", 0.0)
    scans = [[], [], [{"name": "SifiBand_59BC", "id": "E7:97:46:57:1F:5C"}]]
    bridge.list_devices = Mock(side_effect=lambda src: scans.pop(0))

    assert streamer._wait_for_device() is True
    assert bridge.list_devices.call_count == 3


def test_wait_for_device_gives_up_rather_than_blocking_forever(monkeypatch):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    monkeypatch.setattr(sbs, "ADVERTISING_POLL_INTERVAL", 0.0)
    monkeypatch.setattr(sbs, "ADVERTISING_WAIT_TIMEOUT", 0.2)
    bridge.list_devices = Mock(return_value=[])

    assert streamer._wait_for_device() is False


def test_wait_for_device_stops_on_shutdown(monkeypatch):
    """Shutdown must not have to wait out the advertising timeout."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    monkeypatch.setattr(sbs, "ADVERTISING_POLL_INTERVAL", 0.0)
    bridge.list_devices = Mock(side_effect=lambda src: streamer.signal.set() or [])

    assert streamer._wait_for_device() is False
    assert bridge.list_devices.call_count == 1


def test_a_failed_scan_does_not_abort_the_wait(monkeypatch):
    """A scan can fail while the bridge is still settling after a rebuild."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    monkeypatch.setattr(sbs, "ADVERTISING_POLL_INTERVAL", 0.0)
    results = [RuntimeError("adapter busy"), [{"name": "SifiBand_59BC"}]]

    def scan(_src):
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    bridge.list_devices = scan
    assert streamer._wait_for_device() is True


def test_reconnecting_restarts_the_throughput_accounting(monkeypatch):
    """Carrying the counters across an outage divides one connection's samples
    by a span that includes the downtime, flagging a healthy link as short."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    streamer._stream_started = time.time() - 300.0
    streamer._received_samples = {"emg_armband": 5}
    streamer._lost_samples = {"emg_armband": 900}

    streamer.connect()

    assert streamer._received_samples == {}
    assert streamer._lost_samples == {}
    assert streamer._stream_started is None, "the clock must start with the data"


def test_a_healthy_link_is_not_flagged_after_an_outage(monkeypatch, capsys):
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    streamer.ecg_fs = 500
    streamer.connect()                       # as a recovery would
    streamer._stream_started = time.time() - 20.0
    streamer._last_loss_report = streamer._stream_started
    capsys.readouterr()

    _stream(streamer, "ecg", packets=1, rows=10000)   # a full 500 Hz for 20s

    assert capsys.readouterr().out == ""


def test_the_rate_clock_starts_with_the_first_packet(monkeypatch, capsys):
    """Charging the connect-to-first-packet gap to the link reports a healthy
    connection as short of its rate every time it reconnects."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.sb = bridge
    streamer.ecg_fs = 500
    streamer._reset_stream_tracking()

    before = time.time()
    _stream(streamer, "ecg", packets=1, rows=1)
    assert streamer._stream_started >= before
    assert capsys.readouterr().out == "", "judged a rate off the very first packet"


def test_a_trickle_of_loss_on_an_otherwise_healthy_link_is_not_reported(monkeypatch, capsys):
    """Dropped samples already show up as a lower rate; reporting the counter
    too would flag a link running at 99% every ten seconds."""
    bridge = FakeBridge(packets_before_death=0)
    streamer, _ = build(monkeypatch, bridge)
    streamer.emg_fs = 1000
    streamer._stream_started = time.time() - 20.0
    streamer._last_loss_report = streamer._stream_started

    _stream(streamer, "emg_armband", packets=1, rows=19800, lost=200)  # 990 Hz

    assert capsys.readouterr().out == ""
