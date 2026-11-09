"""Coverage for the thread-backed logger used by screen guided training."""
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

import libemg.data_handler as data_handler
from libemg.shared_memory_manager import SharedMemoryManager


class _FakeStream:
    """A newest-first shared-memory buffer fed by a background writer.

    Backed by a real SharedMemoryManager whose variables point at ordinary
    arrays rather than shared pages, so the logger exercises the same read path
    it uses against a live streamer.
    """

    def __init__(self, channels=2, capacity=64, modality="emg"):
        self.capacity = capacity
        self.modality = modality
        self.next_sample = 1.0
        lock = threading.Lock()
        self.smm = SharedMemoryManager()
        self.smm.variables[modality] = {
            "data": np.zeros((capacity, channels)), "shape": (capacity, channels),
            "type": np.double, "smh": None, "lock": lock}
        self.smm.variables[modality + "_count"] = {
            "data": np.zeros((1, 1), dtype=np.int32), "shape": (1, 1),
            "type": np.int32, "smh": None, "lock": lock}

    def push(self, n=1):
        # Samples first, then the counter, in the order the streamers write them.
        buffer = self.smm.variables[self.modality]
        counter = self.smm.variables[self.modality + "_count"]
        for _ in range(n):
            with buffer["lock"]:
                buffer["data"][:] = np.roll(buffer["data"], 1, axis=0)
                buffer["data"][0, :] = self.next_sample
            self.next_sample += 1.0
            with counter["lock"]:
                counter["data"][0, 0] += 1


def _handler(stream, modality="emg"):
    handler = object.__new__(data_handler.OnlineDataHandler)
    handler.modalities = [modality]
    handler.smm = stream.smm
    handler.channel_mask = None
    handler._log_process = None
    handler._log_thread = None
    handler._log_stop_event = None
    handler._log_counts = {}
    handler.log_dropped = {}
    handler.log_signal = Mock()
    return handler


def test_start_log_only_captures_samples_from_the_logging_window(tmp_path):
    stream = _FakeStream()
    handler = _handler(stream)
    prefix = str(tmp_path / "C_0_R_0_")

    stream.push(5)                      # before the rep: must not be logged
    handler.start_log(file_path=prefix, timestamps=False, poll_delay=0.001)
    stream.push(4)
    time.sleep(0.05)
    counts = handler.stop_log()
    stream.push(3)                      # after the rep: must not be logged
    time.sleep(0.05)

    logged = np.loadtxt(prefix + "emg.csv", delimiter=",")
    np.testing.assert_array_equal(logged, np.tile(np.arange(6.0, 10.0), (2, 1)).T)
    assert counts == {"emg": 4}


def test_stop_log_drains_samples_that_arrived_just_before_the_stop(tmp_path):
    stream = _FakeStream()
    handler = _handler(stream)
    prefix = str(tmp_path / "rep_")

    handler.start_log(file_path=prefix, timestamps=False, poll_delay=0.5)
    stream.push(3)
    counts = handler.stop_log()         # no poll has run yet at this point

    logged = np.loadtxt(prefix + "emg.csv", delimiter=",")
    np.testing.assert_array_equal(logged, np.tile(np.arange(1.0, 4.0), (2, 1)).T)
    assert counts == {"emg": 3}
    assert handler._log_thread is None


def test_start_log_writes_a_timestamp_column_when_asked(tmp_path):
    stream = _FakeStream(channels=1)
    handler = _handler(stream)
    prefix = str(tmp_path / "ts_")

    before = time.time()
    handler.start_log(file_path=prefix, timestamps=True, poll_delay=0.001)
    stream.push(2)
    time.sleep(0.05)
    handler.stop_log()
    after = time.time()

    logged = np.loadtxt(prefix + "emg.csv", delimiter=",", ndmin=2)
    assert logged.shape == (2, 2)
    assert np.all((logged[:, 0] >= before) & (logged[:, 0] <= after))
    np.testing.assert_array_equal(logged[:, 1], [1.0, 2.0])


def test_logger_reports_samples_lost_to_buffer_wraparound(tmp_path, capsys):
    stream = _FakeStream(capacity=8)
    handler = _handler(stream)
    prefix = str(tmp_path / "overrun_")

    handler.start_log(file_path=prefix, timestamps=False, poll_delay=0.5)
    stream.push(20)                     # more than the 8-sample buffer holds
    counts = handler.stop_log()

    assert counts == {"emg": 8}
    assert handler.log_dropped == {"emg": 12}
    assert "were overwritten in shared memory" in capsys.readouterr().out
    logged = np.loadtxt(prefix + "emg.csv", delimiter=",")
    np.testing.assert_array_equal(logged[:, 0], np.arange(13.0, 21.0))


def test_stop_log_without_a_logger_still_signals_the_spawned_process():
    handler = _handler(_FakeStream())

    assert handler.stop_log() == {}
    handler.log_signal.set.assert_called_once_with()
    handler.log_signal.clear.assert_called_once_with()


def test_stop_log_of_a_thread_logger_does_not_wait_on_the_process_flag(tmp_path):
    handler = _handler(_FakeStream())
    handler.start_log(file_path=str(tmp_path / "quick_"), timestamps=False, poll_delay=0.001)

    started = time.perf_counter()
    handler.stop_log()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, "stop_log must not pay the process-logger handshake"
    handler.log_signal.set.assert_not_called()
