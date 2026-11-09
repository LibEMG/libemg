import threading
from unittest.mock import Mock

import numpy as np

import libemg.data_handler as data_handler
from libemg.shared_memory_manager import SharedMemoryManager


class _LateArrival(SharedMemoryManager):
    """A manager backed by ordinary arrays, whose samples land after the baseline.

    The spawned logger re-attaches to shared memory, reads the counters to fix
    its starting point, and only then begins polling. Delivering the samples on
    that baseline read puts them on the far side of it, where the device would
    have put them.
    """

    def __init__(self, tag, samples, capacity):
        super().__init__()
        self.tag = tag
        self.samples = samples
        lock = threading.Lock()
        shape = (capacity, samples.shape[1])
        self.variables[tag] = {"data": np.zeros(shape), "shape": shape,
                               "type": np.double, "smh": None, "lock": lock}
        self.variables[tag + "_count"] = {"data": np.zeros((1, 1), dtype=np.int32),
                                          "shape": (1, 1), "type": np.int32,
                                          "smh": None, "lock": lock}

    def find_variable(self, *_args):
        return True

    def get_variable(self, tag):
        baseline = super().get_variable(tag)
        self.variables[self.tag]["data"][:len(self.samples)] = self.samples
        self.variables[self.tag + "_count"]["data"][0, 0] = len(self.samples)
        return baseline


def test_log_to_file_restores_chronological_sample_order(tmp_path, monkeypatch):
    handler = object.__new__(data_handler.OnlineDataHandler)
    handler.shared_memory_items = [["ecg", (5, 2), np.double, Mock()]]
    handler.modalities = ["ecg"]
    handler.channel_mask = None
    handler.file_path = str(tmp_path / "recording_")
    handler.timestamps = False
    handler.delimiter = ","
    handler.log_signal = Mock()
    handler.log_signal.is_set.return_value = True

    # Newest first, as the streamers write it.
    newest_first = np.array([[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]])
    monkeypatch.setattr(data_handler, "SharedMemoryManager",
                        lambda: _LateArrival("ecg", newest_first, capacity=5))

    handler._log_to_file()

    logged = np.loadtxt(tmp_path / "recording_ecg.csv", delimiter=",")
    np.testing.assert_array_equal(
        logged,
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
    )
