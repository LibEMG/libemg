"""Coverage for reading only what a writer has newly produced."""
import threading

import numpy as np
import pytest

from libemg.shared_memory_manager import SharedMemoryManager, assign_shared_memory_locks


def _manager(capacity=8, channels=2, shared_lock=True):
    """A manager whose variables are ordinary arrays instead of shared pages."""
    smm = SharedMemoryManager()
    lock = threading.Lock()
    smm.variables["emg"] = {"data": np.zeros((capacity, channels)),
                            "shape": (capacity, channels), "type": np.double,
                            "smh": None, "lock": lock}
    smm.variables["emg_count"] = {"data": np.zeros((1, 1), dtype=np.int32),
                                  "shape": (1, 1), "type": np.int32, "smh": None,
                                  "lock": lock if shared_lock else threading.Lock()}
    return smm


def _push(smm, n, start=1.0):
    """Write n samples newest-first, then bump the counter, as the streamers do."""
    buffer = smm.variables["emg"]["data"]
    for i in range(n):
        buffer[:] = np.roll(buffer, 1, axis=0)
        buffer[0, :] = start + i
    smm.variables["emg_count"]["data"][0, 0] += n


def test_returns_only_the_samples_that_arrived_since_the_last_read():
    smm = _manager()
    _push(smm, 3)

    count, samples, dropped = smm.get_samples_since("emg", 0)

    assert (count, dropped) == (3, 0)
    np.testing.assert_array_equal(samples[:, 0], [3.0, 2.0, 1.0])


def test_a_read_that_has_seen_everything_comes_back_empty():
    smm = _manager()
    _push(smm, 3)

    count, samples, dropped = smm.get_samples_since("emg", 3)

    assert (count, samples.shape[0], dropped) == (3, 0, 0)


def test_samples_the_buffer_lapped_are_counted_not_invented():
    smm = _manager(capacity=8)
    _push(smm, 20)

    count, samples, dropped = smm.get_samples_since("emg", 0)

    assert (count, dropped) == (20, 12)
    # What survives is the newest 8, not the 20 that were asked for.
    np.testing.assert_array_equal(samples[:, 0], np.arange(20.0, 12.0, -1.0))


def test_the_slice_costs_only_what_arrived():
    smm = _manager(capacity=4096, channels=8)
    _push(smm, 2)

    _, samples, _ = smm.get_samples_since("emg", 0)

    assert samples.shape == (2, 8), "a poller must not copy the whole buffer per read"


def test_counter_and_buffer_are_read_under_one_lock_when_they_share_one():
    smm = _manager()
    held = []
    real_lock = smm.variables["emg"]["lock"]

    class _CountingLock:
        def acquire(self, *args):
            held.append(1)
            return real_lock.acquire(*args)

        def release(self):
            real_lock.release()

    counting = _CountingLock()
    smm.variables["emg"]["lock"] = counting
    smm.variables["emg_count"]["lock"] = counting

    smm.get_samples_since("emg", 0)

    assert len(held) == 1, "a shared lock must be taken once, not once per tag"


def test_reads_still_work_when_the_pair_does_not_share_a_lock():
    smm = _manager(shared_lock=False)
    _push(smm, 2)

    count, samples, dropped = smm.get_samples_since("emg", 0)

    assert (count, dropped) == (2, 0)
    np.testing.assert_array_equal(samples[:, 0], [2.0, 1.0])


def test_streamer_setup_gives_a_modality_and_its_counter_the_same_lock():
    items = assign_shared_memory_locks([["emg", (10, 2), np.double],
                                        ["emg_count", (1, 1), np.int32]])

    assert items[0][-1] is items[1][-1]
