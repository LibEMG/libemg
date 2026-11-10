"""A source that produces samples without a device attached.

Building a pipeline is mostly a matter of getting the shapes and the rates
right, and none of that needs real electrodes. This writes into shared memory
exactly as a device streamer does, so a pipeline built on it is the same
pipeline, and swapping in the real device later changes one block.

It is also what makes the editor testable: an end-to-end run can be exercised
on a machine with no hardware plugged in.
"""

import time
from multiprocessing import Event, Process

import numpy as np


class SyntheticStreamer(Process):
    """Commits generated samples at a fixed rate, like a device would.

    Parameters
    ----------
    shared_memory_items: list
        The items to write into, in the usual ``[tag, shape, dtype, lock]``
        form. The first non-counter item is written to.
    sampling_rate: int (optional), default=1000
        Samples per second.
    num_channels: int (optional), default=8
        Channels produced.
    pattern: str (optional), default='noise'
        ``'noise'`` for gaussian noise, ``'sine'`` for a per-channel sine, and
        ``'bursts'`` for alternating quiet and active periods, which is the
        shape that makes a classifier pipeline visibly do something.
    amplitude: float (optional), default=1.0
        Scale of the generated signal.
    """

    def __init__(self, shared_memory_items, sampling_rate=1000, num_channels=8,
                 pattern="noise", amplitude=1.0):
        super().__init__(daemon=True)
        self.shared_memory_items = shared_memory_items
        self.sampling_rate = int(sampling_rate)
        self.num_channels = int(num_channels)
        self.pattern = pattern
        self.amplitude = float(amplitude)
        self.signal = Event()
        self.notifier_pool = None

    def run(self):
        from libemg.shared_memory_manager import SharedMemoryManager
        smm = SharedMemoryManager(notifier_pool=getattr(self, "notifier_pool", None))
        for item in self.shared_memory_items:
            smm.create_variable(*item)
        tag = next(item[0] for item in self.shared_memory_items
                   if not item[0].endswith("_count"))

        rng = np.random.default_rng(0)
        period = 1.0 / self.sampling_rate
        # Sample times are accumulated from a fixed start rather than by
        # sleeping a period each time, so the stream does not drift slower and
        # slower as each sleep overshoots by a little.
        started = time.perf_counter()
        index = 0
        while not self.signal.is_set():
            index += 1
            target = started + index * period
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            smm.commit(tag, self._sample(rng, index))
        smm.cleanup(parent=False)

    def _sample(self, rng, index):
        t = index / self.sampling_rate
        if self.pattern == "sine":
            frequencies = np.arange(1, self.num_channels + 1) * 5.0
            row = np.sin(2 * np.pi * frequencies * t)
        elif self.pattern == "bursts":
            # Four seconds quiet, four active, so a window of either is easy to
            # recognise in a probe and easy to classify.
            active = (int(t) // 4) % 2 == 1
            row = rng.standard_normal(self.num_channels) * (3.0 if active else 0.3)
        else:
            row = rng.standard_normal(self.num_channels)
        return (row * self.amplitude).reshape(1, -1)


def synthetic_streamer(shared_memory_items=None, sampling_rate=1000,
                       num_channels=8, pattern="noise", amplitude=1.0):
    """Start a source that needs no hardware.

    Matches the shape of the device streamers in :mod:`libemg.streamers`, so
    anything that accepts one of those accepts this.

    Parameters
    ----------
    shared_memory_items: list or None (optional), default=None
        Items to write into. Built for you if omitted.
    sampling_rate: int (optional), default=1000
        Samples per second.
    num_channels: int (optional), default=8
        Channels produced.
    pattern: str (optional), default='noise'
        ``'noise'``, ``'sine'`` or ``'bursts'``.
    amplitude: float (optional), default=1.0
        Scale of the generated signal.

    Returns
    ----------
    SyntheticStreamer
        The running process.
    list
        The shared memory items, to pass to an OnlineDataHandler.

    Examples
    ---------
    >>> streamer, shared_memory = synthetic_streamer(pattern='bursts')
    >>> odh = OnlineDataHandler(shared_memory)
    """
    from libemg.shared_memory_manager import assign_shared_memory_locks
    from libemg.reactive import default_notifier_pool

    if shared_memory_items is None:
        shared_memory_items = [["emg", (2000, num_channels), np.double],
                               ["emg_count", (1, 1), np.int32]]
        assign_shared_memory_locks(shared_memory_items)

    streamer = SyntheticStreamer(shared_memory_items, sampling_rate=sampling_rate,
                                 num_channels=num_channels, pattern=pattern,
                                 amplitude=amplitude)
    streamer.notifier_pool = default_notifier_pool()
    streamer.start()
    return streamer, shared_memory_items
