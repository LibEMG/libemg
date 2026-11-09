"""Mapping the scheduler's monotonic clock onto the recording's clock.

The scheduler runs on ``time.monotonic()``: it never jumps, so a phase lasts
exactly as long as planned whatever the system clock does. The recording does
not use that clock. :meth:`OnlineDataHandler.start_log` stamps every batch with
``time.time()`` as it reads it, and :class:`libemg.session.SessionAnnotations`
fits the device's sample clock against those stamps (sample ``i`` was produced
at ``anchor + i / fs``). So the *collection clock* is ``time.time()`` seconds as
the logger saw it, and a scheduler instant has to be translated into it
explicitly before it can be turned into a sample index.

:class:`ClockMapping` does that translation from paired readings of the two
clocks (anchors), taken at run start and again at every scheduler event. Each
instant is mapped with the newest anchor at or before it, so if the system clock
is stepped mid-run (NTP), boundaries after the step follow the clock the logger
was stamping with by then. The spread of offsets across the run is kept as
``drift`` so a session whose mapping moved can be spotted afterwards.

Nothing here assumes wall-clock time lines up with recorded samples: that link
is the clock fit, done once the recording exists.
"""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass

#: The domain every ``*_collection_time`` field is in.
COLLECTION_CLOCK = "time.time() seconds, as stamped by OnlineDataHandler.start_log"


@dataclass(frozen=True)
class ClockAnchor:
    monotonic: float
    collection: float
    uncertainty: float               # half the time the paired reading took

    @property
    def offset(self):
        return self.collection - self.monotonic

    def to_dict(self):
        return {"monotonic": self.monotonic, "collection": self.collection,
                "uncertainty": self.uncertainty}


class ClockMapping:
    """Monotonic scheduler time -> collection (logger) time.

    Parameters
    ----------
    monotonic, wall: callable
        The two clocks. Injected so tests can drive both.
    """

    def __init__(self, monotonic=time.monotonic, wall=time.time):
        self._monotonic = monotonic
        self._wall = wall
        self.anchors = []
        self._keys = []

    def anchor(self):
        """Read both clocks as close together as possible and keep the pair."""
        before = self._wall()
        mono = self._monotonic()
        after = self._wall()
        return self.add(ClockAnchor(monotonic=mono, collection=(before + after) / 2,
                                    uncertainty=abs(after - before) / 2))

    def add(self, anchor):
        index = bisect.bisect_right(self._keys, anchor.monotonic)
        self._keys.insert(index, anchor.monotonic)
        self.anchors.insert(index, anchor)
        return anchor

    def to_collection(self, monotonic):
        """The collection-clock instant for a scheduler instant."""
        if not self.anchors:
            raise RuntimeError("The clock mapping has no anchors yet; call anchor() first.")
        index = max(0, bisect.bisect_right(self._keys, monotonic) - 1)
        return monotonic + self.anchors[index].offset

    @property
    def drift(self):
        """How far the offset between the clocks moved over the anchors, in seconds."""
        if not self.anchors:
            return 0.0
        offsets = [a.offset for a in self.anchors]
        return max(offsets) - min(offsets)

    def to_dict(self, keep=50):
        """For the session file: the first and newest anchors and the drift."""
        anchors = self.anchors if len(self.anchors) <= keep else (
            self.anchors[: keep // 2] + self.anchors[-keep // 2:])
        return {"collection_clock": COLLECTION_CLOCK,
                "scheduler_clock": "time.monotonic() seconds",
                "method": "collection = monotonic + offset of the newest anchor at or before it",
                "anchor_count": len(self.anchors), "drift_seconds": self.drift,
                "anchors": [a.to_dict() for a in anchors]}


__all__ = ["ClockMapping", "ClockAnchor", "COLLECTION_CLOCK"]
