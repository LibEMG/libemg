"""Cross-process event log for the reactive pipeline.

Every propagation decision the reactive layer makes is recorded here: which
shared-memory item changed, which observer was told about it, what criterion
that observer applied, whether it declared the item dirty, and what ran as a
result. When a pipeline does not fire -- or fires more often than expected --
this log is the record of why, and because the events carry the process that
emitted them it reads correctly when hooks are spread across processes.

The log is deliberately cheap to leave switched on: an event is a small tuple
pushed onto a ``multiprocessing.Queue`` by the process that observed it, and a
single drain thread in the owning process does the formatting and file I/O. No
worker ever formats a string or touches a file.
"""

import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from multiprocessing import Queue


# Event kinds. Kept as plain strings so a log file stays readable and so a
# consumer can filter without importing this module.
COMMIT = "commit"            # a writer changed a stateful item
NOTIFY = "notify"            # a subscriber was woken for an item
EVALUATE = "evaluate"        # an observer applied its criterion
DIRTY = "dirty"              # the criterion said dirty
CLEAN = "clean"              # the criterion said clean
INVOKE = "invoke"            # a hook's step() was entered
COMPLETE = "complete"        # a hook's step() returned
DROP = "drop"                # work was skipped, with a reason
ERROR = "error"              # a hook raised
LIFECYCLE = "lifecycle"      # start/stop of an executor or graph


@dataclass
class Event:
    """One thing that happened in the reactive pipeline.

    Attributes
    ----------
    kind: str
        One of the module-level event kinds.
    timestamp: float
        ``time.time()`` when the event was observed, so events from different
        processes share a clock.
    monotonic: float
        ``time.perf_counter()`` in the observing process. Use this for
        durations within a process; it is not comparable across processes.
    origin: str
        What changed, or what is acting. For a commit this is the shared-memory
        tag that was written.
    observer: str
        The hook or executor the event concerns. Empty for a bare commit.
    criterion: str
        A description of the criterion applied, e.g. ``"OnSamples(40)"``.
    detail: dict
        Kind-specific numbers: generations, sample counts, durations.
    pid: int
        The process that observed the event.
    """

    kind: str
    origin: str = ""
    observer: str = ""
    criterion: str = ""
    detail: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    monotonic: float = field(default_factory=time.perf_counter)
    pid: int = field(default_factory=os.getpid)

    def format(self):
        """Render as one tab-separated line, stable enough to grep and parse."""
        parts = [
            f"{self.timestamp:.6f}",
            f"pid={self.pid}",
            f"{self.kind:<9}",
            f"origin={self.origin or '-'}",
            f"observer={self.observer or '-'}",
            f"criterion={self.criterion or '-'}",
        ]
        if self.detail:
            rendered = " ".join(f"{k}={self._render(v)}" for k, v in sorted(self.detail.items()))
            parts.append(rendered)
        return "\t".join(parts)

    @staticmethod
    def _render(value):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)


class _NullLog:
    """The no-logging case, shaped like an EventLog so callers never branch.

    The reactive layer calls into the log on every commit and every criterion
    evaluation, so the disabled path has to cost as close to nothing as
    possible. These methods are empty and ``enabled`` is False so hot paths can
    skip building a detail dict at all.
    """

    enabled = False

    def emit(self, *args, **kwargs):
        pass

    def record(self, event):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


NULL_LOG = _NullLog()


class EventLog:
    """Collects reactive events from every process and writes them in order.

    Create one in the process that owns the pipeline and hand it to the graph.
    It survives being pickled into a child process: the child keeps the queue
    and drops the drain thread, so a child emits but never writes.

    Parameters
    ----------
    path: str or None (optional), default=None
        File to append events to. If None, events are kept in memory only and
        can be read with :meth:`events`.
    to_stdout: bool (optional), default=False
        Also print each event as it is drained.
    keep: int (optional), default=10000
        How many recent events to retain in memory for :meth:`events`. Set to
        0 to keep none, which is what you want for a long recording that is
        only being written to a file.
    kinds: sequence of str or None (optional), default=None
        Only record these event kinds. None records everything. Filtering here
        happens in the emitting process, so excluded events cost nothing beyond
        the check.

    Examples
    ---------
    >>> log = EventLog(path='reactive.log')
    >>> graph = ReactiveGraph(shared_memory_items, log=log)
    >>> # ... run the pipeline ...
    >>> log.stop()
    >>> for event in log.events()[:5]:
    ...     print(event.format())
    """

    enabled = True

    def __init__(self, path=None, to_stdout=False, keep=10000, kinds=None):
        self.path = path
        self.to_stdout = to_stdout
        self.keep = keep
        self.kinds = set(kinds) if kinds is not None else None
        self._queue = Queue()
        self._records = []
        self._thread = None
        self._stop = threading.Event()
        self._handle = None
        self._owner_pid = os.getpid()

    # ------------------------------------------------------------------
    # emitting (runs in any process)
    # ------------------------------------------------------------------
    def emit(self, kind, origin="", observer="", criterion="", **detail):
        """Record an event. Safe to call from any process."""
        if self.kinds is not None and kind not in self.kinds:
            return
        self.record(Event(kind=kind, origin=origin, observer=observer,
                          criterion=criterion, detail=detail))

    def record(self, event):
        """Queue an already-built event."""
        if self.kinds is not None and event.kind not in self.kinds:
            return
        try:
            self._queue.put_nowait(event)
        except Exception:
            # A full or closed queue must never take the pipeline down with
            # it; losing a debug event is always better than losing a sample.
            pass

    # ------------------------------------------------------------------
    # draining (runs only in the owning process)
    # ------------------------------------------------------------------
    def start(self):
        """Begin draining events. Called for you by ReactiveGraph.start()."""
        if self._thread is not None or os.getpid() != self._owner_pid:
            return
        self._stop.clear()
        if self.path is not None:
            self._handle = open(self.path, "a", buffering=1)
            self._handle.write(f"# libemg reactive event log, opened {time.time():.6f}\n")
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name="libemg-eventlog")
        self._thread.start()

    def _drain(self):
        while not self._stop.is_set():
            self._drain_available(block=True)
        # A stop request has to be followed by a final sweep, or the events
        # that describe the shutdown are the ones you lose.
        self._drain_available(block=False)

    def _drain_available(self, block):
        try:
            event = self._queue.get(timeout=0.1) if block else self._queue.get_nowait()
        except (queue.Empty, OSError, ValueError):
            return
        while True:
            self._write(event)
            try:
                event = self._queue.get_nowait()
            except (queue.Empty, OSError, ValueError):
                return

    def _write(self, event):
        if self.keep:
            self._records.append(event)
            if len(self._records) > self.keep:
                del self._records[:-self.keep]
        line = event.format()
        if self._handle is not None:
            self._handle.write(line + "\n")
        if self.to_stdout:
            print(line)

    def stop(self):
        """Stop draining and flush what is still queued."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3)
        self._thread = None
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    close = stop

    # ------------------------------------------------------------------
    # reading back
    # ------------------------------------------------------------------
    def events(self, kind=None, origin=None, observer=None):
        """Return the retained events, optionally filtered.

        Returns
        ----------
        list
            Matching :class:`Event` objects, oldest first.
        """
        out = self._records
        if kind is not None:
            out = [e for e in out if e.kind == kind]
        if origin is not None:
            out = [e for e in out if e.origin == origin]
        if observer is not None:
            out = [e for e in out if e.observer == observer]
        return list(out)

    def summary(self):
        """Counts per event kind, and per hook for invocations.

        Returns
        ----------
        dict
            ``kinds`` maps event kind to count. ``invocations`` maps hook name
            to how many times it ran. ``dirty`` and ``clean`` map hook name to
            how often its criterion fired or held, which is the pair that tells
            you whether a criterion is set sensibly.
        """
        kinds, invocations, dirty, clean = {}, {}, {}, {}
        for event in self._records:
            kinds[event.kind] = kinds.get(event.kind, 0) + 1
            if event.kind == INVOKE:
                invocations[event.observer] = invocations.get(event.observer, 0) + 1
            elif event.kind == DIRTY:
                dirty[event.observer] = dirty.get(event.observer, 0) + 1
            elif event.kind == CLEAN:
                clean[event.observer] = clean.get(event.observer, 0) + 1
        return {"kinds": kinds, "invocations": invocations,
                "dirty": dirty, "clean": clean}

    def __getstate__(self):
        # The drain thread and the open file belong to the owning process. A
        # child keeps only what it needs to emit.
        state = self.__dict__.copy()
        state["_thread"] = None
        state["_handle"] = None
        state["_records"] = []
        state["_stop"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._stop = threading.Event()
