"""Reactive hooks over stateful shared memory.

The problem this replaces
-------------------------
Online processing in libemg used to be built out of pollers. Each consumer sat
in a loop asking "is there a window yet?", and asking cost a copy of an entire
shared-memory buffer plus, if a filter was installed, a filter pass over all of
it. Several consumers meant several pollers, each independently re-deriving the
same filtered signal and the same features, with no way to share the result.

Here, a write to a shared-memory item announces itself. Anything hooked into
that item is told it changed and decides for itself whether the change matters.

Why "dirty" is not a flag on the item
-------------------------------------
An item can have any number of observers, and a single boolean would be wrong
for all of them: whichever observer cleared the flag would starve the rest.
Worse, observers legitimately disagree about what "changed enough" means. A
filter wants to run on a single new sample. A windowing stage does not care
until a full window increment has accrued. A live plot only wants thirty
updates a second no matter how fast samples arrive.

So the item publishes *facts* -- how many times it has been written, how many
samples have ever arrived, whether its writer has finished -- and each observer
pairs those facts with its own :class:`Criterion` and its own memory of what it
last consumed. Propagation is the notification; dirtiness is a per-observer
judgement made against shared facts.

Because the facts live in the item's state block in shared memory, and the
notification is a synchronization primitive shared between processes, an
observer does not have to live in the process that did the writing.

The cascade
-----------
A hook's output is itself a stateful item, so committing to it notifies that
item's observers in turn. Chaining ``emg`` to ``filtered_emg`` to ``features``
to ``predictions`` needs no coordinating loop; each stage wakes the next. Any
stage may have several observers, and a model is free to hook whichever stage
it wants -- raw samples, filtered samples, or features -- because what a hook
observes is declared rather than hard-wired.

Examples
---------
>>> from libemg.reactive import ReactiveGraph, FilterHook, Input, OnCommit, OnSamples
>>> from libemg.event_log import EventLog
>>> log = EventLog(path='reactive.log')
>>> graph = ReactiveGraph(shared_memory_items, log=log)
>>> graph.add(FilterHook('filter', 'emg', 'filtered_emg', fi=my_filter,
...                      shape=(2000, 8)))
>>> graph.start()
"""

import pickle
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Condition, Event, Process

import numpy as np

from libemg import event_log
from libemg.shared_memory_manager import SharedMemoryManager, MAX_SUBSCRIBERS


# ======================================================================
# Notification
# ======================================================================
class NotifierPool:
    """A fixed set of wake-up slots that any process can signal.

    Each executor claims one slot and waits on it. A writer signals the slots
    subscribed to the item it just wrote, which it discovers from that item's
    state block -- so a writer that started before an observer existed still
    reaches it.

    The slots are made up front and handed to child processes as process
    arguments, because a synchronization primitive cannot be looked up by name
    after the fact on every platform libemg supports. A fixed pool is what
    makes the set of handles known before anything spawns.

    Parameters
    ----------
    size: int (optional), default=16
        How many slots to create. One per executor is enough; the cap is
        :data:`libemg.shared_memory_manager.MAX_SUBSCRIBERS`.
    """

    def __init__(self, size=16):
        if size > MAX_SUBSCRIBERS:
            raise ValueError(f"A notifier pool holds at most {MAX_SUBSCRIBERS} slots.")
        self.size = size
        self._conditions = tuple(Condition() for _ in range(size))
        self._next_slot = 0

    def claim(self):
        """Reserve a slot. Call in the parent, before anything spawns.

        Returns
        ----------
        int
            The claimed slot number.
        """
        if self._next_slot >= self.size:
            raise RuntimeError(
                f"All {self.size} notifier slots are claimed. "
                "Construct the pool with a larger size."
            )
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def signal(self, slot):
        """Wake whoever is waiting on ``slot``."""
        condition = self._conditions[slot]
        with condition:
            condition.notify_all()

    def __len__(self):
        return self.size

    def wait(self, slot, predicate, timeout):
        """Block on ``slot`` until ``predicate`` holds or ``timeout`` elapses.

        The predicate is re-checked under the slot's lock, so a change that
        lands between the check and the wait cannot be missed. The timeout is
        what keeps a dropped notification from becoming a hang: the worst case
        degrades to polling at the timeout interval rather than stopping.

        Returns
        ----------
        bool
            Whether the predicate held.
        """
        condition = self._conditions[slot]
        with condition:
            return condition.wait_for(predicate, timeout=timeout)


_DEFAULT_POOL = None


def default_notifier_pool(size=16):
    """The notifier pool shared by everything built in this process.

    A writer can only wake an observer if the two hold the same pool, and a
    pool cannot be looked up by name after the fact. Routing everything through
    one process-local pool is what lets a streamer started early notify a
    predictor built later, as long as both were set up in the same script --
    which is how libemg pipelines are normally assembled.

    A writer that never obtains the pool still works. Its commits update the
    item's state block, and observers fall back to re-reading that block at
    their fallback interval. The block is a handful of integers, so even that
    fallback is orders of magnitude cheaper than the buffer copy the old
    pollers paid; obtaining the pool turns a cheap poll into no poll at all.

    Parameters
    ----------
    size: int (optional), default=16
        Slots to create, if the pool does not exist yet.

    Returns
    ----------
    NotifierPool
        The process-wide pool.
    """
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = NotifierPool(size)
    return _DEFAULT_POOL


def reset_default_notifier_pool():
    """Discard the process-wide pool. Intended for tests."""
    global _DEFAULT_POOL
    _DEFAULT_POOL = None


# ======================================================================
# Criteria -- how an observer decides an item is dirty
# ======================================================================
@dataclass
class CriterionMemory:
    """What one observer remembers about one item it observes.

    Attributes
    ----------
    generation: int
        The last commit generation this observer has accounted for.
    samples: int
        The sample total this observer has consumed up to. Distinct from
        ``generation`` because a volume-based criterion has to survive commits
        of differing sizes.
    fired_at: float
        ``time.perf_counter()`` when this observer last fired on the item.
    epoch: int
        The item's epoch when this memory was last valid. A mismatch means the
        item's counters were reset and everything here is stale.
    """

    generation: int = 0
    samples: int = 0
    fired_at: float = 0.0
    epoch: int = 0


class Criterion(ABC):
    """An observer's own definition of "this item changed enough to act on".

    Subclass this to express a rule the built-ins do not cover. The contract is
    three methods:

    - :meth:`is_dirty` looks at the item's published facts and the observer's
      memory and returns a judgement. It must not mutate either.
    - :meth:`consume` advances the memory to record that the observer acted.
      Keeping it separate from ``is_dirty`` is what lets the runtime evaluate a
      criterion for logging without committing to running the hook.
    - :meth:`describe` names the criterion for the event log, so a recorded
      decision can be read back and understood.
    """

    @abstractmethod
    def is_dirty(self, snapshot, memory):
        """Whether ``snapshot`` counts as a change this observer cares about."""

    def consume(self, snapshot, memory):
        """Record that the observer has acted on ``snapshot``."""
        memory.generation = snapshot.generation
        memory.samples = snapshot.total_samples
        memory.fired_at = time.perf_counter()
        memory.epoch = snapshot.epoch

    def describe(self):
        return type(self).__name__ + "()"

    def __repr__(self):
        return self.describe()


class OnCommit(Criterion):
    """Dirty on any write at all.

    What a filter or a logger wants: one new sample is a reason to run.
    """

    def is_dirty(self, snapshot, memory):
        return snapshot.generation > memory.generation


class OnSamples(Criterion):
    """Dirty once ``increment`` new samples have accrued.

    What a windowing stage wants. A commit of one sample leaves the item clean
    by this criterion even though it is dirty by :class:`OnCommit`, which is
    exactly the disagreement that makes dirtiness a per-observer judgement
    rather than a property of the item.

    :meth:`consume` advances by exactly ``increment`` rather than jumping to
    the current total, so a burst that delivers three increments at once
    produces three firings instead of one. That keeps window boundaries evenly
    spaced no matter how the device chunks its packets.

    Parameters
    ----------
    increment: int
        How many new samples constitute a change.
    """

    def __init__(self, increment):
        if increment < 1:
            raise ValueError("increment must be at least 1.")
        self.increment = int(increment)

    def is_dirty(self, snapshot, memory):
        return (snapshot.total_samples - memory.samples) >= self.increment

    def consume(self, snapshot, memory):
        memory.samples += self.increment
        memory.generation = snapshot.generation
        memory.fired_at = time.perf_counter()
        memory.epoch = snapshot.epoch

    def describe(self):
        return f"OnSamples({self.increment})"


class AfterSamples(OnSamples):
    """Dirty once ``increment`` samples accrue, then skip whatever piled up.

    Same threshold as :class:`OnSamples`, but :meth:`consume` jumps to the
    current total instead of stepping. Use it where acting on the newest data
    matters more than acting on all of it: a model that has fallen behind
    should predict from the latest window rather than work through a backlog of
    stale ones.
    """

    def consume(self, snapshot, memory):
        memory.samples = snapshot.total_samples
        memory.generation = snapshot.generation
        memory.fired_at = time.perf_counter()
        memory.epoch = snapshot.epoch

    def describe(self):
        return f"AfterSamples({self.increment})"


class Periodic(Criterion):
    """Dirty at most ``hz`` times a second, and only if something changed.

    For observers whose cost is in the presentation rather than the
    computation: a live plot updated faster than the display refreshes is
    wasted work, and a plot of unchanged data is wasted work too.

    Parameters
    ----------
    hz: float
        Maximum firings per second.
    """

    def __init__(self, hz):
        if hz <= 0:
            raise ValueError("hz must be positive.")
        self.hz = float(hz)
        self.period = 1.0 / self.hz

    def is_dirty(self, snapshot, memory):
        if snapshot.generation <= memory.generation:
            return False
        return (time.perf_counter() - memory.fired_at) >= self.period

    def describe(self):
        return f"Periodic({self.hz:g})"


class Always(Criterion):
    """Dirty whenever asked, changed or not.

    Only sensible paired with a rate limit or a hook that must run on a
    heartbeat regardless of input.
    """

    def is_dirty(self, snapshot, memory):
        return True


class WhenClosed(Criterion):
    """Dirty once the item's writer declares itself finished.

    How a finite pipeline finishes: the stage that summarises a run hooks the
    close rather than a data change.
    """

    def is_dirty(self, snapshot, memory):
        return snapshot.closed and memory.generation < snapshot.generation + 1

    def consume(self, snapshot, memory):
        memory.generation = snapshot.generation + 1
        memory.samples = snapshot.total_samples
        memory.fired_at = time.perf_counter()
        memory.epoch = snapshot.epoch


class Custom(Criterion):
    """Wrap a plain function as a criterion.

    Parameters
    ----------
    predicate: callable
        Called as ``predicate(snapshot, memory)``, returning a bool. It is
        given the item's :class:`~libemg.shared_memory_manager.Snapshot` and
        the observer's :class:`CriterionMemory`.
    name: str (optional), default='Custom'
        What the event log should call this criterion.

    Examples
    ---------
    >>> # act only once a second's worth of samples has arrived
    >>> Custom(lambda s, m: s.total_samples - m.samples >= 1000, 'OneSecond')
    """

    def __init__(self, predicate, name="Custom"):
        self.predicate = predicate
        self.name = name

    def is_dirty(self, snapshot, memory):
        return bool(self.predicate(snapshot, memory))

    def describe(self):
        return f"{self.name}()"


class AllOf(Criterion):
    """Dirty only when every wrapped criterion is."""

    def __init__(self, *criteria):
        self.criteria = criteria

    def is_dirty(self, snapshot, memory):
        return all(c.is_dirty(snapshot, memory) for c in self.criteria)

    def consume(self, snapshot, memory):
        for c in self.criteria:
            c.consume(snapshot, memory)

    def describe(self):
        return "AllOf(" + ", ".join(c.describe() for c in self.criteria) + ")"


class AnyOf(Criterion):
    """Dirty when any wrapped criterion is."""

    def __init__(self, *criteria):
        self.criteria = criteria

    def is_dirty(self, snapshot, memory):
        return any(c.is_dirty(snapshot, memory) for c in self.criteria)

    def consume(self, snapshot, memory):
        for c in self.criteria:
            c.consume(snapshot, memory)

    def describe(self):
        return "AnyOf(" + ", ".join(c.describe() for c in self.criteria) + ")"


# ======================================================================
# Hook declarations
# ======================================================================
# How a hook wants its input delivered.
WINDOW = "window"   # the newest `size` rows, oldest first
DELTA = "delta"     # everything committed since this hook last consumed
LATEST = "latest"   # the newest single row
FULL = "full"       # the whole buffer, newest-first, as get_data returns it
STATE = "state"     # no data at all, just the snapshot


@dataclass
class Input:
    """One item a hook observes, and the terms on which it observes it.

    Parameters
    ----------
    tag: str
        The shared-memory item to observe.
    criterion: Criterion
        This hook's rule for when a change to that item matters. Defaults to
        :class:`OnCommit`, i.e. any write.
    mode: str
        How to deliver the data: :data:`WINDOW`, :data:`DELTA`, :data:`LATEST`,
        :data:`FULL`, or :data:`STATE` for no data at all.
    size: int or None
        Rows to deliver in :data:`WINDOW` mode.
    """

    tag: str
    criterion: Criterion = field(default_factory=OnCommit)
    mode: str = WINDOW
    size: int = None

    def __post_init__(self):
        if self.mode == WINDOW and not self.size:
            raise ValueError(f"Input('{self.tag}', mode=WINDOW) needs a size.")


@dataclass
class Output:
    """An item a hook writes, created by the graph if it does not exist.

    Parameters
    ----------
    tag: str
        The shared-memory item to create and commit to.
    shape: tuple
        Buffer shape, as ``(rows, columns)``.
    dtype: type
        Buffer dtype.
    """

    tag: str
    shape: tuple
    dtype: type = np.double


class Hook(ABC):
    """Something that runs when an item it observes is dirty by its criteria.

    A hook declares what it watches and what it writes; the runtime does the
    attaching, the locking and the committing. A hook never takes a lock and
    never commits, which is what makes one testable in isolation: build the
    input dict by hand, call :meth:`step`, and check what comes back.

    Parameters
    ----------
    name: str
        Identifies the hook in the event log. Must be unique in a graph.
    inputs: sequence of Input
        What it observes.
    outputs: sequence of Output (optional)
        What it writes. Committing to these is what continues the cascade.
    trigger: str (optional), default='any'
        With several inputs, whether the hook runs when ``'any'`` of them is
        dirty or only when ``'all'`` are. ``'all'`` is what a hook fusing two
        modalities wants, so it does not run on half its data.
    attach: sequence of str (optional)
        Further items the hook needs access to, which the runtime should make
        available but neither trigger on nor write for it. This is for a value
        the hook sets itself rather than appends to, such as a flag announcing
        that something is ready; the hook writes it with
        :meth:`~libemg.shared_memory_manager.SharedMemoryManager.apply` from
        inside :meth:`step`. Without declaring it here the item is simply not
        attached in the executor's process and the write goes nowhere.
    """

    def __init__(self, name, inputs, outputs=(), trigger="any", attach=()):
        if trigger not in ("any", "all"):
            raise ValueError("trigger must be 'any' or 'all'.")
        self.name = name
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.trigger = trigger
        self.attach = list(attach)

    def setup(self, context):
        """Build whatever cannot be pickled: models, filter state, sockets.

        Runs once, inside the executor's process, before the first step. A
        hook must hold configuration rather than live objects so it can reach
        that process at all; this is where the live objects get made.

        Parameters
        ----------
        context: HookContext
            Access to the executor's shared memory manager and event log.
        """

    @abstractmethod
    def step(self, data, snapshots):
        """Do the work.

        Parameters
        ----------
        data: dict
            Tag to array, delivered per each input's mode. Absent for inputs
            declared :data:`STATE`.
        snapshots: dict
            Tag to :class:`~libemg.shared_memory_manager.Snapshot`, the state
            each input was read at.

        Returns
        ----------
        dict or None
            Tag to array for each output to commit, or None to commit nothing.
            Returning None is the normal way for a hook to decline: a model
            that has not been fitted yet, say.
        """

    def teardown(self):
        """Release anything :meth:`setup` made."""


@dataclass
class HookContext:
    """What a hook is given at setup time."""

    smm: SharedMemoryManager
    log: object
    executor: str


# ======================================================================
# Built-in hooks
# ======================================================================
class CallbackHook(Hook):
    """Run a plain function when an item goes dirty.

    The least ceremony available: useful for logging, for driving something
    outside libemg, and for tests.

    Parameters
    ----------
    name: str
        Hook name.
    inputs: sequence of Input
        What to observe.
    fn: callable
        Called as ``fn(data, snapshots)``; whatever it returns is committed to
        the declared outputs, so returning None is fine for a pure sink.
    outputs: sequence of Output (optional)
        What it writes.
    trigger: str (optional), default='any'
        As :class:`Hook`.

    Examples
    ---------
    >>> CallbackHook('watch', [Input('emg', OnCommit(), mode=LATEST)],
    ...              fn=lambda data, snaps: print(data['emg']))
    """

    def __init__(self, name, inputs, fn, outputs=(), trigger="any"):
        super().__init__(name, inputs, outputs, trigger)
        self.fn = fn

    def step(self, data, snapshots):
        return self.fn(data, snapshots)


class FilterHook(Hook):
    """Filter one item into another.

    Sits between a raw item and everything that wants it conditioned, so the
    filtering happens once for all consumers instead of once per consumer.

    Parameters
    ----------
    name: str
        Hook name.
    source: str
        Item to read.
    target: str
        Item to write.
    fi: libemg.filtering.Filter
        The filter to apply.
    shape: tuple
        Buffer shape for ``target``.
    window: int (optional), default=None
        How many rows to filter per run. Defaults to the target's row count.
        Filtering a margin larger than the increment and keeping the newest
        rows is what stops each run's edge effects from reaching the output.
    increment: int (optional), default=1
        Rows of new input that constitute a change. The default of one sample
        is the point of a filter stage: it is dirty as soon as anything
        arrives.
    dtype: type (optional), default=numpy.double
        Buffer dtype for ``target``.

    Notes
    -----
    The filter is applied to a fresh window on every run rather than carried as
    streaming state, so its output matches what an offline filter would produce
    for the same rows. That is deliberate: libemg's filters are zero-phase,
    which cannot be realised causally, so the honest choice is to keep the
    offline semantics and pay for a margin.
    """

    def __init__(self, name, source, target, fi, shape, window=None,
                 increment=1, dtype=np.double):
        window = window or shape[0]
        super().__init__(
            name,
            inputs=[Input(source, OnSamples(increment), mode=WINDOW, size=window)],
            outputs=[Output(target, shape, dtype)],
        )
        self.source = source
        self.target = target
        self.fi = fi
        self.window = window
        self.increment = increment

    def step(self, data, snapshots):
        samples = data[self.source]
        if samples.shape[0] == 0:
            return None
        filtered = self.fi.filter(samples) if self.fi is not None else samples
        # Only the newest `increment` rows are new; the rest were emitted by an
        # earlier run and re-filtering them here would duplicate them.
        return {self.target: filtered[-self.increment:]}


class FeatureHook(Hook):
    """Extract features from a window of an item.

    Worth having as its own stage when more than one model wants the same
    features: the extraction happens once and both models hook the result.

    Parameters
    ----------
    name: str
        Hook name.
    source: str
        Item to window.
    target: str
        Item to write features to.
    feature_list: list
        Features to extract.
    window_size: int
        Samples per window.
    window_increment: int
        New samples that constitute a change, i.e. the hop between windows.
    num_features: int
        Columns the target buffer needs. Declared rather than inferred because
        the buffer has to exist before the first extraction runs.
    rows: int (optional), default=100
        Rows to retain in the target buffer.
    feature_dic: dict (optional), default=None
        Feature parameters.
    """

    def __init__(self, name, source, target, feature_list, window_size,
                 window_increment, num_features, rows=100, feature_dic=None):
        super().__init__(
            name,
            inputs=[Input(source, OnSamples(window_increment),
                          mode=WINDOW, size=window_size)],
            outputs=[Output(target, (rows, num_features), np.double)],
        )
        self.source = source
        self.target = target
        self.feature_list = feature_list
        self.window_size = window_size
        self.feature_dic = feature_dic or {}
        self._fe = None

    def setup(self, context):
        # Built here rather than in __init__ so the hook stays picklable for a
        # spawned executor.
        from libemg.feature_extractor import FeatureExtractor
        self._fe = FeatureExtractor()

    def step(self, data, snapshots):
        samples = data[self.source]
        if samples.shape[0] < self.window_size:
            return None
        window = samples.transpose()[np.newaxis, :, :]
        features = self._fe.extract_features(
            self.feature_list, window, feature_dic=self.feature_dic, array=True)
        return {self.target: features}


class ProbeHook(Hook):
    """Observe an item without writing anything.

    Rate-limited by default and holds no outputs, so a probe cannot slow the
    pipeline it is watching or change what it produces.

    Parameters
    ----------
    name: str
        Hook name.
    source: str
        Item to observe.
    fn: callable
        Called as ``fn(data, snapshots)``.
    hz: float (optional), default=30.0
        Maximum observations per second.
    mode: str (optional), default=LATEST
        Delivery mode.
    size: int (optional), default=None
        Rows, in :data:`WINDOW` mode.
    """

    def __init__(self, name, source, fn, hz=30.0, mode=LATEST, size=None):
        super().__init__(name,
                         inputs=[Input(source, Periodic(hz), mode=mode, size=size)])
        self.source = source
        self.fn = fn

    def step(self, data, snapshots):
        self.fn(data, snapshots)
        return None


# ======================================================================
# Execution
# ======================================================================
class _ExecutorCore:
    """The wait-and-service loop, independent of what it runs inside.

    An executor owns exactly one wake-up slot and one wait loop. It wakes when
    any item its hooks observe is committed to, asks each hook's criteria
    whether that change matters, and runs the ones that say yes. It never
    polls: the wait is a blocking wait with a timeout, and the timeout exists
    only so a lost notification degrades to a slow check instead of a hang.

    :class:`Executor` runs this in its own process and :class:`ThreadExecutor`
    runs it in a thread of the process that built the graph. The loop is
    identical either way, which is what makes the choice purely about where a
    hook needs to live rather than about what it can do.
    """

    def __init__(self, name, hooks, shared_memory_items, notifier_pool, slot,
                 log, stop_event, poll_fallback=0.05, max_catchup=8):
        self.executor_name = name
        self.hooks = hooks
        self.shared_memory_items = shared_memory_items
        self.notifier_pool = notifier_pool
        self.slot = slot
        self.log = log
        self.stop_event = stop_event
        self.poll_fallback = poll_fallback
        self.max_catchup = max_catchup

    # ------------------------------------------------------------------
    def run(self):
        smm = SharedMemoryManager(notifier_pool=self.notifier_pool, log=self.log)
        self.log.emit(event_log.LIFECYCLE, origin=self.executor_name,
                      observer=self.executor_name, phase="starting", slot=self.slot)
        try:
            self._attach(smm)
            self._setup_hooks(smm)
            self._loop(smm)
        except Exception:
            self.log.emit(event_log.ERROR, origin=self.executor_name,
                          observer=self.executor_name,
                          traceback=traceback.format_exc().replace("\n", " | "))
            raise
        finally:
            for hook in self.hooks:
                try:
                    hook.teardown()
                except Exception:
                    pass
            self.log.emit(event_log.LIFECYCLE, origin=self.executor_name,
                          observer=self.executor_name, phase="stopped")
            smm.cleanup(parent=False)

    def _attach(self, smm):
        """Attach every item this executor's hooks read or write."""
        declared = {item[0]: item for item in self.shared_memory_items}
        wanted = set()
        for hook in self.hooks:
            wanted.update(i.tag for i in hook.inputs)
            wanted.update(o.tag for o in hook.outputs)
            wanted.update(getattr(hook, "attach", ()))
            # A legacy sample counter is kept in step by commit(), so it has to
            # be attached wherever its buffer is.
            for output in hook.outputs:
                if output.tag + "_count" in declared:
                    wanted.add(output.tag + "_count")
            for spec in hook.inputs:
                if spec.tag + "_count" in declared:
                    wanted.add(spec.tag + "_count")
        for tag in sorted(wanted):
            if tag not in declared:
                raise KeyError(
                    f"Executor '{self.executor_name}' needs shared-memory item "
                    f"'{tag}', which the graph did not declare."
                )
            item = declared[tag]
            deadline = time.time() + 10
            while not smm.find_variable(*item):
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Shared-memory item '{tag}' never appeared. Is its writer running?"
                    )
                time.sleep(0.01)
        # Subscribing after attaching means the writer can already be running:
        # subscriptions live in the item's state block, not in the writer.
        self.watched = sorted({i.tag for hook in self.hooks for i in hook.inputs})
        for tag in self.watched:
            smm.subscribe(tag, self.slot)

    def _setup_hooks(self, smm):
        context = HookContext(smm=smm, log=self.log, executor=self.executor_name)
        self.memory = {}
        for hook in self.hooks:
            hook.setup(context)
            for spec in hook.inputs:
                self.memory[(hook.name, spec.tag)] = CriterionMemory()

    # ------------------------------------------------------------------
    def _loop(self, smm):
        seen = {tag: 0 for tag in self.watched}

        def anything_changed():
            for tag in self.watched:
                block = smm._block(tag)
                if int(block[0]) > seen[tag] or int(block[3]):
                    return True
            return False

        while not self.stop_event.is_set():
            self.notifier_pool.wait(self.slot, anything_changed, self.poll_fallback)
            if self.stop_event.is_set():
                break
            snapshots = smm.snapshots(self.watched)
            for tag in self.watched:
                seen[tag] = snapshots[tag].generation
            for hook in self.hooks:
                self._service(hook, smm, snapshots)
            # A pipeline over a finite source has to be able to finish. Once
            # every watched item is closed and no criterion is still dirty,
            # there is nothing left that could ever fire.
            if all(snapshots[t].closed for t in self.watched) and not self._any_dirty(smm):
                self.log.emit(event_log.LIFECYCLE, origin=self.executor_name,
                              observer=self.executor_name, phase="drained")
                break

    def _any_dirty(self, smm):
        for hook in self.hooks:
            snaps = smm.snapshots([i.tag for i in hook.inputs])
            for spec in hook.inputs:
                if spec.criterion.is_dirty(snaps[spec.tag],
                                           self.memory[(hook.name, spec.tag)]):
                    return True
        return False

    def _service(self, hook, smm, cached):
        """Evaluate one hook's criteria and run it while it says dirty."""
        for _ in range(self.max_catchup):
            snaps = {}
            for spec in hook.inputs:
                snaps[spec.tag] = cached.get(spec.tag) or smm.snapshot(spec.tag)
            cached = {}  # only the first pass may use the batch snapshot

            verdicts = {}
            for spec in hook.inputs:
                memory = self.memory[(hook.name, spec.tag)]
                snapshot = snaps[spec.tag]
                # A reset moves the counters backwards on purpose. Without
                # noticing the epoch change an observer would read the reset as
                # "nothing has happened" and stall forever.
                if snapshot.epoch != memory.epoch:
                    memory.generation = 0
                    memory.samples = 0
                    memory.epoch = snapshot.epoch
                dirty = spec.criterion.is_dirty(snapshot, memory)
                verdicts[spec.tag] = dirty
                if self.log.enabled:
                    self.log.emit(event_log.DIRTY if dirty else event_log.CLEAN,
                                  origin=spec.tag, observer=hook.name,
                                  criterion=spec.criterion.describe(),
                                  generation=snapshot.generation,
                                  total_samples=snapshot.total_samples,
                                  seen_generation=memory.generation,
                                  consumed_samples=memory.samples)
            fire = all(verdicts.values()) if hook.trigger == "all" else any(verdicts.values())
            if not fire:
                return

            data = {}
            for spec in hook.inputs:
                if spec.mode == STATE:
                    continue
                memory = self.memory[(hook.name, spec.tag)]
                value, read_at = self._read(smm, spec, memory, snaps[spec.tag])
                data[spec.tag] = value
                # Advance against the state the read itself saw, not the state
                # seen when the criterion was evaluated. A commit landing
                # between the two would otherwise leave the memory behind what
                # was actually handed to the hook, and the next pass would
                # deliver those rows a second time.
                if read_at is not None:
                    snaps[spec.tag] = read_at

            started = time.perf_counter()
            if self.log.enabled:
                self.log.emit(event_log.INVOKE, origin=",".join(
                    t for t, v in verdicts.items() if v),
                    observer=hook.name, executor=self.executor_name)
            try:
                produced = hook.step(data, snaps)
            except Exception:
                self.log.emit(event_log.ERROR, origin=hook.name, observer=hook.name,
                              traceback=traceback.format_exc().replace("\n", " | "))
                # One failing hook must not take the executor's other hooks
                # down with it, so the criterion is consumed and the loop moves
                # on rather than retrying the same bad input forever.
                for spec in hook.inputs:
                    if verdicts[spec.tag]:
                        spec.criterion.consume(snaps[spec.tag],
                                               self.memory[(hook.name, spec.tag)])
                return

            for spec in hook.inputs:
                if verdicts[spec.tag]:
                    spec.criterion.consume(snaps[spec.tag],
                                           self.memory[(hook.name, spec.tag)])

            if produced:
                for tag, value in produced.items():
                    if value is None:
                        continue
                    smm.commit(tag, value)

            if self.log.enabled:
                self.log.emit(event_log.COMPLETE, observer=hook.name,
                              origin=hook.name,
                              duration_ms=(time.perf_counter() - started) * 1e3,
                              committed=",".join(sorted(produced)) if produced else "-")
        else:
            # Still dirty after max_catchup runs: the hook cannot keep up.
            self.log.emit(event_log.DROP, observer=hook.name, origin=hook.name,
                          reason="max_catchup", limit=self.max_catchup)

    def _read(self, smm, spec, memory, snapshot):
        """Deliver an input, and report the state it was actually read at.

        Returns
        ----------
        value: numpy.ndarray
            The data, per the input's mode.
        read_at: Snapshot or None
            The item's state at the instant of the read, or None where the
            mode does not read the state. The caller advances the observer's
            memory against this rather than against an earlier snapshot.
        """
        if spec.mode == WINDOW:
            window, read_at = smm.read_window(spec.tag, spec.size)
            return window, read_at
        if spec.mode == LATEST:
            window, read_at = smm.read_window(spec.tag, 1)
            return window, read_at
        if spec.mode == DELTA:
            samples, read_at, lost = smm.read_since(spec.tag, memory.samples)
            if lost and self.log.enabled:
                self.log.emit(event_log.DROP, origin=spec.tag, observer=spec.tag,
                              reason="buffer_overwritten", lost=lost)
            return samples, read_at
        if spec.mode == FULL:
            return smm.get_variable(spec.tag), None
        raise ValueError(f"Unknown input mode '{spec.mode}'.")


class Executor(Process):
    """Runs a set of hooks in a process of its own.

    The default. Hooks reach the process by being pickled, so a hook must hold
    configuration rather than live objects and build the live ones in
    :meth:`Hook.setup`.

    Constructed by :class:`ReactiveGraph`; not usually built directly.
    """

    def __init__(self, *args, **kwargs):
        name = args[0] if args else kwargs["name"]
        super().__init__(daemon=True, name=f"libemg-executor-{name}")
        # Kept on the wrapper as well as the core so an executor stays
        # introspectable from the process that made it, where the core does
        # not exist yet.
        self.executor_name = name
        self.hooks = args[1] if len(args) > 1 else kwargs.get("hooks", [])
        self._core_args = args
        self._core_kwargs = kwargs

    def run(self):
        _ExecutorCore(*self._core_args, **self._core_kwargs).run()


class ThreadExecutor(threading.Thread):
    """Runs a set of hooks in a thread of the process that built the graph.

    For hooks that cannot or should not be moved to another process. Two cases
    come up in practice:

    - The hook closes over something unpicklable, a lambda or a local
      function. In a process executor that fails at spawn time.
    - The hook has to touch something that only exists here, most often a
      plotting window. A GUI toolkit will not accept calls from another
      process, so a probe that draws has to run in the process that owns the
      window.

    The cost is that the work shares this process's interpreter lock, so a hook
    doing heavy numeric work belongs in a process executor instead. Numpy
    releases the lock for the arithmetic itself, so a probe or a light
    transform is usually fine here.
    """

    def __init__(self, *args, **kwargs):
        name = args[0] if args else kwargs["name"]
        super().__init__(daemon=True, name=f"libemg-thread-executor-{name}")
        self.executor_name = name
        self.hooks = args[1] if len(args) > 1 else kwargs.get("hooks", [])
        self._core_args = args
        self._core_kwargs = kwargs

    def run(self):
        _ExecutorCore(*self._core_args, **self._core_kwargs).run()


class ReactiveGraph:
    """Assembles hooks over stateful shared memory and runs them.

    Hooks are grouped into executors, one process each. Hooks in the same
    executor share a wake-up and run in sequence, which is what you want for
    stages that are individually cheap; a hook that is expensive, or that must
    not be delayed by its neighbours, belongs in its own executor.

    Parameters
    ----------
    shared_memory_items: list
        The items the graph may touch, in the ``[tag, shape, dtype, lock]``
        form the streamers produce. Outputs declared by hooks are appended
        automatically.
    log: EventLog or None (optional), default=None
        Records every commit, every criterion decision and every invocation.
        Defaults to no logging.
    notifier_pool: NotifierPool or None (optional), default=None
        Created for you if omitted.
    poll_fallback: float (optional), default=0.05
        Seconds an executor will wait before re-checking without having been
        woken. This is a safety net, not the mechanism.

    Examples
    ---------
    >>> graph = ReactiveGraph(shared_memory_items, log=EventLog(to_stdout=True))
    >>> graph.add(FilterHook('filt', 'emg', 'filtered_emg', fi, shape=(2000, 8)))
    >>> graph.add(my_model_hook, executor='model')
    >>> graph.start()
    >>> ...
    >>> graph.stop()
    """

    def __init__(self, shared_memory_items, log=None, notifier_pool=None,
                 poll_fallback=0.05):
        self.shared_memory_items = [list(item) for item in shared_memory_items]
        self.log = log if log is not None else event_log.NULL_LOG
        self.notifier_pool = notifier_pool or NotifierPool()
        self.poll_fallback = poll_fallback
        self._groups = {}
        self._names = set()
        self._in_process = {}
        self._executors = []
        self._stop = Event()
        self._smm = None
        self._started = False

    # ------------------------------------------------------------------
    def add(self, hook, executor="main", in_process=False):
        """Register a hook, optionally in a named executor.

        Parameters
        ----------
        hook: Hook
            The hook to run.
        executor: str (optional), default='main'
            Which executor to run it in. Hooks sharing a name share one.
        in_process: bool (optional), default=False
            Run this executor as a thread here instead of as a separate
            process. Needed for a hook that closes over something unpicklable,
            such as a lambda, and for one that has to touch a plotting window,
            which a separate process cannot do. See :class:`ThreadExecutor`.
            All hooks in a given executor must agree on this.

        Returns
        ----------
        ReactiveGraph
            Self, so registrations can be chained.

        Examples
        ---------
        >>> graph.add(FilterHook('filt', 'emg', 'filtered_emg', fi, (2000, 8)))
        >>> graph.add(ProbeHook('scope', 'emg', lambda d, s: plot(d)),
        ...           executor='scope', in_process=True)
        """
        if self._started:
            raise RuntimeError("Cannot add hooks to a graph that is running.")
        if hook.name in self._names:
            raise ValueError(f"A hook named '{hook.name}' is already registered.")
        previous = self._in_process.get(executor)
        if previous is not None and previous != in_process:
            raise ValueError(
                f"Executor '{executor}' was registered with in_process={previous} "
                f"and now with in_process={in_process}. An executor is one thread "
                "or one process, not both."
            )
        self._in_process[executor] = in_process
        self._names.add(hook.name)
        self._groups.setdefault(executor, []).append(hook)
        return self

    def declare(self, tag, shape, dtype=np.double, lock=None):
        """Add a shared-memory item the graph should own.

        Rarely needed: a hook's declared outputs are added for you.
        """
        from multiprocessing import Lock as _Lock
        if any(item[0] == tag for item in self.shared_memory_items):
            return self
        self.shared_memory_items.append([tag, shape, dtype, lock or _Lock()])
        return self

    # ------------------------------------------------------------------
    def _collect_outputs(self):
        """Create shared-memory items for every hook output not already declared."""
        from multiprocessing import Lock as _Lock
        known = {item[0] for item in self.shared_memory_items}
        for hooks in self._groups.values():
            for hook in hooks:
                for output in hook.outputs:
                    if output.tag in known:
                        continue
                    lock = _Lock()
                    self.shared_memory_items.append(
                        [output.tag, output.shape, output.dtype, lock])
                    # A count companion keeps the legacy readers of
                    # "<tag>_count" working against a hook-produced item, so a
                    # derived item is usable anywhere a device item is.
                    self.shared_memory_items.append(
                        [output.tag + "_count", (1, 1), np.int32, lock])
                    known.add(output.tag)
                    known.add(output.tag + "_count")

    def _validate(self):
        """Reject a graph that cannot run, before anything spawns."""
        produced, consumed = {}, set()
        for executor, hooks in self._groups.items():
            for hook in hooks:
                for output in hook.outputs:
                    if output.tag in produced:
                        raise ValueError(
                            f"Both '{produced[output.tag]}' and '{hook.name}' write "
                            f"'{output.tag}'. Two writers to one item would interleave "
                            "their samples."
                        )
                    produced[output.tag] = hook.name
                consumed.update(i.tag for i in hook.inputs)
        declared = {item[0] for item in self.shared_memory_items}
        missing = sorted(consumed - declared)
        if missing:
            raise KeyError(
                f"These items are observed but never declared or produced: {missing}."
            )
        # A cycle would have two stages each waiting for the other. Feedback
        # belongs on a separate control item, not in the data cascade.
        edges = {}
        for hooks in self._groups.values():
            for hook in hooks:
                edges[hook.name] = [produced[i.tag] for i in hook.inputs
                                    if i.tag in produced]
        self._check_acyclic(edges)
        self._check_picklable()

    def _check_picklable(self):
        """Fail here, with an explanation, rather than at spawn time.

        A hook bound for a process executor is pickled to get there, and a
        hook holding a lambda, a local function or a live handle cannot be.
        Left to the platform, that surfaces as a bare PicklingError naming an
        anonymous function, from inside multiprocessing, with nothing to say
        which hook is at fault or what to do about it. Checking up front costs
        one serialisation per hook.
        """
        for executor, hooks in self._groups.items():
            if self._in_process.get(executor, False):
                # A thread executor shares this interpreter, so nothing is
                # serialised and an unpicklable hook is perfectly fine there.
                continue
            for hook in hooks:
                try:
                    pickle.dumps(hook)
                except Exception as error:
                    raise TypeError(
                        f"Hook '{hook.name}' in executor '{executor}' cannot be "
                        f"sent to another process: {error}\n"
                        "A process executor pickles its hooks to reach them. Either "
                        "build the unpicklable part in the hook's setup() instead of "
                        "storing it on the hook, replace a lambda or local function "
                        "with a module-level one, or register the hook with "
                        f"in_process=True to run it as a thread here: "
                        f"graph.add(hook, executor='{executor}', in_process=True)."
                    ) from error

    @staticmethod
    def _check_acyclic(edges):
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {node: WHITE for node in edges}

        def visit(node, path):
            colour[node] = GREY
            for parent in edges.get(node, []):
                if colour.get(parent, BLACK) == GREY:
                    cycle = " -> ".join(path + [parent])
                    raise ValueError(f"The hook graph contains a cycle: {cycle}.")
                if colour.get(parent, BLACK) == WHITE:
                    visit(parent, path + [parent])
            colour[node] = BLACK

        for node in list(edges):
            if colour[node] == WHITE:
                visit(node, [node])

    # ------------------------------------------------------------------
    def start(self, wait=True):
        """Create the graph's items and start every executor.

        Parameters
        ----------
        wait: bool (optional), default=True
            Block until each executor has attached and subscribed. Without
            this, samples committed immediately after ``start()`` can land
            before anything is listening.
        """
        if self._started:
            return self
        self._collect_outputs()
        self._validate()
        self.log.start()
        # Hook outputs have to exist before an executor tries to attach to
        # them, and they have to be created by a process that outlives the
        # executors, so the graph owner creates them.
        self._smm = SharedMemoryManager(notifier_pool=self.notifier_pool, log=self.log)
        produced = {o.tag for hooks in self._groups.values()
                    for hook in hooks for o in hook.outputs}
        for item in self.shared_memory_items:
            base = item[0][:-len("_count")] if item[0].endswith("_count") else item[0]
            if base in produced:
                self._smm.create_variable(*item)
        self._stop.clear()
        for name, hooks in self._groups.items():
            slot = self.notifier_pool.claim()
            in_process = self._in_process.get(name, False)
            factory = ThreadExecutor if in_process else Executor
            executor = factory(name, hooks, self.shared_memory_items,
                               self.notifier_pool, slot, self.log, self._stop,
                               poll_fallback=self.poll_fallback)
            executor.start()
            self._executors.append(executor)
            self.log.emit(event_log.LIFECYCLE, origin="graph", observer=name,
                          phase="spawned" if not in_process else "threaded",
                          slot=slot, hooks=len(hooks))
        self._started = True
        if wait:
            self._await_subscriptions()
        return self

    def _await_subscriptions(self, timeout=10.0):
        """Wait until every executor has registered for its inputs."""
        expected = {}
        for name, hooks in self._groups.items():
            for hook in hooks:
                for spec in hook.inputs:
                    expected.setdefault(spec.tag, set()).add(name)
        if not expected:
            return
        watcher = SharedMemoryManager(log=self.log)
        declared = {item[0]: item for item in self.shared_memory_items}
        deadline = time.time() + timeout
        for tag in expected:
            while not watcher.find_variable(*declared[tag]):
                if time.time() > deadline:
                    return
                time.sleep(0.01)
        while time.time() < deadline:
            if all(len(watcher.subscribers(tag)) >= len(names)
                   for tag, names in expected.items()):
                break
            time.sleep(0.005)
        watcher.cleanup(parent=False)

    def stop(self, timeout=5.0):
        """Ask every executor to finish, then wait for it."""
        if not self._started:
            return
        self._stop.set()
        # An executor blocked in wait_for would otherwise sit there until its
        # fallback timeout expired, so nudge every slot awake.
        for slot in range(self.notifier_pool.size):
            try:
                self.notifier_pool.signal(slot)
            except Exception:
                pass
        for executor in self._executors:
            executor.join(timeout=timeout)
            # A thread cannot be terminated, only asked; it is a daemon, so a
            # thread that ignores the request dies with the interpreter.
            if executor.is_alive() and hasattr(executor, "terminate"):
                executor.terminate()
        self._executors = []
        self._started = False
        self.log.emit(event_log.LIFECYCLE, origin="graph", phase="stopped")
        self.log.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def describe(self):
        """A readable summary of what is hooked to what.

        Returns
        ----------
        str
            One line per hook, listing its executor, what it observes with
            which criterion, and what it writes.
        """
        lines = []
        for executor, hooks in self._groups.items():
            lines.append(f"executor '{executor}':")
            for hook in hooks:
                observes = ", ".join(
                    f"{i.tag} [{i.criterion.describe()}, {i.mode}"
                    + (f"({i.size})" if i.size else "") + "]"
                    for i in hook.inputs)
                writes = ", ".join(o.tag for o in hook.outputs) or "-"
                lines.append(f"  {hook.name}: observes {observes} -> writes {writes}"
                             + (f" (trigger={hook.trigger})" if len(hook.inputs) > 1 else ""))
        return "\n".join(lines)
