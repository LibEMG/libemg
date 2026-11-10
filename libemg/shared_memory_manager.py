import time
from dataclasses import dataclass

import numpy as np
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

from libemg import event_log


# ----------------------------------------------------------------------
# The state block
#
# Every shared-memory variable gets a companion block of int64 counters that
# says what has happened to it. This is what makes a variable "stateful": a
# reader can tell that it changed, and by how much, without copying it.
#
# The block lives in its own shared-memory segment named "<tag>__state", and
# it is guarded by the same lock as the variable itself, so a reader can take
# the data and the state as one consistent snapshot.
#
# Fields are addressed by these indices rather than by name because the block
# has to be readable from any process without shipping a schema.
# ----------------------------------------------------------------------
STATE_FIELDS = 8
_GENERATION = 0     # bumped once per commit; the thing observers compare against
_TOTAL_SAMPLES = 1  # monotonic count of rows ever committed
_EPOCH = 2          # bumped on reset, so a reader can tell a reset from a wrap
_CLOSED = 3         # the writer is finished; lets a finite source terminate a graph
_COMMITS = 4        # diagnostics: how many commits have landed
_DROPPED = 5        # diagnostics: rows overwritten before somebody read them
_T_LAST_NS = 6      # perf_counter_ns of the newest commit, for latency accounting
_SUBSCRIBERS = 7    # bitmask of notifier slots to wake on commit

STATE_SUFFIX = "__state"
MAX_SUBSCRIBERS = 63


@dataclass(frozen=True)
class Snapshot:
    """What a stateful variable's state block said at one instant.

    Attributes
    ----------
    tag: str
        The variable this describes.
    generation: int
        Commits since the variable was created. An observer that has seen
        generation N knows nothing has changed while this still reads N.
    total_samples: int
        Rows ever committed. Criteria that care about volume rather than
        change (a window increment, say) compare against this.
    epoch: int
        Incremented by :meth:`SharedMemoryManager.reset_state`. A change here
        means the counters went backwards deliberately and any observer state
        derived from them is stale.
    closed: bool
        The writer has finished and will not commit again.
    commits: int
        Diagnostic commit count.
    dropped: int
        Diagnostic count of rows that were overwritten before being read.
    t_last_ns: int
        ``time.perf_counter_ns()`` at the newest commit, in the writer's
        process.
    """

    tag: str
    generation: int
    total_samples: int
    epoch: int
    closed: bool
    commits: int
    dropped: int
    t_last_ns: int


def assign_shared_memory_locks(shared_memory_items):
    """Append a synchronization lock to each shared-memory item in place.

    A modality buffer ``<mod>`` and its sample counter ``<mod>_count`` are
    given the *same* lock object so that the (data, count) pair can be read as
    a single atomic snapshot (see :meth:`SharedMemoryManager.get_variables`).
    Without this, a reader can copy the buffer and its count in two separate
    critical sections and observe a count that runs ahead of the data it just
    copied, which splices dropped/duplicated samples into logged signals.

    Parameters
    ----------
    shared_memory_items : list
        A list of ``[tag, shape, dtype]`` items. A lock is appended to each,
        shared between a tag and its matching ``<tag>_count`` entry.

    Returns
    -------
    list
        The same list, with a lock appended to every item.
    """
    locks = {}
    for item in shared_memory_items:
        tag = item[0]
        base = tag[:-len("_count")] if tag.endswith("_count") else tag
        lock = locks.setdefault(base, Lock())
        item.append(lock)
    return shared_memory_items


class SharedMemoryManager:
    """Attaches to shared-memory variables and tracks what has happened to them.

    Beyond holding the buffers, every variable carries a state block (see
    :class:`Snapshot`) recording how many times it has been written and how
    many samples have arrived. That state is what lets an observer in another
    process tell that a variable changed without copying it, and it is the
    foundation the reactive layer in :mod:`libemg.reactive` is built on.

    Parameters
    ----------
    notifier_pool: NotifierPool or None (optional), default=None
        Used by :meth:`commit` to wake observers in other processes. When
        None, commits still update the state block, so an observer that polls
        the block still sees the change; it just is not woken. See
        :mod:`libemg.reactive`.
    log: EventLog or None (optional), default=None
        Records every commit and notification. Defaults to no logging.
    """

    def __init__(self, notifier_pool=None, log=None):
        self.variables = {}
        self.state = {}
        self.notifier_pool = notifier_pool
        self.log = log if log is not None else event_log.NULL_LOG

    def create_variable(self, tag, shape, type, lock, notifier_pool=None):
        if notifier_pool is not None:
            self.notifier_pool = notifier_pool
        if tag in self.variables.keys():
            print(f"Already have access to this variable: {tag}")
            return True

        # if tag exists already
        if self.find_variable(tag, shape, type, lock):
            print(f'{tag} already exists in shared memory, found variable.')
            return True

        try:
            sm = SharedMemory(tag, create=False)
            sm.unlink()
        except:
            pass

        try:
            type_size = type().itemsize
        except TypeError:
            # Passed in non-callable dtype
            type_size = type.itemsize

        smh = SharedMemory(tag, create=True, size=int(type_size * np.prod(shape)))
        data = np.ndarray((shape),dtype=type,buffer=smh.buf)
        data.fill(0)
        self.variables[tag] = {
            "data" : data,
            "shape": shape,
            "type" : type,
            "smh"  : smh,
            "lock" : lock
        }
        # The creator of a variable is the one that gets a fresh state block,
        # so a new session starts at generation zero rather than inheriting
        # counters from whatever ran last.
        self._attach_state(tag, fresh=True)
        return True

    def find_variable(self, tag, shape, type, lock, notifier_pool=None):
        if notifier_pool is not None:
            self.notifier_pool = notifier_pool
        try:
            type_size = type().itemsize
        except TypeError:
            # Passed in non-callable dtype
            type_size = type.itemsize
        try:
            smh = SharedMemory(tag, size=int(type_size * np.prod(shape)))
            # create a new numpy array that uses the shared memory
            data = np.ndarray((shape), dtype=type, buffer=smh.buf)
            self.variables[tag] = {
                "data" : data,
                "shape": shape,
                "type" : type,
                "smh"  : smh,
                "lock" : lock
        }
            self._attach_state(tag, fresh=False)
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    # state blocks
    # ------------------------------------------------------------------
    def _attach_state(self, tag, fresh):
        """Attach (creating if needed) the state block belonging to ``tag``.

        Either side of a variable may attach first, and both may race, so this
        tries to create and falls back to opening what somebody else created.
        ``fresh=True`` additionally discards a block left behind by a previous
        run, which is what the creator of the data variable wants.
        """
        name = tag + STATE_SUFFIX
        size = STATE_FIELDS * np.dtype(np.int64).itemsize
        if fresh:
            try:
                stale = SharedMemory(name, create=False)
                stale.close()
                stale.unlink()
            except Exception:
                pass
        try:
            smh = SharedMemory(name, create=True, size=size)
            created = True
        except FileExistsError:
            smh = SharedMemory(name, create=False)
            created = False
        block = np.ndarray((STATE_FIELDS,), dtype=np.int64, buffer=smh.buf)
        if created:
            block[:] = 0
        self.state[tag] = {"block": block, "smh": smh}
        return True

    def _block(self, tag):
        assert tag in self.state, f"No state block attached for {tag}."
        return self.state[tag]["block"]

    def _read_snapshot(self, tag):
        """Build a Snapshot. The caller must already hold the variable's lock."""
        block = self._block(tag)
        return Snapshot(tag=tag,
                        generation=int(block[_GENERATION]),
                        total_samples=int(block[_TOTAL_SAMPLES]),
                        epoch=int(block[_EPOCH]),
                        closed=bool(block[_CLOSED]),
                        commits=int(block[_COMMITS]),
                        dropped=int(block[_DROPPED]),
                        t_last_ns=int(block[_T_LAST_NS]))

    def snapshot(self, tag):
        """Read a variable's state without copying the variable.

        This is the cheap read the reactive layer uses to decide whether
        anything needs doing: a handful of integers under the variable's lock,
        rather than a copy of the whole buffer.

        Parameters
        ----------
        tag: str
            The variable to inspect.

        Returns
        ----------
        Snapshot
            The state block's contents at the instant it was read.
        """
        with self.variables[tag]["lock"]:
            return self._read_snapshot(tag)

    def snapshots(self, tags):
        """Read several variables' states, each under its own lock.

        Returns
        ----------
        dict
            Mapping from tag to :class:`Snapshot`.
        """
        return {tag: self.snapshot(tag) for tag in tags}

    def reset_state(self, tag):
        """Zero a variable's counters and bump its epoch.

        The epoch is what tells an observer that the counters went backwards on
        purpose, so it can discard state it derived from the old ones instead
        of concluding that nothing has happened for a very long time.
        """
        with self.variables[tag]["lock"]:
            block = self._block(tag)
            epoch = int(block[_EPOCH]) + 1
            subscribers = int(block[_SUBSCRIBERS])
            block[:] = 0
            block[_EPOCH] = epoch
            # Subscriptions describe who is listening, not what has happened,
            # so a reset must not silently unsubscribe everybody.
            block[_SUBSCRIBERS] = subscribers

    def mark_closed(self, tag):
        """Record that no further commits will be made to ``tag``.

        A finite source calls this when it runs out of data. Observers use it
        to shut down once they have consumed everything, which is what lets an
        offline run through a recording terminate rather than wait forever.
        """
        with self.variables[tag]["lock"]:
            self._block(tag)[_CLOSED] = 1
        self._wake(tag)
        self.log.emit(event_log.LIFECYCLE, origin=tag, phase="closed")

    # ------------------------------------------------------------------
    # committing
    # ------------------------------------------------------------------
    def commit(self, tag, samples, count_tag=None):
        """Write new samples and record that the variable changed.

        This is the write path stateful variables are meant to use. It does
        what the streamers' read-modify-write idiom did -- prepend the new rows
        so index 0 stays the newest sample -- and additionally advances the
        state block and wakes anything subscribed to the variable, all under a
        single acquisition of the variable's lock.

        Taking the lock once matters: writing the buffer and its counter
        separately lets a reader observe a count that runs ahead of the data,
        which splices dropped or duplicated samples into whatever it computes.

        Parameters
        ----------
        tag: str
            The variable to write to.
        samples: numpy.ndarray
            The new rows, oldest first, as the device produced them. Pass a 1-D
            array for a single sample.
        count_tag: str or None (optional), default=None
            The legacy sample counter to keep in step. Defaults to
            ``tag + "_count"`` when such a variable is attached, so existing
            readers of that counter keep working untouched.

        Returns
        ----------
        Snapshot
            The variable's state after the commit.
        """
        samples = np.atleast_2d(samples)
        if count_tag is None:
            candidate = tag + "_count"
            count_tag = candidate if candidate in self.variables else None

        with self.variables[tag]["lock"]:
            data = self.variables[tag]["data"]
            capacity = data.shape[0]
            arriving = samples.shape[0]
            # Newest-first is the layout every existing reader expects, so the
            # newest arriving row has to end up at index 0.
            incoming = np.flip(samples, 0)
            if arriving >= capacity:
                # More arrived than the buffer can hold: keep the newest.
                data[:] = incoming[:capacity]
                overwritten = arriving - capacity
            else:
                data[arriving:] = data[:capacity - arriving]
                data[:arriving] = incoming
                overwritten = 0

            block = self._block(tag)
            block[_GENERATION] += 1
            block[_TOTAL_SAMPLES] += arriving
            block[_COMMITS] += 1
            block[_DROPPED] += overwritten
            block[_T_LAST_NS] = time.perf_counter_ns()
            if count_tag is not None and count_tag in self.variables:
                self.variables[count_tag]["data"][:] += arriving
            snapshot = self._read_snapshot(tag)

        # Notifying happens outside the variable's lock on purpose. A waiter
        # holds its notifier's lock and then wants this one; if the writer held
        # this one and then wanted the notifier's, the two orders would close a
        # cycle and deadlock.
        self._wake(tag)
        if self.log.enabled:
            self.log.emit(event_log.COMMIT, origin=tag,
                          generation=snapshot.generation,
                          samples=arriving,
                          total_samples=snapshot.total_samples,
                          dropped=overwritten)
        return snapshot

    def apply(self, tag, fn, count_fn=None, count_tag=None):
        """Transform a variable arbitrarily, and record that it changed.

        :meth:`commit` is the write path for arriving samples. This is the one
        for everything else: a value that is set rather than appended, a
        counter used as a signal, a buffer rewritten by a caller-supplied rule.
        It is what :class:`~libemg.output_writer.SharedMemoryOutputWriter`
        writes through, which is how an adaptation flag or a slice of
        environment feedback becomes something a hook can observe.

        The variable and its counter are updated under a single acquisition of
        the variable's lock, then the state block advances and subscribers are
        woken. Advancing by the counter's own delta means the sample total
        follows whatever rule ``count_fn`` implements, rather than this method
        having to guess what a transform did.

        Parameters
        ----------
        tag: str
            The variable to transform.
        fn: callable
            Called with the current data; its return value is written back.
        count_fn: callable or None (optional), default=None
            Called with the current counter; its return value is written back.
            When None the counter is left alone and the sample total advances
            by one, which is what a flag wants.
        count_tag: str or None (optional), default=None
            The counter variable. Defaults to ``tag + "_count"`` when one is
            attached.

        Returns
        ----------
        Snapshot
            The variable's state after the change.
        """
        if count_tag is None:
            candidate = tag + "_count"
            count_tag = candidate if candidate in self.variables else None

        with self.variables[tag]["lock"]:
            data = self.variables[tag]["data"]
            data[:] = fn(data)
            added = 1
            if count_fn is not None and count_tag is not None:
                counter = self.variables[count_tag]["data"]
                before = int(counter.flat[0])
                counter[:] = count_fn(counter)
                added = max(0, int(counter.flat[0]) - before)

            block = self._block(tag)
            block[_GENERATION] += 1
            block[_TOTAL_SAMPLES] += added
            block[_COMMITS] += 1
            block[_T_LAST_NS] = time.perf_counter_ns()
            snapshot = self._read_snapshot(tag)

        # Outside the lock, for the ordering reason given in commit().
        self._wake(tag)
        if self.log.enabled:
            self.log.emit(event_log.COMMIT, origin=tag,
                          generation=snapshot.generation, samples=added,
                          total_samples=snapshot.total_samples)
        return snapshot

    # ------------------------------------------------------------------
    # subscription and notification
    # ------------------------------------------------------------------
    def subscribe(self, tag, slot):
        """Ask to be woken when ``tag`` is committed to.

        Parameters
        ----------
        tag: str
            The variable to watch.
        slot: int
            A notifier slot from :meth:`NotifierPool.claim`. Slots are recorded
            in the variable's state block, so a writer that started before the
            observer existed still finds it.
        """
        assert 0 <= slot < MAX_SUBSCRIBERS, f"Notifier slot {slot} out of range."
        with self.variables[tag]["lock"]:
            self._block(tag)[_SUBSCRIBERS] |= np.int64(1) << np.int64(slot)

    def unsubscribe(self, tag, slot):
        """Stop being woken when ``tag`` is committed to."""
        with self.variables[tag]["lock"]:
            self._block(tag)[_SUBSCRIBERS] &= ~(np.int64(1) << np.int64(slot))

    def subscribers(self, tag):
        """Slots currently subscribed to ``tag``.

        Returns
        ----------
        list
            The subscribed notifier slot numbers.
        """
        with self.variables[tag]["lock"]:
            mask = int(self._block(tag)[_SUBSCRIBERS])
        return [s for s in range(MAX_SUBSCRIBERS) if mask & (1 << s)]

    def _wake(self, tag):
        """Signal every slot subscribed to ``tag``."""
        if self.notifier_pool is None:
            return
        with self.variables[tag]["lock"]:
            mask = int(self._block(tag)[_SUBSCRIBERS])
        if not mask:
            return
        for slot in range(MAX_SUBSCRIBERS):
            if mask & (1 << slot):
                self.notifier_pool.signal(slot)
                if self.log.enabled:
                    self.log.emit(event_log.NOTIFY, origin=tag,
                                  observer=f"slot{slot}")

    # ------------------------------------------------------------------
    # stateful reads
    # ------------------------------------------------------------------
    def read_window(self, tag, num_samples, chronological=True):
        """Copy the newest ``num_samples`` rows and the state that goes with them.

        This is the read a windowing or model hook wants. Unlike
        :meth:`get_variable` the copy is bounded by the window rather than by
        the buffer, so it holds the writer's lock for less time and allocates
        less.

        Parameters
        ----------
        tag: str
            The variable to read.
        num_samples: int
            How many of the newest rows to take. Clamped to the buffer size.
        chronological: bool (optional), default=True
            Return oldest-first, which is the orientation feature extraction
            and models expect. Pass False for the newest-first layout the
            buffer itself uses.

        Returns
        ----------
        samples: numpy.ndarray
            The requested rows.
        snapshot: Snapshot
            The variable's state at the instant of the read.
        """
        with self.variables[tag]["lock"]:
            data = self.variables[tag]["data"]
            n = int(min(num_samples, data.shape[0]))
            window = data[:n].copy()
            snapshot = self._read_snapshot(tag)
        if chronological:
            window = np.flip(window, 0)
        return window, snapshot

    def read_since(self, tag, last_total, chronological=True):
        """Copy the rows committed since ``last_total`` samples had arrived.

        Parameters
        ----------
        tag: str
            The variable to read.
        last_total: int
            The ``total_samples`` value this caller has already consumed to.
        chronological: bool (optional), default=True
            Return oldest-first.

        Returns
        ----------
        samples: numpy.ndarray
            The rows that arrived since ``last_total``. Empty if none did.
        snapshot: Snapshot
            The variable's state at the instant of the read.
        lost: int
            How many of those rows the buffer had already overwritten, and so
            could not be returned.
        """
        with self.variables[tag]["lock"]:
            data = self.variables[tag]["data"]
            snapshot = self._read_snapshot(tag)
            new = snapshot.total_samples - int(last_total)
            if new <= 0:
                return data[:0].copy(), snapshot, 0
            lost = max(0, new - data.shape[0])
            samples = data[:new - lost].copy()
        if chronological:
            samples = np.flip(samples, 0)
        return samples, snapshot, lost

    def get_variable(self, tag):
        assert tag in self.variables.keys()
        with self.variables[tag]["lock"]:
            return self.variables[tag]["data"].copy()

    def get_variables(self, tags):
        """Atomically copy several variables as one consistent snapshot.

        Every distinct lock guarding the requested tags is held for the whole
        copy, so the returned values reflect the same instant in time. When the
        tags share a single lock (the convention established by
        :func:`assign_shared_memory_locks` for a ``<mod>``/``<mod>_count``
        pair), that lock is acquired exactly once.

        Parameters
        ----------
        tags : list
            The shared-memory tags to copy together.

        Returns
        -------
        dict
            A mapping from each requested tag to a copy of its current data.
        """
        for tag in tags:
            assert tag in self.variables.keys()
        # Collapse duplicate lock objects, then acquire each distinct lock once
        # in a deterministic (id-sorted) order to avoid deadlock if a caller
        # ever groups tags that don't share a lock.
        distinct_locks = {}
        for tag in tags:
            lock = self.variables[tag]["lock"]
            distinct_locks[id(lock)] = lock
        ordered_locks = [distinct_locks[key] for key in sorted(distinct_locks.keys())]
        acquired = []
        try:
            for lock in ordered_locks:
                lock.acquire()
                acquired.append(lock)
            return {tag: self.variables[tag]["data"].copy() for tag in tags}
        finally:
            for lock in reversed(acquired):
                lock.release()

    def get_samples_since(self, tag, last_count, count_tag=None):
        """Copy only the samples a writer has added since ``last_count``.

        The counter and the buffer are read under the same lock acquisition as
        :meth:`get_variables`, so the returned rows and count describe one
        instant. Unlike :meth:`get_variables` the copy is proportional to what
        actually arrived rather than to the whole buffer, which matters for a
        poller running every few milliseconds: it holds the writer's lock for
        less time, and -- because a poller that allocates a fresh copy of the
        entire buffer hundreds of times a second is what drives the interpreter
        into a garbage collection -- it keeps that thread from being the one
        that pays for everyone else's finalizers.

        Parameters
        ----------
        tag : str
            The buffer to read from. Newest sample first, as written by the
            streamers.
        last_count : int
            The counter value this caller has already consumed up to.
        count_tag : str or None
            The counter tag. Defaults to ``tag + "_count"``.

        Returns
        -------
        count : int
            The counter value at the instant of the read.
        samples : numpy.ndarray
            The rows that arrived since ``last_count``, newest first. Empty when
            nothing new arrived.
        dropped : int
            How many of those samples the buffer had already overwritten, and so
            could not be returned.
        """
        count_tag = count_tag if count_tag is not None else tag + "_count"
        for t in (tag, count_tag):
            assert t in self.variables.keys()
        distinct_locks = {}
        for t in (tag, count_tag):
            lock = self.variables[t]["lock"]
            distinct_locks[id(lock)] = lock
        ordered_locks = [distinct_locks[key] for key in sorted(distinct_locks.keys())]
        acquired = []
        try:
            for lock in ordered_locks:
                lock.acquire()
                acquired.append(lock)
            data = self.variables[tag]["data"]
            count = int(self.variables[count_tag]["data"][0, 0])
            new = count - int(last_count)
            if new <= 0:
                return count, data[:0].copy(), 0
            dropped = max(0, new - data.shape[0])
            return count, data[:new - dropped].copy(), dropped
        finally:
            for lock in reversed(acquired):
                lock.release()

    def modify_variable(self, tag, fn):
        assert tag in self.variables.keys()
        with self.variables[tag]["lock"]:
            self.variables[tag]["data"][:] = fn(self.variables[tag]["data"])

    def cleanup(self, parent = True):
        for k in self.variables.keys():
            self.variables[k]["smh"].close()
            if parent:
                self.variables[k]["smh"].unlink()
        # State blocks are separate segments, so they leak unless they are
        # released alongside the variables they describe.
        for k in self.state.keys():
            try:
                self.state[k]["smh"].close()
                if parent:
                    self.state[k]["smh"].unlink()
            except Exception:
                pass
        self.variables = {}
        self.state = {}
    
    def get_variable_list(self):
        return [[k, self.variables[k]["shape"], self.variables[k]["type"]] for k in self.variables.keys()]
    
    def get_shared_memory_items(self):
        return [[i, self.variables[i]["shape"], self.variables[i]["type"], self.variables[i]["lock"]] for i in self.variables]