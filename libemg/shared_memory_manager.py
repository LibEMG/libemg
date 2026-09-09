import numpy as np
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory


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
    def __init__(self):
        self.variables = {}

    def create_variable(self, tag, shape, type, lock):
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
        return True

    def find_variable(self, tag, shape, type, lock):
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
            return True
        except FileNotFoundError:
            return False
        
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
        self.variables = {}
    
    def get_variable_list(self):
        return [[k, self.variables[k]["shape"], self.variables[k]["type"]] for k in self.variables.keys()]
    
    def get_shared_memory_items(self):
        return [[i, self.variables[i]["shape"], self.variables[i]["type"], self.variables[i]["lock"]] for i in self.variables]