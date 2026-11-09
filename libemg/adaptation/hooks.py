"""Hooks for incremental (user-in-the-loop) learning.

What the adaptation loop is
---------------------------
Three parties pass work between them while a person is using the system:

1. The **environment** decides, from what the person did, what the model
   *should* have said, and writes that judgement to ``environment_feedback``.
2. The **memory** side pairs each judgement with the model input that produced
   it, accumulating a slice per trial and announcing a finished slice on
   ``memory_flag``.
3. The **adaptation** side folds finished slices into the model, trains, saves
   the new weights and announces them on ``adapt_flag``.

Then the running predictor picks up the new weights and the cycle closes.

Every one of those handoffs used to be a poll. :class:`MemoryManager` and
:class:`AdaptationManager` are unthrottled loops, and the memory loop copies
both the feedback buffer and the whole model-input buffer on every pass just to
learn whether the feedback counter moved. The predictor re-read the adaptation
flag on every iteration of its own loop.

The hooks here express the same three handoffs as reactions. Each stage
declares what it observes and the terms on which a change counts, and the
writes that used to be polled for now announce themselves. The chain is
``environment_feedback`` to ``memory_flag`` to ``adapt_flag``, each stage
waking the next.

Why the criteria differ
-----------------------
The stages disagree about what a change means, which is the same disagreement
the rest of the reactive layer is built around.

- Memory is dirty on **any** feedback row, because every judgement the person
  produced is data worth keeping.
- Adaptation is dirty on a **completed slice**, not on a row. Training on a
  fraction of a trial is worse than waiting for the whole one.
- The predictor is dirty only on a **new model**, which is rare and expensive
  to act on, so it must not be confused with either of the above.

Examples
---------
>>> from libemg.adaptation.hooks import MemoryHook, AdaptationHook
>>> from libemg.reactive import ReactiveGraph
>>> graph = ReactiveGraph(adaptation_items, log=log)
>>> graph.add(MemoryHook(memory, save_dir='memories/'), executor='memory')
>>> graph.add(AdaptationHook(model, load_dir='memories/', save_dir='models/'),
...           executor='adapt')
>>> graph.start()
"""

import os
import pickle
import time

import numpy as np

from libemg.reactive import (DELTA, FULL, STATE, Criterion, Hook, Input,
                             OnCommit)


def ensure_directory(directory):
    """Create ``directory`` if it is not already there.

    exist_ok, not a prior existence check. The memory side's save directory and
    the adaptation side's load directory are by design the same one, and their
    executors start at the same moment, so a check followed by a create lets
    both see it missing and the loser die with FileExistsError.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)


class OnNewSlice(Criterion):
    """Dirty when the slice counter has advanced past what was consumed.

    The adaptation side's definition of a meaningful change. The counter it
    watches is written as a running total of completed trials, so the gap
    between it and what this observer has folded in is exactly the number of
    slices still to load. Acting on anything less than a completed slice means
    training on a fragment of a trial.
    """

    def is_dirty(self, snapshot, memory):
        return self._value(snapshot) > memory.samples

    def consume(self, snapshot, memory):
        memory.samples = self._value(snapshot)
        memory.generation = snapshot.generation
        memory.fired_at = time.perf_counter()
        memory.epoch = snapshot.epoch

    @staticmethod
    def _value(snapshot):
        # The flag's own value is carried by the hook, which reads it; from the
        # state block all that is visible is that it moved. Generation is the
        # conservative stand-in and is what the hook reconciles against.
        return snapshot.generation

    def describe(self):
        return "OnNewSlice()"


class MemoryHook(Hook):
    """Assemble training slices from what the environment judged.

    Observes ``environment_feedback`` and, for each judgement, finds the model
    input that produced it and appends the pair to a memory. A judgement
    carrying a new trial number closes the previous slice: the memory is saved
    and ``memory_flag`` is advanced, which is what wakes the adaptation side.

    This is :class:`~libemg.adaptation.managers.MemoryManager` expressed as a
    reaction. The behaviour is the same; what changes is that it runs when
    feedback arrives instead of asking whether any has, and that asking used to
    copy the whole model-input buffer as well as the feedback buffer.

    Parameters
    ----------
    memory: Any
        The memory object. Needs ``append``, ``save``, ``reset`` and ``__add__``,
        matching what :class:`~libemg.adaptation.managers.MemoryManager` expects.
    save_dir: str
        Where finished slices are written. Must match the adaptation side's
        ``load_dir``.
    name: str (optional), default='memory'
        Hook name, as it appears in the event log.
    feedback_tag: str (optional), default='environment_feedback'
        The item the environment writes its judgements to. Each row is
        ``[timestamp, trial, feedback...]``.
    input_tag: str (optional), default='model_input'
        The item the predictor writes its inputs to. Each row is
        ``[timestamp, features...]``.
    flag_tag: str (optional), default='memory_flag'
        The item advanced when a slice is finished.

    Notes
    -----
    Feedback rows are joined to model inputs by timestamp rather than by
    position, because the two are written by different processes at different
    rates and their indices do not correspond. A row whose timestamp is not
    found is dropped and counted; see :meth:`unmatched`.
    """

    def __init__(self, memory, save_dir, name="memory",
                 feedback_tag="environment_feedback", input_tag="model_input",
                 flag_tag="memory_flag"):
        super().__init__(
            name,
            inputs=[
                # Any judgement at all is worth keeping, so this is dirty on a
                # single row.
                Input(feedback_tag, OnCommit(), mode=DELTA),
                # The inputs are looked up by timestamp, not consumed, so this
                # is read without being a trigger.
                Input(input_tag, _NeverDirty(), mode=FULL),
            ],
            outputs=[],
            # The slice flag is set, not appended to, so this hook writes it
            # itself rather than returning it. It still has to be attached in
            # the executor's process, which is what declaring it here does.
            attach=[flag_tag],
        )
        self.memory = memory
        self.save_dir = save_dir
        self.feedback_tag = feedback_tag
        self.input_tag = input_tag
        self.flag_tag = flag_tag
        self.trial_counter = 1
        self._unmatched = 0
        self._appended = 0
        self._smm = None
        self._log = None

    def setup(self, context):
        ensure_directory(self.save_dir)
        self.memory.reset()
        # The slice flag is advanced directly rather than declared as an
        # output, because it is a running count that this hook owns rather than
        # a buffer of samples the runtime should append to.
        self._smm = context.smm
        self._log = context.log

    def step(self, data, snapshots):
        feedback = data[self.feedback_tag]
        if feedback.shape[0] == 0:
            return None
        inputs = data[self.input_tag]

        for row in feedback:
            timestamp, trial = row[0], row[1]
            if trial != self.trial_counter:
                self._close_slice()
                self.trial_counter = trial
            match = np.where(inputs[:, 0] == timestamp)[0]
            if match.size == 0:
                # The predictor's buffer has already rolled past this
                # timestamp, so the pair cannot be reconstructed. Dropping it
                # silently would look like the person simply produced less
                # data, so it is counted instead.
                self._unmatched += 1
                continue
            self.memory.append(row[2:], inputs[match, 1:], trial - 1)
            self._appended += 1
        return None

    def _close_slice(self):
        """Save the finished slice and tell the adaptation side it exists."""
        path = os.path.join(self.save_dir, f"memory_{int(self.trial_counter)}.pkl")
        self.memory.save(path)
        if self._smm is not None:
            if self.flag_tag not in self._smm.variables:
                # Announcing is the whole point of closing a slice. Silently
                # skipping it produced a run where slices piled up on disk and
                # nothing downstream ever heard about them, with no error
                # anywhere to explain why adaptation never happened.
                raise KeyError(
                    f"'{self.name}' cannot announce a finished slice: shared-memory "
                    f"item '{self.flag_tag}' is not attached. Declare it on the hook "
                    "(it is passed to attach=) and in the graph's items."
                )
            value = int(self.trial_counter)
            self._smm.apply(self.flag_tag, lambda block: value)
        self.memory.reset()

    def flush(self):
        """Close the slice in progress. Call when a session ends."""
        self._close_slice()

    def unmatched(self):
        """How many feedback rows had no model input to pair with.

        Returns
        ----------
        int
            A non-zero count means the predictor's input buffer is too small
            for the delay between a prediction and the environment's judgement
            of it, and that training data is being lost.
        """
        return self._unmatched

    def appended(self):
        """How many feedback and input pairs reached the memory."""
        return self._appended


class _NeverDirty(Criterion):
    """For an input that is read but never triggers.

    A hook sometimes needs a second item's contents without that item's
    arrival being a reason to run. Declaring it with this criterion delivers
    the data on every run without ever firing one.
    """

    def is_dirty(self, snapshot, memory):
        return False

    def describe(self):
        return "NeverDirty()"


class AdaptationHook(Hook):
    """Fold finished slices into the model and publish the result.

    Observes ``memory_flag``, loads whatever slices have appeared since it last
    looked, trains, saves the new weights and advances ``adapt_flag``, which is
    what tells a running predictor to swap models.

    This is :class:`~libemg.adaptation.managers.AdaptationManager` expressed as
    a reaction, with one deliberate behavioural difference. The manager calls
    ``adapt`` on every pass of its loop, including passes where no new memory
    arrived, so it retrains continuously on unchanged data and republishes a
    model each time. This hook trains when a slice arrives. Pass
    ``continuous=True`` for the old behaviour.

    Parameters
    ----------
    model: Any
        Needs ``adapt``, ``save`` and ``load``.
    load_dir: str
        Where slices are read from. Must match the memory side's ``save_dir``.
    save_dir: str
        Where adapted models are written. Must match the predictor's
        ``file_path``, since that is where it looks for them.
    initial_memory_loc: str or None (optional), default=None
        A memory to start from, typically from screen-guided training. Strongly
        worth providing: adapting from nothing is unstable early on.
    name: str (optional), default='adaptation'
        Hook name, as it appears in the event log.
    flag_tag: str (optional), default='memory_flag'
        The item announcing finished slices.
    notify_tag: str (optional), default='adapt_flag'
        The item advanced when a new model is ready.
    notify: bool (optional), default=True
        Whether to announce new models at all. False trains without ever
        swapping the running model, which is what an offline comparison wants.
    stop_after: int or None (optional), default=None
        Stop adapting once this many slices have been folded in.
    continuous: bool (optional), default=False
        Retrain on every wake rather than only when a slice arrives.
    """

    def __init__(self, model, load_dir, save_dir, initial_memory_loc=None,
                 name="adaptation", flag_tag="memory_flag",
                 notify_tag="adapt_flag", notify=True, stop_after=None,
                 continuous=False):
        super().__init__(
            name,
            inputs=[Input(flag_tag, OnNewSlice(), mode=FULL)],
            outputs=[],
            attach=[notify_tag],
        )
        self.model = model
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.initial_memory_loc = initial_memory_loc
        self.flag_tag = flag_tag
        self.notify_tag = notify_tag
        self.notify = notify
        self.stop_after = stop_after
        self.continuous = continuous
        self.memory = None
        self.memory_count = 0
        self.adaptation_count = 0
        # True while a slice was announced but could not be read yet. The
        # retry in _load_when_ready normally clears it within the same step;
        # this records the case where it did not, so pending() can report it.
        self._pending = False
        self._started = None
        self._smm = None
        self._log = None

    def setup(self, context):
        ensure_directory(self.save_dir)
        ensure_directory(self.load_dir)
        # Loaded here rather than in __init__ so the hook stays picklable for a
        # spawned executor, and so a large initial memory is not carried across
        # the process boundary.
        if self.initial_memory_loc is not None:
            self.memory = self._load(self.initial_memory_loc)
        self._smm = context.smm
        self._log = context.log
        self._started = time.time()

    def step(self, data, snapshots):
        # Checked before loading, not after. Checking afterwards left slices
        # being read off disk and folded into memory forever while adapt() was
        # never called again, so a long session grew its held memory without
        # bound for no benefit.
        if self.stop_after is not None and self.memory_count >= self.stop_after:
            return None

        flag = data[self.flag_tag]
        available = int(np.asarray(flag).flat[0])
        if self.stop_after is not None:
            available = min(available, self.stop_after)
        loaded = 0
        while self.memory_count < available:
            self.memory_count += 1
            path = os.path.join(self.load_dir, f"memory_{self.memory_count}.pkl")
            slice_ = self._load_when_ready(path)
            if slice_ is None:
                # The announcement beat the file to disk. Step back so this
                # slice is loaded rather than skipped.
                self.memory_count -= 1
                self._pending = True
                break
            self.memory = slice_ if self.memory is None else self.memory + slice_
            loaded += 1
        else:
            self._pending = False
        if self.memory is None:
            return None
        if loaded == 0 and not self.continuous:
            return None

        losses = self.model.adapt(self.memory)
        self.adaptation_count += 1
        model_path = os.path.join(self.save_dir, f"mdl{self.adaptation_count}.pkl")
        self.model.save(model_path)
        self._record_losses(losses)
        if self.notify and self._smm is not None \
                and self.notify_tag in self._smm.variables:
            number = self.adaptation_count
            self._smm.apply(self.notify_tag, lambda block: number)
        return None

    def _record_losses(self, losses):
        try:
            with open(os.path.join(self.save_dir, "losses.txt"), "a") as handle:
                handle.write(f"{time.time() - self._started}\t{losses}\n")
        except Exception:
            # Losing the loss record must not stop adaptation.
            pass

    def teardown(self):
        if self.model is not None:
            try:
                self.model.save(os.path.join(self.save_dir, "model_final.pkl"))
            except Exception:
                pass

    def pending(self):
        """Whether a slice was announced but could not be read.

        Returns
        ----------
        bool
            True means a slice is still outstanding. It is retried on the next
            wake, which arrives with the next completed trial; if that was the
            last trial of a session, call :meth:`step` once more or check this
            before shutting down.
        """
        return self._pending

    def _load_when_ready(self, location, attempts=20, interval=0.01):
        """Load a slice, tolerating an announcement that beat the file to disk.

        The memory side writes a slice and then advances the flag, so by the
        time this runs the file is normally there. A brief retry covers the
        window where it is not, and also the window where the file exists but
        is still being written, which surfaces as an unpickling error rather
        than a missing file.

        Returns
        ----------
        object or None
            The loaded slice, or None if it never became readable. The caller
            steps its counter back so the slice is retried rather than skipped.
        """
        for attempt in range(attempts):
            try:
                return self._load(location)
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                if attempt == attempts - 1:
                    return None
                time.sleep(interval)
        return None

    @staticmethod
    def _load(location):
        with open(location, "rb") as handle:
            return pickle.load(handle)


class ModelSwapHook(Hook):
    """React to a newly adapted model becoming available.

    The last link in the chain. The predictor itself watches ``adapt_flag``
    from inside its own streaming process, because that is where the model it
    would replace actually lives; see
    :meth:`~libemg.emg_predictor.OnlineStreamer.on_model_update`. This hook is
    for everything *else* that wants to know: a plot marking when the model
    changed, a log, a counter of how often adaptation reached the user.

    Parameters
    ----------
    name: str
        Hook name.
    fn: callable
        Called as ``fn(number)`` with the new model number.
    notify_tag: str (optional), default='adapt_flag'
        The item to observe.
    """

    def __init__(self, name, fn, notify_tag="adapt_flag"):
        super().__init__(name, inputs=[Input(notify_tag, OnCommit(), mode=FULL)])
        self.fn = fn
        self.notify_tag = notify_tag

    def step(self, data, snapshots):
        number = int(np.asarray(data[self.notify_tag]).flat[0])
        if number >= 0:
            self.fn(number)
        return None
