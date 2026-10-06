"""Coverage for the incremental-learning hooks and the write path they need.

Three layers, in the order they depend on each other. First
:meth:`SharedMemoryManager.apply`, because every handoff in the chain is a
value that is *set* rather than appended and that is the write path for it.
Then each hook on its own, driven by hand: ``setup()`` with a context the test
builds and ``step()`` with input dicts the test builds, so a claim about
pairing or about when training happens is checked arithmetically rather than by
racing a scheduler. Last, the chain across processes, where the only thing that
can be asserted is what actually arrived.

Everything a spawned executor needs -- hooks, the memory, the model -- is
defined at module level, because Windows spawns rather than forks and anything
nested inside a test function cannot be pickled into the child. Every item is
named after the test that made it, so a segment leaked by one test cannot be
found by another.
"""
import itertools
import os
import pickle
import threading
import time

import numpy as np
import pytest
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

from libemg import event_log
from libemg.adaptation.hooks import (AdaptationHook, MemoryHook, ModelSwapHook,
                                     OnNewSlice, _NeverDirty)
from libemg.event_log import EventLog
from libemg.reactive import (CriterionMemory, FULL, HookContext, NotifierPool,
                             OnCommit, ReactiveGraph)
from libemg.shared_memory_manager import SharedMemoryManager, Snapshot


# ======================================================================
# helpers
# ======================================================================
def _snapshot(generation=1, total_samples=1, epoch=0, closed=False, dropped=0,
              tag="item"):
    """A state block reading, built by hand rather than by a writer."""
    return Snapshot(tag=tag, generation=generation, total_samples=total_samples,
                    epoch=epoch, closed=closed, commits=generation,
                    dropped=dropped, t_last_ns=1)


def _latest_training(save_dir):
    """The most recently saved model, or None if nothing has trained yet."""
    if not os.path.isdir(save_dir):
        return None
    models = [name for name in os.listdir(save_dir) if name.startswith("mdl")]
    if not models:
        return None
    latest = max(int(name[3:-4]) for name in models)
    return _load(os.path.join(save_dir, f"mdl{latest}.pkl"))


def _wait_until(predicate, timeout=30.0, interval=0.01):
    """Poll ``predicate`` until it holds or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _context(smm):
    """What an executor hands a hook at setup time, without an executor."""
    return HookContext(smm=smm, log=event_log.NULL_LOG, executor="hand-driven")


def _value(smm, tag):
    """The scalar a flag item currently holds."""
    return int(np.asarray(smm.get_variable(tag)).flat[0])


class Memory:
    """The smallest thing MemoryHook will accept: append, reset, save and +.

    A real libemg memory holds windows and one-hot targets; what the hooks
    require of it is only this interface, so the test uses something it can
    check the contents of.
    """

    def __init__(self):
        self.feedback = []
        self.inputs = []
        self.trials = []

    def append(self, feedback, model_input, trial):
        self.feedback.append(np.asarray(feedback).ravel())
        self.inputs.append(np.asarray(model_input).ravel())
        self.trials.append(int(trial))

    def reset(self):
        self.feedback = []
        self.inputs = []
        self.trials = []

    def __add__(self, other):
        combined = Memory()
        combined.feedback = self.feedback + other.feedback
        combined.inputs = self.inputs + other.inputs
        combined.trials = self.trials + other.trials
        return combined

    def __len__(self):
        return len(self.feedback)

    def save(self, location):
        with open(location, "wb") as handle:
            pickle.dump(self, handle)


class Model:
    """Records what it was trained on, so a test can assert values not counts."""

    def __init__(self):
        self.fits = []
        self.seen = []

    def adapt(self, memory):
        self.fits.append(len(memory))
        self.seen.append(list(memory.trials))
        return [0.5 / len(self.fits)]

    def save(self, location):
        with open(location, "wb") as handle:
            pickle.dump(self, handle)


def _load(location):
    with open(location, "rb") as handle:
        return pickle.load(handle)


def _slice_of(count, trial, first_index=1):
    """A memory holding ``count`` rows, indexed so its contents are identifiable."""
    memory = Memory()
    for index in range(first_index, first_index + count):
        memory.append([100.0 * index, 100.0 * index + 1],
                      [10.0 * index, 10.0 * index + 1], trial)
    return memory


def _write_slice(directory, number, count, trial=0, first_index=1):
    """Put a slice on disk where an AdaptationHook will look for it."""
    os.makedirs(directory, exist_ok=True)
    memory = _slice_of(count, trial, first_index)
    memory.save(os.path.join(directory, f"memory_{number}.pkl"))
    return memory


def _input_row(stamp, index):
    """One model input: a timestamp and four features that name their index."""
    return np.hstack(([float(stamp)], 10.0 * index + np.arange(4.0)))


def _feedback_row(stamp, trial, index):
    """One judgement: a timestamp, the trial it belongs to, two feedback values."""
    return np.array([float(stamp), float(trial), 100.0 * index, 100.0 * index + 1])


def _expected_features(index):
    return 10.0 * index + np.arange(4.0)


def _expected_feedback(index):
    return np.array([100.0 * index, 100.0 * index + 1])


def _push(smm, tag, row):
    """Prepend a row the way a streamer would, and count it.

    The counter is advanced so ``apply`` moves ``total_samples`` by one per
    row, which is what a DELTA reader downstream reconciles against.
    """
    smm.apply(tag, lambda block: np.vstack((row, block))[:block.shape[0]],
              lambda counter: counter + 1)


def _set_flag(smm, tag, value):
    smm.apply(tag, lambda block: int(value))


# ======================================================================
# item naming and teardown
# ======================================================================
_COUNTER = itertools.count()

FEEDBACK = "feedback"
INPUT = "input"
MEMORY_FLAG = "memory_flag"
ADAPT_FLAG = "adapt_flag"

_SHAPES = {FEEDBACK: ((200, 4), np.float64),
           INPUT: ((200, 5), np.float64),
           MEMORY_FLAG: ((1, 1), np.int32),
           ADAPT_FLAG: ((1, 1), np.int32)}


def _chain(name):
    """The four items the chain runs on, named after the test that wants them.

    Returns
    ----------
    tags: dict
        Role to shared-memory tag.
    items: list
        ``[tag, shape, dtype, lock]`` entries, each buffer sharing its lock
        with its ``_count`` companion.
    """
    suffix = f"{name}_{os.getpid()}_{next(_COUNTER)}"
    tags = {role: f"libemgtest_ad_{role}_{suffix}" for role in _SHAPES}
    items = []
    for role, tag in tags.items():
        shape, dtype = _SHAPES[role]
        lock = Lock()
        items.append([tag, shape, dtype, lock])
        items.append([tag + "_count", (1, 1), np.int32, lock])
    return tags, items


def _memory_kwargs(tags):
    return dict(feedback_tag=tags[FEEDBACK], input_tag=tags[INPUT],
                flag_tag=tags[MEMORY_FLAG])


def _adapt_kwargs(tags):
    return dict(flag_tag=tags[MEMORY_FLAG], notify_tag=tags[ADAPT_FLAG])


class _Environment:
    """Owns every manager and graph a test made, and tears them down."""

    def __init__(self):
        self.managers = []
        self.graphs = []

    def create(self, items, **kwargs):
        smm = SharedMemoryManager(**kwargs)
        for item in items:
            assert smm.create_variable(*item)
        self.managers.append(smm)
        return smm

    def graph(self, *args, **kwargs):
        graph = ReactiveGraph(*args, **kwargs)
        self.graphs.append(graph)
        return graph

    def release(self):
        for graph in reversed(self.graphs):
            try:
                graph.stop(timeout=10)
            except Exception:
                pass
            owned = getattr(graph, "_smm", None)
            if owned is not None:
                try:
                    owned.cleanup(parent=True)
                except Exception:
                    pass
        for smm in reversed(self.managers):
            try:
                smm.cleanup(parent=True)
            except Exception:
                pass


@pytest.fixture
def env():
    environment = _Environment()
    try:
        yield environment
    finally:
        environment.release()


@pytest.fixture
def log():
    made = EventLog(keep=200000)
    try:
        yield made
    finally:
        try:
            made.stop()
        except Exception:
            pass


@pytest.fixture
def crossprocess():
    """Skip rather than fail where the platform cannot give us the pieces."""
    name = f"libemgtest_ad_probe_{os.getpid()}_{next(_COUNTER)}"
    try:
        handle = SharedMemory(name, create=True, size=8)
    except Exception as problem:  # pragma: no cover - environment dependent
        pytest.skip(f"shared memory is unavailable here: {problem}")
    handle.close()
    handle.unlink()


# ======================================================================
# 1. SharedMemoryManager.apply -- the write path everything here depends on
# ======================================================================
def test_apply_writes_back_what_the_transform_returned(env):
    tags, items = _chain("apply_writes")
    smm = env.create(items)

    smm.apply(tags[INPUT], lambda block: block + 3.5)
    smm.apply(tags[MEMORY_FLAG], lambda block: 7)

    np.testing.assert_array_equal(smm.get_variable(tags[INPUT]),
                                  np.full((200, 5), 3.5))
    assert _value(smm, tags[MEMORY_FLAG]) == 7


def test_apply_advances_the_total_by_the_counters_own_delta(env):
    tags, items = _chain("apply_delta")
    smm = env.create(items)
    before = smm.snapshot(tags[FEEDBACK])
    assert (before.generation, before.total_samples, before.commits) == (0, 0, 0)

    after = smm.apply(tags[FEEDBACK], lambda block: block,
                      lambda counter: counter + 3)

    assert after.generation == 1
    assert after.commits == 1
    # Not one, and not three-because-we-said-so: the total follows whatever
    # rule count_fn implements, read back off the counter it wrote.
    assert after.total_samples == 3
    assert _value(smm, tags[FEEDBACK] + "_count") == 3
    assert smm.snapshot(tags[FEEDBACK]) == after

    again = smm.apply(tags[FEEDBACK], lambda block: block,
                      lambda counter: counter + 3)
    assert (again.generation, again.commits, again.total_samples) == (2, 2, 6)


def test_apply_without_a_count_fn_leaves_the_counter_alone_and_counts_one(env):
    tags, items = _chain("apply_flag")
    smm = env.create(items)

    first = smm.apply(tags[MEMORY_FLAG], lambda block: 4)
    second = smm.apply(tags[MEMORY_FLAG], lambda block: 9)

    assert _value(smm, tags[MEMORY_FLAG] + "_count") == 0, "a flag has no samples"
    assert (first.total_samples, second.total_samples) == (1, 2)
    assert (first.generation, second.generation) == (1, 2)


def test_apply_wakes_a_subscriber_that_is_already_waiting(env):
    tags, items = _chain("apply_wakes")
    pool = NotifierPool(size=2)
    smm = env.create(items, notifier_pool=pool)
    slot = pool.claim()
    smm.subscribe(tags[MEMORY_FLAG], slot)
    assert smm.subscribers(tags[MEMORY_FLAG]) == [slot]

    outcome = {}
    waiting = threading.Event()

    def observer():
        # A real waiter, blocking on the slot the way an executor does, with a
        # timeout long enough that expiry and being woken cannot be confused.
        waiting.set()
        started = time.monotonic()
        outcome["woken"] = pool.wait(
            slot, lambda: smm.snapshot(tags[MEMORY_FLAG]).generation > 0,
            timeout=60.0)
        outcome["elapsed"] = time.monotonic() - started

    waiter = threading.Thread(target=observer, daemon=True)
    waiter.start()
    assert waiting.wait(timeout=10), "the observer thread never started"

    smm.apply(tags[MEMORY_FLAG], lambda block: 1)

    waiter.join(timeout=30)
    assert not waiter.is_alive(), "the waiter was never woken"
    assert outcome["woken"] is True
    assert outcome["elapsed"] < 30, "it returned on its timeout, not on the write"


# ======================================================================
# 2. the criteria the hooks judge with, against hand-built snapshots
# ======================================================================
def test_onnewslice_is_dirty_until_the_slice_it_saw_is_consumed():
    criterion = OnNewSlice()
    memory = CriterionMemory()
    snapshot = _snapshot(generation=2, total_samples=2)

    assert criterion.is_dirty(snapshot, memory) is True
    criterion.consume(snapshot, memory)

    assert criterion.is_dirty(snapshot, memory) is False
    assert memory.samples == 2
    assert criterion.is_dirty(_snapshot(generation=3, total_samples=3), memory) is True
    assert criterion.describe() == "OnNewSlice()"


def test_neverdirty_reads_without_ever_firing():
    criterion = _NeverDirty()

    assert criterion.is_dirty(_snapshot(generation=99, total_samples=99),
                              CriterionMemory()) is False
    assert criterion.describe() == "NeverDirty()"


# ======================================================================
# 3. MemoryHook, driven by hand
# ======================================================================
def test_a_feedback_row_is_paired_with_the_input_sharing_its_timestamp(env, tmp_path):
    tags, items = _chain("join_by_stamp")
    smm = env.create(items)
    memory = Memory()
    hook = MemoryHook(memory, save_dir=str(tmp_path / "memories"),
                      **_memory_kwargs(tags))
    hook.setup(_context(smm))

    # The predictor and the environment write at different rates, so the two
    # buffers do not line up. Five inputs have arrived, newest first; the
    # judgements that came back are for the second and fourth of them. Joining
    # by position would pair the first judgement with stamp 5.
    inputs = np.vstack([_input_row(stamp, stamp) for stamp in (5, 4, 3, 2, 1)])
    feedback = np.vstack([_feedback_row(2, 1, 2), _feedback_row(4, 1, 4)])

    assert hook.step({tags[FEEDBACK]: feedback, tags[INPUT]: inputs}, {}) is None

    assert (hook.appended(), hook.unmatched()) == (2, 0)
    np.testing.assert_array_equal(memory.inputs,
                                  [_expected_features(2), _expected_features(4)])
    np.testing.assert_array_equal(memory.feedback,
                                  [_expected_feedback(2), _expected_feedback(4)])
    assert memory.trials == [0, 0], "the trial is stored zero-based"


def test_a_feedback_row_with_no_input_is_counted_and_the_rest_still_land(env, tmp_path):
    tags, items = _chain("unmatched")
    smm = env.create(items)
    memory = Memory()
    hook = MemoryHook(memory, save_dir=str(tmp_path / "memories"),
                      **_memory_kwargs(tags))
    hook.setup(_context(smm))

    inputs = np.vstack([_input_row(stamp, stamp) for stamp in (3, 2, 1)])
    # The middle judgement is for a stamp the input buffer has rolled past.
    feedback = np.vstack([_feedback_row(1, 1, 1), _feedback_row(99, 1, 99),
                          _feedback_row(3, 1, 3)])

    hook.step({tags[FEEDBACK]: feedback, tags[INPUT]: inputs}, {})

    assert hook.unmatched() == 1, "a dropped pair has to be visible"
    assert hook.appended() == 2
    assert len(memory) == 2
    np.testing.assert_array_equal(memory.inputs,
                                  [_expected_features(1), _expected_features(3)])


def test_appended_counts_the_pairs_that_reached_the_memory(env, tmp_path):
    tags, items = _chain("appended")
    smm = env.create(items)
    memory = Memory()
    hook = MemoryHook(memory, save_dir=str(tmp_path / "memories"),
                      **_memory_kwargs(tags))
    hook.setup(_context(smm))
    inputs = np.vstack([_input_row(stamp, stamp) for stamp in (3, 2, 1)])

    assert hook.appended() == 0
    hook.step({tags[FEEDBACK]: np.vstack([_feedback_row(1, 1, 1),
                                          _feedback_row(2, 1, 2)]),
               tags[INPUT]: inputs}, {})
    assert hook.appended() == 2
    hook.step({tags[FEEDBACK]: np.vstack([_feedback_row(3, 1, 3)]),
               tags[INPUT]: inputs}, {})

    assert hook.appended() == 3
    assert len(memory) == 3
    # An empty delivery is declined without disturbing the count.
    assert hook.step({tags[FEEDBACK]: np.zeros((0, 4)),
                      tags[INPUT]: inputs}, {}) is None
    assert hook.appended() == 3


def test_flush_closes_the_slice_in_progress(env, tmp_path):
    tags, items = _chain("flush")
    smm = env.create(items)
    save_dir = tmp_path / "memories"
    memory = Memory()
    hook = MemoryHook(memory, save_dir=str(save_dir), **_memory_kwargs(tags))
    hook.setup(_context(smm))
    inputs = np.vstack([_input_row(stamp, stamp) for stamp in (2, 1)])
    hook.step({tags[FEEDBACK]: np.vstack([_feedback_row(1, 1, 1),
                                          _feedback_row(2, 1, 2)]),
               tags[INPUT]: inputs}, {})
    assert not (save_dir / "memory_1.pkl").exists(), "nothing has ended the trial"

    hook.flush()

    written = _load(save_dir / "memory_1.pkl")
    assert len(written) == 2
    np.testing.assert_array_equal(written.inputs,
                                  [_expected_features(1), _expected_features(2)])
    assert _value(smm, tags[MEMORY_FLAG]) == 1
    assert smm.snapshot(tags[MEMORY_FLAG]).generation == 1
    assert len(memory) == 0, "the hook starts the next slice empty"


def test_a_new_trial_number_closes_the_slice_and_advances_the_flag(
        env, crossprocess, log, tmp_path):
    tags, items = _chain("trial_change")
    save_dir = tmp_path / "memories"
    pool = NotifierPool(size=4)
    owner = env.create(items, notifier_pool=pool)
    graph = env.graph(items, log=log, notifier_pool=pool)
    graph.add(MemoryHook(Memory(), save_dir=str(save_dir), **_memory_kwargs(tags)),
              executor="memory")
    graph.start()

    for index in (1, 2):
        _push(owner, tags[INPUT], _input_row(index, index))
        _push(owner, tags[FEEDBACK], _feedback_row(index, 1, index))
    # Nothing can close the slice while the trial number is unchanged, so this
    # holds whether or not the hook has run yet.
    assert _value(owner, tags[MEMORY_FLAG]) == 0, "an open trial announces nothing"
    assert not (save_dir / "memory_1.pkl").exists()

    # The first judgement of the next trial is what ends the previous one.
    _push(owner, tags[INPUT], _input_row(3, 3))
    _push(owner, tags[FEEDBACK], _feedback_row(3, 2, 3))

    assert _wait_until(lambda: _value(owner, tags[MEMORY_FLAG]) == 1), \
        "the finished slice was never announced"
    assert _wait_until(lambda: (save_dir / "memory_1.pkl").exists())
    written = _load(save_dir / "memory_1.pkl")
    assert len(written) == 2
    np.testing.assert_array_equal(written.feedback,
                                  [_expected_feedback(1), _expected_feedback(2)])
    assert written.trials == [0, 0]

    graph.stop(timeout=10)


# ======================================================================
# 4. AdaptationHook, driven by hand
# ======================================================================
def test_by_default_it_trains_only_when_a_slice_arrives(env, tmp_path):
    tags, items = _chain("only_on_slice")
    smm = env.create(items)
    load_dir, save_dir = tmp_path / "memories", tmp_path / "models"
    _write_slice(str(load_dir), 1, count=2)
    model = Model()
    hook = AdaptationHook(model, load_dir=str(load_dir), save_dir=str(save_dir),
                          **_adapt_kwargs(tags))
    hook.setup(_context(smm))
    flag = {tags[MEMORY_FLAG]: np.array([[1]])}

    hook.step(flag, {})
    hook.step(flag, {})
    hook.step(flag, {})

    assert model.fits == [2], "unchanged data was retrained on"
    assert hook.adaptation_count == 1
    assert sorted(os.listdir(save_dir)) == ["losses.txt", "mdl1.pkl"]


def test_continuous_retrains_on_every_wake(env, tmp_path):
    tags, items = _chain("continuous")
    smm = env.create(items)
    load_dir, save_dir = tmp_path / "memories", tmp_path / "models"
    _write_slice(str(load_dir), 1, count=2)
    model = Model()
    hook = AdaptationHook(model, load_dir=str(load_dir), save_dir=str(save_dir),
                          continuous=True, **_adapt_kwargs(tags))
    hook.setup(_context(smm))
    flag = {tags[MEMORY_FLAG]: np.array([[1]])}

    hook.step(flag, {})
    hook.step(flag, {})
    hook.step(flag, {})

    # Three wakes, one slice: the old manager's behaviour, kept behind a flag.
    assert model.fits == [2, 2, 2]
    assert hook.adaptation_count == 3
    assert _value(smm, tags[ADAPT_FLAG]) == 3


def test_notify_false_trains_without_announcing_a_new_model(env, tmp_path):
    tags, items = _chain("no_notify")
    smm = env.create(items)
    load_dir = tmp_path / "memories"
    _write_slice(str(load_dir), 1, count=2)
    quiet = AdaptationHook(Model(), load_dir=str(load_dir),
                           save_dir=str(tmp_path / "quiet"), notify=False,
                           name="quiet", **_adapt_kwargs(tags))
    quiet.setup(_context(smm))

    quiet.step({tags[MEMORY_FLAG]: np.array([[1]])}, {})

    assert quiet.adaptation_count == 1
    assert (tmp_path / "quiet" / "mdl1.pkl").exists(), "it still trained and saved"
    assert _value(smm, tags[ADAPT_FLAG]) == 0
    assert smm.snapshot(tags[ADAPT_FLAG]).generation == 0, "nothing was announced"

    # The same hook that does announce, for contrast.
    loud = AdaptationHook(Model(), load_dir=str(load_dir),
                          save_dir=str(tmp_path / "loud"), name="loud",
                          **_adapt_kwargs(tags))
    loud.setup(_context(smm))
    loud.step({tags[MEMORY_FLAG]: np.array([[1]])}, {})

    assert _value(smm, tags[ADAPT_FLAG]) == 1
    assert smm.snapshot(tags[ADAPT_FLAG]).generation == 1


def test_stop_after_stops_adapting(env, tmp_path):
    tags, items = _chain("stop_after")
    smm = env.create(items)
    load_dir, save_dir = tmp_path / "memories", tmp_path / "models"
    for number in (1, 2, 3):
        _write_slice(str(load_dir), number, count=2, trial=number - 1,
                     first_index=10 * number)
    model = Model()
    hook = AdaptationHook(model, load_dir=str(load_dir), save_dir=str(save_dir),
                          stop_after=2, **_adapt_kwargs(tags))
    hook.setup(_context(smm))

    hook.step({tags[MEMORY_FLAG]: np.array([[1]])}, {})
    assert model.fits == [2]
    hook.step({tags[MEMORY_FLAG]: np.array([[2]])}, {})
    hook.step({tags[MEMORY_FLAG]: np.array([[3]])}, {})

    # stop_after=2 means two slices are folded in and trained on, then nothing
    # further happens. Loading stops together with training, so a long session
    # cannot keep reading slices off disk and growing the held memory to do
    # work that will never happen.
    assert model.fits == [2, 4], "adaptation continued past stop_after"
    assert hook.adaptation_count == 2
    assert sorted(name for name in os.listdir(save_dir) if name.startswith("mdl")) \
        == ["mdl1.pkl", "mdl2.pkl"]
    assert _value(smm, tags[ADAPT_FLAG]) == 2
    assert hook.memory_count == 2
    assert len(hook.memory) == 4


def test_a_missing_slice_is_retried_rather_than_skipped(env, tmp_path):
    tags, items = _chain("retry_missing")
    smm = env.create(items)
    load_dir, save_dir = tmp_path / "memories", tmp_path / "models"
    os.makedirs(load_dir, exist_ok=True)
    model = Model()
    hook = AdaptationHook(model, load_dir=str(load_dir), save_dir=str(save_dir),
                          **_adapt_kwargs(tags))
    hook.setup(_context(smm))

    # The flag moved before the write finished.
    hook.step({tags[MEMORY_FLAG]: np.array([[1]])}, {})

    assert hook.memory_count == 0, "a slice that was not there was counted anyway"
    assert model.fits == []

    _write_slice(str(load_dir), 1, count=3, trial=0)
    hook.step({tags[MEMORY_FLAG]: np.array([[1]])}, {})

    assert hook.memory_count == 1
    assert model.fits == [3], "the slice was skipped instead of retried"


def test_an_initial_memory_is_loaded_and_trained_on_with_the_slices(env, tmp_path):
    tags, items = _chain("initial_memory")
    smm = env.create(items)
    load_dir, save_dir = tmp_path / "memories", tmp_path / "models"
    os.makedirs(load_dir, exist_ok=True)
    initial = tmp_path / "screen_guided.pkl"
    _slice_of(4, trial=99).save(initial)
    _write_slice(str(load_dir), 1, count=2, trial=0)
    model = Model()
    hook = AdaptationHook(model, load_dir=str(load_dir), save_dir=str(save_dir),
                          initial_memory_loc=str(initial), **_adapt_kwargs(tags))

    hook.setup(_context(smm))
    assert len(hook.memory) == 4, "the initial memory is loaded in setup"
    hook.step({tags[MEMORY_FLAG]: np.array([[1]])}, {})

    assert model.fits == [6]
    assert model.seen == [[99, 99, 99, 99, 0, 0]], \
        "the initial memory has to survive the first slice being folded in"


def test_it_loads_every_slice_that_appeared_and_announces_the_new_model(
        env, crossprocess, log, tmp_path):
    tags, items = _chain("adapt_e2e")
    load_dir, save_dir = tmp_path / "memories", tmp_path / "models"
    _write_slice(str(load_dir), 1, count=2, trial=0)
    _write_slice(str(load_dir), 2, count=3, trial=1, first_index=100)
    pool = NotifierPool(size=4)
    owner = env.create(items, notifier_pool=pool)
    graph = env.graph(items, log=log, notifier_pool=pool)
    graph.add(AdaptationHook(Model(), load_dir=str(load_dir),
                             save_dir=str(save_dir), **_adapt_kwargs(tags)),
              executor="adapt")
    graph.start()

    _set_flag(owner, tags[MEMORY_FLAG], 2)

    assert _wait_until(lambda: _value(owner, tags[ADAPT_FLAG]) == 1), \
        "the new model was never announced"
    assert _wait_until(lambda: (save_dir / "mdl1.pkl").exists())
    trained = _load(save_dir / "mdl1.pkl")
    # Both slices that had appeared since it last looked, in one training.
    assert trained.fits == [5]
    assert trained.seen == [[0, 0, 1, 1, 1]]

    graph.stop(timeout=10)


def test_a_slice_written_after_the_flag_moved_is_picked_up_on_the_next_wake(
        env, crossprocess, log, tmp_path):
    tags, items = _chain("late_write")
    load_dir, save_dir = tmp_path / "memories", tmp_path / "models"
    os.makedirs(load_dir, exist_ok=True)
    pool = NotifierPool(size=4)
    owner = env.create(items, notifier_pool=pool)
    graph = env.graph(items, log=log, notifier_pool=pool)
    graph.add(AdaptationHook(Model(), load_dir=str(load_dir),
                             save_dir=str(save_dir), **_adapt_kwargs(tags)),
              executor="adapt")
    graph.start()

    # The flag moves first, so the hook goes looking for a file that is not
    # there yet.
    _set_flag(owner, tags[MEMORY_FLAG], 1)
    assert _wait_until(lambda: log.summary()["invocations"].get("adaptation", 0) >= 1), \
        "the hook never ran on the flag that moved"
    assert not (save_dir / "mdl1.pkl").exists()
    assert _value(owner, tags[ADAPT_FLAG]) == 0

    # Now the writes land, and the next slice announces itself.
    _write_slice(str(load_dir), 1, count=2, trial=0)
    _write_slice(str(load_dir), 2, count=3, trial=1, first_index=100)
    _set_flag(owner, tags[MEMORY_FLAG], 2)

    assert _wait_until(lambda: (save_dir / "mdl1.pkl").exists()), \
        "the model was never written"
    # What matters is that slice 1 is not skipped, not exactly when it lands.
    # The hook retries briefly within a step, so a file appearing during that
    # window is picked up there rather than on a later wake. Either way both
    # slices must end up folded in, and the announced model number must match
    # how many trainings actually happened.
    assert _wait_until(lambda: _value(owner, tags[ADAPT_FLAG]) >= 1)
    assert _wait_until(lambda: _latest_training(save_dir) is not None
                       and _latest_training(save_dir).seen[-1] == [0, 0, 1, 1, 1]), \
        "slice 1 was skipped rather than retried"
    latest = max(int(name[3:-4]) for name in os.listdir(save_dir)
                 if name.startswith("mdl"))
    assert _wait_until(lambda: _value(owner, tags[ADAPT_FLAG]) == latest)

    graph.stop(timeout=10)


# ======================================================================
# 5. ModelSwapHook
# ======================================================================
def test_a_swap_fires_with_the_model_number_and_ignores_the_sentinel():
    seen = []
    hook = ModelSwapHook("swap", seen.append, notify_tag="adapt_flag")

    assert hook.step({"adapt_flag": np.array([[1]])}, {}) is None
    hook.step({"adapt_flag": np.array([[2]])}, {})
    hook.step({"adapt_flag": np.array([[-1]])}, {})
    hook.step({"adapt_flag": np.array([[3]])}, {})

    assert seen == [1, 2, 3], "the sentinel is not a model"
    assert [spec.tag for spec in hook.inputs] == ["adapt_flag"]
    assert isinstance(hook.inputs[0].criterion, OnCommit)
    assert hook.inputs[0].mode == FULL
    assert hook.outputs == []


# ======================================================================
# 6. the whole chain, across processes
# ======================================================================
N_TRIALS = 3
PER_TRIAL = 4


def test_feedback_becomes_slices_becomes_models_becomes_swaps(
        env, crossprocess, log, tmp_path):
    tags, items = _chain("full_chain")
    mem_dir, mdl_dir = tmp_path / "memories", tmp_path / "models"
    # Deliberately NOT created here. The memory side's save_dir and the
    # adaptation side's load_dir are by design the same directory and their
    # executors start together, so leaving the creation to the hooks is what
    # exercises ensure_directory being safe against that race.
    pool = NotifierPool(size=8)
    owner = env.create(items, notifier_pool=pool)
    # -1 is what a predictor writes when no adapted model exists yet, so the
    # chain starts with a commit the swap hook has to decline.
    _set_flag(owner, tags[ADAPT_FLAG], -1)

    swaps = []
    graph = env.graph(items, log=log, notifier_pool=pool)
    graph.add(MemoryHook(Memory(), save_dir=str(mem_dir), **_memory_kwargs(tags)),
              executor="memory")
    graph.add(AdaptationHook(Model(), load_dir=str(mem_dir), save_dir=str(mdl_dir),
                             **_adapt_kwargs(tags)), executor="adapt")
    # A list appended to by a bound method cannot be pickled into a process,
    # and does not need to be: this one only has to reach the test.
    graph.add(ModelSwapHook("swap", swaps.append, notify_tag=tags[ADAPT_FLAG]),
              executor="swap", in_process=True)
    graph.start()
    assert len(graph._executors) == 3
    assert len({getattr(e, "pid", None) for e in graph._executors}) == 3

    # The environment judges PER_TRIAL predictions per trial. The first
    # judgement of trial k is what closes slice k-1, so one extra trial is
    # written to close the last one. Each slice is waited for before the next
    # trial starts, so a burst cannot fold two slices into one training.
    for trial in range(1, N_TRIALS + 2):
        for position in range(PER_TRIAL):
            index = (trial - 1) * PER_TRIAL + position + 1
            _push(owner, tags[INPUT], _input_row(index, index))
            _push(owner, tags[FEEDBACK], _feedback_row(index, trial, index))
        if trial == 1:
            continue
        finished = trial - 1
        assert _wait_until(lambda n=finished: _value(owner, tags[MEMORY_FLAG]) == n), \
            f"slice {finished} was never announced"
        assert _wait_until(lambda n=finished: _value(owner, tags[ADAPT_FLAG]) == n), \
            f"model {finished} was never announced"

    assert _wait_until(lambda: swaps == [1, 2, 3]), f"swaps were {swaps}"

    # Every completed trial produced exactly one slice, holding exactly the
    # pairs written during it, joined by timestamp.
    assert sorted(os.listdir(mem_dir)) == [f"memory_{k}.pkl"
                                           for k in range(1, N_TRIALS + 1)]
    for trial in range(1, N_TRIALS + 1):
        written = _load(mem_dir / f"memory_{trial}.pkl")
        indices = [(trial - 1) * PER_TRIAL + position + 1
                   for position in range(PER_TRIAL)]
        assert written.trials == [trial - 1] * PER_TRIAL
        np.testing.assert_array_equal(written.feedback,
                                      [_expected_feedback(i) for i in indices])
        np.testing.assert_array_equal(written.inputs,
                                      [_expected_features(i) for i in indices])

    # One model per slice, each trained on everything folded in so far.
    assert sorted(name for name in os.listdir(mdl_dir) if name.startswith("mdl")) \
        == [f"mdl{k}.pkl" for k in range(1, N_TRIALS + 1)]
    final = _load(mdl_dir / f"mdl{N_TRIALS}.pkl")
    assert final.fits == [PER_TRIAL * k for k in range(1, N_TRIALS + 1)]
    assert final.seen[-1] == [0] * PER_TRIAL + [1] * PER_TRIAL + [2] * PER_TRIAL

    assert _value(owner, tags[MEMORY_FLAG]) == N_TRIALS
    assert _value(owner, tags[ADAPT_FLAG]) == N_TRIALS

    graph.stop(timeout=10)
    summary = log.summary()
    assert summary["invocations"]["memory"] >= 1
    assert summary["invocations"]["adaptation"] >= N_TRIALS
    assert summary["invocations"]["swap"] >= N_TRIALS
