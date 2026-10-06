"""Coverage for the reactive layer over stateful shared memory.

Two halves. The first is arithmetic: a criterion is a pure judgement about an
item's published facts and an observer's memory of what it consumed, so every
one of them is tested against hand-built snapshots with no processes and no
shared memory in sight. The second is the machinery: a graph must refuse a
pipeline it cannot run before it spawns anything, and a graph it can run has to
carry a change all the way through a cascade that spans processes.

Everything a spawned executor needs -- hooks, callbacks, the writer -- is
defined at module level, because Windows spawns rather than forks and anything
nested inside a test function cannot be pickled into the child.
"""
import itertools
import os
import time

import numpy as np
import pytest
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

from libemg import event_log
from libemg.event_log import EventLog
from libemg.reactive import (AfterSamples, AllOf, Always, AnyOf, CallbackHook,
                             CriterionMemory, Custom, DELTA, Executor, FULL,
                             FeatureHook, FilterHook, Hook, Input, LATEST,
                             NotifierPool, OnCommit, OnSamples, Output, Periodic,
                             ProbeHook, ReactiveGraph, STATE, WINDOW, WhenClosed,
                             default_notifier_pool, reset_default_notifier_pool)
from libemg.shared_memory_manager import (MAX_SUBSCRIBERS, SharedMemoryManager,
                                          Snapshot, assign_shared_memory_locks)


# ======================================================================
# helpers
# ======================================================================
def _snapshot(generation=1, total_samples=1, epoch=0, closed=False, dropped=0,
              tag="item"):
    """A state block reading, built by hand rather than by a writer."""
    return Snapshot(tag=tag, generation=generation, total_samples=total_samples,
                    epoch=epoch, closed=closed, commits=generation,
                    dropped=dropped, t_last_ns=1)


def _wait_until(predicate, timeout=30.0, interval=0.01):
    """Poll ``predicate`` until it holds or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _noop(data, snapshots):
    return None


class _Doubler:
    """Stands in for a libemg Filter: the interface is one ``filter`` method."""

    def filter(self, samples):
        return samples * 2.0


# ======================================================================
# module-level hooks, so a spawned executor can unpickle them
# ======================================================================
E2E_SRC = "libemgtest_rx_e2e_emg"
E2E_MID = "libemgtest_rx_e2e_doubled"
E2E_OUT = "libemgtest_rx_e2e_means"
E2E_INCREMENT = 4
E2E_SAMPLES = 20

ERR_SRC = "libemgtest_rx_err_src"
ERR_OUT = "libemgtest_rx_err_ok"

CLOSE_SRC = "libemgtest_rx_close_src"
CLOSE_OUT = "libemgtest_rx_close_summary"

RUNNING_SRC = "libemgtest_rx_running_src"


class DoublingHook(Hook):
    """Doubles whatever arrived since it last ran. Dirty on any commit."""

    def __init__(self, name, source, target, shape):
        super().__init__(name,
                         inputs=[Input(source, OnCommit(), mode=DELTA)],
                         outputs=[Output(target, shape, np.double)])
        self.source = source
        self.target = target

    def step(self, data, snapshots):
        samples = data[self.source]
        if samples.shape[0] == 0:
            return None
        return {self.target: samples * 2.0}


class WindowMeanHook(Hook):
    """Averages a window, once per ``increment`` new samples."""

    def __init__(self, name, source, target, shape, increment):
        super().__init__(name,
                         inputs=[Input(source, OnSamples(increment),
                                       mode=WINDOW, size=increment)],
                         outputs=[Output(target, shape, np.double)])
        self.source = source
        self.target = target
        self.increment = increment

    def step(self, data, snapshots):
        window = data[self.source]
        if window.shape[0] < self.increment:
            return None
        return {self.target: window.mean(axis=0)}


class RaisingHook(Hook):
    """A hook that always fails, to prove it cannot take its neighbours down."""

    def __init__(self, name, source):
        super().__init__(name, inputs=[Input(source, OnCommit(), mode=STATE)])

    def step(self, data, snapshots):
        raise ValueError("deliberate failure from " + self.name)


class PassThroughHook(Hook):
    """Copies the first column of whatever arrived into its own item."""

    def __init__(self, name, source, target, shape):
        super().__init__(name,
                         inputs=[Input(source, OnCommit(), mode=DELTA)],
                         outputs=[Output(target, shape, np.double)])
        self.source = source
        self.target = target

    def step(self, data, snapshots):
        samples = data[self.source]
        if samples.shape[0] == 0:
            return None
        return {self.target: samples[:, :1]}


class ClosingSummaryHook(Hook):
    """Runs once, when its source declares itself finished."""

    def __init__(self, name, source, target):
        super().__init__(name,
                         inputs=[Input(source, WhenClosed(), mode=FULL)],
                         outputs=[Output(target, (4, 1), np.double)])
        self.source = source
        self.target = target

    def step(self, data, snapshots):
        return {self.target: np.array([[float(data[self.source].sum())]])}


def _paced_writer(items, pool, source, watched, increment, total):
    """Commit ``total`` single samples, letting the cascade catch up each time.

    Waiting on the downstream items rather than sleeping is what makes the
    end-to-end assertions exact: the next sample is not written until the
    previous one has been through both hooks, so a window boundary always falls
    where the arithmetic says it does.
    """
    smm = SharedMemoryManager(notifier_pool=pool)
    try:
        for item in items:
            deadline = time.monotonic() + 30
            while not smm.find_variable(*item):
                if time.monotonic() > deadline:
                    raise TimeoutError("never saw shared-memory item " + item[0])
                time.sleep(0.01)
        for i in range(1, total + 1):
            smm.commit(source, np.array([float(i), 10.0 * i]))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                totals = [smm.snapshot(tag).total_samples for tag in watched]
                if totals == [i, i // increment]:
                    break
                time.sleep(0.001)
    finally:
        smm.cleanup(parent=False)


# ======================================================================
# fixtures
# ======================================================================
_COUNTER = itertools.count()


def _unique(base):
    return f"libemgtest_rx_{base}_{os.getpid()}_{next(_COUNTER)}"


def _items(tag, shape=(64, 2), dtype=np.double, with_count=True):
    spec = [[tag, shape, dtype]]
    if with_count:
        spec.append([tag + "_count", (1, 1), np.int32])
    return assign_shared_memory_locks(spec)


class _Environment:
    """Owns every manager, graph and process a test made, and tears them down."""

    def __init__(self):
        self.managers = []
        self.attached = []
        self.graphs = []
        self.processes = []

    def create(self, items, **kwargs):
        smm = SharedMemoryManager(**kwargs)
        for item in items:
            assert smm.create_variable(*item)
        self.managers.append(smm)
        return smm

    def attach(self, items, **kwargs):
        smm = SharedMemoryManager(**kwargs)
        for item in items:
            assert _wait_until(lambda item=item: smm.find_variable(*item), 30), item[0]
        self.attached.append(smm)
        return smm

    def graph(self, *args, **kwargs):
        graph = ReactiveGraph(*args, **kwargs)
        self.graphs.append(graph)
        return graph

    def spawn(self, target, args):
        process = Process(target=target, args=args, daemon=True)
        self.processes.append(process)
        process.start()
        return process

    def release(self):
        for process in reversed(self.processes):
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for graph in reversed(self.graphs):
            try:
                graph.stop(timeout=10)
            except Exception:
                pass
            # stop() joins the executors but does not release the items the
            # graph itself created for hook outputs, so the test does.
            owned = getattr(graph, "_smm", None)
            if owned is not None:
                try:
                    owned.cleanup(parent=True)
                except Exception:
                    pass
        for smm in reversed(self.attached):
            try:
                smm.cleanup(parent=False)
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
    made = EventLog(keep=100000)
    try:
        yield made
    finally:
        try:
            made.stop()
        except Exception:
            pass


def _require_shared_memory():
    name = _unique("probe")
    try:
        handle = SharedMemory(name, create=True, size=8)
    except Exception as problem:  # pragma: no cover - environment dependent
        pytest.skip(f"shared memory is unavailable here: {problem}")
    handle.close()
    handle.unlink()


@pytest.fixture
def crossprocess():
    """Skip rather than fail where the platform cannot give us the pieces."""
    _require_shared_memory()


# ======================================================================
# 1. criteria, against hand-built snapshots
# ======================================================================
def test_a_single_new_sample_is_dirty_to_oncommit_and_clean_to_onsamples():
    # The whole reason dirtiness is a per-observer judgement rather than a flag
    # on the item: these two disagree about the very same snapshot.
    snapshot = _snapshot(generation=1, total_samples=1)
    memory = CriterionMemory()

    assert OnCommit().is_dirty(snapshot, memory) is True
    assert OnSamples(40).is_dirty(snapshot, memory) is False


def test_oncommit_is_clean_until_the_generation_moves():
    memory = CriterionMemory()
    criterion = OnCommit()
    snapshot = _snapshot(generation=3, total_samples=9)

    assert criterion.is_dirty(snapshot, memory)
    criterion.consume(snapshot, memory)

    assert criterion.is_dirty(snapshot, memory) is False
    assert (memory.generation, memory.samples, memory.epoch) == (3, 9, 0)
    assert criterion.is_dirty(_snapshot(generation=4, total_samples=10), memory)


def test_onsamples_waits_for_a_whole_increment():
    criterion = OnSamples(4)
    memory = CriterionMemory()

    assert criterion.is_dirty(_snapshot(generation=3, total_samples=3), memory) is False
    assert criterion.is_dirty(_snapshot(generation=4, total_samples=4), memory) is True


def test_a_criterion_judges_without_mutating_anything():
    snapshot = _snapshot(generation=5, total_samples=50)
    memory = CriterionMemory(generation=1, samples=10, fired_at=0.0, epoch=0)

    OnSamples(4).is_dirty(snapshot, memory)
    OnCommit().is_dirty(snapshot, memory)

    assert memory == CriterionMemory(generation=1, samples=10, fired_at=0.0, epoch=0)
    assert snapshot == _snapshot(generation=5, total_samples=50)


def test_onsamples_steps_so_a_burst_of_three_increments_fires_three_times():
    criterion = OnSamples(10)
    memory = CriterionMemory()
    snapshot = _snapshot(generation=1, total_samples=30)

    firings = 0
    while criterion.is_dirty(snapshot, memory) and firings < 10:
        criterion.consume(snapshot, memory)
        firings += 1

    assert firings == 3, "window boundaries have to stay evenly spaced"
    assert memory.samples == 30


def test_aftersamples_skips_the_backlog_so_a_burst_fires_once():
    criterion = AfterSamples(10)
    memory = CriterionMemory()
    snapshot = _snapshot(generation=1, total_samples=30)

    firings = 0
    while criterion.is_dirty(snapshot, memory) and firings < 10:
        criterion.consume(snapshot, memory)
        firings += 1

    assert firings == 1, "a model that fell behind should act on the newest data"
    assert memory.samples == 30
    assert criterion.increment == 10


def test_periodic_respects_its_rate_and_needs_a_change():
    criterion = Periodic(hz=50.0)
    memory = CriterionMemory()

    assert criterion.is_dirty(_snapshot(generation=1, total_samples=1), memory)
    criterion.consume(_snapshot(generation=1, total_samples=1), memory)

    # Changed again, but too soon.
    assert criterion.is_dirty(_snapshot(generation=2, total_samples=2), memory) is False
    time.sleep(criterion.period * 1.5)
    assert criterion.is_dirty(_snapshot(generation=2, total_samples=2), memory) is True
    # Long enough since the last firing, but nothing has changed.
    assert criterion.is_dirty(_snapshot(generation=1, total_samples=1), memory) is False


def test_always_is_dirty_even_when_nothing_happened():
    assert Always().is_dirty(_snapshot(generation=0, total_samples=0),
                             CriterionMemory()) is True


def test_whenclosed_fires_exactly_once():
    criterion = WhenClosed()
    memory = CriterionMemory()
    open_item = _snapshot(generation=2, total_samples=7)
    closed_item = _snapshot(generation=2, total_samples=7, closed=True)

    assert criterion.is_dirty(open_item, memory) is False
    assert criterion.is_dirty(closed_item, memory) is True

    criterion.consume(closed_item, memory)
    assert criterion.is_dirty(closed_item, memory) is False
    assert memory.samples == 7


def test_custom_wraps_a_plain_predicate():
    criterion = Custom(lambda snapshot, memory:
                       snapshot.total_samples - memory.samples >= 1000,
                       "OneSecond")

    assert criterion.is_dirty(_snapshot(total_samples=999), CriterionMemory()) is False
    assert criterion.is_dirty(_snapshot(total_samples=1000), CriterionMemory()) is True
    assert criterion.describe() == "OneSecond()"


def test_allof_needs_every_member_and_anyof_needs_one():
    snapshot = _snapshot(generation=1, total_samples=1)
    memory = CriterionMemory()

    assert AllOf(OnCommit(), OnSamples(4)).is_dirty(snapshot, memory) is False
    assert AnyOf(OnCommit(), OnSamples(4)).is_dirty(snapshot, memory) is True

    both = _snapshot(generation=1, total_samples=4)
    assert AllOf(OnCommit(), OnSamples(4)).is_dirty(both, memory) is True


def test_a_combination_consumes_every_member_it_wraps():
    combined = AnyOf(OnCommit(), OnSamples(4))
    memory = CriterionMemory()
    snapshot = _snapshot(generation=1, total_samples=4)

    combined.consume(snapshot, memory)

    assert combined.is_dirty(snapshot, memory) is False


def test_every_criterion_names_itself_for_the_event_log():
    assert OnCommit().describe() == "OnCommit()"
    assert OnSamples(40).describe() == "OnSamples(40)"
    assert AfterSamples(40).describe() == "AfterSamples(40)"
    assert Periodic(30).describe() == "Periodic(30)"
    assert Always().describe() == "Always()"
    assert WhenClosed().describe() == "WhenClosed()"
    assert repr(OnSamples(2)) == "OnSamples(2)"
    assert AllOf(OnCommit(), OnSamples(2)).describe() == "AllOf(OnCommit(), OnSamples(2))"
    assert AnyOf(OnCommit()).describe() == "AnyOf(OnCommit())"


def test_a_criterion_that_cannot_work_is_refused_when_it_is_built():
    with pytest.raises(ValueError):
        OnSamples(0)
    with pytest.raises(ValueError):
        Periodic(0)
    with pytest.raises(ValueError):
        Input("emg", OnCommit(), mode=WINDOW)


def test_an_input_observes_any_write_unless_told_otherwise():
    spec = Input("emg", mode=STATE)

    assert isinstance(spec.criterion, OnCommit)
    assert spec.size is None


# ======================================================================
# 6. a graph refuses what it cannot run, before it spawns anything
# ======================================================================
def _assert_nothing_spawned(graph):
    assert graph._executors == []
    assert graph._started is False


def test_two_hooks_writing_one_item_is_refused(env):
    graph = env.graph([["src", (8, 1), np.double, None]])
    graph.add(CallbackHook("first", [Input("src", OnCommit(), mode=STATE)], _noop,
                           outputs=[Output("shared", (4, 1))]))
    graph.add(CallbackHook("second", [Input("src", OnCommit(), mode=STATE)], _noop,
                           outputs=[Output("shared", (4, 1))]))

    with pytest.raises(ValueError, match="write"):
        graph.start()

    _assert_nothing_spawned(graph)


def test_observing_an_item_nobody_declares_or_produces_is_refused(env):
    graph = env.graph([])
    graph.add(CallbackHook("orphan", [Input("nowhere", OnCommit(), mode=STATE)], _noop))

    with pytest.raises(KeyError, match="never declared or produced"):
        graph.start()

    _assert_nothing_spawned(graph)


def test_a_cycle_is_refused(env):
    graph = env.graph([])
    graph.add(CallbackHook("a", [Input("b_item", OnCommit(), mode=STATE)], _noop,
                           outputs=[Output("a_item", (4, 1))]))
    graph.add(CallbackHook("b", [Input("a_item", OnCommit(), mode=STATE)], _noop,
                           outputs=[Output("b_item", (4, 1))]))

    with pytest.raises(ValueError, match="cycle"):
        graph.start()

    _assert_nothing_spawned(graph)


def test_two_hooks_cannot_share_a_name(env):
    graph = env.graph([["src", (8, 1), np.double, None]])
    graph.add(CallbackHook("twin", [Input("src", OnCommit(), mode=STATE)], _noop))

    with pytest.raises(ValueError, match="already registered"):
        graph.add(CallbackHook("twin", [Input("src", OnCommit(), mode=STATE)], _noop))

    _assert_nothing_spawned(graph)


def test_a_hook_cannot_be_added_once_the_graph_is_running(env, crossprocess, log):
    items = _items(RUNNING_SRC, shape=(16, 1))
    env.create(items)
    graph = env.graph(items, log=log, notifier_pool=NotifierPool(size=4))
    graph.add(CallbackHook("sink", [Input(RUNNING_SRC, OnCommit(), mode=STATE)], _noop))
    graph.start()

    with pytest.raises(RuntimeError, match="running"):
        graph.add(CallbackHook("late", [Input(RUNNING_SRC, OnCommit(), mode=STATE)],
                               _noop))

    assert len(graph._executors) == 1


# ======================================================================
# 7. describe
# ======================================================================
def test_describe_names_every_hook_its_criterion_and_its_outputs(env):
    graph = env.graph([["raw", (100, 2), np.double, None]])
    graph.add(FilterHook("conditioning", "raw", "filtered", fi=None,
                         shape=(100, 2), increment=5), executor="signal")
    graph.add(WindowMeanHook("averaging", "filtered", "averages", (10, 2), 20),
              executor="features")
    graph.add(ProbeHook("watching", "raw", _noop, hz=15.0), executor="features")

    described = graph.describe()

    assert "executor 'signal':" in described
    assert "executor 'features':" in described
    assert "conditioning: observes raw [OnSamples(5), window(100)] -> writes filtered" in described
    assert "averaging: observes filtered [OnSamples(20), window(20)] -> writes averages" in described
    assert "watching: observes raw [Periodic(15), latest] -> writes -" in described


def test_describe_mentions_the_trigger_when_a_hook_fuses_two_items(env):
    graph = env.graph([["a", (8, 1), np.double, None], ["b", (8, 1), np.double, None]])
    graph.add(CallbackHook("fusion",
                           [Input("a", OnCommit(), mode=LATEST),
                            Input("b", OnSamples(3), mode=WINDOW, size=3)],
                           _noop, outputs=[Output("fused", (8, 1))], trigger="all"))

    described = graph.describe()

    assert "trigger=all" in described
    assert "fused" in described


# ======================================================================
# 11. a hook's step is testable on its own
# ======================================================================
def test_a_filter_hook_filters_and_emits_only_the_new_rows():
    hook = FilterHook("filter", "raw", "clean", fi=_Doubler(), shape=(50, 2),
                      window=10, increment=3)
    window = np.arange(20.0).reshape(10, 2)

    produced = hook.step({"raw": window}, {})

    assert set(produced) == {"clean"}
    # Filtered over the whole window, but only the newest three rows are new.
    np.testing.assert_array_equal(produced["clean"], window[-3:] * 2.0)


def test_a_filter_hook_with_no_filter_passes_the_newest_rows_through():
    hook = FilterHook("filter", "raw", "clean", fi=None, shape=(50, 2), increment=1)
    window = np.arange(20.0).reshape(10, 2)

    produced = hook.step({"raw": window}, {})

    np.testing.assert_array_equal(produced["clean"], window[-1:])


def test_a_filter_hook_declines_when_there_is_nothing_to_filter():
    hook = FilterHook("filter", "raw", "clean", fi=_Doubler(), shape=(50, 2))

    assert hook.step({"raw": np.zeros((0, 2))}, {}) is None


def test_a_filter_hook_declares_what_it_reads_and_writes():
    hook = FilterHook("filter", "raw", "clean", fi=None, shape=(50, 2), increment=7)

    assert [spec.tag for spec in hook.inputs] == ["raw"]
    assert isinstance(hook.inputs[0].criterion, OnSamples)
    assert hook.inputs[0].criterion.increment == 7
    assert hook.inputs[0].size == 50, "the window defaults to the target's rows"
    assert [(o.tag, o.shape) for o in hook.outputs] == [("clean", (50, 2))]


def test_a_callback_hook_returns_whatever_its_function_returns():
    seen = {}

    def remember(data, snapshots):
        seen["data"] = data
        seen["snapshots"] = snapshots
        return {"out": np.array([[1.0]])}

    hook = CallbackHook("callback", [Input("emg", OnCommit(), mode=LATEST)],
                        remember, outputs=[Output("out", (4, 1))])
    snapshots = {"emg": _snapshot()}

    produced = hook.step({"emg": np.array([[3.0]])}, snapshots)

    np.testing.assert_array_equal(produced["out"], [[1.0]])
    np.testing.assert_array_equal(seen["data"]["emg"], [[3.0]])
    assert seen["snapshots"] is snapshots


def test_a_callback_hook_that_returns_nothing_commits_nothing():
    hook = CallbackHook("sink", [Input("emg", OnCommit(), mode=STATE)], _noop)

    assert hook.step({}, {"emg": _snapshot()}) is None
    assert hook.outputs == []


def test_a_hook_needs_a_trigger_it_understands():
    with pytest.raises(ValueError, match="trigger"):
        CallbackHook("bad", [Input("emg", OnCommit(), mode=STATE)], _noop,
                     trigger="sometimes")


def test_a_feature_hook_declares_a_buffer_wide_enough_for_its_features():
    hook = FeatureHook("features", "emg", "feature_item", ["MAV"], window_size=40,
                       window_increment=10, num_features=8, rows=50)

    assert [(o.tag, o.shape) for o in hook.outputs] == [("feature_item", (50, 8))]
    assert hook.inputs[0].size == 40
    assert hook.inputs[0].criterion.increment == 10
    assert hook.step({"emg": np.zeros((39, 8))}, {}) is None, "an incomplete window"


# ======================================================================
# 12. the notifier pool
# ======================================================================
def test_the_pool_hands_out_distinct_slots_until_it_runs_out():
    pool = NotifierPool(size=3)

    slots = [pool.claim() for _ in range(3)]

    assert slots == [0, 1, 2]
    assert len(set(slots)) == 3
    assert len(pool) == 3
    with pytest.raises(RuntimeError, match="claimed"):
        pool.claim()


def test_a_pool_cannot_be_wider_than_the_subscriber_mask():
    with pytest.raises(ValueError):
        NotifierPool(size=MAX_SUBSCRIBERS + 1)


def test_waiting_on_a_slot_gives_up_at_the_timeout():
    pool = NotifierPool(size=1)
    slot = pool.claim()

    started = time.monotonic()
    held = pool.wait(slot, lambda: False, timeout=0.05)

    assert held is False
    assert time.monotonic() - started >= 0.04
    assert pool.wait(slot, lambda: True, timeout=0.05) is True


def test_the_process_wide_pool_is_the_same_pool_every_time():
    reset_default_notifier_pool()
    try:
        first = default_notifier_pool()
        assert default_notifier_pool() is first
        reset_default_notifier_pool()
        assert default_notifier_pool() is not first
    finally:
        reset_default_notifier_pool()


# ======================================================================
# 8. end to end, across processes
# ======================================================================
def test_a_change_cascades_through_two_executors_to_the_far_end(env, crossprocess, log):
    items = _items(E2E_SRC, shape=(64, 2))
    env.create(items)
    graph = env.graph(items, log=log, notifier_pool=NotifierPool(size=8))
    graph.add(DoublingHook("double", E2E_SRC, E2E_MID, (64, 2)), executor="filtering")
    graph.add(WindowMeanHook("mean", E2E_MID, E2E_OUT, (32, 2), E2E_INCREMENT),
              executor="windowing")
    graph.start()

    assert len(graph._executors) == 2, "the two hooks must be in separate processes"
    assert len({executor.pid for executor in graph._executors}) == 2

    declared = {item[0]: item for item in graph.shared_memory_items}
    reader = env.attach([declared[E2E_SRC], declared[E2E_MID], declared[E2E_OUT]])
    writer = env.spawn(_paced_writer,
                       ([declared[E2E_SRC], declared[E2E_MID], declared[E2E_OUT]],
                        graph.notifier_pool, E2E_SRC, [E2E_MID, E2E_OUT],
                        E2E_INCREMENT, E2E_SAMPLES))
    writer.join(timeout=120)
    assert not writer.is_alive(), "the writer never finished"
    assert writer.exitcode == 0

    expected_source = np.array([[float(i), 10.0 * i]
                                for i in range(1, E2E_SAMPLES + 1)])
    expected_doubled = expected_source * 2.0
    expected_means = expected_doubled.reshape(
        E2E_SAMPLES // E2E_INCREMENT, E2E_INCREMENT, 2).mean(axis=1)

    assert reader.snapshot(E2E_SRC).total_samples == E2E_SAMPLES
    assert reader.snapshot(E2E_MID).total_samples == E2E_SAMPLES
    assert reader.snapshot(E2E_OUT).total_samples == E2E_SAMPLES // E2E_INCREMENT

    source, _ = reader.read_window(E2E_SRC, E2E_SAMPLES)
    doubled, _ = reader.read_window(E2E_MID, E2E_SAMPLES)
    means, _ = reader.read_window(E2E_OUT, E2E_SAMPLES // E2E_INCREMENT)
    np.testing.assert_allclose(source, expected_source)
    np.testing.assert_allclose(doubled, expected_doubled)
    np.testing.assert_allclose(means, expected_means)

    graph.stop(timeout=10)
    summary = log.summary()
    assert summary["invocations"]["double"] >= 1
    assert summary["invocations"]["mean"] == E2E_SAMPLES // E2E_INCREMENT
    assert summary["dirty"]["mean"] == E2E_SAMPLES // E2E_INCREMENT
    # The windowing stage was told about every commit and declined most of them:
    # that disagreement is the point of a per-observer criterion.
    assert summary["clean"].get("mean", 0) > 0


# ======================================================================
# 9. a failing hook is contained
# ======================================================================
def test_a_hook_that_raises_leaves_its_executor_and_its_neighbour_running(
        env, crossprocess, log):
    items = _items(ERR_SRC, shape=(16, 1))
    owner = env.create(items, notifier_pool=None)
    graph = env.graph(items, log=log, notifier_pool=NotifierPool(size=4))
    graph.add(RaisingHook("boom", ERR_SRC), executor="together")
    graph.add(PassThroughHook("survivor", ERR_SRC, ERR_OUT, (16, 1)),
              executor="together")
    graph.start()
    owner.notifier_pool = graph.notifier_pool

    for value in (1.0, 2.0, 3.0):
        owner.commit(ERR_SRC, np.array([value]))
    declared = {item[0]: item for item in graph.shared_memory_items}
    reader = env.attach([declared[ERR_OUT]])

    assert _wait_until(lambda: reader.snapshot(ERR_OUT).total_samples >= 3), \
        "the surviving hook stopped running"
    assert graph._executors[0].is_alive(), "one bad hook took the executor down"
    np.testing.assert_array_equal(reader.read_window(ERR_OUT, 3)[0][:, 0],
                                  [1.0, 2.0, 3.0])

    graph.stop(timeout=10)
    errors = log.events(kind=event_log.ERROR)
    assert errors, "a failing hook has to leave a record"
    assert all(event.origin == "boom" for event in errors)
    assert "deliberate failure" in errors[0].detail["traceback"]
    # The survivor ran, and everything committed to the source reached its
    # output (asserted above). Its invocation count is deliberately NOT
    # checked against the number of commits: a DELTA input hands over
    # everything that accrued since the hook last consumed, so three commits
    # that land before the executor first wakes are correctly delivered as one
    # batched invocation. Counting invocations here would be asserting how the
    # scheduler happened to interleave, not that the data got through.
    assert log.summary()["invocations"]["survivor"] >= 1


# ======================================================================
# 10. a finite source lets the graph finish by itself
# ======================================================================
def test_closing_the_source_lets_a_graph_drain_and_its_executor_exit(
        env, crossprocess, log):
    items = _items(CLOSE_SRC, shape=(4, 1))
    owner = env.create(items)
    graph = env.graph(items, log=log, notifier_pool=NotifierPool(size=4))
    graph.add(ClosingSummaryHook("summarise", CLOSE_SRC, CLOSE_OUT),
              executor="finishing")
    graph.start()
    owner.notifier_pool = graph.notifier_pool
    executor = graph._executors[0]
    declared = {item[0]: item for item in graph.shared_memory_items}
    reader = env.attach([declared[CLOSE_OUT]])

    for value in (1.0, 2.0, 3.0):
        owner.commit(CLOSE_SRC, np.array([value]))
    assert executor.is_alive(), "nothing has closed yet, so nothing may exit"

    owner.mark_closed(CLOSE_SRC)

    executor.join(timeout=60)
    assert not executor.is_alive(), "a drained executor has to exit on its own"
    assert executor.exitcode == 0
    assert reader.snapshot(CLOSE_OUT).total_samples == 1
    assert reader.get_variable(CLOSE_OUT)[0, 0] == pytest.approx(6.0)

    graph.stop(timeout=10)
    drained = [event for event in log.events(kind=event_log.LIFECYCLE)
               if event.detail.get("phase") == "drained"]
    assert drained, "the executor should record that it drained"


# ======================================================================
# the executor is a process, and a hook context is what setup() is given
# ======================================================================
def test_an_executor_is_a_named_daemon_process():
    pool = NotifierPool(size=2)
    executor = Executor("probe", [], [], pool, pool.claim(), event_log.NULL_LOG, None)

    assert isinstance(executor, Process)
    assert executor.daemon is True
    assert executor.name == "libemg-executor-probe"
    assert executor.executor_name == "probe"
