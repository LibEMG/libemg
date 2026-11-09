"""Coverage for the reactive layer's debug log.

The log exists to answer "why did this stage fire?" and "why did it not?", so
what matters is that a decision made in one process is still readable in
another, that the record names the item, the observer and the criterion, and
that switching the log off costs nothing.

Everything a spawned process needs is defined at module level, because Windows
spawns rather than forks and anything nested inside a test function cannot be
pickled into the child.
"""
import os
import pickle
import time

import pytest

from libemg import event_log
from libemg.event_log import NULL_LOG, Event, EventLog


# ======================================================================
# helpers
# ======================================================================
def _wait_until(predicate, timeout=15.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _emit_from_child(log, count):
    """Emit from a spawned process, which is the case the log exists to serve."""
    for index in range(count):
        log.emit(event_log.DIRTY, origin="emg", observer="child",
                 criterion="OnSamples(50)", index=index)


def _emit_kinds_from_child(log):
    log.emit(event_log.COMMIT, origin="emg")
    log.emit(event_log.DIRTY, origin="emg", observer="hook")
    log.emit(event_log.CLEAN, origin="emg", observer="hook")


@pytest.fixture
def log():
    """A started log, stopped and closed however the test ends."""
    made = EventLog(keep=10000)
    made.start()
    try:
        yield made
    finally:
        made.stop()


# ======================================================================
# 1. an event renders readably
# ======================================================================
def test_an_event_renders_its_identifying_fields():
    event = Event(kind=event_log.DIRTY, origin="filtered_emg", observer="model",
                  criterion="OnSamples(40)",
                  detail={"generation": 7, "total_samples": 400})
    line = event.format()

    assert event_log.DIRTY in line
    assert "origin=filtered_emg" in line
    assert "observer=model" in line
    assert "criterion=OnSamples(40)" in line
    assert "generation=7" in line
    assert "total_samples=400" in line
    assert f"pid={os.getpid()}" in line
    assert "\n" not in line, "an event has to stay on one line to be greppable"


def test_an_event_renders_absent_fields_as_placeholders():
    line = Event(kind=event_log.COMMIT, origin="emg").format()

    assert "observer=-" in line
    assert "criterion=-" in line


def test_event_detail_is_ordered_so_lines_are_comparable():
    first = Event(kind=event_log.DIRTY, detail={"b": 2, "a": 1}).format()
    second = Event(kind=event_log.DIRTY, detail={"a": 1, "b": 2}).format()

    assert first.split("\t")[-1] == second.split("\t")[-1]


def test_floats_are_rendered_without_noise():
    line = Event(kind=event_log.COMPLETE, detail={"duration_ms": 1.0 / 3.0}).format()

    assert "duration_ms=0.333333" in line


# ======================================================================
# 2. events cross a process boundary
# ======================================================================
def test_events_emitted_in_another_process_are_collected_here(log):
    from multiprocessing import Process

    child = Process(target=_emit_from_child, args=(log, 5), daemon=True)
    child.start()
    child.join(timeout=30)
    assert child.exitcode == 0

    assert _wait_until(lambda: len(log.events(observer="child")) >= 5), \
        "events emitted in a child never reached the owning process"
    collected = log.events(observer="child")
    assert {event.origin for event in collected} == {"emg"}
    assert {event.criterion for event in collected} == {"OnSamples(50)"}
    # The pid on the record has to be the emitter's, or a log spanning
    # processes cannot be untangled.
    assert all(event.pid != os.getpid() for event in collected)


def test_stop_drains_what_is_still_queued(log):
    for index in range(50):
        log.emit(event_log.COMMIT, origin="emg", index=index)
    log.stop()

    assert len(log.events(kind=event_log.COMMIT)) == 50, \
        "stopping must flush the queue, not discard it"


# ======================================================================
# 3. filtering happens in the emitting process
# ======================================================================
def test_kind_filtering_excludes_unwanted_events():
    log = EventLog(keep=1000, kinds=[event_log.DIRTY])
    log.start()
    try:
        log.emit(event_log.DIRTY, origin="emg", observer="hook")
        log.emit(event_log.CLEAN, origin="emg", observer="hook")
        log.emit(event_log.COMMIT, origin="emg")
        log.stop()
    finally:
        log.stop()

    kinds = {event.kind for event in log.events()}
    assert kinds == {event_log.DIRTY}


def test_kind_filtering_applies_to_a_child_process_too():
    from multiprocessing import Process

    log = EventLog(keep=1000, kinds=[event_log.COMMIT])
    log.start()
    try:
        child = Process(target=_emit_kinds_from_child, args=(log,), daemon=True)
        child.start()
        child.join(timeout=30)
        assert child.exitcode == 0
        _wait_until(lambda: len(log.events()) >= 1)
        log.stop()
    finally:
        log.stop()

    assert {event.kind for event in log.events()} == {event_log.COMMIT}


# ======================================================================
# 4. writing to a file
# ======================================================================
def test_events_are_appended_to_a_file(tmp_path):
    path = tmp_path / "reactive.log"
    log = EventLog(path=str(path), keep=100)
    log.start()
    try:
        log.emit(event_log.DIRTY, origin="emg", observer="model",
                 criterion="OnSamples(40)", generation=3)
        log.stop()
    finally:
        log.stop()

    assert path.exists()
    contents = path.read_text()
    assert contents.startswith("#"), "the file should say when it was opened"
    body = [line for line in contents.splitlines() if not line.startswith("#")]
    assert len(body) == 1
    assert "origin=emg" in body[0]
    assert "observer=model" in body[0]
    assert "criterion=OnSamples(40)" in body[0]


def test_a_second_run_appends_rather_than_truncating(tmp_path):
    path = tmp_path / "reactive.log"
    for _ in range(2):
        log = EventLog(path=str(path), keep=10)
        log.start()
        log.emit(event_log.COMMIT, origin="emg")
        log.stop()

    body = [line for line in path.read_text().splitlines()
            if not line.startswith("#")]
    assert len(body) == 2


# ======================================================================
# 5. retention is bounded
# ======================================================================
def test_retention_is_bounded_so_a_long_session_cannot_grow_without_limit():
    log = EventLog(keep=25)
    log.start()
    try:
        for index in range(300):
            log.emit(event_log.COMMIT, origin="emg", index=index)
        log.stop()
    finally:
        log.stop()

    retained = log.events()
    assert len(retained) == 25
    # The newest are the ones worth keeping.
    assert retained[-1].detail["index"] == 299


def test_keep_zero_retains_nothing_but_still_writes(tmp_path):
    path = tmp_path / "reactive.log"
    log = EventLog(path=str(path), keep=0)
    log.start()
    try:
        for _ in range(10):
            log.emit(event_log.COMMIT, origin="emg")
        log.stop()
    finally:
        log.stop()

    assert log.events() == []
    body = [line for line in path.read_text().splitlines()
            if not line.startswith("#")]
    assert len(body) == 10, "keep=0 bounds memory, it does not stop the file"


# ======================================================================
# 6. summaries
# ======================================================================
def test_summary_reports_invocations_and_the_dirty_clean_split(log):
    for _ in range(3):
        log.emit(event_log.INVOKE, origin="emg", observer="model")
        log.emit(event_log.DIRTY, origin="emg", observer="model")
    for _ in range(7):
        log.emit(event_log.CLEAN, origin="emg", observer="model")
    log.emit(event_log.INVOKE, origin="emg", observer="filter")
    log.stop()

    summary = log.summary()
    assert summary["invocations"] == {"model": 3, "filter": 1}
    assert summary["dirty"] == {"model": 3}
    assert summary["clean"] == {"model": 7}
    assert summary["kinds"][event_log.CLEAN] == 7
    assert summary["kinds"][event_log.INVOKE] == 4


def test_events_can_be_filtered_when_read_back(log):
    log.emit(event_log.DIRTY, origin="emg", observer="a")
    log.emit(event_log.DIRTY, origin="imu", observer="b")
    log.emit(event_log.CLEAN, origin="emg", observer="a")
    log.stop()

    assert len(log.events(kind=event_log.DIRTY)) == 2
    assert len(log.events(origin="emg")) == 2
    assert len(log.events(observer="b")) == 1
    assert len(log.events(kind=event_log.DIRTY, origin="emg")) == 1


def test_summary_of_an_empty_log_is_empty(log):
    log.stop()
    summary = log.summary()

    assert summary["kinds"] == {}
    assert summary["invocations"] == {}


# ======================================================================
# 7. the disabled case has to cost nothing and break nothing
# ======================================================================
def test_the_null_log_absorbs_every_call():
    assert NULL_LOG.enabled is False
    # The reactive layer calls all of these on hot paths without checking.
    NULL_LOG.emit(event_log.COMMIT, origin="emg", observer="x", criterion="y", n=1)
    NULL_LOG.record(Event(kind=event_log.COMMIT))
    NULL_LOG.start()
    NULL_LOG.stop()
    NULL_LOG.close()


def test_the_null_log_survives_being_sent_to_another_process():
    # A hook carrying the null log is pickled into its executor, so the null
    # log has to pickle even though it holds nothing.
    revived = pickle.loads(pickle.dumps(NULL_LOG))

    assert revived.enabled is False
    revived.emit(event_log.COMMIT, origin="emg")


def test_a_real_log_hands_a_child_what_it_needs_and_nothing_it_should_not_have():
    # A real log carries a multiprocessing queue, which the platform only
    # transfers while a process is actually being spawned, never through a
    # plain pickle round-trip. So the transferred state is checked directly;
    # that it genuinely arrives in a child is covered by the cross-process
    # test above, which passes it through Process args as the library does.
    log = EventLog(path="unused.log", keep=5, kinds=[event_log.DIRTY])
    log.start()
    try:
        state = log.__getstate__()
    finally:
        log.stop()

    # The drain thread and the open file belong to the owning process; a child
    # emits but must never write.
    assert state["_thread"] is None
    assert state["_handle"] is None
    assert state["_records"] == []
    # What the child does need: the queue to emit onto, and the filter so
    # excluded events cost nothing on its side.
    assert state["_queue"] is not None
    assert state["kinds"] == {event_log.DIRTY}

    revived = EventLog.__new__(EventLog)
    revived.__setstate__(state)
    assert revived._thread is None
    revived.emit(event_log.DIRTY, origin="emg")


def test_an_unstarted_log_accepts_events_without_raising():
    log = EventLog(keep=10)
    log.emit(event_log.COMMIT, origin="emg")
    log.stop()


def test_the_log_is_enabled_flag_lets_hot_paths_skip_work():
    # The reactive layer guards detail-building with this, so it has to be a
    # plain truthy attribute on both shapes.
    assert EventLog().enabled is True
    assert NULL_LOG.enabled is False
