"""Coverage for the state block that makes a shared-memory item observable.

Every variable carries a companion block of counters saying what has happened
to it. These tests pin down two things: that the new write path (``commit``)
lands the same bytes the streamers' old read-modify-write idiom did, and that
the counters, subscriptions and stateful reads built on top of it say what the
reactive layer relies on them saying.
"""
import itertools
import os
import threading

import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory

from libemg.shared_memory_manager import (MAX_SUBSCRIBERS, STATE_SUFFIX, Snapshot,
                                          SharedMemoryManager,
                                          assign_shared_memory_locks)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
_COUNTER = itertools.count()


def _unique(base):
    """A shared-memory tag no other test (or run) can be holding."""
    return f"libemgtest_{base}_{os.getpid()}_{next(_COUNTER)}"


def _items(base, capacity=8, channels=2, with_count=True):
    """A ``[tag, shape, dtype, lock]`` set shaped like the streamers produce."""
    spec = [[_unique(base), (capacity, channels), np.double]]
    if with_count:
        spec.append([spec[0][0] + "_count", (1, 1), np.int32])
    return assign_shared_memory_locks(spec)


def _legacy_write(buffer, new):
    """The read-modify-write idiom the streamers used before ``commit``."""
    new = np.atleast_2d(new)
    capacity = buffer.shape[0]
    buffer[:] = np.vstack((np.flip(new, 0), buffer))[:capacity, :]


class _Managers:
    """Hands out managers and guarantees every segment is released."""

    def __init__(self):
        self._created = []
        self._attached = []

    def create(self, items, **kwargs):
        smm = SharedMemoryManager(**kwargs)
        for item in items:
            assert smm.create_variable(*item)
        self._created.append(smm)
        return smm

    def attach(self, items, **kwargs):
        smm = SharedMemoryManager(**kwargs)
        for item in items:
            assert smm.find_variable(*item)
        self._attached.append(smm)
        return smm

    def release(self):
        # Attached managers only ever borrowed the segments, so they close;
        # the creators are the ones that unlink.
        for smm in reversed(self._attached):
            try:
                smm.cleanup(parent=False)
            except Exception:
                pass
        for smm in reversed(self._created):
            try:
                smm.cleanup(parent=True)
            except Exception:
                pass


@pytest.fixture
def managers():
    handout = _Managers()
    try:
        yield handout
    finally:
        handout.release()


# ----------------------------------------------------------------------
# 1. commit lands the same bytes the old idiom did
# ----------------------------------------------------------------------
_WRITES = {
    "single_sample": [np.array([1.0, 2.0])],
    "multi_row": [np.arange(6.0).reshape(3, 2)],
    "larger_than_the_buffer": [np.arange(100.0, 124.0).reshape(12, 2)],
    "repeated_until_it_wraps": [np.arange(6.0).reshape(3, 2),
                                np.arange(10.0, 20.0).reshape(5, 2),
                                np.arange(50.0, 58.0).reshape(4, 2),
                                np.array([99.0, 98.0])],
}


@pytest.mark.parametrize("case", sorted(_WRITES))
def test_commit_writes_what_the_read_modify_write_idiom_wrote(managers, case):
    writes = _WRITES[case]
    items = _items("legacy_" + case, capacity=8)
    tag = items[0][0]
    smm = managers.create(items)
    reference = np.zeros((8, 2))

    for new in writes:
        smm.commit(tag, new)
        _legacy_write(reference, new)

    np.testing.assert_array_equal(smm.get_variable(tag), reference)
    # Index 0 is the contract every existing reader is written against.
    newest = np.atleast_2d(writes[-1])[-1]
    np.testing.assert_array_equal(smm.get_variable(tag)[0], newest)


def test_a_write_larger_than_the_buffer_keeps_the_newest_rows(managers):
    items = _items("oversized", capacity=4, channels=1)
    tag = items[0][0]
    smm = managers.create(items)

    smm.commit(tag, np.arange(1.0, 11.0).reshape(10, 1))

    np.testing.assert_array_equal(smm.get_variable(tag)[:, 0], [10.0, 9.0, 8.0, 7.0])


# ----------------------------------------------------------------------
# 2. the legacy counter is kept in step
# ----------------------------------------------------------------------
def test_commit_advances_the_legacy_sample_counter(managers):
    items = _items("counted")
    tag = items[0][0]
    smm = managers.create(items)

    smm.commit(tag, np.zeros(2))
    smm.commit(tag, np.zeros((3, 2)))

    assert int(smm.get_variable(tag + "_count")[0, 0]) == 4
    assert smm.snapshot(tag).total_samples == 4


def test_commit_does_not_need_a_counter_companion(managers):
    items = _items("uncounted", with_count=False)
    tag = items[0][0]
    smm = managers.create(items)

    snapshot = smm.commit(tag, np.ones(2))

    assert tag + "_count" not in smm.variables
    assert snapshot.total_samples == 1


def test_an_explicitly_named_counter_is_the_one_advanced(managers):
    items = _items("explicit_counter")
    tag, count_tag = items[0][0], items[1][0]
    smm = managers.create(items)

    smm.commit(tag, np.zeros((2, 2)), count_tag=count_tag)

    assert int(smm.get_variable(count_tag)[0, 0]) == 2


# ----------------------------------------------------------------------
# 3. the state block's counters
# ----------------------------------------------------------------------
def test_generation_advances_once_per_commit_however_big_it_is(managers):
    items = _items("generation", capacity=8)
    tag = items[0][0]
    smm = managers.create(items)

    assert smm.snapshot(tag).generation == 0
    smm.commit(tag, np.zeros(2))
    assert smm.snapshot(tag).generation == 1
    smm.commit(tag, np.zeros((5, 2)))
    assert smm.snapshot(tag).generation == 2


def test_total_samples_accumulates_row_counts(managers):
    items = _items("totals", capacity=16)
    tag = items[0][0]
    smm = managers.create(items)

    for rows in (1, 3, 7):
        smm.commit(tag, np.zeros((rows, 2)))

    snapshot = smm.snapshot(tag)
    assert (snapshot.total_samples, snapshot.commits) == (11, 3)


def test_rows_are_only_reported_dropped_when_a_write_outruns_the_buffer(managers):
    items = _items("dropped", capacity=4)
    tag = items[0][0]
    smm = managers.create(items)

    smm.commit(tag, np.zeros((4, 2)))
    smm.commit(tag, np.zeros((4, 2)))
    # Wrapping the buffer over several commits drops nothing: each of those
    # writes fitted, whatever it displaced.
    assert smm.snapshot(tag).dropped == 0

    smm.commit(tag, np.zeros((6, 2)))
    snapshot = smm.snapshot(tag)
    assert (snapshot.dropped, snapshot.total_samples, snapshot.commits) == (2, 14, 3)


def test_a_commit_stamps_the_time_and_leaves_epoch_and_closed_alone(managers):
    items = _items("stamp")
    tag = items[0][0]
    smm = managers.create(items)

    snapshot = smm.commit(tag, np.zeros(2))

    assert snapshot.t_last_ns > 0
    assert snapshot.epoch == 0
    assert snapshot.closed is False
    assert snapshot.tag == tag


# ----------------------------------------------------------------------
# 4. snapshots are pure reads
# ----------------------------------------------------------------------
def test_reading_the_state_changes_nothing(managers):
    items = _items("pure_read", capacity=8)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.arange(6.0).reshape(3, 2))
    before = smm.get_variable(tag)

    first = smm.snapshot(tag)
    second = smm.snapshot(tag)

    assert first == second
    assert isinstance(first, Snapshot)
    np.testing.assert_array_equal(smm.get_variable(tag), before)
    assert int(smm.get_variable(tag + "_count")[0, 0]) == 3


def test_a_snapshot_reflects_the_latest_commit(managers):
    items = _items("latest_commit", capacity=8)
    tag = items[0][0]
    smm = managers.create(items)

    returned = smm.commit(tag, np.zeros((2, 2)))

    assert smm.snapshot(tag) == returned


def test_snapshots_reads_several_items_at_once(managers):
    items = _items("multi_snapshot")
    tag, count_tag = items[0][0], items[1][0]
    smm = managers.create(items)
    smm.commit(tag, np.zeros(2))

    snapshots = smm.snapshots([tag, count_tag])

    assert set(snapshots) == {tag, count_tag}
    assert snapshots[tag].total_samples == 1
    # The counter is a variable in its own right and nobody commits to it.
    assert snapshots[count_tag].total_samples == 0


# ----------------------------------------------------------------------
# 5. reset_state
# ----------------------------------------------------------------------
def test_reset_zeros_the_counters_and_bumps_the_epoch(managers):
    items = _items("reset", capacity=8)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.zeros((3, 2)))

    smm.reset_state(tag)

    snapshot = smm.snapshot(tag)
    assert (snapshot.generation, snapshot.total_samples, snapshot.commits,
            snapshot.dropped, snapshot.t_last_ns) == (0, 0, 0, 0, 0)
    assert snapshot.epoch == 1
    smm.reset_state(tag)
    assert smm.snapshot(tag).epoch == 2


def test_reset_does_not_silently_unsubscribe_observers(managers):
    items = _items("reset_subscribers")
    tag = items[0][0]
    smm = managers.create(items)
    smm.subscribe(tag, 0)
    smm.subscribe(tag, 5)

    smm.reset_state(tag)

    assert smm.subscribers(tag) == [0, 5]


# ----------------------------------------------------------------------
# 6. mark_closed
# ----------------------------------------------------------------------
def test_mark_closed_records_that_the_writer_is_finished(managers):
    items = _items("closed")
    tag = items[0][0]
    smm = managers.create(items)
    assert smm.snapshot(tag).closed is False

    smm.mark_closed(tag)

    assert smm.snapshot(tag).closed is True
    # Closing says nothing about the data, so the counters stand.
    assert smm.snapshot(tag).total_samples == 0


# ----------------------------------------------------------------------
# 7. read_window
# ----------------------------------------------------------------------
def test_read_window_returns_the_newest_rows_oldest_first(managers):
    items = _items("window", capacity=8, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.arange(1.0, 6.0).reshape(5, 1))

    window, snapshot = smm.read_window(tag, 3)

    np.testing.assert_array_equal(window[:, 0], [3.0, 4.0, 5.0])
    assert snapshot.total_samples == 5


def test_read_window_can_return_the_buffers_own_newest_first_layout(managers):
    items = _items("window_reversed", capacity=8, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.arange(1.0, 6.0).reshape(5, 1))

    window, _ = smm.read_window(tag, 3, chronological=False)

    np.testing.assert_array_equal(window[:, 0], [5.0, 4.0, 3.0])


def test_read_window_clamps_to_the_buffer(managers):
    items = _items("window_clamped", capacity=4, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.arange(1.0, 4.0).reshape(3, 1))

    window, _ = smm.read_window(tag, 1000)

    assert window.shape == (4, 1)
    # The one row never written is still the zero it was created as.
    np.testing.assert_array_equal(window[:, 0], [0.0, 1.0, 2.0, 3.0])


def test_read_window_hands_back_a_copy(managers):
    items = _items("window_copy", capacity=8, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.ones((2, 1)))

    window, _ = smm.read_window(tag, 2)
    window[:] = 42.0

    np.testing.assert_array_equal(smm.get_variable(tag)[:2, 0], [1.0, 1.0])


# ----------------------------------------------------------------------
# 8. read_since
# ----------------------------------------------------------------------
def test_read_since_returns_exactly_what_arrived(managers):
    items = _items("since", capacity=8, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.arange(1.0, 4.0).reshape(3, 1))

    samples, snapshot, lost = smm.read_since(tag, 0)

    assert (snapshot.total_samples, lost) == (3, 0)
    np.testing.assert_array_equal(samples[:, 0], [1.0, 2.0, 3.0])

    samples, snapshot, lost = smm.read_since(tag, snapshot.total_samples)
    assert (samples.shape[0], lost) == (0, 0)

    smm.commit(tag, np.array([4.0]))
    samples, _, lost = smm.read_since(tag, 3)
    np.testing.assert_array_equal(samples[:, 0], [4.0])
    assert lost == 0


def test_read_since_comes_back_empty_when_nothing_new_arrived(managers):
    items = _items("since_empty", capacity=8, channels=2)
    tag = items[0][0]
    smm = managers.create(items)

    samples, snapshot, lost = smm.read_since(tag, 0)

    assert samples.shape == (0, 2)
    assert (snapshot.total_samples, lost) == (0, 0)
    # A reader that is somehow ahead of the writer is not an error either.
    samples, _, lost = smm.read_since(tag, 99)
    assert (samples.shape[0], lost) == (0, 0)


def test_read_since_reports_the_rows_the_buffer_overwrote(managers):
    items = _items("since_lost", capacity=4, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    for value in np.arange(1.0, 11.0):
        smm.commit(tag, np.array([value]))

    samples, snapshot, lost = smm.read_since(tag, 0)

    assert (snapshot.total_samples, lost) == (10, 6)
    # What survives is the newest four, not the ten that were asked for.
    np.testing.assert_array_equal(samples[:, 0], [7.0, 8.0, 9.0, 10.0])


def test_read_since_can_return_the_newest_first_layout(managers):
    items = _items("since_reversed", capacity=8, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.arange(1.0, 4.0).reshape(3, 1))

    samples, _, _ = smm.read_since(tag, 0, chronological=False)

    np.testing.assert_array_equal(samples[:, 0], [3.0, 2.0, 1.0])


# ----------------------------------------------------------------------
# 9. subscription
# ----------------------------------------------------------------------
def test_subscriptions_round_trip(managers):
    items = _items("subscriptions")
    tag = items[0][0]
    smm = managers.create(items)
    assert smm.subscribers(tag) == []

    smm.subscribe(tag, 0)
    smm.subscribe(tag, 3)
    smm.subscribe(tag, MAX_SUBSCRIBERS - 1)
    assert smm.subscribers(tag) == [0, 3, MAX_SUBSCRIBERS - 1]

    smm.unsubscribe(tag, 3)
    assert smm.subscribers(tag) == [0, MAX_SUBSCRIBERS - 1]

    # Subscribing twice is not two subscriptions.
    smm.subscribe(tag, 0)
    assert smm.subscribers(tag) == [0, MAX_SUBSCRIBERS - 1]


def test_a_subscription_is_visible_to_another_manager_on_the_same_item(managers):
    items = _items("subscription_shared")
    tag = items[0][0]
    writer = managers.create(items)
    observer = managers.attach(items)

    observer.subscribe(tag, 2)

    assert writer.subscribers(tag) == [2]


@pytest.mark.parametrize("slot", [-1, MAX_SUBSCRIBERS, MAX_SUBSCRIBERS + 10])
def test_an_out_of_range_notifier_slot_is_refused(managers, slot):
    items = _items("subscription_range")
    tag = items[0][0]
    smm = managers.create(items)

    with pytest.raises(AssertionError):
        smm.subscribe(tag, slot)


class _RecordingPool:
    """A notifier pool that only remembers who it was asked to wake."""

    def __init__(self):
        self.signalled = []

    def signal(self, slot):
        self.signalled.append(slot)


def test_committing_wakes_every_subscribed_slot(managers):
    pool = _RecordingPool()
    items = _items("wake")
    tag = items[0][0]
    smm = managers.create(items, notifier_pool=pool)
    smm.subscribe(tag, 1)
    smm.subscribe(tag, 4)

    smm.commit(tag, np.zeros(2))

    assert pool.signalled == [1, 4]
    smm.mark_closed(tag)
    assert pool.signalled == [1, 4, 1, 4], "closing has to wake observers too"


# ----------------------------------------------------------------------
# 10. the pre-existing interface, unchanged
# ----------------------------------------------------------------------
def test_get_variable_returns_a_copy_of_the_whole_buffer(managers):
    items = _items("get_variable", capacity=4, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.array([7.0]))

    copy = smm.get_variable(tag)
    copy[0, 0] = -1.0

    assert copy.shape == (4, 1)
    assert smm.get_variable(tag)[0, 0] == 7.0


def test_get_variables_returns_every_tag_asked_for(managers):
    items = _items("get_variables", capacity=4, channels=1)
    tag, count_tag = items[0][0], items[1][0]
    smm = managers.create(items)
    smm.commit(tag, np.array([3.0]))

    values = smm.get_variables([tag, count_tag])

    assert set(values) == {tag, count_tag}
    assert values[tag][0, 0] == 3.0
    assert values[count_tag][0, 0] == 1


def test_get_samples_since_still_reads_through_the_legacy_counter(managers):
    items = _items("get_samples_since", capacity=8, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.arange(1.0, 4.0).reshape(3, 1))

    count, samples, dropped = smm.get_samples_since(tag, 0)

    assert (count, dropped) == (3, 0)
    # Newest first, which is what this call has always returned.
    np.testing.assert_array_equal(samples[:, 0], [3.0, 2.0, 1.0])

    count, samples, dropped = smm.get_samples_since(tag, 3)
    assert (count, samples.shape[0], dropped) == (3, 0, 0)


def test_modify_variable_still_writes_through_a_function(managers):
    items = _items("modify", capacity=4, channels=1)
    tag = items[0][0]
    smm = managers.create(items)
    smm.commit(tag, np.array([2.0]))

    smm.modify_variable(tag, lambda data: data + 1.0)

    np.testing.assert_array_equal(smm.get_variable(tag)[:, 0], [3.0, 1.0, 1.0, 1.0])
    # Writing behind commit's back deliberately leaves the state block alone.
    assert smm.snapshot(tag).generation == 1


def test_get_variable_list_entries_still_unpack_into_three(managers):
    items = _items("variable_list", capacity=4, channels=1)
    tag, count_tag = items[0][0], items[1][0]
    smm = managers.create(items)

    listing = smm.get_variable_list()

    assert len(listing) == 2
    for entry in listing:
        assert len(entry) == 3
    unpacked = {name: (shape, dtype) for name, shape, dtype in listing}
    assert unpacked[tag] == ((4, 1), np.double)
    assert unpacked[count_tag] == ((1, 1), np.int32)


def test_get_shared_memory_items_entries_still_unpack_into_four(managers):
    items = _items("smi", capacity=4, channels=1)
    tag, count_tag = items[0][0], items[1][0]
    smm = managers.create(items)

    listing = smm.get_shared_memory_items()

    assert len(listing) == 2
    for entry in listing:
        assert len(entry) == 4
    unpacked = {name: (shape, dtype, lock)
                for name, shape, dtype, lock in listing}
    assert unpacked[tag][0] == (4, 1)
    assert unpacked[tag][1] == np.double
    # The lock handed in is the lock handed back, and the pair still shares it.
    assert unpacked[tag][2] is items[0][3]
    assert unpacked[count_tag][2] is unpacked[tag][2]


def test_finding_a_variable_that_was_never_created_fails_quietly():
    smm = SharedMemoryManager()

    found = smm.find_variable(_unique("absent"), (4, 1), np.double, threading.Lock())

    assert found is False
    assert smm.variables == {}


# ----------------------------------------------------------------------
# 11. cleanup releases the state segments too
# ----------------------------------------------------------------------
def _segment_exists(name):
    try:
        handle = SharedMemory(name, create=False)
    except FileNotFoundError:
        return False
    handle.close()
    return True


def test_cleanup_as_parent_releases_the_state_segments(managers):
    items = _items("cleanup_parent", capacity=4, channels=1)
    tag, count_tag = items[0][0], items[1][0]
    smm = managers.create(items)
    smm.commit(tag, np.array([1.0]))
    assert _segment_exists(tag + STATE_SUFFIX)

    smm.cleanup(parent=True)

    assert smm.variables == {} and smm.state == {}
    for name in (tag, count_tag, tag + STATE_SUFFIX, count_tag + STATE_SUFFIX):
        assert not _segment_exists(name), name + " leaked"


def test_cleanup_as_child_leaves_the_item_usable_by_whoever_else_holds_it(managers):
    items = _items("cleanup_child", capacity=8, channels=1)
    tag = items[0][0]
    creator = managers.create(items)
    other = managers.attach(items)
    creator.commit(tag, np.array([1.0]))

    creator.cleanup(parent=False)

    assert creator.variables == {}
    # The segment is still there for the manager that is still attached.
    other.commit(tag, np.array([2.0]))
    snapshot = other.snapshot(tag)
    assert (snapshot.total_samples, snapshot.generation) == (2, 2)
    np.testing.assert_array_equal(other.get_variable(tag)[:2, 0], [2.0, 1.0])

    other.cleanup(parent=True)
    assert not _segment_exists(tag)
    assert not _segment_exists(tag + STATE_SUFFIX)


def test_a_second_manager_sees_the_state_the_first_one_committed(managers):
    items = _items("cross_manager", capacity=8, channels=1)
    tag = items[0][0]
    writer = managers.create(items)
    reader = managers.attach(items)

    writer.commit(tag, np.arange(1.0, 4.0).reshape(3, 1))

    assert reader.snapshot(tag) == writer.snapshot(tag)
    samples, _, lost = reader.read_since(tag, 0)
    assert lost == 0
    np.testing.assert_array_equal(samples[:, 0], [1.0, 2.0, 3.0])
