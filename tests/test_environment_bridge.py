"""Coverage for running an environment offscreen and showing it in the GUI.

The feature is two halves that have to agree. A
:class:`~libemg._gui._environments.frame_bridge.FrameBridge` is a piece of
memory two processes both point at, so almost everything about it is only
interesting when somebody else is looking: writing a frame and reading it back
through the same object proves nothing a plain numpy array would not. Every
round-trip here is therefore also asserted through a *second* bridge attached
to the same name, and the one that matters most is asserted from a spawned
process.

The other half is the environment itself, which is a real pygame game with its
window taken away. Those tests launch one, wait for it to draw, and stop it.
They poll to a deadline rather than sleeping a fixed time, because a pygame
process takes as long as the machine takes to start it, and they skip rather
than fail where pygame cannot initialise at all.

Everything a spawned process needs is defined at module level, because Windows
spawns rather than forks and nothing nested inside a test function can be
pickled into the child.
"""
import os

# Before pygame is imported anywhere in this process: the video driver is
# chosen when the display module initialises, and the dummy driver is what lets
# set_mode produce a surface with no window on screen.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import itertools
import time
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from libemg._gui._environments import (FORWARDED_KEYS, ControllerSpec,
                                       EmbeddedEnvironment, FrameBridge,
                                       build_factory, build_registry)
from libemg._gui._environments.embedded import _KeyState, _forward_key_state


# ======================================================================
# helpers
# ======================================================================
_COUNTER = itertools.count()


def _unique(base):
    """A segment name no other test, and no earlier run, can collide with."""
    return f"libemgtest_env_{base}_{os.getpid()}_{next(_COUNTER)}"


def _wait_until(predicate, timeout=60.0, interval=0.02):
    """Poll ``predicate`` until it holds or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _pixel(frame, row, column, width):
    """One RGBA pixel out of the flat buffer, laid out the way publish writes it."""
    start = ((row * width) + column) * 4
    return [round(float(value), 4) for value in frame[start:start + 4]]


# ======================================================================
# module-level target, so a spawned process can unpickle it
# ======================================================================
CHILD_COLOUR = (255, 128, 0)
CHILD_MESSAGE = "written from the other process"


def _attach_and_publish(name, width, height):
    """Attach to an existing bridge by name and write a frame into it.

    Carrying frames between processes is the whole point of the bridge, so it
    is exercised the way it is used: the child is given nothing but a name, and
    everything it writes has to be visible to the parent that created the
    segments.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
    import pygame
    pygame.init()
    try:
        bridge = FrameBridge(name)
        assert (bridge.width, bridge.height) == (width, height)
        surface = pygame.Surface((width, height))
        surface.fill(CHILD_COLOUR)
        bridge.mark_running(True)
        bridge.publish(surface)
        bridge.report_error(CHILD_MESSAGE)
        bridge.close()
    finally:
        pygame.quit()


# ======================================================================
# fixtures
# ======================================================================
class _Segments:
    """Owns every bridge, environment and process a test made, and releases them.

    Shared memory outlives the process that forgot it, so a test that fails
    half way through still has to hand its segments back, or the next run
    inherits them.
    """

    def __init__(self):
        self.created = []
        self.attached = []
        self.embedded = []
        self.processes = []

    def create(self, name, width, height):
        bridge = FrameBridge(name, width, height, create=True)
        self.created.append(bridge)
        return bridge

    def attach(self, name):
        bridge = FrameBridge(name)
        self.attached.append(bridge)
        return bridge

    def embed(self, name, factory, width, height, **kwargs):
        embedded = EmbeddedEnvironment(name, factory, width, height, **kwargs)
        self.embedded.append(embedded)
        return embedded.start()

    def spawn(self, target, args):
        process = Process(target=target, args=args, daemon=True)
        self.processes.append(process)
        process.start()
        return process

    def release(self):
        for process in reversed(self.processes):
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for embedded in reversed(self.embedded):
            try:
                embedded.stop(timeout=10)
            except Exception:
                pass
        for bridge in reversed(self.attached):
            try:
                bridge.close()
            except Exception:
                pass
        for bridge in reversed(self.created):
            try:
                bridge.close(unlink=True)
            except Exception:
                pass


@pytest.fixture
def segments():
    owned = _Segments()
    try:
        yield owned
    finally:
        owned.release()


@pytest.fixture
def shared_memory():
    """Skip rather than fail where the platform cannot give us a segment."""
    name = _unique("probe")
    try:
        handle = SharedMemory(name, create=True, size=8)
    except Exception as problem:  # pragma: no cover - environment dependent
        pytest.skip(f"shared memory is unavailable here: {problem}")
    handle.close()
    handle.unlink()


@pytest.fixture
def pygame_module(shared_memory):
    """The pygame module, or a skipped test where it cannot start."""
    try:
        import pygame
    except Exception as problem:  # pragma: no cover - environment dependent
        pytest.skip(f"pygame is unavailable here: {problem}")
    try:
        pygame.init()
        pygame.display.init()
    except Exception as problem:  # pragma: no cover - environment dependent
        pytest.skip(f"pygame cannot initialise a display here: {problem}")
    return pygame


def _surface(pygame, width, height, colour=(255, 0, 0)):
    surface = pygame.Surface((width, height))
    surface.fill(colour)
    return surface


# ======================================================================
# 1. the frame is sized for what a raw texture wants
# ======================================================================
def test_a_new_bridge_sizes_its_frame_for_float_rgba(segments, shared_memory):
    bridge = segments.create(_unique("size"), 40, 30)

    pixels = bridge.pixels()

    assert pixels.shape == (40 * 30 * 4,), "DearPyGui wants one flat array"
    assert pixels.dtype == np.float32
    assert pixels.nbytes == 40 * 30 * 4 * 4
    np.testing.assert_array_equal(pixels, np.zeros(40 * 30 * 4, dtype=np.float32))
    assert (bridge.width, bridge.height) == (40, 30)


def test_creating_a_bridge_without_a_frame_size_is_refused(shared_memory):
    with pytest.raises(ValueError):
        FrameBridge(_unique("nosize"), 0, 0, create=True)


# ======================================================================
# 2. the control block
# ======================================================================
def test_control_fields_round_trip(segments, shared_memory):
    bridge = segments.create(_unique("control"), 8, 4)

    assert (bridge.running(), bridge.finished(), bridge.quit_requested()) == \
        (False, False, False)
    assert (bridge.frames(), bridge.generation()) == (0, 0)

    bridge.mark_running(True)
    assert bridge.running() is True
    bridge.mark_running(False)
    assert bridge.running() is False

    bridge.request_quit()
    assert bridge.quit_requested() is True

    bridge.mark_running(True)
    bridge.mark_finished()
    assert bridge.finished() is True
    assert bridge.running() is False, "finishing has to clear running"


def test_publishing_bumps_the_frame_counters(segments, pygame_module):
    bridge = segments.create(_unique("counters"), 8, 4)
    surface = _surface(pygame_module, 8, 4)

    bridge.publish(surface)
    assert (bridge.generation(), bridge.frames()) == (1, 1)

    bridge.publish(surface)
    assert (bridge.generation(), bridge.frames()) == (2, 2)
    assert bridge.fps() > 0, "a measured rate appears once frames flow"


# ======================================================================
# 3. the failure message
# ======================================================================
def test_an_unreported_error_reads_as_an_empty_string(segments, shared_memory):
    bridge = segments.create(_unique("noerror"), 8, 4)

    assert bridge.error() == ""


def test_an_error_round_trips_and_an_oversized_one_is_truncated(segments,
                                                                shared_memory):
    bridge = segments.create(_unique("error"), 8, 4)

    bridge.report_error("Radius between ISO Fitts targets is too large.")
    assert bridge.error() == "Radius between ISO Fitts targets is too large."

    # A traceback is easily longer than the block reserved for it. Losing the
    # tail of the message is fine; raising inside the handler that is trying to
    # explain a failure is not.
    bridge.report_error("x" * 10000)
    truncated = bridge.error()
    assert len(truncated) == 2047, "the message is cut to the space reserved"
    assert set(truncated) == {"x"}

    bridge.report_error("")
    assert bridge.error() == "", "reporting nothing clears the old message"


# ======================================================================
# 4. input
# ======================================================================
def test_every_forwarded_key_round_trips(segments, shared_memory):
    bridge = segments.create(_unique("keys"), 8, 4)

    for name in FORWARDED_KEYS:
        bridge.send_input({name})
        assert bridge.held_keys() == {name}, name

    bridge.send_input(set(FORWARDED_KEYS))
    assert bridge.held_keys() == set(FORWARDED_KEYS)

    bridge.send_input([])
    assert bridge.held_keys() == set()


def test_a_key_the_bridge_does_not_carry_is_ignored(segments, shared_memory):
    bridge = segments.create(_unique("strangekey"), 8, 4)

    bridge.send_input({"left", "f13", "", "LEFT"})

    assert bridge.held_keys() == {"left"}, "unknown names are dropped, not raised on"


def test_the_pointer_round_trips(segments, shared_memory):
    bridge = segments.create(_unique("pointer"), 8, 4)

    assert bridge.pointer() == (0, 0, False)

    bridge.send_input([], mouse=(37, 412), mouse_down=True)
    assert bridge.pointer() == (37, 412, True)

    bridge.send_input([], mouse_down=False)
    assert bridge.pointer() == (37, 412, False), \
        "a report with no position leaves the last one alone"


# ======================================================================
# 5. two bridges, one segment
# ======================================================================
def test_a_second_bridge_sees_pixels_control_and_input_written_by_the_first(
        segments, shared_memory):
    name = _unique("shared")
    writer = segments.create(name, 12, 6)

    reader = segments.attach(name)

    assert (reader.width, reader.height) == (12, 6)
    assert reader.pixels().shape == writer.pixels().shape

    writer.pixels()[5] = 0.25
    assert reader.pixels()[5] == np.float32(0.25), "the frame is not a copy"

    writer.mark_running(True)
    writer.report_error("something to say")
    assert reader.running() is True
    assert reader.error() == "something to say"

    reader.send_input({"up", "space"}, mouse=(9, 3), mouse_down=True)
    reader.request_quit()
    assert writer.held_keys() == {"up", "space"}, "input travels the other way"
    assert writer.pointer() == (9, 3, True)
    assert writer.quit_requested() is True


# ======================================================================
# 6. publishing a surface
# ======================================================================
def test_publish_writes_a_surface_as_float_rgba(segments, pygame_module):
    width, height = 6, 4
    bridge = segments.create(_unique("publish"), width, height)
    surface = _surface(pygame_module, width, height, colour=(255, 0, 0))
    # A shape, so the assertions are about position and not only about colour.
    pygame_module.draw.rect(surface, (0, 0, 255), pygame_module.Rect(1, 2, 2, 1))

    bridge.publish(surface)

    frame = bridge.pixels()
    assert frame.min() >= 0.0 and frame.max() <= 1.0, "colours are scaled to 0..1"
    # tobytes gives row-major RGBA, so pixel (row, column) starts at
    # ((row * width) + column) * 4.
    assert _pixel(frame, 0, 0, width) == [1.0, 0.0, 0.0, 1.0]
    assert _pixel(frame, 2, 1, width) == [0.0, 0.0, 1.0, 1.0]
    assert _pixel(frame, 2, 2, width) == [0.0, 0.0, 1.0, 1.0]
    assert _pixel(frame, 2, 3, width) == [1.0, 0.0, 0.0, 1.0], "outside the shape"
    assert _pixel(frame, 3, 5, width) == [1.0, 0.0, 0.0, 1.0]

    reshaped = frame.reshape(height, width, 4)
    assert reshaped[2, 1].tolist() == [0.0, 0.0, 1.0, 1.0]


# ======================================================================
# 7. releasing the segments
# ======================================================================
def test_closing_with_unlink_releases_the_segments(shared_memory):
    name = _unique("release")
    bridge = FrameBridge(name, 8, 4, create=True)
    attached = FrameBridge(name)
    attached.close()

    bridge.close(unlink=True)

    with pytest.raises(FileNotFoundError):
        FrameBridge(name)


# ======================================================================
# 8. across a process boundary
# ======================================================================
def test_a_spawned_process_writes_the_frame_the_parent_is_holding(segments,
                                                                  pygame_module):
    width, height = 10, 5
    name = _unique("crossprocess")
    bridge = segments.create(name, width, height)

    child = segments.spawn(_attach_and_publish, (name, width, height))
    child.join(timeout=120)

    assert not child.is_alive(), "the child never finished"
    assert child.exitcode == 0
    assert bridge.generation() == 1, "the parent sees the child's frame count"
    assert bridge.frames() == 1
    assert bridge.running() is True
    assert bridge.error() == CHILD_MESSAGE
    expected = [round(channel / 255.0, 4) for channel in CHILD_COLOUR] + [1.0]
    assert _pixel(bridge.pixels(), 0, 0, width) == expected
    assert _pixel(bridge.pixels(), 4, 9, width) == expected


# ======================================================================
# 9. key state, indexed by whatever code a controller cares about
# ======================================================================
def test_key_state_answers_for_any_code_including_the_arrow_keys():
    # The regression: a list sized to the forwarded keys looks equivalent and
    # is not. A controller indexes what get_pressed returns with pygame's own
    # key codes, and an arrow key's code is in the millions, so the list raised
    # IndexError where the answer wanted was simply "not held".
    left, right = 1073741904, 1073741903
    state = _KeyState({left})

    assert state[left] is True
    assert state[right] is False
    assert state[0] is False
    assert state[1 << 30] is False, "a code far past any slot must answer, not raise"
    assert left in state
    assert len(state) > (1 << 20), "a caller checking bounds has to be satisfied"


def test_key_state_refuses_to_be_iterated():
    state = _KeyState({97})

    with pytest.raises(TypeError):
        list(state)


def test_forwarded_key_state_reports_what_the_bridge_was_told(segments,
                                                              pygame_module,
                                                              monkeypatch):
    bridge = segments.create(_unique("keystate"), 8, 4)
    # _forward_key_state replaces a module function, so monkeypatch is told to
    # put the real one back and nothing else in this session is affected.
    monkeypatch.setattr(pygame_module.key, "get_pressed",
                        pygame_module.key.get_pressed)

    _forward_key_state(pygame_module, bridge)

    bridge.send_input({"left", "space", "a"})
    pressed = pygame_module.key.get_pressed()

    # Both spellings have to be found: pygame names these K_LEFT and K_SPACE
    # but K_a, and getting that wrong silently drops the arrow keys, which are
    # the ones a Fitts task is actually driven with.
    assert pressed[pygame_module.K_LEFT] is True
    assert pressed[pygame_module.K_SPACE] is True
    assert pressed[pygame_module.K_a] is True
    assert pressed[pygame_module.K_RIGHT] is False
    assert pressed[pygame_module.K_F13] is False

    bridge.send_input([])
    assert pygame_module.key.get_pressed()[pygame_module.K_LEFT] is False


# ======================================================================
# 10. a real environment, running offscreen
# ======================================================================
def _iso_fitts(**overrides):
    """An ISO Fitts task small enough to run quickly, and its frame size."""
    spec = build_registry()["iso_fitts"]
    values = spec.defaults()
    values.update({"num_trials": 100, "width": 640, "height": 480, "fps": 60,
                   "target_distance_radius": 180, "save_file": None})
    values.update(overrides)
    width, height = spec.frame_size(values)
    return build_factory(spec, ControllerSpec(kind="keyboard"), values), width, height


def test_an_embedded_environment_runs_publishes_frames_and_stops(segments,
                                                                 pygame_module):
    factory, width, height = _iso_fitts()

    embedded = segments.embed(_unique("isofitts"), factory, width, height)

    assert embedded.pixels().shape == (width * height * 4,)
    assert _wait_until(lambda: embedded.status()["running"], timeout=90), \
        "the environment never reported itself running"
    assert _wait_until(lambda: embedded.status()["frames"] >= 150, timeout=90), \
        "the environment started but never drew"

    status = embedded.status()
    assert status["error"] == "", status["error"]
    assert status["finished"] is False
    assert status["generation"] >= 150
    # The smoothed rate takes a few dozen frames to settle, which waiting for
    # 150 of them has already allowed for.
    assert 5.0 < status["fps"] < 400.0, status["fps"]

    frame = embedded.pixels().reshape(height, width, 4)
    lit = int((frame[:, :, :3].sum(axis=2) > 0.15).sum())
    assert lit > 200, "the frame the GUI would draw is empty"

    before = embedded.status()["generation"]
    assert _wait_until(lambda: embedded.status()["generation"] > before + 20,
                       timeout=30), "frames stopped advancing"

    embedded.stop(timeout=15)

    assert not embedded.alive, "the environment process outlived its stop()"
    assert embedded.runner.exitcode is not None


def test_forwarded_keys_change_what_the_environment_draws(segments, pygame_module):
    factory, width, height = _iso_fitts()
    embedded = segments.embed(_unique("isokeys"), factory, width, height)
    assert _wait_until(lambda: embedded.status()["frames"] >= 30, timeout=90), \
        "the environment never started drawing"
    frame = embedded.pixels().reshape(height, width, 4)

    embedded.send_input({"right"})
    moved_right = embedded.status()["generation"] + 60
    assert _wait_until(lambda: embedded.status()["generation"] > moved_right,
                       timeout=30), "frames stopped advancing"
    moving_right = np.array(frame, copy=True)

    embedded.send_input({"left"})
    moved_left = embedded.status()["generation"] + 60
    assert _wait_until(lambda: embedded.status()["generation"] > moved_left,
                       timeout=30), "frames stopped advancing"
    moving_left = np.array(frame, copy=True)

    assert not np.allclose(moving_right, moving_left), \
        "driving the cursor with forwarded keys did not change the frame"


# ======================================================================
# 11. an environment that refuses its settings
# ======================================================================
def _impossible_iso_fitts():
    """An ISO Fitts whose ring does not fit on its screen, which it rejects."""
    spec = build_registry()["iso_fitts"]
    values = spec.defaults()
    values.update({"num_trials": 5, "width": 640, "height": 480,
                   "target_distance_radius": 400})
    width, height = spec.frame_size(values)
    return build_factory(spec, ControllerSpec(kind="keyboard"), values), width, height


def test_an_environment_that_refuses_its_settings_stops_instead_of_hanging(
        segments, pygame_module):
    factory, width, height = _impossible_iso_fitts()

    embedded = segments.embed(_unique("badiso"), factory, width, height)

    assert _wait_until(lambda: not embedded.alive, timeout=90), \
        "an environment that cannot be built has to stop, not sit there"
    assert embedded.status()["running"] is False
    assert embedded.status()["frames"] == 0


def test_an_environment_that_refuses_its_settings_says_why(segments, pygame_module):
    # The failure happens in a process with no window and no terminal anybody
    # is watching, so the reason reaching status()['error'] is the difference
    # between a panel that stays black and a sentence saying what to change.
    factory, width, height = _impossible_iso_fitts()

    embedded = segments.embed(_unique("badisowhy"), factory, width, height)
    assert _wait_until(lambda: not embedded.alive, timeout=90)

    reported = embedded.status()["error"]

    assert reported, "the environment failed silently"
    assert reported.startswith("ValueError"), reported
    assert "Radius between ISO Fitts targets" in reported, reported
    assert "reduce target distance radius" in reported, \
        "the message has to say what the user should change"
