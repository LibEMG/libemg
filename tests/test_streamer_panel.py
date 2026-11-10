"""Coverage for the panel that starts a device from inside the GUI.

The panel's claim is that nothing about a device is written down twice. The
device list is the streamers module filtered to the ones that write shared
memory, and each device's controls are its own signature. A test that listed
today's devices by hand would pass just as happily against a hardcoded copy,
which is the failure being guarded against, so the expectations here are
derived live from ``_streamer_functions`` and ``_params_from_signature`` and a
test fails when the panel and the library drift apart.

The other half is what the panel reports. A device streamer spawns a process
that connects on its own, so a device that is unplugged starts perfectly well
and simply never delivers a sample. The panel is therefore asserted to say it
is waiting until samples actually arrive, and to say so out loud once five
seconds have passed with nothing.

Only the synthetic device is ever started here. The real streamers spawn
processes that look for hardware that is not on this machine, so the "no
samples" path is exercised by driving the panel's own state instead of by
waiting on a device that will never answer.
"""
import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui")

from libemg._gui._pipeline.registry import (_params_from_signature,
                                            _streamer_functions)
from libemg._gui._streamer_panel import (MESSAGE_TAG, STATUS_TAG, SYNTHETIC,
                                         WINDOW_TAG, StreamerPanel)

#: Every wait in this module has one. A test that hangs is worse than a test
#: that fails, and the synthetic device answers in well under a second.
TIMEOUT = 20.0


# ======================================================================
# helpers
# ======================================================================
class _FakeState:
    """A state block reading, built by hand rather than by a writer."""

    def __init__(self, total_samples=0, commits=0):
        self.total_samples = total_samples
        self.commits = commits


class _SilentHandler:
    """Stands in for an OnlineDataHandler attached to a device that never talks."""

    def __init__(self, modalities=("emg",)):
        self.modalities = list(modalities)

    def get_state(self, modality=None):
        return {name: _FakeState() for name in self.modalities}


def _pump(panel, frames=2):
    """Give the panel the per-frame slice the GUI's render loop would."""
    for _ in range(frames):
        panel.poll()
        dpg.render_dearpygui_frame()


def _wait_until(panel, predicate, timeout=TIMEOUT, interval=0.02):
    """Poll to a deadline rather than sleeping a fixed time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _pump(panel, 1)
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _total(panel, modality="emg"):
    return panel.odh.get_state(modality).total_samples


# ======================================================================
# fixtures
# ======================================================================
@pytest.fixture(scope="module", autouse=True)
def dpg_context():
    """One context for the module, and no window left behind afterwards."""
    try:
        dpg.create_context()
        dpg.create_viewport(title="streamer tests", width=820, height=680)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    except Exception as error:      # no display, no GPU, no dearpygui backend
        pytest.skip(f"DearPyGui cannot create a context here: {error}")
    yield
    dpg.destroy_context()


@pytest.fixture
def panel():
    """A panel with its window up, stopped and torn down even on failure."""
    published = {}
    made = StreamerPanel(
        on_started=lambda odh, items: published.update(odh=odh, items=items),
        on_stopped=published.clear)
    made.published = published
    made.spawn_window()
    _pump(made)
    try:
        yield made
    finally:
        made.cleanup()
        _pump(made, 1)


@pytest.fixture
def running(panel):
    """The synthetic device, started and delivering, at 1000 Hz on 6 channels."""
    panel._choose(None, SYNTHETIC)
    panel._changed(None, 1000, "sampling_rate")
    panel._changed(None, 6, "num_channels")
    panel.start()
    assert panel.handle is not None, dpg.get_value(MESSAGE_TAG)
    assert _wait_until(panel, lambda: _total(panel) > 0), \
        "the synthetic device delivered no samples"
    return panel


# ======================================================================
# 1. the device list is generated
# ======================================================================
def test_the_device_list_is_the_streamers_module_plus_the_synthetic_one(panel):
    # Derived live, so replacing the generation with a hardcoded copy of
    # today's devices fails the moment a streamer is added or removed.
    assert panel.devices == [SYNTHETIC] + sorted(_streamer_functions())
    assert panel.devices[0] == SYNTHETIC
    assert dpg.get_value("__streamer_choice") == SYNTHETIC


def test_a_streamer_with_no_shared_memory_is_not_offered(panel):
    # mock_emg_stream sends over a socket and returns nothing the GUI could
    # attach a handler to, so it is not a device this panel can start.
    assert "mock_emg_stream" not in panel.devices
    assert "mock_emg_stream" not in panel.functions
    for name in panel.functions:
        assert "shared_memory_items" in _signature_names(panel.functions[name])


def _signature_names(function):
    import inspect
    return inspect.signature(function).parameters


# ======================================================================
# 2. a device's own arguments become its controls
# ======================================================================
@pytest.mark.parametrize("device", ["delsys_streamer", "sifi_biopoint_streamer"])
def test_selecting_a_device_draws_one_control_per_parameter(panel, device):
    if device not in panel.functions:
        pytest.skip(f"{device} is not in this build of libemg.streamers")

    panel._choose(None, device)
    _pump(panel)

    expected = [spec.name for spec in _params_from_signature(panel.functions[device])]
    assert expected, f"{device} declares no parameters to draw"
    missing = [name for name in expected
               if not dpg.does_item_exist(f"__streamer_set_{name}")]
    assert missing == [], f"{device} drew no control for {missing}"
    assert panel.selected == device


def test_the_synthetic_device_draws_its_four_controls(panel):
    panel._choose(None, SYNTHETIC)
    _pump(panel)

    for name in ("sampling_rate", "num_channels", "pattern", "amplitude"):
        assert dpg.does_item_exist(f"__streamer_set_{name}"), f"no control for {name}"
    assert dpg.get_value("__streamer_set_pattern") == "bursts"
    assert dpg.get_value("__streamer_set_num_channels") == 8


# ======================================================================
# 3. editing, and switching away
# ======================================================================
def test_editing_a_control_records_the_value(panel):
    panel._choose(None, SYNTHETIC)
    panel._changed(None, 2000, "sampling_rate")
    assert panel.values["sampling_rate"] == 2000


def test_switching_device_clears_the_old_values_and_rebuilds(panel):
    panel._choose(None, SYNTHETIC)
    panel._changed(None, 2000, "sampling_rate")

    panel._choose(None, "delsys_streamer")
    _pump(panel)

    # delsys_streamer has no sampling rate of its own, so a value carried over
    # from the previous device would be passed to a function that never asked
    # for it.
    assert "sampling_rate" not in panel.values
    assert set(panel.values) == {spec.name for spec in
                                 _params_from_signature(panel.functions["delsys_streamer"])}
    assert not dpg.does_item_exist("__streamer_set_sampling_rate")


# ======================================================================
# 4. starting publishes a handler the rest of the window can use
# ======================================================================
def test_starting_publishes_a_handler_and_its_shared_memory(running):
    assert "odh" in running.published, "nothing was handed to the GUI"
    assert running.published["odh"] is running.odh
    assert running.published["items"] is running.items
    assert running._modalities == ["emg"]


def test_the_published_handler_delivers_the_requested_channels(running):
    data, _ = running.published["odh"].get_data(N=200)
    assert "emg" in data
    assert data["emg"].shape[1] == 6, "the channel count asked for is not what arrived"


def test_the_status_table_reports_a_rate_near_the_one_requested(running):
    assert _wait_until(running,
                       lambda: float(dpg.get_value("__streamer_r_emg") or 0) > 0)
    rate = float(dpg.get_value("__streamer_r_emg"))
    assert 700 < rate < 1300, f"reported {rate:.0f} Hz for a 1000 Hz device"
    assert int(dpg.get_value("__streamer_n_emg")) > 0
    assert int(dpg.get_value("__streamer_w_emg")) > 0
    assert "samples per second" in dpg.get_value(STATUS_TAG)


# ======================================================================
# 5. starting resets the counters
# ======================================================================
def test_starting_again_does_not_inherit_the_previous_runs_counts(running):
    # Shared-memory segments outlive the process that made them and every
    # device writes to the same modality names, so without a reset the second
    # device attaches to whatever the first one left behind. A device that is
    # not connected would then read as live data.
    assert _wait_until(running, lambda: _total(running) > 1000)
    first = _total(running)
    running.stop()

    running.start()
    assert running.handle is not None, dpg.get_value(MESSAGE_TAG)
    second = _total(running)

    assert first > 1000
    assert second < 500, (f"the second run started at {second} samples, "
                          f"carried over from the first run's {first}")
    assert _wait_until(running, lambda: _total(running) > second), \
        "the restarted device delivered nothing"


# ======================================================================
# 6. switching device while running is refused
# ======================================================================
def test_switching_device_while_running_is_refused_and_the_combo_is_put_back(running):
    running._choose(None, "delsys_streamer")
    _pump(running)

    assert running.selected == SYNTHETIC
    assert dpg.get_value("__streamer_choice") == SYNTHETIC
    assert "Stop the running device" in dpg.get_value(MESSAGE_TAG)
    assert running.handle is not None


def test_starting_twice_is_refused(running):
    handle = running.handle
    running.start()
    assert running.handle is handle
    assert "Already streaming" in dpg.get_value(MESSAGE_TAG)


# ======================================================================
# 7. stopping withdraws the handler
# ======================================================================
def test_stopping_withdraws_the_handler_and_clears_the_modalities(running):
    running.stop()

    assert running.published == {}, "the GUI was left holding a dead handler"
    assert running.odh is None
    assert running.handle is None
    assert running._modalities == []
    assert dpg.get_value(MESSAGE_TAG) == "Stopped."
    assert dpg.get_value(STATUS_TAG) == "Incoming data    nothing streaming"


def test_the_panel_can_start_again_after_it_has_been_stopped(running):
    running.stop()

    running.start()
    assert running.handle is not None, dpg.get_value(MESSAGE_TAG)
    assert "odh" in running.published
    assert _wait_until(running, lambda: _total(running) > 0)


def test_stopping_when_nothing_runs_does_nothing(panel):
    panel.stop()
    assert panel.handle is None


# ======================================================================
# 8. cleanup
# ======================================================================
def test_cleanup_stops_the_device_and_removes_the_window(running):
    running.cleanup()

    assert running.handle is None
    assert running.odh is None
    assert running.published == {}
    assert not dpg.does_alias_exist(WINDOW_TAG), "the streamer window was left open"


# ======================================================================
# 9. a device that never delivers is reported, not hidden
# ======================================================================
def test_right_after_start_the_panel_says_it_is_waiting(panel):
    panel._choose(None, SYNTHETIC)
    panel.start()

    # Said before a single frame has been rendered, because a streamer process
    # starts whether or not the device it is looking for is there.
    message = dpg.get_value(MESSAGE_TAG)
    assert "Waiting for the first samples" in message
    assert panel._seen_data is False


def test_the_message_turns_to_streaming_only_once_samples_arrive(panel):
    panel._choose(None, SYNTHETIC)
    panel._changed(None, 500, "sampling_rate")
    panel.start()
    assert "is streaming" not in dpg.get_value(MESSAGE_TAG)

    assert _wait_until(panel, lambda: panel._seen_data), \
        "the synthetic device never reported samples"
    assert "is streaming" in dpg.get_value(MESSAGE_TAG)


def test_a_device_that_delivers_nothing_is_reported_after_five_seconds(panel):
    # Driven through the panel's own state rather than by starting a real
    # device and waiting five seconds for hardware that is not on this machine.
    panel.odh = _SilentHandler()
    panel._modalities = ["emg"]
    panel._rebuild_table()
    panel._started_at = time.perf_counter() - 6.0
    panel._previous_at = time.perf_counter() - 1.0
    panel._seen_data = False
    panel._warned = False

    panel.poll()

    message = dpg.get_value(MESSAGE_TAG)
    assert "no samples have arrived in five seconds" in message
    assert "left running in case it is still connecting" in message
    assert panel._warned is True
    # Said once, not on every frame from here on.
    panel._say("something else")
    panel._previous_at = time.perf_counter() - 1.0
    panel.poll()
    assert dpg.get_value(MESSAGE_TAG) == "something else"
    panel.odh = None


def test_a_device_that_will_not_start_reports_the_reason(panel):
    panel._choose(None, SYNTHETIC)
    # A channel count that is not a number fails inside the launch, so this is
    # a launch failure with no hardware involved.
    panel.values["num_channels"] = "not a number"
    panel.start()

    assert panel.handle is None
    assert "Could not start that device" in dpg.get_value(MESSAGE_TAG)


# ======================================================================
# 6. the menu does not orphan what is already running
# ======================================================================
def test_opening_the_panel_twice_reuses_the_one_already_there(dpg_context):
    """A second menu click must not leave the first panel invisible.

    Building a second panel runs its setup, which deletes the window tag the
    first one owns. The first panel would stay in the GUI's poll list, writing
    into widgets the second one now owns, and would still hold its streaming
    device with no window able to stop it.
    """
    from libemg.gui import GUI

    gui = GUI(online_data_handler=None, args={}, width=900, height=700)
    gui._streamer_callback()
    first = gui.sp
    _pump(first)
    first._choose(None, SYNTHETIC)
    first.start()
    assert first.handle is not None

    try:
        gui._streamer_callback()
        assert gui.sp is first
        assert gui.panels.count(first) == 1
        assert first.handle is not None
        assert dpg.does_alias_exist(first.window_tag)
    finally:
        first.cleanup()
        _pump(first, 1)


def test_a_closed_panel_is_replaced_rather_than_revived(dpg_context):
    from libemg.gui import GUI

    gui = GUI(online_data_handler=None, args={}, width=900, height=700)
    gui._streamer_callback()
    first = gui.sp
    _pump(first)
    first.cleanup()
    _pump(first, 1)

    gui._streamer_callback()
    try:
        assert gui.sp is not first
        assert first not in gui.panels
        assert gui.panels.count(gui.sp) == 1
    finally:
        gui.sp.cleanup()


def test_stopping_releases_the_shared_memory_handles(running):
    """Otherwise a start-stop cycle accumulates OS handles."""
    handler = running.published["odh"]
    assert handler.smm.variables, "the handler had no segments open"
    running.stop()
    _pump(running, 1)
    assert not handler.smm.variables
