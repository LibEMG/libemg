"""Coverage for the native dearpygui live-signal viewer."""
import time

import dearpygui.dearpygui as dpg
import numpy as np
import pytest

from libemg._gui._visualization_panel import VisualizationPanel


class _FakeHandler:
    """Stands in for an OnlineDataHandler holding newest-first buffers."""

    def __init__(self, shapes=(("emg", (3000, 8)), ("ecg", (1000, 1)))):
        self.modalities = [name for name, _ in shapes]
        # Column 0 counts down from the buffer length so orientation is checkable:
        # index 0 is the newest sample.
        self.buffers = {}
        for name, (rows, cols) in shapes:
            buf = np.zeros((rows, cols))
            buf[:, 0] = np.arange(rows, 0, -1)
            self.buffers[name] = buf
        self.reads = 0

    def get_data(self, N=0, filter=True):
        self.reads += 1
        vals = {m: b.copy() for m, b in self.buffers.items()}
        counts = {m: np.array([[b.shape[0]]], dtype=np.int32) for m, b in self.buffers.items()}
        return vals, counts


@pytest.fixture
def dpg_context():
    dpg.create_context()
    dpg.create_viewport(title="test", width=900, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    yield
    dpg.destroy_context()


def test_window_offers_a_control_row_per_detected_modality(dpg_context):
    panel = VisualizationPanel(_FakeHandler(), num_samples=400)
    panel.spawn_window()

    assert panel.modalities == ["emg", "ecg"]
    assert panel.channels == {"emg": 8, "ecg": 1}
    for mod in ("emg", "ecg"):
        assert dpg.does_item_exist(panel._enable_tag(mod)), f"no checkbox for {mod}"
        assert dpg.does_item_exist(panel._samples_tag(mod)), f"no sample control for {mod}"
        assert dpg.does_item_exist(panel._plot_tag(mod)), f"no plot for {mod}"
        assert dpg.get_value(panel._enable_tag(mod)) is True
    assert dpg.does_item_exist("__vls_start_button")
    assert dpg.does_item_exist("__vls_stop_button")
    # One series per channel, so every channel is separately legible.
    assert all(dpg.does_item_exist(panel._series_tag("emg", ch)) for ch in range(8))


def test_each_modality_keeps_its_own_sample_count(dpg_context):
    panel = VisualizationPanel(_FakeHandler(), num_samples=400)
    panel.spawn_window()

    dpg.set_value(panel._samples_tag("emg"), 250)
    dpg.set_value(panel._samples_tag("ecg"), 60)
    panel._update_plots()

    assert len(dpg.get_value(panel._series_tag("emg", 0))[0]) == 250
    assert len(dpg.get_value(panel._series_tag("ecg", 0))[0]) == 60


def test_samples_are_plotted_oldest_to_newest(dpg_context):
    """Shared memory is newest-first; plotted straight through, the trace runs
    backwards and a live signal appears to scroll the wrong way."""
    panel = VisualizationPanel(_FakeHandler(shapes=(("emg", (100, 1)),)), num_samples=10)
    panel.spawn_window()
    dpg.set_value(panel._samples_tag("emg"), 5)

    panel._update_plots()

    y = dpg.get_value(panel._series_tag("emg", 0))[1][:5]
    # Buffer column counts down from 100, so the newest sample is 100.
    assert y == [96.0, 97.0, 98.0, 99.0, 100.0]


def test_sample_count_is_clamped_to_the_buffer(dpg_context):
    panel = VisualizationPanel(_FakeHandler(shapes=(("ecg", (200, 1)),)), num_samples=50)
    panel.spawn_window()

    assert dpg.get_item_configuration(panel._samples_tag("ecg"))["max_value"] == 200
    dpg.set_value(panel._samples_tag("ecg"), 100000)
    panel._update_plots()

    assert len(dpg.get_value(panel._series_tag("ecg", 0))[0]) == 200


def test_unchecked_modalities_are_hidden_and_not_updated(dpg_context):
    panel = VisualizationPanel(_FakeHandler(), num_samples=100)
    panel.spawn_window()
    panel._update_plots()
    before = dpg.get_value(panel._series_tag("ecg", 0))[0]

    dpg.set_value(panel._enable_tag("ecg"), False)
    panel._toggle_modality_callback(None, False, "ecg")
    dpg.set_value(panel._samples_tag("ecg"), 25)
    panel._update_plots()

    assert not dpg.is_item_shown(panel._plot_tag("ecg"))
    assert dpg.get_value(panel._series_tag("ecg", 0))[0] == before, "hidden plot still updated"
    assert dpg.is_item_shown(panel._plot_tag("emg"))


def test_start_and_stop_control_the_updater(dpg_context):
    handler = _FakeHandler(shapes=(("emg", (500, 2)),))
    panel = VisualizationPanel(handler, num_samples=100, refresh_rate=200)
    panel.spawn_window()

    assert dpg.get_value("__vls_status") == "Stopped"
    panel.start_callback()
    assert dpg.get_value("__vls_status") == "Running"
    time.sleep(0.2)
    panel.stop_callback()

    reads_at_stop = handler.reads
    assert reads_at_stop > 1, "the updater never ran"
    assert dpg.get_value("__vls_status") == "Stopped"
    time.sleep(0.15)
    assert handler.reads == reads_at_stop, "the updater kept running after Stop"
    assert panel.get_frame_stats()["frames"] > 0


def test_start_is_idempotent(dpg_context):
    panel = VisualizationPanel(_FakeHandler(shapes=(("emg", (500, 2)),)), refresh_rate=100)
    panel.spawn_window()

    panel.start_callback()
    first = panel._thread
    panel.start_callback()

    assert panel._thread is first, "a second Start spawned a competing updater"
    panel.stop_callback()


def test_cleanup_stops_the_updater_before_deleting_the_plots(dpg_context):
    panel = VisualizationPanel(_FakeHandler(shapes=(("emg", (500, 2)),)), refresh_rate=200)
    panel.spawn_window()
    panel.start_callback()
    time.sleep(0.05)

    panel.cleanup_window()

    assert panel._thread is None
    assert not dpg.does_alias_exist("__vls_visualize_window")


def test_visualizing_without_data_says_so(dpg_context):
    class _Empty:
        modalities = []

        def get_data(self, N=0, filter=True):
            return {}, {}

    panel = VisualizationPanel(_Empty())
    with pytest.raises(ConnectionError, match="no modalities"):
        panel.spawn_window()
