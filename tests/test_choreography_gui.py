"""The choreography mode inside Collect Data: mode selection, the run window, and
a recording made through the same session pipeline as the image-folder mode."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from libemg._gui import _choreography_panel as choreo_module
from libemg._gui import _data_collection_panel as panel_module
from libemg._gui import _style
from libemg._gui._choreography_panel import (CHOREOGRAPHY_MODE, IMAGE_FOLDER_MODE,
                                             ChoreographyConfigSection, ChoreographyRunWindow,
                                             PanelCollectionLink)
from libemg._gui._data_collection_panel import DataCollectionPanel
from libemg.choreography import ChoreographyConfig, load_assets
from libemg.choreography import annotations as notes
from libemg.choreography.controller import PREVIEW, RECORDING
from libemg.choreography.schedule import COMPLETED, FAILED, PAUSED_STATE, RUNNING
from libemg.session import PROMPT, SessionAnnotations

from tests.test_data_collection_panel import _StreamingHandler

ASSETS = load_assets()


class _Handler(_StreamingHandler):
    """The streaming test double, plus what the GUI calls before recording."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prepared = 0

    def prepare_smm(self):
        self.prepared += 1


@pytest.fixture
def dpg(monkeypatch):
    fake = MagicMock()
    values = {}
    fake.does_item_exist.side_effect = lambda tag: tag in values
    fake.get_value.side_effect = lambda tag: values.get(tag)
    fake.set_value.side_effect = lambda tag, value: values.__setitem__(tag, value)
    fake.delete_item.side_effect = lambda tag, **_: values.pop(tag, None)
    fake.values = values

    # Items exist once made, holding their default value, as in DearPyGui.
    def maker(name):
        def make(*args, **kwargs):
            tag = kwargs.get("tag")
            if tag is not None:
                default = kwargs.get("default_value", args[0] if args and name == "add_text"
                                     else None)
                values.setdefault(tag, default)
            return MagicMock()
        return make
    for name in ("add_text", "add_progress_bar", "add_image", "add_button", "add_combo",
                 "add_input_text", "add_input_int", "add_input_float", "add_checkbox",
                 "add_radio_button", "add_raw_texture", "window", "table", "group",
                 "child_window", "file_dialog"):
        setattr(fake, name, MagicMock(side_effect=maker(name)))
    for module in (panel_module, choreo_module, _style):
        monkeypatch.setattr(module, "dpg", fake)
    monkeypatch.setattr(choreo_module, "get_texture_registry", Mock(return_value="registry"))
    monkeypatch.setattr(panel_module, "set_texture", Mock())
    return fake


def _config(**kwargs):
    kwargs.setdefault("gestures", ["hand_closed", "wrist_flexion"])
    kwargs.setdefault("warmup", 0.15)
    kwargs.setdefault("ramp_in", 0.1)
    kwargs.setdefault("hold", 0.2)
    kwargs.setdefault("ramp_out", 0.1)
    kwargs.setdefault("rest", 0.1)
    kwargs.setdefault("repetitions", 2)
    kwargs.setdefault("shuffle_seed", 3)
    return ChoreographyConfig(**kwargs)


class _StubRenderer:
    """Stands in for the 3D renderer: right size, no drawing."""

    def __init__(self, fail_after=None):
        self.width = self.height = 0
        self.calls = 0
        self.fail_after = fail_after

    def resize(self, width, height):
        self.width, self.height = width, height

    def render(self, clip, time, out=None):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("device lost")
        if out is None:
            out = np.zeros(self.width * self.height * 4, np.float32)
        return out

    def close(self):
        pass


def _drive(window, until, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        window.poll()
        if until():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


# --- the mode -------------------------------------------------------------------------

def test_the_choreography_is_the_default_mode():
    dcp = DataCollectionPanel(_Handler())
    assert dcp.collection_mode == CHOREOGRAPHY_MODE


def test_a_protocol_asks_for_the_image_folder_backup(tmp_path):
    from libemg.protocol import CollectionProtocol
    dcp = DataCollectionPanel(_Handler(), protocol=CollectionProtocol(name="study"))
    assert dcp.collection_mode == IMAGE_FOLDER_MODE
    with pytest.raises(ValueError):
        DataCollectionPanel(_Handler(), collection_mode="video")


def test_the_mode_cannot_change_while_a_collection_is_running(dpg):
    dcp = DataCollectionPanel(_Handler())
    dcp.run_window = Mock(active=True)
    assert dcp.set_collection_mode(IMAGE_FOLDER_MODE) is False
    assert dcp.collection_mode == CHOREOGRAPHY_MODE
    dcp.run_window = Mock(active=False)
    assert dcp.set_collection_mode(IMAGE_FOLDER_MODE) is True
    assert dcp.collection_mode == IMAGE_FOLDER_MODE
    # A running image-folder collection blocks it the same way.
    dcp.run_window = None
    dcp.spawn_collection_thread = Mock(is_alive=Mock(return_value=True))
    assert dcp.set_collection_mode(CHOREOGRAPHY_MODE) is False


def test_the_configuration_window_builds_in_both_modes(tmp_path, dpg):
    for mode in (CHOREOGRAPHY_MODE, IMAGE_FOLDER_MODE):
        dcp = DataCollectionPanel(_Handler(), collection_mode=mode, data_folder=str(tmp_path))
        dcp.spawn_configuration_window()
        assert (dcp.choreography is not None) == True     # built either way, shown per mode


def test_the_image_folder_session_says_which_mode_it_was(tmp_path, dpg):
    handler = _Handler()
    dcp = DataCollectionPanel(handler, num_reps=1, rest_time=0, collection_mode=IMAGE_FOLDER_MODE)
    dcp.output_folder = str(tmp_path)
    dcp.num_motions = 1
    dpg.run_callbacks.side_effect = lambda *_args: setattr(dcp, "advance", True)
    media = Mock()
    media.fps = 1000
    dcp.run_sgt([[media, "class_0", 0, 0, 0.05]])
    session = SessionAnnotations.load(tmp_path)
    mode = session.metadata["collection_mode"]
    assert mode["mode"] == IMAGE_FOLDER_MODE and mode["config"]["protocol"] is None
    assert session.metadata["collection_session_id"]
    # Its stages are exactly what that mode always wrote: no choreography fields.
    raw = json.loads((tmp_path / "_session_annotations.json").read_text())
    assert all("phase" not in stage and "status" not in stage for stage in raw["stages"])


# --- the settings section ---------------------------------------------------------------

def test_the_settings_read_overrides_and_report_text_that_is_not_a_number(dpg):
    section = ChoreographyConfigSection(ASSETS, config=_config())
    section.build()
    section.rep_text["hand_closed"] = "4"
    section.timing_text["wrist_flexion"]["hold"] = "2.5"
    section.timing_text["hand_closed"]["rest"] = "soon"
    config, problems = section.read_config()
    assert config.gestures == ["hand_closed", "wrist_flexion"]
    assert config.repetition_overrides == {"hand_closed": 4}
    assert config.timing_overrides == {"wrist_flexion": {"hold": 2.5}}
    assert problems == ["Hand Closed: rest 'soon' is not a number."]


def test_select_all_clear_all_and_reordering(dpg):
    section = ChoreographyConfigSection(ASSETS, config=_config())
    section.build()
    section._clear_all()
    assert section.read_config()[0].gestures == []
    _, plan, problems = section.validate()
    assert plan is None and "Select at least one gesture." in problems
    section._select_all()
    assert len(section.read_config()[0].gestures) == 16
    first, second = section.order[:2]
    section._move(None, None, (second, -1))
    assert section.order[:2] == [second, first]


def test_start_and_preview_are_refused_without_gestures(tmp_path, dpg):
    dcp = DataCollectionPanel(_Handler(), data_folder=str(tmp_path),
                              choreography_config=_config(gestures=[]))
    dcp.spawn_configuration_window()
    assert dcp.open_choreography(recording=False) is None
    assert dcp.run_window is None


def test_presets_load_into_the_section(tmp_path, dpg):
    path = _config(gestures=["thumb_flex"], hold=4.0).save(tmp_path / "p.json")
    section = ChoreographyConfigSection(ASSETS, config=_config())
    section.build()
    section._preset_chosen(None, {"file_path_name": str(path)}, False)
    config, _ = section.read_config()
    assert config.gestures == ["thumb_flex"] and config.hold == 4.0


# --- the run window ---------------------------------------------------------------------

def _window(dpg, mode=PREVIEW, renderer=None, **kwargs):
    renderer = renderer or _StubRenderer()
    window = ChoreographyRunWindow(ASSETS, _config(**kwargs), mode=mode,
                                   renderer_factory=lambda: renderer)
    window.spawn()
    _drive(window, lambda: window._renderer_ready)
    return window, renderer


def test_a_preview_runs_to_completion_and_shows_each_phase(dpg):
    window, renderer = _window(dpg)
    phases = set()
    window._start()
    assert window.controller.state == RUNNING

    def watch():
        phases.add(dpg.values.get("__ch_phase"))
        return window.controller.result is not None and window._finalizer is None
    _drive(window, watch)
    assert {"Current phase: Warm-up", "Current phase: Ramping in", "Current phase: Hold",
            "Current phase: Ramping out", "Current phase: Rest"} <= phases
    assert renderer.calls > 5
    assert window.controller.result.record_path is None, "a preview writes nothing"
    window.destroy()
    assert not window.worker.alive


def test_pause_and_resume_from_the_window(dpg):
    window, _ = _window(dpg, hold=1.0)
    window._start()
    _drive(window, lambda: window.controller.view().phase == "hold")
    window._pause()
    assert window.controller.state == PAUSED_STATE
    window.poll()
    assert dpg.values["__ch_phase"] == "Current phase: Paused"
    frozen = window.controller.pose()
    time.sleep(0.05)
    window.poll()
    assert window.controller.pose() == frozen
    window._resume()
    assert window.controller.state == RUNNING
    window._stop()
    _drive(window, lambda: window.controller.result is not None)
    window.destroy()


def test_reset_returns_the_hand_to_rest_and_allows_a_new_run(dpg):
    window, _ = _window(dpg, hold=2.0)
    window._start()
    _drive(window, lambda: window.controller.view().phase == "ramp_in")
    first = window.controller.run_id
    window._reset()
    _drive(window, lambda: window.controller.state == "ready")
    assert window.controller.run_id != first and window.controller.pose() == ("rest", 0.0)
    window.destroy()


def test_a_render_failure_fails_the_run_and_says_so(dpg):
    window, _ = _window(dpg, renderer=_StubRenderer(fail_after=5), hold=2.0)
    window._start()
    _drive(window, lambda: window.controller.result is not None)
    assert window.controller.result.status == FAILED
    assert window.controller.result.reason.startswith("render_failure")
    window.destroy()


# --- a recording through the shared pipeline -----------------------------------------

def test_a_choreography_recording_uses_the_existing_session_pipeline(tmp_path, dpg):
    handler = _Handler(fs=500.0, batch=5)
    dcp = DataCollectionPanel(handler, data_folder=str(tmp_path), choreography_config=_config(),
                              signal_check="off")
    dcp.output_folder = str(tmp_path)
    for name in ("subject_number", "subject_name", "session_number", "day_number",
                 "condition_number"):
        dpg.values[f"__dc_{name}"] = "1"
    dpg.values.update({"__dc_output_folder": str(tmp_path), "__dc_num_reps": 1,
                       "__dc_rep_time": 1.0, "__dc_rest_time": 0.0, "__dc_media_folder": "",
                       "__dc_auto_advance": True})
    dcp.spawn_configuration_window()
    window = dcp.open_choreography(recording=True)
    window.worker.close()
    stub = _StubRenderer()
    from libemg.choreography.renderer import RenderWorker
    window.worker = RenderWorker(lambda: stub, 64, 64)
    _drive(window, lambda: window._renderer_ready)
    assert dcp.collection_active is False, "opening the window records nothing yet"
    assert not (tmp_path / "_session_annotations.json").exists()

    window._start()
    _drive(window, lambda: window.controller.state == RUNNING)
    assert dcp.collection_active and handler.prepared == 1
    assert dcp.set_collection_mode(IMAGE_FOLDER_MODE) is False
    _drive(window, lambda: window.controller.result is not None, timeout=30)
    result = window.controller.result
    assert result.status == COMPLETED, result

    session = SessionAnnotations.load(tmp_path)
    assert session.metadata["collection_mode"]["mode"] == CHOREOGRAPHY_MODE
    assert session.metadata["choreography_run"]["run_id"] == result.run_id
    assert session.class_map == {"1": "hand_closed", "5": "wrist_flexion"}
    holds = session.select(kinds=(PROMPT,))
    assert len(holds) == 4 and all(stage.phase == "hold" for stage in holds)
    for cls, rep in ((1, 0), (1, 1), (5, 0), (5, 1)):
        for mod in handler.modalities:
            cut = tmp_path / f"C_{cls}_R_{rep}_{mod}.csv"
            assert cut.exists(), cut
            # A 0.2 s hold at 500 Hz.
            assert abs(len(np.loadtxt(cut, delimiter=",", ndmin=2)) - 100) <= 2

    record = json.loads((tmp_path / notes.RUN_RECORD_NAME).read_text())
    assert record["collection_session_id"] == session.metadata["collection_session_id"]
    phases = [iv for iv in record["intervals"] if iv["track"] == "phase"]
    assert all(iv["start_sample_index"] is not None for iv in phases)
    for before, after in zip(phases, phases[1:]):
        assert after["start_sample_index"] == before["end_sample_index"]
    hold = next(iv for iv in phases if iv["phase"] == "hold")
    assert hold["end_sample_index"] - hold["start_sample_index"] in range(98, 103)
    details = json.loads((tmp_path / "collection_details.json").read_text())
    assert details["collection_mode"] == CHOREOGRAPHY_MODE
    assert details["class_map"] == {"1": "hand_closed", "5": "wrist_flexion"}
    window.destroy()
    assert dcp.run_window is None


def test_shutting_the_application_down_saves_a_running_recording(tmp_path, dpg):
    handler = _Handler(fs=500.0, batch=5)
    dcp = DataCollectionPanel(handler, data_folder=str(tmp_path),
                              choreography_config=_config(hold=5.0), signal_check="off")
    dcp.output_folder = str(tmp_path)
    plan_config = _config(hold=5.0)
    from libemg.choreography.schedule import build_plan
    link = PanelCollectionLink(dcp, build_plan(plan_config, ASSETS.gesture_ids), ASSETS)
    dcp.run_window = ChoreographyRunWindow(ASSETS, plan_config, mode=RECORDING, link=link,
                                           output_folder=str(tmp_path),
                                           renderer_factory=lambda: _StubRenderer())
    dcp.run_window.spawn()
    _drive(dcp.run_window, lambda: dcp.run_window._renderer_ready)
    dcp.run_window._start()
    _drive(dcp.run_window, lambda: dcp.run_window.controller.view().phase == "hold")
    dcp.shutdown()
    record = json.loads((tmp_path / notes.RUN_RECORD_NAME).read_text())
    assert record["status"] == "cancelled" and record["end_reason"] == "application_shutdown"
    last = [iv for iv in record["intervals"] if iv["track"] == "phase"][-1]
    assert last["phase"] == "hold" and last["completion_status"] == "cancelled"
    # The cancelled hold is in the annotations but not cut as a repetition.
    session = SessionAnnotations.load(tmp_path)
    assert any(s.status == "cancelled" for s in session.stages)
    assert not any(Path(tmp_path).glob("C_*_R_*_emg.csv"))


# --- intensity ------------------------------------------------------------------------

def test_intensity_settings_read_validate_and_round_trip(dpg):
    from libemg._gui._choreography_panel import parse_levels
    assert parse_levels("30, 60 90%") == ([0.3, 0.6, 0.9], [])
    assert parse_levels("30, lots")[1] == ["Intensity level 'lots' is not a number."]
    section = ChoreographyConfigSection(ASSETS, config=_config())
    section.build()
    assert section.read_config()[0].intensity_enabled is False
    dpg.values["__ch_intensity_levels"] = "25, oops"
    assert section.read_config()[1] == [], "level typos only matter once intensity is on"
    dpg.values["__ch_intensity_enabled"] = True
    _, plan, problems = section.validate()
    assert plan is None and "Intensity level 'oops' is not a number." in problems
    dpg.values["__ch_intensity_levels"] = "25, 75"
    dpg.values["__ch_intensity_order"] = "Seeded shuffle"
    config, plan, problems = section.validate()
    assert problems == [] and config.intensity_levels == [0.25, 0.75]
    assert config.intensity_order == "shuffle"
    assert sorted(r.intensity for r in plan.repetitions) == [0.25, 0.25, 0.75, 0.75]
    section.describe()
    assert "Intensity targets 25, 75% of maximum" in dpg.values["__ch_summary"]
    dpg.values["__ch_repetitions"] = 3
    section.describe()
    assert "not a multiple of 2 levels" in dpg.values["__ch_summary"]
    section.apply_config(_config(intensity_enabled=True, intensity_levels=[0.4]))
    assert dpg.values["__ch_intensity_levels"] == "40"
    assert section.read_config()[0].intensity_levels == [0.4]


def test_the_intensity_bar_is_only_drawn_when_asked_for(dpg):
    window, _ = _window(dpg)
    assert window.bar is None
    assert not any(c.kwargs.get("tag") == "__ch_intensity_bar"
                   for c in dpg.add_drawlist.call_args_list)
    window.destroy()
    window, _ = _window(dpg, intensity_enabled=True, intensity_levels=[0.5, 1.0])
    assert window.bar is not None
    assert any(c.kwargs.get("tag") == "__ch_intensity_bar"
               for c in dpg.add_drawlist.call_args_list)
    window.destroy()


def test_the_intensity_bar_follows_the_target(dpg):
    from libemg._gui._choreography_panel import IntensityBar
    from libemg.choreography.schedule import HOLD, RAMP_IN, REST
    view = Mock(intensity_level=0.6, intensity=0.3, phase=RAMP_IN)
    fill, level, current, color = IntensityBar.appearance(view, RUNNING)
    assert (fill, level, current, color) == (0.3, 0.6, True, _style.ACCENT)
    view = Mock(intensity_level=0.6, intensity=0.6, phase=HOLD)
    assert IntensityBar.appearance(view, RUNNING)[3] == _style.SUCCESS
    view = Mock(intensity_level=0.9, intensity=0.0, phase=REST)
    assert IntensityBar.appearance(view, RUNNING)[:3] == (0.0, 0.9, False), "previews the next"
    assert IntensityBar.appearance(view, PAUSED_STATE)[3] == _style.MUTED
    assert IntensityBar.appearance(view, "ready")[1] is None
    bar = IntensityBar()
    bar.height = 480
    assert bar.y(0.0) == 480 - IntensityBar.BOTTOM and bar.y(1.0) == IntensityBar.TOP
    assert bar.y(0.5) == pytest.approx((bar.y(0.0) + bar.y(1.0)) / 2)


def test_the_window_names_the_target_effort(dpg):
    window, _ = _window(dpg, hold=0.5, intensity_enabled=True, intensity_levels=[0.5, 1.0])
    seen = set()
    window._start()

    def watch():
        seen.add(dpg.values.get("__ch_intensity_text"))
        seen.add(dpg.values.get("__ch_next"))
        return window.controller.result is not None and window._finalizer is None
    _drive(window, watch)
    assert "Target effort: 50% of maximum" in seen and "Target effort: 100% of maximum" in seen
    assert any(text and text.endswith("at 100% effort") for text in seen)
    window.destroy()


def test_a_recording_with_intensity_writes_per_sample_targets(tmp_path, dpg):
    from libemg.choreography import targets as tg
    handler = _Handler(fs=500.0, batch=5)
    cfg = _config(intensity_enabled=True, intensity_levels=[0.5, 1.0])
    dcp = DataCollectionPanel(handler, data_folder=str(tmp_path / "rec"),
                              choreography_config=cfg, signal_check="off")
    folder = tmp_path / "rec"
    dcp.output_folder = str(folder)
    for name in ("subject_number", "subject_name", "session_number", "day_number",
                 "condition_number"):
        dpg.values[f"__dc_{name}"] = "1"
    dpg.values.update({"__dc_output_folder": str(folder), "__dc_num_reps": 1,
                       "__dc_rep_time": 1.0, "__dc_rest_time": 0.0, "__dc_media_folder": "",
                       "__dc_auto_advance": True})
    dcp.spawn_configuration_window()
    window = dcp.open_choreography(recording=True)
    window.worker.close()
    from libemg.choreography.renderer import RenderWorker
    window.worker = RenderWorker(lambda: _StubRenderer(), 64, 64)
    _drive(window, lambda: window._renderer_ready)
    window._start()
    _drive(window, lambda: window.controller.result is not None, timeout=30)
    assert window.controller.result.status == COMPLETED

    record = json.loads((folder / notes.RUN_RECORD_NAME).read_text())
    session = SessionAnnotations.load(folder)
    for mod in handler.modalities:
        targets = tg.load_targets(folder, mod)
        _, samples = session.read(mod)
        assert len(targets) == len(samples) == len(targets.samples)
        ramps = targets.phase_code == tg.PHASE_CODES["ramp_in"]
        rising = np.diff(targets.position[ramps])
        # Rising within each ramp; the only drops are from one ramp to the next.
        assert (rising >= -1e-9).sum() >= len(rising) - 4
        # Six decimals on disk.
        np.testing.assert_allclose(targets.intensity, targets.position * targets.intensity_level,
                                   atol=1e-6)
        holds = targets.phase_code == tg.PHASE_CODES["hold"]
        assert set(np.round(targets.intensity[holds], 6)) == {0.5, 1.0}
    details = json.loads((folder / "collection_details.json").read_text())
    assert details["intensity"]["levels"] == [0.5, 1.0]
    assert record["target_files"]["emg"] == "_choreography_targets_emg.csv"
    # Fixed order: hand_closed, wrist_flexion at 50 %, then both at 100 %.
    assert [s.factors.get("intensity") for s in session.select(kinds=(PROMPT,))] \
        == [0.5, 0.5, 1.0, 1.0]

    out = tmp_path / "repetitions"
    written = tg.cut_repetitions(folder, out, modalities=["emg"])
    assert len(written) == 8, "a data file and a target file per repetition"
    data = np.loadtxt(out / "C_1_R_0_emg.csv", delimiter=",", ndmin=2)
    labels = np.loadtxt(out / "C_1_R_0_emg_targets.csv", delimiter=",", ndmin=2)
    assert len(data) == len(labels) > 0
    position = labels[:, tg.columns(True).index("position")]
    assert position[0] < 0.1 and position.max() == pytest.approx(1.0)
    window.destroy()


# --- no rest between gestures -------------------------------------------------------------

def test_the_rest_checkbox_turns_rests_off_and_says_so(dpg):
    section = ChoreographyConfigSection(ASSETS, config=_config())
    section.build()
    assert section.read_config()[0].rest_between_gestures is True
    dpg.values["__ch_rest_between_gestures"] = False
    config, plan, problems = section.validate()
    assert problems == [] and config.rest_between_gestures is False
    assert not any(p.phase == "rest" for p in plan.phases)
    section.describe()
    assert "No rest between gestures: 3 direct transitions" in dpg.values["__ch_summary"]
    dpg.values["__ch_warmup"] = 0.0
    _, plan, problems = section.validate()
    assert plan is None and any("rest class" in p for p in problems)
    section.apply_config(_config(rest_between_gestures=True))
    assert dpg.values["__ch_rest_between_gestures"] is True


def test_the_window_warns_of_a_direct_transition(dpg):
    window, _ = _window(dpg, hold=0.3, rest_between_gestures=False)
    seen = set()
    window._start()

    def watch():
        seen.add(dpg.values.get("__ch_next"))
        return window.controller.result is not None and window._finalizer is None
    _drive(window, watch)
    assert "Then straight into: Wrist Flexion" in seen
    assert "Then straight into: Hand Closed" in seen
    assert window.controller.result.status == COMPLETED
    window.destroy()
