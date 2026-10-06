"""Coverage for designing a collection protocol in the window."""
from unittest.mock import MagicMock, Mock

import pytest

from libemg._gui import _protocol_panel as panel_module
from libemg._gui._protocol_panel import ProtocolPanel, _numbers, _parse_factors
from libemg.protocol import PROTOCOL_NAME, CollectionProtocol, Segment


@pytest.fixture
def dpg(monkeypatch):
    # MagicMock rather than Mock: the window is built out of dpg's context
    # managers.
    fake = MagicMock()
    fake.does_item_exist.return_value = True
    monkeypatch.setattr(panel_module, "dpg", fake)
    return fake


def _filled(dpg, **typed):
    """Stand in for the window with these values showing."""
    values = {"__protocol_name": "study", "__protocol_kind": "regression",
              "__protocol_notes": "pilot",
              "__protocol_dof_labels": "Flex/Ext, Pro/Sup",
              "__protocol_animator": "scatter", "__protocol_fps": 30,
              "__protocol_tpd": 1.5, "__protocol_show_direction": True,
              "__protocol_show_countdown": False, "__protocol_show_boundary": True,
              "__protocol_figure_width": 8, "__protocol_figure_height": 6,
              "__protocol_dpi": 100, "__protocol_video_format": ".gif",
              "__protocol_num_reps": 4, "__protocol_rep_time": 6.0,
              "__protocol_rest_time": 2.0, "__protocol_rest_jitter": 0.75,
              "__protocol_reaction_time": 0.3, "__protocol_order": "counterbalanced",
              "__protocol_seed": 4242,
              "__protocol_factors": "limb_position: neutral, elevated\nintensity: low",
              "__protocol_media": "animation/", "__protocol_path": "",
              "__protocol_template": "Within and combined DOF"}
    values.update({f"__protocol_{name}": value for name, value in typed.items()})
    for name, _ in panel_module.ACQUISITION_FIELDS:
        values.setdefault(f"__protocol_acq_{name}", "")
    dpg.get_value.side_effect = lambda tag: values.get(tag)
    return values


# --- reading the window ------------------------------------------------

def test_the_window_is_read_back_into_the_protocol(dpg):
    panel = ProtocolPanel()
    _filled(dpg, acq_device="Delsys Trigno", acq_sampling_rate_hz="2000")

    panel._read_window()
    protocol = panel.protocol

    assert protocol.name == "study" and protocol.kind == "regression"
    assert protocol.dof_labels == ["Flex/Ext", "Pro/Sup"]
    assert protocol.media.animator == "scatter" and protocol.media.fps == 30
    assert protocol.media.figsize == (8, 6) and protocol.media.video_format == ".gif"
    assert protocol.media.show_countdown is False
    assert protocol.timing.num_reps == 4 and protocol.timing.rest_jitter == 0.75
    assert protocol.order == "counterbalanced" and protocol.seed == 4242
    assert protocol.factors == {"limb_position": ["neutral", "elevated"],
                                "intensity": ["low"]}
    assert protocol.acquisition["device"] == "Delsys Trigno"
    assert protocol.acquisition["sampling_rate_hz"] == "2000"
    assert panel.media_folder == "animation/"


def test_a_blank_name_does_not_produce_a_nameless_file(dpg):
    panel = ProtocolPanel()
    _filled(dpg, name="   ")
    panel._read_window()
    assert panel.protocol.name == "collection"


def test_nonsense_in_a_number_field_cannot_produce_an_impossible_protocol(dpg):
    panel = ProtocolPanel()
    _filled(dpg, fps=0, num_reps=-3, rest_jitter=-1.0, dpi=1)
    panel._read_window()

    assert panel.protocol.media.fps >= 1
    assert panel.protocol.timing.num_reps >= 1
    assert panel.protocol.timing.rest_jitter == 0.0


def test_factor_lines_that_are_not_factors_are_ignored():
    assert _parse_factors("limb_position: a, b\nnonsense\n: nothing\nempty:") == {
        "limb_position": ["a", "b"]}
    assert _parse_factors("") == {} and _parse_factors(None) == {}


def test_a_target_is_parsed_from_whatever_was_typed():
    assert _numbers("1, -0.5, 0") == [1.0, -0.5, 0.0]
    assert _numbers("1; 2") == [1.0, 2.0]
    assert _numbers("") == [] and _numbers("x, 1") == [1.0]


# --- trajectory editing ------------------------------------------------

def test_applying_a_template_fills_the_trajectory_for_the_dofs_in_the_window(dpg):
    panel = ProtocolPanel()
    _filled(dpg)

    panel._apply_template()

    assert panel.protocol.trajectory
    # Two DOFs were named, so the template drives both of them and their pairs.
    held = {tuple(s.target) for s in panel.protocol.trajectory if s.kind == "hold"}
    assert (1.0, 0.0) in held and (1.0, 1.0) in held


def test_segments_can_be_added_edited_and_removed(dpg):
    panel = ProtocolPanel()
    _filled(dpg)
    panel._read_window()

    panel._add_segment()
    panel._add_segment()
    assert len(panel.protocol.trajectory) == 2
    assert len(panel.protocol.trajectory[0].target) == 2, "one target per DOF"

    dpg.get_value.side_effect = lambda tag: {"__protocol_seg_kind_0": "ramp"}.get(tag)
    panel._segment_changed("__protocol_seg_seconds_0", 3.5, 0)
    panel._segment_changed("__protocol_seg_target_0", "1, -1", 0)
    assert panel.protocol.trajectory[0].seconds == 3.5
    assert panel.protocol.trajectory[0].target == [1.0, -1.0]

    panel._remove_segment(None, None, 0)
    assert len(panel.protocol.trajectory) == 1
    panel._clear_segments()
    assert panel.protocol.trajectory == []


def test_a_segment_cannot_be_given_a_negative_duration_or_a_zero_period(dpg):
    panel = ProtocolPanel(protocol=CollectionProtocol(
        trajectory=[Segment("sinusoid", 1.0, [1.0])]))
    _filled(dpg)

    panel._segment_changed("__protocol_seg_seconds_0", -5.0, 0)
    panel._segment_changed("__protocol_seg_period_0", 0.0, 0)

    assert panel.protocol.trajectory[0].seconds == 0.0
    assert panel.protocol.trajectory[0].period > 0


def test_the_summary_warns_when_the_countdown_has_nothing_to_count_down_to(dpg):
    panel = ProtocolPanel(protocol=CollectionProtocol(
        dof_labels=["a"], trajectory=[Segment("sinusoid", 4.0, [1.0])]))
    panel.protocol.media.show_countdown = True

    panel._refresh_summary()

    said = dpg.set_value.call_args[0][1]
    assert "add a hold or a rest" in said
    # With a hold in it there is nothing to warn about.
    panel.protocol.trajectory.append(Segment("hold", 1.0, [1.0]))
    panel._refresh_summary()
    assert "add a hold or a rest" not in dpg.set_value.call_args[0][1]


def test_the_summary_says_how_long_a_repetition_will_take(dpg):
    panel = ProtocolPanel(protocol=CollectionProtocol(
        dof_labels=["a"], trajectory=[Segment("hold", 3.0, [1.0])]))
    panel.protocol.media.fps = 10

    panel._refresh_summary()

    assert "30 frames, 3.0 s" in dpg.set_value.call_args[0][1]


# --- files and media ---------------------------------------------------

def test_saving_and_loading_round_trips_through_the_window(tmp_path, dpg):
    panel = ProtocolPanel()
    _filled(dpg, path=str(tmp_path / "study.yaml"))
    panel._apply_template()

    path = panel.save()

    assert path.exists()
    fresh = ProtocolPanel()
    _filled(dpg, path=str(path))
    loaded = fresh.load()
    assert loaded.name == "study" and loaded.seed == 4242
    assert loaded.order == "counterbalanced"
    assert len(loaded.trajectory) == len(panel.protocol.trajectory)


def test_saving_with_no_path_falls_back_to_the_protocol_name(tmp_path, dpg,
                                                             monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = ProtocolPanel()
    _filled(dpg, path="")

    assert panel.save().name == "study.yaml"


def test_loading_a_file_that_is_not_there_says_so_rather_than_raising(dpg):
    panel = ProtocolPanel()
    _filled(dpg, path="no/such/protocol.yaml")

    assert panel.load() is None
    assert "Could not load" in dpg.set_value.call_args[0][1]


def test_a_loaded_protocol_is_published_to_the_rest_of_the_gui(tmp_path, dpg):
    saved = CollectionProtocol(name="ready", dof_labels=["a"])
    saved.save(tmp_path)
    published = []
    panel = ProtocolPanel(on_ready=lambda protocol, folder: published.append(
        (protocol.name, folder)))
    _filled(dpg, path=str(tmp_path / PROTOCOL_NAME))

    panel.load()

    assert published == [("ready", "animation/")]


def test_building_media_refuses_a_protocol_with_nothing_in_it(dpg):
    panel = ProtocolPanel()
    _filled(dpg)

    panel.build()

    assert panel._worker is None
    assert "no trajectory" in dpg.set_value.call_args[0][1]


def test_building_media_runs_off_the_render_thread_and_reports_back(tmp_path, dpg):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    published = []
    panel = ProtocolPanel(on_ready=lambda protocol, folder: published.append(folder))
    _filled(dpg, media=str(tmp_path / "animation"), video_format=".gif", fps=5,
            dof_labels="a, b", show_countdown=False, show_direction=False)
    panel.protocol.trajectory = [Segment("rest", 0.2), Segment("hold", 0.2, [1.0, 0.0])]

    panel.build()
    panel._worker.join(timeout=120)
    panel.poll()

    assert (tmp_path / "animation" / "study.gif").exists()
    assert (tmp_path / "animation" / "study.txt").exists()
    assert published == [str(tmp_path / "animation")]


def test_a_failed_build_reports_instead_of_dying_in_the_thread(dpg, monkeypatch):
    panel = ProtocolPanel()
    _filled(dpg)
    panel.protocol.trajectory = [Segment("rest", 0.2)]
    monkeypatch.setattr(panel.protocol, "build_media",
                        Mock(side_effect=RuntimeError("no encoder")))

    panel.build()
    panel._worker.join(timeout=30)
    panel.poll()

    assert "no encoder" in dpg.set_value.call_args[0][1]


def test_the_panel_owns_a_window_the_gui_can_find(dpg):
    panel = ProtocolPanel()
    assert panel.window_tag == "__protocol_window"
    panel.cleanup()
    dpg.delete_item.assert_called_with("__protocol_window")


def test_the_window_offers_every_section_the_protocol_has(dpg):
    ProtocolPanel().spawn_window()

    headers = [call.kwargs.get("label")
               for call in dpg.collapsing_header.call_args_list]
    assert headers == ["What is being collected", "Prompt trajectory",
                       "How it is drawn", "Timing and order", "Design factors",
                       "Acquisition (ISEK reporting)"]
    combos = {call.kwargs.get("tag") for call in dpg.add_combo.call_args_list}
    assert combos == {"__protocol_kind", "__protocol_template",
                      "__protocol_animator", "__protocol_video_format",
                      "__protocol_order"}
    # Every ISEK field gets a box, so the form cannot quietly lose one.
    tagged = {call.kwargs.get("tag") for call in dpg.add_input_text.call_args_list}
    for name, _ in panel_module.ACQUISITION_FIELDS:
        assert f"__protocol_acq_{name}" in tagged
