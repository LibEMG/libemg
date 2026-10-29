"""Coverage for how screen guided training drives the online data handler's logger."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from libemg._gui import _data_collection_panel as panel_module
from libemg._gui import _style
from libemg._gui._data_collection_panel import DataCollectionPanel
from libemg.session import PAUSE, PROMPT, REST, SessionAnnotations


class _RecordingHandler:
    """Stands in for an OnlineDataHandler, writing a file per logging window."""

    def __init__(self, modalities=("emg", "imu"), samples_per_rep=5):
        self.modalities = list(modalities)
        self.samples_per_rep = samples_per_rep
        self.calls = []
        self._prefix = None

    def reset(self):
        self.calls.append("reset")

    def start_log(self, file_path='', timestamps=True, **_kwargs):
        self.calls.append(("start_log", file_path, timestamps))
        self._prefix = file_path

    def stop_log(self):
        self.calls.append("stop_log")
        counts = {}
        for mod in self.modalities:
            rows = np.arange(self.samples_per_rep, dtype=float).reshape(-1, 1)
            with open(self._prefix + mod + ".csv", "a", newline='') as handle:
                np.savetxt(handle, rows, delimiter=",")
            counts[mod] = self.samples_per_rep
        return counts


class _StreamingHandler:
    """Stands in for an OnlineDataHandler delivering at a fixed rate, in batches.

    One recording is written for the whole logging window, as a continuous
    collection produces: sample ``i`` was produced at ``t0 + i / fs``, and a whole
    batch of them shares the arrival time it was read at. The first channel holds
    the sample's own index, so a cut file says where in the recording it came
    from.
    """

    def __init__(self, modalities=("emg", "imu"), fs=200.0, batch=10, channels=2):
        self.modalities = list(modalities)
        self.fs = fs
        self.batch = batch
        self.channels = channels
        self.log_dropped = {}
        self.calls = []
        self.alive = True
        self._prefix = None
        self._t0 = None
        self._frozen = 0

    def reset(self):
        self.calls.append("reset")

    def start_log(self, file_path='', timestamps=True, **_kwargs):
        self.calls.append(("start_log", file_path, timestamps))
        self._prefix = file_path
        self._t0 = time.time()

    def kill_stream(self):
        """Stop delivering, without closing the log, as an unplugged device does."""
        self._frozen = self._delivered()
        self.alive = False

    def _delivered(self):
        if not self.alive:
            return self._frozen
        whole_batches = int((time.time() - self._t0) * self.fs) // self.batch
        return whole_batches * self.batch

    def log_counts(self):
        return {mod: self._delivered() for mod in self.modalities}

    def stop_log(self):
        self.calls.append("stop_log")
        n = self._delivered()
        counts = {}
        for mod in self.modalities:
            np.savetxt(self._prefix + mod + ".csv", self._rows(n), delimiter=",")
            counts[mod] = n
        return counts

    def _rows(self, n):
        index = np.arange(n, dtype=float)
        # A batch is read once its last sample exists, plus a delivery lag that
        # varies from batch to batch -- the thing the clock fit has to see past.
        batch = index // self.batch
        arrival = self._t0 + ((batch + 1) * self.batch - 1) / self.fs + 0.002 + 0.001 * (batch % 3)
        samples = np.column_stack([index + 1000 * channel for channel in range(self.channels)])
        return np.column_stack([arrival, samples])


def _panel(handler, tmp_path, dpg, **kwargs):
    kwargs.setdefault("rest_time", 0)
    dcp = DataCollectionPanel(handler, num_reps=1, **kwargs)
    dcp.output_folder = str(tmp_path)
    dcp.num_motions = 1
    # Stand in for the user clicking "Continue" once a rep ends.
    dpg.run_callbacks.side_effect = lambda *_args: setattr(dcp, "advance", True)
    return dcp


def _media_list(n_classes=2, n_reps=1, rep_time=0.01):
    media_list = []
    for rep in range(n_reps):
        for cls in range(n_classes):
            media = Mock()
            media.fps = 1000
            media_list.append([media, f"class_{cls}", cls, rep, rep_time])
    return media_list


def _annotations(tmp_path):
    return SessionAnnotations.load(tmp_path)


@pytest.fixture
def dpg(monkeypatch):
    # MagicMock rather than Mock: the configuration window builds itself out of
    # dpg's context managers.
    fake = MagicMock()
    monkeypatch.setattr(panel_module, "dpg", fake)
    # The panel draws through libemg._gui._style too, which holds its own
    # handle on DearPyGui. Left real, it would draw with no context, and that
    # takes the interpreter down rather than raising.
    monkeypatch.setattr(_style, "dpg", fake)
    monkeypatch.setattr(panel_module, "set_texture", Mock())
    return fake


# --- one log per repetition (continuous_log=False) ---------------------------

def test_run_sgt_brackets_each_rep_with_a_logging_window(tmp_path, dpg):
    handler = _RecordingHandler()
    dcp = _panel(handler, tmp_path, dpg, timestamps=True, continuous_log=False)
    dcp.run_sgt(_media_list(n_classes=2))

    start_calls = [c for c in handler.calls if isinstance(c, tuple)]
    assert [Path(c[1]).name for c in start_calls] == ["C_0_R_0_", "C_1_R_0_"]
    assert all(c[2] is True for c in start_calls), "timestamps flag must reach the logger"
    # Every start_log is closed by a stop_log before the next one opens.
    bracketed = [c[0] if isinstance(c, tuple) else c for c in handler.calls]
    assert bracketed.count("start_log") == bracketed.count("stop_log") == 2
    assert bracketed.index("start_log") < bracketed.index("stop_log")

    for cls in range(2):
        for mod in handler.modalities:
            assert (tmp_path / f"C_{cls}_R_0_{mod}.csv").exists()


def test_redoing_a_rep_replaces_the_previous_files_instead_of_appending(tmp_path, dpg):
    handler = _RecordingHandler(samples_per_rep=5)
    dcp = _panel(handler, tmp_path, dpg, continuous_log=False)
    stale = tmp_path / "C_0_R_0_emg.csv"
    stale.write_text("999\n999\n999\n999\n999\n999\n999\n", encoding="utf-8")

    dcp.run_sgt(_media_list(n_classes=1))

    logged = np.loadtxt(stale, delimiter=",")
    np.testing.assert_array_equal(logged, np.arange(5.0))


def test_run_sgt_raises_when_a_modality_logged_nothing(tmp_path, dpg):
    handler = _RecordingHandler(samples_per_rep=0)
    dcp = _panel(handler, tmp_path, dpg, continuous_log=False)
    with pytest.raises(ConnectionError, match="emg, imu"):
        dcp.run_sgt(_media_list(n_classes=1))


def test_logger_is_stopped_even_when_the_media_playback_fails(tmp_path, monkeypatch, dpg):
    handler = _RecordingHandler()
    dcp = _panel(handler, tmp_path, dpg, continuous_log=False)
    monkeypatch.setattr(dcp, "play_collection_visual", Mock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        dcp.run_sgt(_media_list(n_classes=1))

    assert "stop_log" in handler.calls


# --- one continuous recording (the default) ----------------------------------

def test_the_session_file_keeps_the_rests_and_pauses_as_well_as_the_reps(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, rest_time=0.1, auto_advance=False)
    dcp.run_sgt(_media_list(n_classes=2, rep_time=0.2))

    # One logging window for the whole collection, not one per repetition.
    opened = [c for c in handler.calls if isinstance(c, tuple)]
    assert len(opened) == 1 and Path(opened[0][1]).name == "_session_"

    session = np.loadtxt(tmp_path / "_session_emg.csv", delimiter=",", ndmin=2)
    cut = sum(len(np.loadtxt(tmp_path / f"C_{cls}_R_0_emg.csv", delimiter=",", ndmin=2))
              for cls in range(2))
    # The rests, the pauses and the overruns are all still in the recording, so
    # it holds considerably more than the repetitions cut out of it.
    assert len(session) > cut * 1.5
    # And it is one unbroken run of samples: index is the first channel.
    np.testing.assert_array_equal(session[:, 1], np.arange(len(session), dtype=float))


def test_every_stage_of_the_collection_is_annotated_with_its_indices(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, rest_time=0.1, auto_advance=False)
    dcp.run_sgt(_media_list(n_classes=2, rep_time=0.2))

    session = _annotations(tmp_path)
    kinds = [stage.kind for stage in session.stages]
    assert kinds == [REST, PROMPT, PAUSE, REST, PROMPT, PAUSE]
    assert [s.motion for s in session.select(kinds=(PROMPT,))] == [0, 1]
    assert session.class_map == {"0": "class_0", "1": "class_1"}

    for stage in session.stages:
        for mod in handler.modalities:
            start, end = stage.span(mod)
            assert 0 <= start <= end <= session.clocks[mod].n_samples
    # A prompt is indexed to the duration it was asked for, not to the frame the
    # render loop happened to stop on.
    for stage in session.select(kinds=(PROMPT,)):
        start, end = stage.span("emg")
        assert end - start == pytest.approx(0.2 * session.clocks["emg"].fs, abs=2)


def test_annotations_are_on_disk_before_the_collection_finishes(tmp_path, dpg, monkeypatch):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, rest_time=0)
    seen = []
    play = dcp.play_collection_visual

    def watched(media, active=True):
        # What the annotation file holds at the moment a stage begins -- i.e.
        # everything collected before it, written as it went.
        path = tmp_path / "_session_annotations.json"
        seen.append(len(json.loads(path.read_text())["stages"]) if path.exists() else None)
        return play(media, active=active)

    monkeypatch.setattr(dcp, "play_collection_visual", watched)
    dcp.run_sgt(_media_list(n_classes=3, rep_time=0.1))

    # The file exists from the first prompt, and grows a stage at a time rather
    # than appearing whole at the end.
    assert seen[0] == 0
    assert seen == sorted(seen) and seen[-1] > seen[0]


def test_the_timeline_accounts_for_every_sample_in_the_recording(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, rest_time=0.1, auto_advance=False)
    dcp.run_sgt(_media_list(n_classes=2, rep_time=0.2))

    session = _annotations(tmp_path)
    spans = session.timeline("emg")
    assert spans[0].start == 0
    assert spans[-1].end == session.clocks["emg"].n_samples
    for before, after in zip(spans, spans[1:]):
        assert before.end == after.start, "the stages must tile the recording"
    assert {span.kind for span in spans} == {REST, PROMPT, PAUSE}


def test_a_session_can_be_cut_again_from_the_annotations_alone(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, rest_time=0.1)
    dcp.run_sgt(_media_list(n_classes=2, rep_time=0.2))

    recut = tmp_path / "recut"
    written = _annotations(tmp_path).cut(output_folder=recut)

    assert {cut.path.name for cut in written} == {
        f"C_{cls}_R_0_{mod}.csv" for cls in range(2) for mod in handler.modalities}
    for cut in written:
        original = np.loadtxt(tmp_path / cut.path.name, delimiter=",", ndmin=2)
        again = np.loadtxt(cut.path, delimiter=",", ndmin=2)
        np.testing.assert_array_equal(original, again)
    # And the cut lands where the prompt was: the first channel is the sample's
    # own index in the recording.
    session = _annotations(tmp_path)
    for stage in session.select(kinds=(PROMPT,)):
        chunk = np.loadtxt(recut / f"C_{stage.motion}_R_0_emg.csv", delimiter=",", ndmin=2)
        assert chunk[0, 0] == pytest.approx(stage.span("emg")[0], abs=1)


def test_cutting_can_take_the_rests_and_extra_context_either_side(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, rest_time=0.2)
    dcp.run_sgt(_media_list(n_classes=2, rep_time=0.2))

    session = _annotations(tmp_path)
    fs = session.clocks["emg"].fs
    plain = {cut.path.name: cut.got for cut in
             session.cut(output_folder=tmp_path / "plain", modalities=["emg"])}
    padded = {cut.path.name: cut.got for cut in
              session.cut(output_folder=tmp_path / "padded", modalities=["emg"],
                          kinds=(PROMPT, REST), lead_in=0.05, lead_out=0.05)}

    assert set(plain) == {"C_0_R_0_emg.csv", "C_1_R_0_emg.csv"}
    # A rest is written under a name that carries no C_/R_ tokens, so a regex
    # parser pointed at the folder cannot mistake one for a repetition.
    assert set(padded) - set(plain) == {"rest_0_0_emg.csv", "rest_1_0_emg.csv"}
    for name, got in plain.items():
        assert padded[name] - got == pytest.approx(0.1 * fs, abs=2)


def test_a_redone_rep_is_cut_from_the_last_take_but_still_annotated(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, rest_time=0, auto_advance=False)
    redone = []

    def advance(*_args):
        # Redo the first prompt once, then continue through the rest.
        if not redone:
            redone.append(True)
            dcp.redo_collection_callback()
        else:
            dcp.continue_collection_callback()

    dpg.run_callbacks.side_effect = advance
    dcp.run_sgt(_media_list(n_classes=2, rep_time=0.1))

    session = _annotations(tmp_path)
    takes = [s for s in session.stages if s.kind == PROMPT and s.motion == 0]
    assert [t.attempt for t in takes] == [0, 1]
    assert [t.superseded for t in takes] == [True, False]
    # The abandoned take is still in the recording and still described, but the
    # file that gets cut is the one that was collected last.
    written = session.cut(output_folder=tmp_path / "recut", modalities=["emg"])
    assert [cut.stage for cut in written if cut.stage.motion == 0] == [takes[-1]]
    assert len(session.cut(output_folder=tmp_path / "all", modalities=["emg"],
                           include_superseded=True)) == 3


def test_a_stream_that_dies_mid_session_is_caught_while_it_can_be_redone(tmp_path, dpg,
                                                                        monkeypatch):
    monkeypatch.setattr(panel_module, "STREAM_SILENCE_LIMIT", 0.15)
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg)
    play = dcp.play_collection_visual
    monkeypatch.setattr(dcp, "play_collection_visual",
                        lambda media, active=True: (handler.kill_stream(),
                                                    play(media, active=active))[1])

    with pytest.raises(ConnectionError, match="no samples have arrived"):
        dcp.run_sgt(_media_list(n_classes=3, rep_time=0.2))

    # The collection still closed its log and left annotations for what it got.
    assert "stop_log" in handler.calls
    assert (tmp_path / "_session_annotations.json").exists()
    assert _annotations(tmp_path).stages


# --- who the collection was of ------------------------------------------------

def _fill_configuration(dpg, tmp_path, **typed):
    """Stand in for the configuration window with these fields typed into it."""
    values = {"__dc_num_reps": "2", "__dc_rep_time": "1.5", "__dc_rest_time": "0.5",
              "__dc_media_folder": "media/", "__dc_output_folder": str(tmp_path),
              "__dc_auto_advance": True, "__dc_subject_number": "7",
              "__dc_subject_name": "Ada", "__dc_session_number": "2",
              "__dc_day_number": "3", "__dc_condition_number": "4"}
    values.update({f"__dc_{name}": value for name, value in typed.items()})
    dpg.get_value.side_effect = lambda tag: values[tag]


def test_the_configuration_window_offers_a_field_for_each_subject_detail(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg, subject_number=9, subject_name="Grace")
    dcp.spawn_configuration_window()

    tagged = {call.kwargs["tag"]: call.kwargs["default_value"]
              for call in dpg.add_input_text.call_args_list}
    assert tagged["__dc_subject_number"] == 9
    assert tagged["__dc_subject_name"] == "Grace"
    for field in ("session_number", "day_number", "condition_number"):
        assert tagged[f"__dc_{field}"] == 1, "defaults stand in for a first session"
    # And they are torn down with the rest of the window.
    assert "__dc_subject_name" in dcp.widget_tags["configuration"]


def test_subject_details_are_read_off_the_window_when_start_is_pressed(tmp_path, dpg):
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg)
    _fill_configuration(dpg, tmp_path)

    dcp.get_settings()

    assert (dcp.subject_number, dcp.subject_name) == (7, "Ada")
    assert (dcp.session_number, dcp.day_number, dcp.condition_number) == (2, 3, 4)


def test_subject_details_are_written_as_yaml_beside_the_data(tmp_path, dpg):
    yaml = pytest.importorskip("yaml")
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg)
    _fill_configuration(dpg, tmp_path)
    dcp.get_settings()

    path = dcp.write_subject_details()
    written = yaml.safe_load(path.read_text())

    assert path == tmp_path / "subject_details.yaml"
    assert written["subject_number"] == 7 and written["subject_name"] == "Ada"
    assert written["session_number"] == 2 and written["day_number"] == 3
    assert written["condition_number"] == 4
    assert written["collected"], "the file says when the collection happened"
    # Written in the order the window asks for them, not alphabetically.
    assert list(written) == ["subject_number", "subject_name", "session_number",
                            "day_number", "condition_number", "collected"]


def test_a_subject_labelled_by_something_other_than_a_number_is_kept(tmp_path, dpg):
    yaml = pytest.importorskip("yaml")
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg)
    _fill_configuration(dpg, tmp_path, subject_number="P01", condition_number="")
    dcp.get_settings()

    # A study that labels its subjects P01 is not a mistake, and a collection is
    # not the moment to refuse one.
    assert dcp.subject_number == "P01"
    assert dcp.condition_number == ""
    written = yaml.safe_load(dcp.write_subject_details().read_text())
    assert written["subject_number"] == "P01"


def test_the_yaml_lands_in_the_output_folder_the_collection_creates(tmp_path, dpg,
                                                                   monkeypatch):
    yaml = pytest.importorskip("yaml")
    handler = _StreamingHandler()
    dcp = _panel(handler, tmp_path, dpg)
    media = tmp_path / "media"
    media.mkdir()
    (media / "flexion.png").write_bytes(b"")
    output = tmp_path / "does_not_exist_yet"
    _fill_configuration(dpg, tmp_path)
    dcp.get_settings()
    dcp.media_folder, dcp.output_folder, dcp.data_folder = str(media), str(output), str(output)
    monkeypatch.setattr(panel_module, "Media", Mock(return_value=Mock(type="png")))

    dcp.gather_media()

    assert yaml.safe_load((output / "subject_details.yaml").read_text())["subject_name"] == "Ada"


# --- a protocol driving the collection ---------------------------------

def _protocol(num_reps=2, **kwargs):
    from libemg.protocol import CollectionProtocol
    kwargs.setdefault("name", "study")
    kwargs.setdefault("seed", 5)
    protocol = CollectionProtocol(**kwargs)
    protocol.timing.num_reps = num_reps
    return protocol


def _media_folder(tmp_path, n_classes=3):
    folder = tmp_path / "media"
    folder.mkdir(exist_ok=True)
    for index in range(n_classes):
        (folder / f"gesture_{index}.png").write_bytes(b"")
    return folder


def _gather(dcp, tmp_path, monkeypatch, n_classes=3):
    """Run gather_media against a folder of stub images."""
    dcp.media_folder = str(_media_folder(tmp_path, n_classes))
    dcp.data_folder = dcp.output_folder = str(tmp_path)
    monkeypatch.setattr(panel_module, "Media", Mock(return_value=Mock(type="png")))
    return dcp.gather_media()


def test_without_a_protocol_the_order_is_what_it_always_was(tmp_path, dpg, monkeypatch):
    dcp = _panel(_StreamingHandler(), tmp_path, dpg)
    dcp.num_reps = 2

    media_list = _gather(dcp, tmp_path, monkeypatch)

    assert [(m[2], m[3]) for m in media_list] == [
        (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    assert all(prompt.rest_time == dcp.rest_time for prompt in dcp.prompt_plan)


def test_a_protocol_shuffles_the_prompts_and_jitters_the_rests(tmp_path, dpg,
                                                               monkeypatch):
    from libemg.protocol import PROTOCOL_NAME
    protocol = _protocol(factors={"limb_position": ["neutral", "elevated"]})
    protocol.timing.rest_jitter = 1.0
    dcp = _panel(_StreamingHandler(), tmp_path, dpg, protocol=protocol)

    media_list = _gather(dcp, tmp_path, monkeypatch)

    orders = [[m[2] for m in media_list if m[3] == rep] for rep in range(2)]
    assert all(sorted(order) == [0, 1, 2] for order in orders), "every class, each rep"
    assert orders[0] != orders[1], "and not in the same order"
    assert len({prompt.rest_time for prompt in dcp.prompt_plan}) > 1
    # The protocol is kept with the data it was collected under.
    assert (tmp_path / PROTOCOL_NAME).exists()


def test_the_protocols_timings_are_what_the_window_opens_on(tmp_path, dpg):
    protocol = _protocol(num_reps=7)
    protocol.timing.rep_time, protocol.timing.rest_time = 4.5, 2.5

    dcp = DataCollectionPanel(_StreamingHandler(), num_reps=1, rest_time=0,
                              protocol=protocol)

    assert (dcp.num_reps, dcp.rep_time, dcp.rest_time) == (7, 4.5, 2.5)


def test_a_protocol_can_be_handed_over_as_a_path(tmp_path, dpg):
    path = _protocol().save(tmp_path / "study.yaml")
    dcp = DataCollectionPanel(_StreamingHandler(), protocol=str(path))
    assert dcp.protocol.name == "study"


def test_the_factor_level_reaches_the_prompt_and_the_annotations(tmp_path, dpg):
    protocol = _protocol(factors={"limb_position": ["neutral", "elevated"]})
    dcp = _panel(_StreamingHandler(), tmp_path, dpg, protocol=protocol, rest_time=0.05)
    available = _media_list(n_classes=2, rep_time=0.1)
    dcp.prompt_plan = protocol.schedule(["class_0", "class_1"], num_reps=2)
    # One entry per prompt, in the plan's order, as gather_media builds it.
    media_list = [[available[prompt.motion][0], prompt.name, prompt.motion,
                   prompt.rep, 0.1] for prompt in dcp.prompt_plan]

    dcp.run_sgt(media_list)

    session = _annotations(tmp_path)
    prompts = session.select(kinds=(PROMPT,))
    assert {stage.factors["limb_position"] for stage in prompts} == {"neutral",
                                                                     "elevated"}
    for stage in prompts:
        assert stage.factors["limb_position"] == ("neutral" if stage.rep == 0
                                                  else "elevated")
    # The rests carry the level too, so a rest cut out of the session is not
    # left unattributable.
    assert all(stage.factors for stage in session.select(kinds=(REST,)))
    # And the subject was told which level to be in.
    shown = [call.kwargs.get("value") for call in dpg.set_value.call_args_list]
    assert any(value and "(neutral)" in str(value) for value in shown)


def test_the_jittered_rest_is_the_one_that_gets_annotated(tmp_path, dpg):
    protocol = _protocol(num_reps=1)
    protocol.timing.rest_time, protocol.timing.rest_jitter = 0.05, 0.05
    dcp = _panel(_StreamingHandler(), tmp_path, dpg, protocol=protocol, rest_time=0.05)
    dcp.prompt_plan = protocol.schedule(["class_0", "class_1"], num_reps=1)
    available = _media_list(n_classes=2, rep_time=0.1)
    media_list = [[available[prompt.motion][0], prompt.name, prompt.motion,
                   prompt.rep, 0.1] for prompt in dcp.prompt_plan]

    dcp.run_sgt(media_list)

    rests = _annotations(tmp_path).select(kinds=(REST,))
    assert [stage.duration for stage in rests] == [
        pytest.approx(prompt.rest_time) for prompt in dcp.prompt_plan]


# --- checking the signal before collecting ------------------------------

class _QualityHandler(_StreamingHandler):
    """A handler whose buffer holds whatever the test wants checked."""

    def __init__(self, buffer, rate=1000.0, **kwargs):
        super().__init__(**kwargs)
        self.buffer = buffer
        self.rate = rate
        # Opened a second ago, so the buffer already holds data -- a handler
        # that has delivered nothing yet is a different test.
        self._opened = time.time() - 1.0

    def get_data(self, N=0, filter=True):
        # Counts advance with the clock, so the measured rate is the real one.
        delivered = int((time.time() - self._opened) * self.rate)
        return ({mod: self.buffer for mod in self.modalities},
                {mod: delivered for mod in self.modalities})


def _band_limited(n=4000, centre=90.0, seed=0, fs=1000.0):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(spectrum * np.exp(-((freqs - centre) ** 2) / (2 * 70 ** 2)), n)


def _healthy(channels=4, n=4000):
    return np.column_stack([_band_limited(n, seed=index) for index in range(channels)])


def test_the_signal_check_measures_the_rate_it_is_actually_getting(tmp_path, dpg):
    dcp = _panel(_QualityHandler(_healthy()), tmp_path, dpg)

    rates = dcp.measure_sampling_rates(seconds=0.2)

    assert set(rates) == set(dcp.online_data_handler.modalities)
    assert all(rate > 0 for rate in rates.values())


def test_a_clean_signal_passes_the_check_and_is_recorded(tmp_path, dpg):
    yaml = pytest.importorskip("yaml")
    dcp = _panel(_QualityHandler(_healthy()), tmp_path, dpg)

    reports = dcp.check_signal_quality(seconds=2.0, rate_seconds=0.1)

    assert all(report.ok for report in reports)
    written = yaml.safe_load((tmp_path / "signal_quality.yaml").read_text())
    assert set(written["modalities"][0]) >= {"channels", "problems", "samples"}
    assert written["measured_rates_hz"]


def test_a_dead_channel_is_caught_before_the_first_prompt(tmp_path, dpg):
    buffer = _healthy()
    buffer[:, 1] *= 1e-6
    dcp = _panel(_QualityHandler(buffer), tmp_path, dpg)

    problems = [problem for report in dcp.check_signal_quality(seconds=2.0, rate_seconds=0.1)
                for problem in report.problems]

    assert any("Channel(s) 1" in problem for problem in problems)


def test_start_refuses_a_bad_signal_only_when_asked_to(tmp_path, dpg, monkeypatch):
    buffer = _healthy()
    buffer[:, 1] *= 1e-6
    dcp = _panel(_QualityHandler(buffer), tmp_path, dpg, signal_check="block")
    _fill_configuration(dpg, tmp_path)
    monkeypatch.setattr(dcp, "gather_media", Mock(return_value=[]))
    monkeypatch.setattr(dcp, "spawn_collection_window", Mock())

    with pytest.raises(ConnectionError, match="does not look collectable"):
        dcp.start_callback()

    # The default warns and collects anyway: a heuristic must not be able to
    # cancel a booked session hour.
    dcp.signal_check = "warn"
    dcp.start_callback()
    dcp.spawn_collection_thread.join(timeout=5)


def test_the_signal_check_can_be_turned_off_entirely(tmp_path, dpg, monkeypatch):
    dcp = _panel(_QualityHandler(_healthy()), tmp_path, dpg, signal_check="off")
    _fill_configuration(dpg, tmp_path)
    monkeypatch.setattr(dcp, "gather_media", Mock(return_value=[]))
    monkeypatch.setattr(dcp, "spawn_collection_window", Mock())
    monkeypatch.setattr(dcp, "check_signal_quality",
                        Mock(side_effect=AssertionError("should not be called")))

    dcp.start_callback()
    dcp.spawn_collection_thread.join(timeout=5)


def test_a_session_that_shifted_the_spectrum_is_reported_as_fatigue(tmp_path, dpg,
                                                                    capsys):
    dcp = _panel(_StreamingHandler(), tmp_path, dpg, rest_time=0)
    dcp.run_sgt(_media_list(n_classes=1, n_reps=2, rep_time=0.2))
    session = _annotations(tmp_path)

    # The recording the collection produced is a ramp, not EMG, so the shift is
    # imposed here: what is under test is that the first rep is compared with
    # the last and the result is reported.
    from libemg import signal_quality
    calls = []

    def fake_shift(early, late, fs):
        calls.append((len(early), len(late), fs))
        return -0.3, "Median frequency fell 30%"

    original = signal_quality.fatigue_shift
    signal_quality.fatigue_shift = fake_shift
    try:
        shift = dcp.report_fatigue(session)
    finally:
        signal_quality.fatigue_shift = original

    assert shift == -0.3
    assert calls and calls[0][0] > 0 and calls[0][1] > 0
    assert "Median frequency fell 30%" in capsys.readouterr().out


def test_fatigue_is_not_reported_when_there_is_nothing_to_compare(tmp_path, dpg):
    dcp = _panel(_StreamingHandler(), tmp_path, dpg, rest_time=0)
    dcp.run_sgt(_media_list(n_classes=2, n_reps=1, rep_time=0.1))
    session = _annotations(tmp_path)

    assert dcp.report_fatigue(session) is None, "one rep is not a comparison"
    assert dcp.report_fatigue(session, modality="not_a_modality") is None
