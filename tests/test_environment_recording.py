"""Coverage for recording an environment and reading the run back for training.

A run is recorded in two halves -- frames from the environment, EMG from the
online data handler -- and joined by when each EMG sample was produced. What
matters is that every sample ends up beside the frame that was on screen when
it was produced, and that the result loads straight into an
OfflineDataHandler. The join is checked against EMG with a known clock, so the
right frame for every sample is known exactly rather than approximately.

The environments are run headless, a frame at a time, from a scripted
controller, so what they record can be checked against what they were sent.
"""
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from libemg.adaptation._base import ciil_direction
from libemg.environments._base import PredictionMap
from libemg.environments.controllers import ClassifierController, RegressorController
from libemg.environments.recording import (EMG_NAME, FRAMES_NAME, MANIFEST_NAME,
                                           EnvironmentRecording, FrameLog,
                                           RecordingConfig, assemble, load_recording)

FRAME_START, FRAME_PERIOD, FRAMES = 100.0, 0.02, 60          # 50 fps, 1.2 s
EMG_FS, CHANNELS = 1000.0, 4


# ======================================================================
# helpers
# ======================================================================
def _frames(folder, **config):
    """60 frames, 10 per trial, each recording its own index as the prediction."""
    log = FrameLog()
    for k in range(FRAMES):
        log.add(FRAME_START + k * FRAME_PERIOD, k // 10,
                model={"prediction": k, "speed": 1.0},
                game={"cursor_x": 10 * k, "in_target": k % 2 == 0},
                adaptation={"optimal": [1.0, -1.0], "ciil_class": None if k == 0 else 3})
    config = RecordingConfig(folder=str(folder), **config)
    log.save(config, "Synthetic", {"actions": {"up": 1}})
    return config


def _emg(folder, start=FRAME_START - 0.1, seconds=1.6, batch=20):
    """EMG with an exact clock: each batch is read the instant its last sample is produced.

    That makes the production-clock fit exact, so the frame every sample belongs
    to is known. Starts before the first frame and runs past the last, so the
    join has to drop both ends.
    """
    n = int(seconds * EMG_FS)
    produced = start + 0.0005 + np.arange(n) / EMG_FS       # never exactly on a frame
    arrivals = np.empty(n)
    for a in range(0, n, batch):
        arrivals[a:a + batch] = produced[min(n, a + batch) - 1]
    samples = np.random.default_rng(0).normal(size=(n, CHANNELS))
    np.savetxt(folder / EMG_NAME, np.hstack([arrivals[:, None], samples]), delimiter=",")
    return produced, samples


class ScriptedClassifier(ClassifierController):
    """A classifier controller that reads from a list instead of a socket."""

    def __init__(self, messages, num_classes=5):
        super().__init__("predictions", num_classes, port=0)
        self.messages = list(messages)

    def _get_action(self):
        return self.messages.pop(0) if self.messages else None


class ScriptedRegressor(RegressorController):
    def __init__(self, messages):
        super().__init__(port=0)
        self.messages = list(messages)

    def _get_action(self):
        return self.messages.pop(0) if self.messages else None


@pytest.fixture
def pygame_module():
    pygame = pytest.importorskip("pygame")
    try:
        pygame.init()
        pygame.font.init()
    except Exception as error:      # pragma: no cover - a machine without SDL
        pytest.skip(f"pygame cannot initialise here: {error}")
    yield pygame
    pygame.quit()


def _play(environment, frames):
    environment.game_setup()
    for _ in range(frames):
        environment._run_loop()
        if environment.done:
            break
    environment.save_results()


# ======================================================================
# 1. the frame log
# ======================================================================
def test_the_frame_log_spreads_sequences_and_keeps_only_the_groups_asked_for(tmp_path):
    _frames(tmp_path, game=False)
    frames = pd.read_csv(tmp_path / FRAMES_NAME)

    assert list(frames.columns) == ["time", "trial", "prediction", "speed",
                                    "optimal_0", "optimal_1", "ciil_class"]
    assert len(frames) == FRAMES
    assert np.isnan(frames["ciil_class"][0]), "None is recorded as missing, not as zero"
    assert frames["time"][1] - frames["time"][0] == pytest.approx(FRAME_PERIOD)

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["environment"] == "Synthetic"
    assert manifest["frames"]["columns"]["prediction"] == "model"
    assert manifest["frames"]["columns"]["optimal_0"] == "adaptation"
    assert manifest["details"] == {"actions": {"up": 1}}


# ======================================================================
# 2. joining EMG to frames
# ======================================================================
def test_every_emg_sample_gets_the_frame_on_screen_when_it_was_produced(tmp_path):
    _frames(tmp_path)
    produced, samples = _emg(tmp_path)

    written = assemble(tmp_path)

    assert [p.name for p in written] == [f"trial_{t}.csv" for t in range(6)]
    table = pd.concat([pd.read_csv(p) for p in written], ignore_index=True)
    expected_frame = np.floor((table["timestamp"] - FRAME_START) / FRAME_PERIOD).astype(int)
    assert (table["prediction"] == expected_frame).all()
    assert (table["trial"] == expected_frame // 10).all()

    # Nothing from before the first frame or after the last one.
    inside = (produced >= FRAME_START) & (produced <= FRAME_START + FRAMES * FRAME_PERIOD)
    assert len(table) == inside.sum()
    np.testing.assert_allclose(table["timestamp"], produced[inside], atol=1e-6)
    np.testing.assert_allclose(table[[f"emg_{c}" for c in range(CHANNELS)]].to_numpy(),
                               samples[inside])

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["emg"]["fs"] == pytest.approx(EMG_FS, rel=1e-6)
    assert [t["trial"] for t in manifest["trials"]] == list(range(6))
    assert manifest["trial_files"]["data_columns"] == [1, 2, 3, 4]


def test_a_run_loads_into_an_offline_data_handler_ready_for_windowing(tmp_path):
    _frames(tmp_path)
    _emg(tmp_path)
    assemble(tmp_path)

    odh = load_recording(tmp_path)

    assert len(odh.data) == 6
    assert all(trial.shape[1] == CHANNELS for trial in odh.data)
    assert {"trial", "timestamp", "prediction", "speed", "cursor_x", "in_target",
            "optimal_0", "optimal_1", "ciil_class"} == set(odh.extra_attributes)
    for i, trial in enumerate(odh.data):
        assert odh.prediction[i].shape == (trial.shape[0], 1)
        assert len(np.unique(odh.trial[i])) == 1, "one file, one trial"
    assert sorted(int(t[0, 0]) for t in odh.trial) == list(range(6))

    windows, metadata = odh.parse_windows(50, 10, metadata_operations={"prediction": "last_sample"})
    assert windows.shape[1:] == (CHANNELS, 50)
    assert len(metadata["prediction"]) == len(windows)

    only = load_recording(tmp_path, fields=["ciil_class"])
    assert only.extra_attributes == ["ciil_class"]


def test_assembling_again_gives_the_same_files(tmp_path):
    _frames(tmp_path)
    _emg(tmp_path)
    first = [p.read_text() for p in assemble(tmp_path)]
    second = [p.read_text() for p in assemble(tmp_path)]

    assert first == second


def test_the_checkboxes_decide_the_columns(tmp_path):
    _frames(tmp_path, timestamps=False, model_outputs=False, adaptation=False)
    _emg(tmp_path)

    columns = list(pd.read_csv(assemble(tmp_path)[0]).columns)

    assert columns == [f"emg_{c}" for c in range(CHANNELS)] + ["trial", "cursor_x", "in_target"]


def test_without_emg_a_run_is_one_row_per_frame_and_says_it_cannot_be_trained_on(tmp_path):
    _frames(tmp_path, emg=False)
    _emg(tmp_path)                  # on disk, but the run said not to use it

    table = pd.read_csv(assemble(tmp_path)[0])

    assert len(table) == 10
    assert table["timestamp"][0] == pytest.approx(FRAME_START)
    with pytest.raises(ValueError, match="without EMG"):
        load_recording(tmp_path)


def test_a_run_that_recorded_no_frames_writes_no_trials(tmp_path):
    FrameLog().save(RecordingConfig(folder=str(tmp_path)), "Synthetic")

    assert assemble(tmp_path) == []
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())["trials"] == []


def test_the_recording_logs_emg_around_the_run_and_assembles_when_stopped(tmp_path):
    class LoggingHandler:
        """Stands in for an OnlineDataHandler: writes the EMG file when the log stops."""

        def start_log(self, file_path):
            self.prefix = file_path

        def stop_log(self):
            _emg(tmp_path / "run")
            assert self.prefix == str(tmp_path / "run" / "_session_")

    config = RecordingConfig(folder=str(tmp_path / "run"))
    with EnvironmentRecording(config, LoggingHandler()) as recording:
        assert recording.records_emg
        _frames(tmp_path / "run")

    assert len(list((tmp_path / "run").glob("trial_*.csv"))) == 6
    assert not EnvironmentRecording(config).records_emg, "no handler, no EMG"


# ======================================================================
# 3. the CIIL label for a discrete controller
# ======================================================================
@pytest.mark.parametrize("to_target, in_target, moving, label", [
    ([50, 10], True, "E", "NM"),          # in the target: dwell
    ([50, 10], False, "N", "N"),          # up helps (target is above), so it stands
    ([50, 10], False, "E", "E"),
    ([50, 10], False, "W", "E"),          # left hurts: the larger gap is to the right
    ([50, 10], False, "NM", "E"),         # standing still never helps outside the target
    ([-5, -80], False, "S", "S"),
    ([-5, -80], False, "E", "S"),
    ([-5, -80], False, None, "S"),
])
def test_ciil_keeps_any_direction_that_helps_and_otherwise_takes_the_longer_gap(
        to_target, in_target, moving, label):
    assert ciil_direction(to_target, in_target, moving) == label


# ======================================================================
# 4. prediction maps
# ======================================================================
def test_a_prediction_map_sends_what_it_does_not_list_to_its_default():
    mapping = PredictionMap.from_actions({"N": 8, "S": 7, "E": 1, "W": -1, "NM": 0}, default="NM")

    assert dict(mapping) == {8: "N", 7: "S", 1: "E", 0: "NM"}
    assert mapping[11] == "NM" and mapping[8.0] == "N"
    restored = pickle.loads(pickle.dumps(mapping))
    assert restored == mapping and restored[42] == "NM"


# ======================================================================
# 5. environments, run a frame at a time
# ======================================================================
def test_fitts_records_what_it_was_sent_where_things_were_and_the_ciil_class(tmp_path,
                                                                            pygame_module):
    from libemg.environments.fitts import Fitts, FittsConfig

    np.random.seed(3)
    # Class 7 is outside the default map: it has to hold the cursor still, not end the task.
    messages = ["3 1"] * 30 + ["7 1"] * 10 + ["1 0.5"] * 30
    config = RecordingConfig(folder=str(tmp_path))
    fitts = Fitts(ScriptedClassifier(messages, num_classes=8),
                  FittsConfig(num_trials=50, width=600, height=400, velocity=4), recording=config)

    _play(fitts, len(messages))

    frames = pd.read_csv(tmp_path / FRAMES_NAME)
    assert len(frames) == len(messages)
    assert list(frames["prediction"][[0, 30, 40]]) == [3, 7, 1]
    assert list(frames["speed"][[0, 40]]) == [1.0, 0.5]
    assert (frames["velocity_x"][1:30] > 0).all(), "class 3 is right"
    assert (frames[["velocity_x", "velocity_y"]][30:40] == 0).all().all(), \
        "an unmapped class holds the cursor still"

    # optimal points from cursor to target, with y up as the model's 'N' is.
    dx = frames["target_x"] - frames["cursor_x"]
    dy = frames["cursor_y"] - frames["target_y"]
    outside = frames["in_target"] == 0
    assert (np.sign(frames["optimal_x"][outside]) == np.sign(dx[outside])).all()
    assert (np.sign(frames["optimal_y"][outside]) == np.sign(dy[outside])).all()
    np.testing.assert_allclose(np.hypot(frames["optimal_x"], frames["optimal_y"])[outside], 1)

    command = {3: "E", 7: "NM", 1: "N"}
    class_of = {"S": 0, "N": 1, "NM": 2, "E": 3, "W": 4}
    for i, row in frames.iterrows():
        label = ciil_direction([dx[i], dy[i]], bool(row["in_target"]), command[row["prediction"]])
        assert row["ciil_class"] == class_of[label]

    details = json.loads((tmp_path / MANIFEST_NAME).read_text())["details"]
    assert details["actions"] == {"down": 0, "up": 1, "no motion": 2, "right": 3, "left": 4}
    assert details["controller"] == "ScriptedClassifier"


def test_fitts_from_a_regressor_records_its_outputs_and_the_tolerant_ciil_label(tmp_path,
                                                                               pygame_module):
    from libemg.environments.fitts import Fitts, FittsConfig

    np.random.seed(5)
    messages = ["0.5 -0.25 1730000000.0"] * 20
    fitts = Fitts(ScriptedRegressor(messages), FittsConfig(num_trials=50),
                  recording=RecordingConfig(folder=str(tmp_path)))

    _play(fitts, len(messages))

    frames = pd.read_csv(tmp_path / FRAMES_NAME)
    assert (frames["output_x"] == 0.5).all() and (frames["output_y"] == -0.25).all()
    assert "prediction" not in frames and "ciil_class" not in frames
    assert np.isfinite(frames[["ciil_x", "ciil_y"]]).all().all()


def test_fitts_without_a_recording_records_nothing(tmp_path, pygame_module):
    from libemg.environments.fitts import Fitts, FittsConfig

    fitts = Fitts(ScriptedClassifier(["3 1"] * 5), FittsConfig(num_trials=5))
    _play(fitts, 5)

    assert fitts.frame_log is None
    assert list(tmp_path.iterdir()) == []


def test_emg_hero_records_the_lane_and_the_lane_it_should_have_played(tmp_path, pygame_module):
    from libemg.environments.emg_hero import EMGHero, _Note

    hero = EMGHero(ScriptedClassifier(["3"] * 3), recording=RecordingConfig(folder=str(tmp_path)))
    hero.game_setup()
    note = _Note(2)
    note.y_pos, note.length = 520, 10       # on the hit line, and the speed keeps it there a frame
    hero.notes = [note]
    hero.last_note = float("inf")           # no new notes
    hero._run_loop()
    hero.notes = []
    hero._run_loop()
    hero.save_results()

    frames = pd.read_csv(tmp_path / FRAMES_NAME)
    assert list(frames["lane"]) == [2, 2], "class 3 plays lane 3 (index 2) by default"
    assert list(frames["note_2"]) == [1, 0]
    assert list(frames["ciil_lane"]) == [2, -1]
    assert list(frames["ciil_class"]) == [3, 2], "nothing to play is class 2 by default"


def test_curricular_fitts_finishes_a_trial_without_a_save_file(tmp_path, pygame_module):
    from libemg.environments.curricular_fitts import CurricularFitts, CurricularFittsConfig

    # A timeout this short ends the first trial within a few frames. Ending a
    # trial with no save_file used to fail on None + "/".
    config = CurricularFittsConfig(num_trials=3, default_timeout=0.01)
    task = CurricularFitts(ScriptedRegressor(["0.1 0.2 1.0"] * 400), config,
                           recording=RecordingConfig(folder=str(tmp_path)))
    task.target_generator.save_yields = lambda filename=None: None   # keep the test folder clean
    task.game_setup()
    import time as _time
    deadline = _time.monotonic() + 10
    while task.trial_number < 3 and _time.monotonic() < deadline:
        task._run_loop()
        _time.sleep(0.002)
    task.save_results()

    frames = pd.read_csv(tmp_path / FRAMES_NAME)
    assert task.trial_number >= 3
    assert set(frames["trial"]) >= {1, 2}
    assert (frames["output_x"].dropna() == 0.1).all()
    assert {"cursor_x", "target_radius", "optimal_x", "ciil_x"} <= set(frames.columns)
