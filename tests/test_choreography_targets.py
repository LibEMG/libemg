"""Regression targets for the 3D hand choreography: position, intensity, per-sample files.

Runs on the fake clock from test_choreography -- no window, no device.
"""
import json

import numpy as np
import pytest

from libemg.choreography import ChoreographyConfig
from libemg.choreography import annotations as notes
from libemg.choreography import targets as tg
from libemg.choreography.config import IN_TURN, SHUFFLE
from libemg.choreography.controller import ChoreographyController
from libemg.choreography.schedule import (COMPLETED, CYCLE_PAUSE, HOLD, PAUSED, PAUSED_STATE,
                                          RAMP_IN, RAMP_OUT, REST, RUNNING, WARMUP, build_plan,
                                          position_for)
from libemg.session import ModalityClock

from tests.test_choreography import (ASSETS, WALL_OFFSET, FakeClock, _preview_run, _recording,
                                     config, phase_intervals, run_to_end)


def _advance_to(controller, clock, at, step=0.01):
    """Tick until ``at`` seconds into the plan."""
    while controller.scheduler.elapsed(clock.t) < at - 1e-9:
        clock.advance(min(step, at - controller.scheduler.elapsed(clock.t)))
        controller.tick()


def _timed(**kwargs):
    # warm-up 2, ramp-in 0.5, hold 1, ramp-out 0.5, rest 1.5
    kwargs.setdefault("gestures", ["hand_closed"])
    kwargs.setdefault("repetitions", 2)
    return config(**kwargs)


# --- position --------------------------------------------------------------------------

def test_position_rises_linearly_holds_at_one_and_falls_linearly():
    assert [position_for(RAMP_IN, p) for p in (0, 0.25, 0.5, 1)] == [0, 0.25, 0.5, 1]
    assert position_for(HOLD, 0.3) == 1.0
    assert [position_for(RAMP_OUT, p) for p in (0, 0.25, 1)] == [1, 0.75, 0]
    for resting in (WARMUP, REST, CYCLE_PAUSE, None):
        assert position_for(resting, 0.5) == 0.0
    assert position_for(RAMP_IN, 1.7) == 1.0 and position_for(RAMP_IN, -1) == 0.0


def test_the_view_reports_position_through_a_repetition():
    clock = FakeClock()
    controller = ChoreographyController(ASSETS, _timed(), clock=clock, wall=clock.wall)
    controller.start()
    expected = {1.0: 0.0,                     # warm-up
                2.125: 0.25, 2.25: 0.5,       # a quarter and half of the 0.5 s ramp-in
                2.9: 1.0,                     # hold
                3.625: 0.75,                  # a quarter into the ramp-out
                4.5: 0.0}                     # rest
    for at, position in expected.items():
        _advance_to(controller, clock, at)
        assert controller.view().position == pytest.approx(position, abs=1e-9), at
        assert controller.view().intensity is None


def test_a_pause_freezes_the_position():
    clock = FakeClock()
    controller = ChoreographyController(ASSETS, _timed(), clock=clock, wall=clock.wall)
    controller.start()
    _advance_to(controller, clock, 2.2)       # 40 % into the ramp-in
    controller.pause()
    clock.advance(3.0)
    controller.tick()
    assert controller.view().position == pytest.approx(0.4)
    controller.resume()
    _advance_to(controller, clock, 2.35)
    assert controller.view().position == pytest.approx(0.7)


# --- intensity settings -------------------------------------------------------------------

def test_intensity_is_off_by_default_and_old_presets_load(tmp_path):
    cfg = ChoreographyConfig(gestures=["hand_closed"])
    assert cfg.intensity_enabled is False and cfg.intensities_for("hand_closed") == [None] * 3
    old = cfg.to_dict()
    for key in ("intensity_enabled", "intensity_levels", "intensity_order"):
        old.pop(key)
    old["schema_version"] = 1
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(old))
    loaded = ChoreographyConfig.load(path)
    assert loaded.intensity_enabled is False and loaded.validate(ASSETS.gesture_ids) == []


def test_levels_are_dealt_in_turn_and_unequal_counts_stay_as_even_as_possible():
    cfg = config(intensity_enabled=True, intensity_levels=[0.3, 0.6, 0.9], repetitions=5,
                 repetition_overrides={"index_flex": 3})
    assert cfg.intensities_for("hand_closed") == [0.3, 0.6, 0.9, 0.3, 0.6]
    assert cfg.intensities_for("index_flex") == [0.3, 0.6, 0.9]


def test_a_shuffled_deal_is_seeded_balanced_and_per_gesture():
    cfg = config(intensity_enabled=True, intensity_levels=[0.25, 0.5, 0.75], repetitions=6,
                 intensity_order=SHUFFLE, shuffle_seed=7)
    first = cfg.intensities_for("hand_closed")
    assert first == cfg.snapshot().intensities_for("hand_closed"), "reproducible"
    assert sorted(first) == [0.25, 0.25, 0.5, 0.5, 0.75, 0.75], "balanced"
    other = config(intensity_enabled=True, intensity_levels=[0.25, 0.5, 0.75], repetitions=6,
                   intensity_order=SHUFFLE, shuffle_seed=7, gestures=["hand_closed"])
    assert other.intensities_for("hand_closed") == first, "other gestures do not reshuffle it"


@pytest.mark.parametrize("levels, order, message", [
    ([], IN_TURN, "at least one intensity level"),
    ([0.0], IN_TURN, "above 0%"),
    ([1.2], IN_TURN, "at most 100%"),
    ([float("nan")], IN_TURN, "above 0%"),
    ([0.5], "sideways", "Intensity order"),
])
def test_bad_intensity_settings_are_refused(levels, order, message):
    cfg = config(intensity_enabled=True, intensity_levels=levels, intensity_order=order)
    assert any(message in problem for problem in cfg.validate(ASSETS.gesture_ids))
    # ...but only when intensity is on.
    assert config(intensity_levels=levels, intensity_order=order).validate(ASSETS.gesture_ids) == []


def test_the_plan_gives_each_repetition_its_level_and_each_rest_the_next_one():
    plan = build_plan(_timed(intensity_enabled=True, intensity_levels=[0.4, 0.8]),
                      ASSETS.gesture_ids)
    assert [r.intensity for r in plan.repetitions] == [0.4, 0.8]
    by_phase = [(p.phase, p.intensity) for p in plan.phases]
    assert by_phase == [(WARMUP, 0.4), (RAMP_IN, 0.4), (HOLD, 0.4), (RAMP_OUT, 0.4),
                        (REST, 0.8), (RAMP_IN, 0.8), (HOLD, 0.8), (RAMP_OUT, 0.8)]
    assert plan.to_dict()["intensity_levels"] == [0.4, 0.8]


def test_the_view_reports_intensity_as_position_times_level():
    clock = FakeClock()
    controller = ChoreographyController(
        ASSETS, _timed(intensity_enabled=True, intensity_levels=[0.4, 0.8]),
        clock=clock, wall=clock.wall)
    controller.start()
    _advance_to(controller, clock, 1.0)
    view = controller.view()
    assert view.intensity == 0.0 and view.intensity_level == 0.4, "the warm-up previews it"
    _advance_to(controller, clock, 2.25)
    assert controller.view().intensity == pytest.approx(0.2)
    _advance_to(controller, clock, 3.0)
    assert controller.view().intensity == pytest.approx(0.4)
    _advance_to(controller, clock, 5.75)            # half-way up the second ramp-in (5.5-6.0)
    view = controller.view()
    assert (view.phase, view.intensity_level) == (RAMP_IN, 0.8)
    assert view.intensity == pytest.approx(0.4)


# --- intervals ---------------------------------------------------------------------------

def test_every_interval_carries_its_position_and_intensity_at_both_ends():
    clock = FakeClock()
    _, result = _preview_run(clock, _timed(intensity_enabled=True, intensity_levels=[0.5, 1.0]))
    phases = phase_intervals(result.intervals)
    ends = [(iv["phase"], iv["position_start"], iv["position_end"], iv["intensity_level"],
             iv["intensity_start"], iv["intensity_end"]) for iv in phases]
    approx = [(p, pytest.approx(a), pytest.approx(b), lvl, pytest.approx(c), pytest.approx(d))
              for p, a, b, lvl, c, d in [
                  (WARMUP, 0, 0, None, 0, 0),
                  (RAMP_IN, 0, 1, 0.5, 0, 0.5), (HOLD, 1, 1, 0.5, 0.5, 0.5),
                  (RAMP_OUT, 1, 0, 0.5, 0.5, 0),
                  (REST, 0, 0, None, 0, 0),
                  (RAMP_IN, 0, 1, 1.0, 0, 1), (HOLD, 1, 1, 1.0, 1, 1),
                  (RAMP_OUT, 1, 0, 1.0, 1, 0)]]
    assert ends == approx
    assert phases[0]["next_intensity_level"] == 0.5 and phases[4]["next_intensity_level"] == 1.0
    parents = [iv for iv in result.intervals if iv["track"] == notes.REPETITION_TRACK]
    assert [p["intensity_level"] for p in parents] == [0.5, 1.0]
    assert all(p["position_start"] is None for p in parents)


def test_without_intensity_the_intensity_fields_stay_empty():
    clock = FakeClock()
    _, result = _preview_run(clock, _timed())
    for iv in phase_intervals(result.intervals):
        assert iv["intensity_level"] is None and iv["intensity_start"] is None
        assert iv["position_start"] is not None


def test_a_pause_splits_a_ramp_into_segments_that_meet_at_the_frozen_position():
    clock = FakeClock()
    cfg = _timed(repetitions=1, intensity_enabled=True, intensity_levels=[0.6])
    controller = ChoreographyController(ASSETS, cfg, clock=clock, wall=clock.wall)
    controller.start()
    _advance_to(controller, clock, 2.2)
    controller.pause()
    clock.advance(2.0)
    controller.tick()
    controller.resume()
    run_to_end(controller, clock, step=0.01)
    phases = phase_intervals(controller.finalize().intervals)
    ramp = [iv for iv in phases if iv["phase"] in (RAMP_IN, PAUSED)]
    assert [iv["phase"] for iv in ramp] == [RAMP_IN, PAUSED, RAMP_IN]
    first, pause, second = ramp
    assert first["position_start"] == pytest.approx(0.0)
    assert first["position_end"] == pytest.approx(0.4)
    assert pause["position_start"] == pause["position_end"] == pytest.approx(0.4)
    assert pause["intensity_start"] == pytest.approx(0.24) and not pause["training_eligible"]
    assert second["position_start"] == pytest.approx(0.4)
    assert second["position_end"] == pytest.approx(1.0)
    assert second["intensity_end"] == pytest.approx(0.6)


# --- per-sample targets -----------------------------------------------------------------

def _clocked(intervals, fs=1000.0):
    for iv in intervals:
        iv["start_collection_time"] = iv["start_monotonic"] + WALL_OFFSET
        iv["end_collection_time"] = iv["end_monotonic"] + WALL_OFFSET
    clock = ModalityClock(fs=fs, anchor=WALL_OFFSET + 99.5, n_samples=int(14 * fs))
    notes.assign_sample_indices(intervals, {"emg": clock})
    return clock


def test_per_sample_targets_follow_the_intervals_exactly():
    clock = FakeClock()
    controller, result = _preview_run(clock, _timed(intensity_enabled=True,
                                                    intensity_levels=[0.5, 1.0]))
    modality_clock = _clocked(result.intervals)
    order = {r.repetition_id: r.order_index for r in controller.plan.repetitions}
    targets = tg.sample_targets(result.intervals, modality_clock, "emg", order)
    assert len(targets) == modality_clock.n_samples
    # Half a second before the run: outside, rest, not for training.
    assert targets.phase_code[:500].tolist() == [tg.OUTSIDE] * 500
    assert not targets.training_eligible[:500].any()
    t = targets.collection_time - (WALL_OFFSET + 100.0)    # seconds into the plan
    ramp = (t >= 2.0) & (t < 2.5)
    np.testing.assert_allclose(targets.position[ramp], (t[ramp] - 2.0) / 0.5, atol=1e-9)
    assert (targets.motion_class_id[ramp] == ASSETS.class_id("hand_closed")).all()
    assert (targets.phase_code[ramp] == tg.PHASE_CODES[RAMP_IN]).all()
    np.testing.assert_allclose(targets.intensity[ramp], 0.5 * targets.position[ramp])
    hold = (t >= 2.5) & (t < 3.5)
    assert (targets.position[hold] == 1.0).all() and (targets.intensity[hold] == 0.5).all()
    out = (t >= 3.5) & (t < 4.0)
    np.testing.assert_allclose(targets.position[out], 1.0 - (t[out] - 3.5) / 0.5, atol=1e-9)
    rest = (t >= 4.0) & (t < 5.5)
    assert (targets.position[rest] == 0).all() and (targets.motion_class_id[rest] == 0).all()
    assert (targets.repetition_index[rest] == -1).all()
    assert set(targets.repetition_index[(t >= 5.5) & (t < 7.5)]) == {1}
    second = (t >= 6.0) & (t < 7.0)
    assert (targets.intensity[second] == 1.0).all()
    within = (t >= 0) & (t < 7.5)
    assert targets.training_eligible[within].all()


def test_targets_without_intensity_have_no_intensity_columns(tmp_path):
    clock = FakeClock()
    _, result = _preview_run(clock, _timed())
    modality_clock = _clocked(result.intervals)
    written = tg.write_target_files(tmp_path, result.intervals, {"emg": modality_clock})
    path = tmp_path / written["emg"]
    header = path.read_text().splitlines()[0]
    assert header == "# " + ",".join(tg.columns(False))
    back = tg.read_target_file(path)
    assert back.intensity is None and len(back) == modality_clock.n_samples


def test_a_recording_writes_target_files_that_line_up_with_its_samples(tmp_path):
    clock = FakeClock()
    controller, link = _recording(tmp_path, clock, _timed(intensity_enabled=True,
                                                          intensity_levels=[0.5, 1.0]))
    controller.start()
    run_to_end(controller, clock, step=0.02)
    result = controller.finalize()
    record = json.loads(result.record_path.read_text())
    assert record["intensity"]["enabled"] and record["intensity"]["levels"] == [0.5, 1.0]
    assert record["targets"]["columns"] == list(tg.columns(True))
    assert set(record["target_files"]) == {"emg", "imu"}
    for modality in ("emg", "imu"):
        path = tmp_path / record["target_files"][modality]
        table = np.loadtxt(path, delimiter=",")
        assert len(table) == record["sample_clocks"][modality]["n_samples"]
        assert table[:, 0].tolist() == list(range(len(table)))
    # The file and a rebuild from the record agree.
    from_file = tg.load_targets(tmp_path, "emg", with_samples=False)
    (tmp_path / record["target_files"]["emg"]).unlink()
    rebuilt = tg.load_targets(tmp_path, "emg", with_samples=False)
    np.testing.assert_allclose(from_file.position, rebuilt.position, atol=1e-6)
    np.testing.assert_array_equal(from_file.motion_class_id, rebuilt.motion_class_id)
    np.testing.assert_allclose(from_file.intensity, rebuilt.intensity, atol=1e-6)
    # Intensity reaches the session stages as a factor.
    stages, _ = link.closed_with
    holds = [s for s in stages if s.phase == HOLD]
    assert [s.factors for s in holds] == [{"intensity": 0.5}, {"intensity": 1.0}]


def test_repetitions_are_never_cut_into_the_recording_folder(tmp_path):
    with pytest.raises(ValueError, match="outside the recording folder"):
        tg.cut_repetitions(tmp_path, tmp_path / "repetitions")
    with pytest.raises(ValueError, match="outside the recording folder"):
        tg.cut_repetitions(tmp_path, tmp_path)


# --- the displayed motion ------------------------------------------------------------------

def test_the_displayed_hand_moves_linearly_with_the_position_target():
    from libemg.choreography.animation import MotionProfile, clip_time
    from libemg.choreography.gltf import GltfModel
    model = GltfModel.load(ASSETS.glb_path)
    gesture = "hand_closed"
    profile = MotionProfile.measure(model, clips=[gesture])

    def vertices(t):
        world, joints = model.pose(gesture if t is not None else "rest", t or 0.0)
        return np.vstack([model.skin_vertices(p, world, joints)[0] for p in model.primitives])

    rest = vertices(None).astype(float)
    full = vertices(1.0).astype(float) - rest
    total = (full * full).sum()

    def travel(t):
        return float(((vertices(t) - rest) * full).sum() / total)

    authored_gap = 0.0
    for phase in (RAMP_IN, RAMP_OUT):
        for progress in (0.2, 0.5, 0.8):
            nominal = clip_time(phase, progress)
            want = position_for(phase, progress)
            assert travel(profile.clip_time(gesture, nominal)) == pytest.approx(want, abs=0.01)
            authored_gap = max(authored_gap, abs(travel(nominal) - want))
    assert authored_gap > 0.05, "the authored clip is eased; the profile is what makes it linear"
    # The hold and the rest frames are untouched.
    assert profile.clip_time(gesture, 1.2) == 1.2 and profile.clip_time("rest", 0.0) == 0.0
    assert profile.record()["motion_profile"] == "linear"
