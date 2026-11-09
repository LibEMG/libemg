"""The 3D hand choreography: plan, scheduler, annotations, clock mapping and controller.

Everything here runs on a fake clock -- no window, no device, no sleeping.
"""
import json
import math
from collections import Counter

import numpy as np
import pytest

from libemg.choreography import ChoreographyConfig, load_assets
from libemg.choreography import annotations as notes
from libemg.choreography.animation import clip_time, pose_for
from libemg.choreography.assets import EXCLUDED_CLIPS
from libemg.choreography.clock import ClockAnchor, ClockMapping
from libemg.choreography.controller import (PREVIEW, RECORDING, ChoreographyController,
                                            CollectionLink)
from libemg.choreography.schedule import (CANCELLED, COMPLETED, CYCLE_PAUSE, FAILED, HOLD,
                                          PAUSED, PAUSED_STATE, RAMP_IN, RAMP_OUT, REST, RUNNING,
                                          WARMUP, ChoreographyScheduler, build_plan)
from libemg.choreography.config import SHUFFLE
from libemg.session import PAUSE, PROMPT, RAMP_IN as RAMP_IN_KIND, REST as REST_KIND
from libemg.session import ModalityClock

ASSETS = load_assets()
GESTURES = ASSETS.gesture_ids
WALL_OFFSET = 1_700_000_000.0


class FakeClock:
    """A monotonic clock and a wall clock that only move when told to."""

    def __init__(self, start=100.0, wall_offset=WALL_OFFSET):
        self.t = start
        self.wall_offset = wall_offset

    def __call__(self):
        return self.t

    def wall(self):
        return self.t + self.wall_offset

    def advance(self, seconds):
        self.t += seconds
        return self.t


def config(**kwargs):
    kwargs.setdefault("gestures", ["hand_closed", "wrist_flexion", "index_flex"])
    kwargs.setdefault("shuffle_seed", 1234)
    kwargs.setdefault("warmup", 2.0)
    kwargs.setdefault("ramp_in", 0.5)
    kwargs.setdefault("hold", 1.0)
    kwargs.setdefault("ramp_out", 0.5)
    kwargs.setdefault("rest", 1.5)
    kwargs.setdefault("repetitions", 3)
    return ChoreographyConfig(**kwargs)


def run_to_end(scheduler_or_controller, clock, step=0.05, limit=100000):
    for _ in range(limit):
        if scheduler_or_controller.state not in (RUNNING, PAUSED_STATE):
            return
        clock.advance(step)
        scheduler_or_controller.tick(clock.t) if isinstance(
            scheduler_or_controller, ChoreographyScheduler) else scheduler_or_controller.tick()
    raise AssertionError("the run never finished")


def phase_intervals(intervals):
    return [iv for iv in intervals if iv["track"] == notes.PHASE_TRACK]


def repetition_intervals(intervals):
    return [iv for iv in intervals if iv["track"] == notes.REPETITION_TRACK]


# --- the asset ---------------------------------------------------------------------

def test_the_packaged_asset_loads_from_inside_the_package():
    import libemg
    package = __import__("pathlib").Path(libemg.__file__).parent
    assert ASSETS.glb_path.is_file() and ASSETS.manifest_path.is_file()
    assert package in ASSETS.glb_path.parents, "never read from the authoring machine's paths"
    assert len(ASSETS.glb_sha256) == 64


def test_rest_and_the_preview_timeline_are_not_selectable():
    for excluded in EXCLUDED_CLIPS:
        assert excluded not in GESTURES
    assert len(GESTURES) == 16
    assert ASSETS.label("hand_closed") == "Hand Closed"


def test_class_ids_are_fixed_by_the_versioned_map_not_by_selection_order():
    assert ASSETS.class_map[0] == "rest"
    assert ASSETS.class_id("hand_closed") == 1
    assert ASSETS.class_id("wrist_flexion") == 5
    assert ASSETS.class_map_version == "right_hand_choreography.v1"
    reordered = config(gestures=["wrist_flexion", "hand_closed"])
    controller = ChoreographyController(ASSETS, reordered)
    assert controller.class_ids["wrist_flexion"] == 5 and controller.class_ids["hand_closed"] == 1


def test_every_manifest_gesture_has_a_clip_in_the_model():
    from libemg.choreography.gltf import GltfModel
    model = GltfModel.load(ASSETS.glb_path)
    for gesture in GESTURES + ["rest"]:
        assert gesture in model.clips


# --- configuration ------------------------------------------------------------------

@pytest.mark.parametrize("change, message", [
    ({"gestures": []}, "Select at least one gesture"),
    ({"warmup": -1.0}, "cannot be negative"),
    ({"hold": float("nan")}, "finite"),
    ({"rest": float("inf")}, "finite"),
    ({"ramp_in": 0.0}, "Ramp-in must be longer than zero"),
    ({"ramp_out": 0.0}, "Ramp-out must be longer than zero"),
    ({"repetitions": 0}, "at least 1"),
    ({"repetitions": 1.5}, "whole number"),
    ({"repetitions": True}, "whole number"),
    ({"repetition_overrides": {"hand_closed": 0}}, "hand_closed"),
    ({"timing_overrides": {"hand_closed": {"ramp_in": 0}}}, "hand_closed: Ramp-in"),
    ({"timing_overrides": {"hand_closed": {"hold": -2}}}, "hand_closed: Hold cannot"),
    ({"gestures": ["rest"]}, "Not selectable"),
    ({"gestures": ["Pose_Library_Preview"]}, "Not selectable"),
    ({"order_mode": "random"}, "Order must be"),
])
def test_invalid_settings_are_refused_with_a_reason(change, message):
    problems = config(**change).validate(GESTURES)
    assert any(message in p for p in problems), problems
    with pytest.raises(ValueError):
        build_plan(config(**change), GESTURES)


def test_zero_hold_rest_warmup_and_cycle_pause_are_valid():
    assert config(hold=0, rest=0, warmup=0, cycle_pause=0, cycle_pause_enabled=True).validate(GESTURES) == []


def test_presets_round_trip(tmp_path):
    original = config(order_mode=SHUFFLE, repetition_overrides={"hand_closed": 5},
                      timing_overrides={"index_flex": {"hold": 4.0}}, cycle_pause_enabled=True)
    path = original.save(tmp_path / "preset.json")
    loaded = ChoreographyConfig.load(path)
    assert loaded.to_dict() == original.to_dict()
    raw = json.loads(path.read_text())
    raw["surprise"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Unknown"):
        ChoreographyConfig.load(path)


def test_a_snapshot_does_not_follow_later_edits():
    live = config()
    frozen = live.snapshot()
    live.gestures.append("hand_open")
    live.timing_overrides["hand_closed"] = {"hold": 9}
    assert "hand_open" not in frozen.gestures and frozen.timing_overrides == {}


# --- the plan -----------------------------------------------------------------------

def test_three_gestures_three_reps_make_three_cycles_of_one_each():
    plan = build_plan(config(), GESTURES)
    assert plan.cycles == [["hand_closed", "wrist_flexion", "index_flex"]] * 3
    holds = Counter(p.motion_class for p in plan.phases if p.phase == HOLD)
    assert holds == {"hand_closed": 3, "wrist_flexion": 3, "index_flex": 3}
    assert plan.total_repetitions == 9


def test_unequal_counts_are_exact_and_finished_gestures_drop_out():
    plan = build_plan(config(repetition_overrides={"hand_closed": 4, "index_flex": 1}), GESTURES)
    assert plan.cycles == [["hand_closed", "wrist_flexion", "index_flex"],
                           ["hand_closed", "wrist_flexion"],
                           ["hand_closed", "wrist_flexion"],
                           ["hand_closed"]]
    for phase in (RAMP_IN, HOLD, RAMP_OUT):
        counts = Counter(p.motion_class for p in plan.phases if p.phase == phase)
        assert counts == {"hand_closed": 4, "wrist_flexion": 3, "index_flex": 1}
    # Each gesture's repetitions are numbered 0..n-1.
    for gesture, n in plan.requested.items():
        indices = [r.gesture_repetition_index for r in plan.repetitions if r.motion_id == gesture]
        assert indices == list(range(n))


def test_fixed_order_is_the_listed_order_every_cycle():
    plan = build_plan(config(gestures=["index_flex", "hand_closed", "wrist_flexion"]), GESTURES)
    assert all(cycle == ["index_flex", "hand_closed", "wrist_flexion"] for cycle in plan.cycles)


def test_a_seeded_shuffle_is_reproducible_and_recorded():
    many = list(GESTURES)
    first = build_plan(config(gestures=many, order_mode=SHUFFLE, shuffle_seed=7), GESTURES)
    again = build_plan(config(gestures=many, order_mode=SHUFFLE, shuffle_seed=7), GESTURES)
    other = build_plan(config(gestures=many, order_mode=SHUFFLE, shuffle_seed=8), GESTURES)
    assert first.cycles == again.cycles
    assert first.cycles != other.cycles
    assert first.cycles[0] != first.cycles[1], "each cycle is shuffled on its own"
    assert sorted(first.cycles[0]) == sorted(many)
    stored = first.to_dict()
    assert stored["shuffle_seed"] == 7 and stored["cycles"] == first.cycles
    assert [p.phase for p in first.phases] == [p.phase for p in again.phases]


def test_the_phase_sequence_follows_the_repetition_rules():
    plan = build_plan(config(gestures=["hand_closed", "index_flex"], repetitions=2), GESTURES)
    sequence = [(p.phase, p.motion_class) for p in plan.phases]
    assert sequence == [
        (WARMUP, "rest"),
        (RAMP_IN, "hand_closed"), (HOLD, "hand_closed"), (RAMP_OUT, "hand_closed"),
        (REST, "rest"),
        (RAMP_IN, "index_flex"), (HOLD, "index_flex"), (RAMP_OUT, "index_flex"),
        (REST, "rest"),                       # between cycles, no cycle pause configured
        (RAMP_IN, "hand_closed"), (HOLD, "hand_closed"), (RAMP_OUT, "hand_closed"),
        (REST, "rest"),
        (RAMP_IN, "index_flex"), (HOLD, "index_flex"), (RAMP_OUT, "index_flex"),
    ], "warm-up first, rests between, and no trailing rest after the last ramp-out"
    # Contiguous: every phase starts where the last one ended.
    for before, after in zip(plan.phases, plan.phases[1:]):
        assert after.start == before.end
    assert plan.total_duration == pytest.approx(2.0 + 4 * 2.0 + 3 * 1.5)


def test_the_cycle_pause_replaces_the_rest_between_cycles_and_is_not_stacked():
    plan = build_plan(config(gestures=["hand_closed", "index_flex"], repetitions=3,
                             cycle_pause_enabled=True, cycle_pause=7.0), GESTURES)
    between = [p for p in plan.phases if p.phase in (REST, CYCLE_PAUSE)]
    assert [p.phase for p in between] == [REST, CYCLE_PAUSE, REST, CYCLE_PAUSE, REST]
    assert all(p.duration == 7.0 for p in between if p.phase == CYCLE_PAUSE)
    for index, phase in enumerate(plan.phases):
        if phase.phase == CYCLE_PAUSE:
            assert plan.phases[index - 1].phase == RAMP_OUT and plan.phases[index + 1].phase == RAMP_IN


def test_zero_length_phases_are_left_out():
    plan = build_plan(config(warmup=0, hold=0, rest=0, cycle_pause_enabled=True, cycle_pause=0,
                             gestures=["hand_closed", "index_flex"], repetitions=2), GESTURES)
    phases = {p.phase for p in plan.phases}
    assert phases == {RAMP_IN, RAMP_OUT}
    assert all(p.duration > 0 for p in plan.phases)
    assert plan.phases[0].phase == RAMP_IN


def test_per_gesture_timing_overrides_and_the_rest_that_follows_a_gesture():
    plan = build_plan(config(gestures=["hand_closed", "index_flex"], repetitions=1,
                             timing_overrides={"hand_closed": {"hold": 4.0, "rest": 3.0}}),
                      GESTURES)
    holds = {p.motion_class: p.duration for p in plan.phases if p.phase == HOLD}
    assert holds == {"hand_closed": 4.0, "index_flex": 1.0}
    rest = next(p for p in plan.phases if p.phase == REST)
    assert rest.duration == 3.0, "the rest after hand_closed is hand_closed's own"


def test_the_estimate_is_the_plan():
    plan = build_plan(config(repetitions=2), GESTURES)
    assert plan.total_duration == pytest.approx(sum(p.duration for p in plan.phases))


# --- the scheduler -----------------------------------------------------------------

def test_phase_boundaries_land_exactly_where_the_plan_puts_them_however_ticks_fall():
    clock = FakeClock()
    plan = build_plan(config(), GESTURES)
    scheduler = ChoreographyScheduler(plan)
    scheduler.start(clock.t)
    t0 = clock.t
    # Coarse, irregular ticks: boundaries must still be exact.
    rng = np.random.default_rng(0)
    while scheduler.state == RUNNING:
        scheduler.tick(clock.advance(float(rng.uniform(0.01, 0.9))))
    assert scheduler.state == COMPLETED
    starts = [e for e in scheduler.events if e["type"] == "phase_started"]
    ends = [e for e in scheduler.events if e["type"] == "phase_ended"]
    assert len(starts) == len(ends) == len(plan.phases)
    for phase, start, end in zip(plan.phases, starts, ends):
        assert start["monotonic"] == pytest.approx(t0 + phase.start, abs=1e-9)
        assert end["monotonic"] == pytest.approx(t0 + phase.end, abs=1e-9)
        assert end["status"] == "completed"
    finish = scheduler.events[-1]
    assert finish["type"] == "run_completed"
    assert finish["monotonic"] == pytest.approx(t0 + plan.total_duration)
    assert finish["repetitions_completed"] == 9


def test_events_are_ordered_unique_and_never_duplicated():
    clock = FakeClock()
    scheduler = ChoreographyScheduler(build_plan(config(), GESTURES))
    scheduler.start(clock.t)
    run_to_end(scheduler, clock, step=0.013)
    ids = [e["event_id"] for e in scheduler.events]
    assert len(ids) == len(set(ids))
    assert [e["sequence"] for e in scheduler.events] == list(range(1, len(ids) + 1))
    started = Counter((e["plan_index"], e["segment_index"]) for e in scheduler.events
                      if e["type"] == "phase_started")
    assert max(started.values()) == 1


def test_pause_freezes_the_schedule_and_resume_continues_from_the_same_point():
    clock = FakeClock()
    plan = build_plan(config(gestures=["hand_closed"], repetitions=1, warmup=1.0), GESTURES)
    scheduler = ChoreographyScheduler(plan)
    scheduler.start(clock.t)
    t0 = clock.t
    scheduler.tick(clock.advance(1.0 + 0.5 + 0.4))            # 0.4 s into the hold
    before = scheduler.view(clock.t)
    assert before.phase == HOLD and before.progress == pytest.approx(0.4)
    assert scheduler.pause(clock.t)
    clock.advance(5.0)
    scheduler.tick(clock.t)
    frozen = scheduler.view(clock.t)
    assert frozen.display_phase == PAUSED and frozen.phase == HOLD
    assert frozen.progress == pytest.approx(0.4), "nothing moves while paused"
    assert pose_for(frozen) == pose_for(before), "and the hand stays on the same frame"
    scheduler.resume(clock.t)
    run_to_end(scheduler, clock, step=0.01)
    ended = [e for e in scheduler.events if e["type"] == "phase_ended" and e["phase"] == HOLD]
    assert [e["status"] for e in ended] == ["paused", "completed"]
    # The rest of the schedule is shifted by exactly the pause.
    ramp_out = next(e for e in scheduler.events
                    if e["type"] == "phase_started" and e["phase"] == RAMP_OUT)
    assert ramp_out["monotonic"] == pytest.approx(t0 + 2.5 + 5.0)
    assert scheduler.events[-1]["monotonic"] == pytest.approx(t0 + plan.total_duration + 5.0)


def test_stopping_cancels_closes_the_phase_and_shows_rest():
    clock = FakeClock()
    scheduler = ChoreographyScheduler(build_plan(config(), GESTURES))
    scheduler.start(clock.t)
    scheduler.tick(clock.advance(2.2))                         # in the first ramp-in
    assert scheduler.stop(clock.t) == CANCELLED
    ended = [e for e in scheduler.events if e["type"] == "phase_ended"][-1]
    assert ended["phase"] == RAMP_IN and ended["status"] == "cancelled"
    assert scheduler.events[-1]["type"] == "run_cancelled"
    assert pose_for(scheduler.view(clock.t)) == ("rest", 0.0)


def test_stopping_after_the_end_does_not_turn_a_finished_run_into_a_cancelled_one():
    clock = FakeClock()
    plan = build_plan(config(gestures=["hand_closed"], repetitions=1), GESTURES)
    scheduler = ChoreographyScheduler(plan)
    scheduler.start(clock.t)
    clock.advance(plan.total_duration + 3)
    assert scheduler.stop(clock.t) == COMPLETED


def test_a_failed_run_is_distinguished_from_a_cancelled_one():
    clock = FakeClock()
    scheduler = ChoreographyScheduler(build_plan(config(), GESTURES))
    scheduler.start(clock.t)
    scheduler.tick(clock.advance(3.0))
    assert scheduler.stop(clock.t, status=FAILED, reason="render_failure: boom") == FAILED
    assert scheduler.events[-1]["type"] == "run_failed"
    assert scheduler.events[-1]["reason"] == "render_failure: boom"


def test_the_view_reports_cycle_and_repetition_progress():
    clock = FakeClock()
    plan = build_plan(config(repetitions=2), GESTURES)
    scheduler = ChoreographyScheduler(plan)
    scheduler.start(clock.t)
    # Into cycle 2, second gesture's ramp-in: warmup 2 + 3 reps*(2 + rest 1.5) ...
    target = next(p for p in plan.phases if p.phase == RAMP_IN and p.cycle_index == 1
                  and p.motion_class == "wrist_flexion")
    scheduler.tick(clock.advance(target.start + 0.1))
    view = scheduler.view(clock.t)
    assert view.cycle_index == 1 and view.cycle_count == 2
    assert view.motion_class == "wrist_flexion"
    assert view.gesture_repetition_index == 1 and view.gesture_repetition_count == 2
    assert view.repetitions_done == 4 and view.repetitions_total == 6
    assert view.remaining == pytest.approx(0.4)


# --- animation ---------------------------------------------------------------------

def test_phases_map_onto_their_own_clip_intervals():
    assert clip_time(RAMP_IN, 0.0) == pytest.approx(0.3)
    assert clip_time(RAMP_IN, 1.0) == pytest.approx(1.0)
    assert clip_time(HOLD, 0.0) == pytest.approx(1.0)
    assert clip_time(HOLD, 1.0) == pytest.approx(1.5)
    assert clip_time(RAMP_OUT, 0.0) == pytest.approx(1.5)
    assert clip_time(RAMP_OUT, 1.0) == pytest.approx(2.2)
    assert clip_time(REST, 0.7) == 0.0 and clip_time(WARMUP, 0.2) == 0.0


def test_a_longer_hold_does_not_change_how_the_ramps_play():
    clock = FakeClock()
    poses = []
    for hold in (0.5, 6.0):
        plan = build_plan(config(gestures=["hand_closed"], repetitions=1, warmup=1.0, hold=hold),
                          GESTURES)
        scheduler = ChoreographyScheduler(plan)
        start = clock.t
        scheduler.start(start)
        scheduler.tick(clock.advance(1.0 + 0.25))           # half-way through the 0.5 s ramp-in
        poses.append(pose_for(scheduler.view(clock.t)))
        clock.advance(100)
    assert poses[0] == poses[1] == ("hand_closed", pytest.approx(0.65))


def test_resting_phases_show_the_rest_frame():
    clock = FakeClock()
    scheduler = ChoreographyScheduler(build_plan(config(), GESTURES))
    assert pose_for(scheduler.view(clock.t)) == ("rest", 0.0)
    scheduler.start(clock.t)
    scheduler.tick(clock.advance(0.5))                          # warm-up
    assert pose_for(scheduler.view(clock.t)) == ("rest", 0.0)


def test_the_clip_rest_and_target_frames_follow_the_documented_contract():
    from libemg.choreography.gltf import GltfModel
    model = GltfModel.load(ASSETS.glb_path)

    def joints(clip, t):
        world, _ = model.pose(clip, t)
        return world

    rest = joints("rest", 0.0)
    for gesture in ("hand_closed", "wrist_flexion", "thumb_extend"):
        for t in (0.0, 0.3, 2.2, 2.5):
            np.testing.assert_allclose(joints(gesture, t), rest, atol=1e-6)
        assert np.abs(joints(gesture, 1.0) - rest).max() > 1e-3
        np.testing.assert_allclose(joints(gesture, 1.0), joints(gesture, 1.5), atol=1e-6)


# --- intervals ---------------------------------------------------------------------

def _preview_run(clock, cfg=None, actions=None, step=0.02):
    controller = ChoreographyController(ASSETS, cfg or config(), clock=clock, wall=clock.wall)
    controller.start()
    for at, action in sorted((actions or {}).items()):
        while clock.t < controller.scheduler._t0 + at and controller.state in (RUNNING, PAUSED_STATE):
            clock.advance(step)
            controller.tick()
        action(controller)
    run_to_end(controller, clock, step=step)
    return controller, controller.finalize()


def test_derived_labels_and_separate_motion_and_phase_fields():
    clock = FakeClock()
    controller, result = _preview_run(clock, config(gestures=["hand_closed"], repetitions=2,
                                                    cycle_pause_enabled=True, cycle_pause=3))
    labels = [(iv["motion_class"], iv["phase"], iv["derived_label"])
              for iv in phase_intervals(result.intervals)]
    assert labels[:5] == [("rest", "warmup", "rest/warmup"),
                          ("hand_closed", "ramp_in", "hand_closed/ramp_in"),
                          ("hand_closed", "hold", "hand_closed/hold"),
                          ("hand_closed", "ramp_out", "hand_closed/ramp_out"),
                          ("rest", "cycle_pause", "rest/cycle_pause")]
    rest_run = _preview_run(FakeClock(), config(gestures=["hand_closed", "index_flex"],
                                                repetitions=1))[1]
    rest = next(iv for iv in phase_intervals(rest_run.intervals) if iv["phase"] == REST)
    assert (rest["motion_class"], rest["derived_label"]) == ("rest", "rest")
    hold = next(iv for iv in phase_intervals(result.intervals) if iv["phase"] == HOLD)
    assert hold["motion_class_id"] == 1


def test_phase_intervals_tile_the_run_without_overlap_or_gaps():
    clock = FakeClock()
    def pause_and_resume(c):
        c.pause()
        clock.advance(1.7)
        c.tick()
        c.resume()

    _, result = _preview_run(clock, config(), actions={5.3: pause_and_resume})
    assert any(iv["phase"] == PAUSED for iv in phase_intervals(result.intervals))
    phases = phase_intervals(result.intervals)
    for before, after in zip(phases, phases[1:]):
        assert after["start_monotonic"] == pytest.approx(before["end_monotonic"], abs=1e-9)
        assert after["start_monotonic"] >= before["end_monotonic"] - 1e-9


def test_every_repetition_gets_one_ramp_in_hold_ramp_out_and_a_parent_region():
    clock = FakeClock()
    _, result = _preview_run(clock, config(repetition_overrides={"index_flex": 1}))
    phases = phase_intervals(result.intervals)
    parents = repetition_intervals(result.intervals)
    assert len(parents) == 3 + 3 + 1
    for parent in parents:
        children = [iv for iv in phases if iv["parent_repetition_id"] == parent["interval_id"]]
        assert [c["phase"] for c in children] == [RAMP_IN, HOLD, RAMP_OUT]
        assert parent["start_monotonic"] == children[0]["start_monotonic"]
        assert parent["end_monotonic"] == children[-1]["end_monotonic"]
        assert parent["completion_status"] == COMPLETED and parent["training_eligible"]
    for iv in phases:
        assert iv["completion_status"] == "completed" and iv["training_eligible"]


def test_parent_regions_can_be_turned_off():
    _, result = _preview_run(FakeClock(), config(parent_regions=False))
    assert repetition_intervals(result.intervals) == []


def test_a_paused_repetition_is_split_into_linked_segments_and_the_pause_is_excluded():
    clock = FakeClock()
    cfg = config(gestures=["hand_closed"], repetitions=1, warmup=1.0)
    paused_at = 1.0 + 0.5 + 0.3

    def pause(c):
        c.pause()
        clock.advance(4.0)
        c.tick()
        c.resume()

    _, result = _preview_run(clock, cfg, actions={paused_at: pause}, step=0.01)
    phases = phase_intervals(result.intervals)
    holds = [iv for iv in phases if iv["phase"] == HOLD]
    assert [h["segment_index"] for h in holds] == [0, 1]
    assert [h["completion_status"] for h in holds] == ["paused", "completed"]
    assert holds[0]["parent_repetition_id"] == holds[1]["parent_repetition_id"]
    pause_iv = next(iv for iv in phases if iv["phase"] == PAUSED)
    assert pause_iv["training_eligible"] is False
    assert pause_iv["end_monotonic"] - pause_iv["start_monotonic"] == pytest.approx(4.0, abs=0.02)
    assert pause_iv["parent_repetition_id"] == holds[0]["parent_repetition_id"]
    assert holds[0]["end_monotonic"] == pause_iv["start_monotonic"]
    assert holds[1]["start_monotonic"] == pause_iv["end_monotonic"]
    parent = repetition_intervals(result.intervals)[0]
    assert parent["completion_status"] == COMPLETED and parent["pause_count"] == 1
    # The hold's total on-screen time is still the planned hold.
    assert sum(h["end_monotonic"] - h["start_monotonic"] for h in holds) == pytest.approx(1.0)


def test_cancelling_mid_repetition_never_counts_it_as_completed():
    clock = FakeClock()
    cfg = config(gestures=["hand_closed"], repetitions=2, warmup=1.0)
    _, result = _preview_run(clock, cfg, actions={1.2: lambda c: c.stop()})
    assert result.status == CANCELLED and result.reason == "user_stop"
    phases = phase_intervals(result.intervals)
    assert phases[-1]["phase"] == RAMP_IN and phases[-1]["completion_status"] == "cancelled"
    assert not phases[-1]["training_eligible"]
    parents = repetition_intervals(result.intervals)
    assert [p["completion_status"] for p in parents] == [CANCELLED]


def test_cancelling_while_paused_closes_the_pause_without_a_dangling_interval():
    clock = FakeClock()

    def pause_then_stop(c):
        c.pause()
        clock.advance(2.0)
        c.tick()
        c.stop()

    _, result = _preview_run(clock, config(), actions={2.7: pause_then_stop})
    phases = phase_intervals(result.intervals)
    assert phases[-1]["phase"] == PAUSED and phases[-1]["completion_status"] == CANCELLED
    assert all(iv["end_monotonic"] is not None for iv in result.intervals)


def test_intervals_rebuilt_from_a_truncated_event_stream_are_closed_as_interrupted():
    clock = FakeClock()
    scheduler = ChoreographyScheduler(build_plan(config(), GESTURES))
    scheduler.start(clock.t)
    scheduler.tick(clock.advance(3.0))          # mid-hold of the first gesture (2.5-3.5 s)
    # As if the app died here; its journal's last heartbeat was at 3.0 s.
    intervals = notes.build_intervals(scheduler.events, last_alive={"monotonic": clock.t})
    last = phase_intervals(intervals)[-1]
    assert last["phase"] == HOLD and last["completion_status"] == "interrupted"
    assert last["end_monotonic"] == clock.t
    assert last["end_reason"] == "events_ended_without_run_end"
    assert repetition_intervals(intervals)[-1]["completion_status"] == "interrupted"


def test_duplicate_and_unmatched_events_do_not_duplicate_intervals():
    clock = FakeClock()
    scheduler = ChoreographyScheduler(build_plan(config(), GESTURES))
    scheduler.start(clock.t)
    run_to_end(scheduler, clock)
    events = scheduler.events + scheduler.events[3:9]              # replayed
    stray = dict(scheduler.events[4], event_id="stray", sequence=10**6)
    intervals = notes.build_intervals(events + [stray])
    clean = notes.build_intervals(scheduler.events)
    assert len(intervals) == len(clean)
    ids = [iv["interval_id"] for iv in intervals]
    assert len(ids) == len(set(ids))


# --- clocks and samples ------------------------------------------------------------

def test_scheduler_time_maps_explicitly_onto_the_collection_clock():
    clock = FakeClock(start=10.0, wall_offset=5000.0)
    mapping = ClockMapping(monotonic=clock, wall=clock.wall)
    mapping.anchor()
    assert mapping.to_collection(12.5) == pytest.approx(5012.5)
    # The system clock is stepped by +2 s mid-run: later instants follow it.
    clock.advance(10.0)
    clock.wall_offset += 2.0
    mapping.anchor()
    assert mapping.to_collection(15.0) == pytest.approx(5015.0), "before the step"
    assert mapping.to_collection(21.0) == pytest.approx(5023.0), "after the step"
    assert mapping.drift == pytest.approx(2.0)
    assert mapping.to_dict()["anchor_count"] == 2


def test_a_sample_on_a_boundary_belongs_to_the_later_interval():
    clock = ModalityClock(fs=1000.0, anchor=50.0, n_samples=10_000)
    # Sample 2000 is produced at exactly 52.0 s.
    first = notes.sample_bounds(clock, 51.0, 52.0)
    second = notes.sample_bounds(clock, 52.0, 53.0)
    assert first == (1000, 2000) and second == (2000, 3000)
    # Between samples: the first sample at or after the instant.
    assert notes.sample_bounds(clock, 51.0004, 51.0016) == (1001, 1002)
    # Clamped to the recording.
    assert notes.sample_bounds(clock, 40.0, 70.0) == (0, 10_000)


def test_sample_indices_tile_the_recording_like_the_intervals_do():
    clock = FakeClock()
    _, result = _preview_run(clock, config())
    phases = phase_intervals(result.intervals)
    # Pretend these were recorded: map monotonic onto a collection clock.
    for iv in result.intervals:
        iv["start_collection_time"] = iv["start_monotonic"] + 1000.0
        iv["end_collection_time"] = iv["end_monotonic"] + 1000.0
    modality_clock = ModalityClock(fs=512.0, anchor=1000.0 + 100.0 - 0.0137, n_samples=10**6)
    primary = notes.assign_sample_indices(result.intervals, {"emg": modality_clock})
    assert primary == "emg"
    for before, after in zip(phases, phases[1:]):
        assert after["start_sample_index"] == before["end_sample_index"]
    total = sum(iv["end_sample_index"] - iv["start_sample_index"] for iv in phases)
    assert total == phases[-1]["end_sample_index"] - phases[0]["start_sample_index"]


# --- the controller ----------------------------------------------------------------

class FakeLink(CollectionLink):
    """A collection service that counts samples at a fixed rate on the fake clock."""

    def __init__(self, clock, fs=1000.0, modalities=("emg", "imu")):
        self.clock = clock
        self.fs = fs
        self.modalities = list(modalities)
        self.calls = []
        self.alive = True
        self.frozen_at = None
        self.saved = []
        self.closed_with = None
        self.t_open = None

    def open(self, header):
        self.calls.append("open")
        self.t_open = self.clock.t
        self.header = header
        return {"collection_session_id": "session-1", "modalities": self.modalities}

    def _n(self):
        end = self.clock.t if self.frozen_at is None else self.frozen_at
        return int((end - self.t_open) * self.fs)

    def counts(self):
        return {mod: self._n() for mod in self.modalities}

    def logger_alive(self):
        return self.alive

    def unplug(self):
        self.frozen_at = self.clock.t

    def replug(self, fs=None):
        # Counting continues from where it stopped.
        offset = self._n()
        self.frozen_at = None
        self.t_open = self.clock.t - offset / self.fs

    def save_progress(self, stages):
        self.saved.append(len(stages))

    def close(self, stages, metadata):
        self.calls.append("close")
        self.closed_with = (stages, metadata)
        # Sample i is produced at anchor + i / fs, from the moment the log opened,
        # for as long as the run went (a cable pull loses no rows here).
        produced = int((self.clock.t - self.t_open) * self.fs) + 1
        return {mod: ModalityClock(fs=self.fs, anchor=WALL_OFFSET + self.t_open,
                                   n_samples=produced)
                for mod in self.modalities}


def _recording(tmp_path, clock, cfg=None, **kwargs):
    link = FakeLink(clock)
    controller = ChoreographyController(ASSETS, cfg or config(), mode=RECORDING, link=link,
                                        output_folder=tmp_path, clock=clock, wall=clock.wall,
                                        **kwargs)
    return controller, link


def test_preview_records_nothing(tmp_path):
    clock = FakeClock()
    with pytest.raises(ValueError):
        ChoreographyController(ASSETS, config(), mode=RECORDING, clock=clock)
    controller, result = _preview_run(clock)
    assert result.mode == PREVIEW and result.record_path is None
    assert all(iv["start_collection_time"] is None for iv in result.intervals)
    assert list(tmp_path.iterdir()) == []


def test_a_recording_opens_before_the_warmup_and_writes_a_complete_record(tmp_path):
    clock = FakeClock()
    controller, link = _recording(tmp_path, clock)
    controller.start()
    assert link.calls == ["open"]
    run_to_end(controller, clock, step=0.02)
    result = controller.finalize()
    assert link.calls == ["open", "close"]
    assert result.status == COMPLETED
    record = json.loads((tmp_path / notes.RUN_RECORD_NAME).read_text())
    for key in ("schema_version", "run_id", "collection_session_id", "config", "schedule",
                "asset", "class_map", "shuffle_seed", "timing", "clock", "intervals", "events"):
        assert key in record, key
    assert record["asset"]["glb_sha256"] == ASSETS.glb_sha256
    assert record["class_map"]["version"] == "right_hand_choreography.v1"
    assert record["interval_convention"].startswith("half-open")
    required = ("schema_version", "run_id", "collection_session_id", "interval_id",
                "parent_repetition_id", "motion_class", "phase", "cycle_index",
                "gesture_repetition_index", "start_collection_time", "end_collection_time",
                "start_sample_index", "end_sample_index", "completion_status")
    for interval in record["intervals"]:
        for key in required:
            assert key in interval, key
        assert interval["collection_session_id"] == "session-1"
        assert interval["start_sample_index"] is not None
    # Collection times are the logger's clock, not the scheduler's.
    first = record["intervals"][0]
    assert first["start_collection_time"] == pytest.approx(first["start_monotonic"] + WALL_OFFSET)
    # The journal holds the header and every event.
    header, events, footer = notes.read_journal(tmp_path / notes.JOURNAL_NAME)
    assert header["run_id"] == record["run_id"] and footer["status"] == COMPLETED
    assert [e["event_id"] for e in events] == [e["event_id"] for e in record["events"]]
    assert link.saved, "annotation progress was saved while the run went"


def test_recorded_stages_feed_the_existing_session_format(tmp_path):
    clock = FakeClock()
    controller, link = _recording(tmp_path, clock, config(gestures=["hand_closed", "index_flex"],
                                                          repetitions=2))
    controller.start()
    run_to_end(controller, clock)
    controller.finalize()
    stages, metadata = link.closed_with
    kinds = Counter(s.kind for s in stages)
    assert kinds[PROMPT] == 4, "the holds are the prompts, so C_ files mean steady holds"
    assert kinds[RAMP_IN_KIND] == 4 and kinds["ramp_out"] == 4
    assert kinds[REST_KIND] == 4                     # warm-up + 3 rests
    holds = [s for s in stages if s.kind == PROMPT]
    assert {(s.motion, s.rep) for s in holds} == {(1, 0), (1, 1), (7, 0), (7, 1)}
    warmup = next(s for s in stages if s.phase == WARMUP)
    assert (warmup.kind, warmup.motion, warmup.rep, warmup.label) == (REST_KIND, 1, 0, "hand_closed")
    assert metadata["choreography_run"]["status"] == COMPLETED
    assert all(s.duration == pytest.approx(s.t_end - s.t_start) for s in stages)


def test_a_display_stall_pauses_at_the_last_tick_and_marks_the_repetition(tmp_path):
    clock = FakeClock()
    controller, _ = _recording(tmp_path, clock, config(gestures=["hand_closed"], repetitions=2,
                                                       warmup=1.0))
    controller.start()
    t0 = clock.t
    while clock.t < t0 + 1.7:
        clock.advance(0.02)
        controller.tick()
    last_tick = clock.t
    clock.advance(2.0)                         # the GUI thread was blocked for 2 s
    controller.tick()
    assert controller.state == PAUSED_STATE
    assert controller.scheduler.pause_reason == "ui_stall"
    view = controller.view()
    assert view.elapsed == pytest.approx(last_tick - t0), "nothing unseen was scheduled"
    clock.advance(1.0)
    controller.tick()
    assert controller.resume()
    run_to_end(controller, clock)
    result = controller.finalize()
    phases = phase_intervals(result.intervals)
    interrupted = [iv for iv in phases if iv["completion_status"] == "interrupted"]
    assert len(interrupted) == 1 and interrupted[0]["end_monotonic"] == pytest.approx(last_tick)
    assert not interrupted[0]["training_eligible"]
    parents = repetition_intervals(result.intervals)
    assert parents[0]["completion_status"] == "interrupted", "not silently completed"
    assert parents[1]["completion_status"] == COMPLETED


def test_a_lagging_renderer_pauses_and_flags_what_was_not_shown(tmp_path):
    clock = FakeClock()
    controller, _ = _recording(tmp_path, clock, config(gestures=["hand_closed"], repetitions=1,
                                                       warmup=1.0), render_stall_threshold=0.5)
    controller.start()
    presented = clock.t
    for _ in range(80):                         # ticks keep coming, frames do not
        clock.advance(0.02)
        controller.tick()
        if clock.t < presented + 0.3:
            controller.frame_presented(clock.t)
            presented_last = clock.t
    assert controller.state == PAUSED_STATE
    assert controller.scheduler.pause_reason == "render_stall"
    controller.resume()
    controller.render_stall_threshold = None    # frames flow again from here on
    run_to_end(controller, clock)
    result = controller.finalize()
    flagged = [iv for iv in phase_intervals(result.intervals) if "presentation_gap" in iv["flags"]]
    assert flagged and not any(iv["training_eligible"] for iv in flagged)
    gap = next(e for e in controller.scheduler.events if e["type"] == "gap")
    assert gap["start_monotonic"] == pytest.approx(presented_last)


def test_a_lost_stream_pauses_the_recording_and_resume_waits_for_data(tmp_path):
    clock = FakeClock()
    controller, link = _recording(tmp_path, clock, config(gestures=["hand_closed"], repetitions=2,
                                                          warmup=1.0))
    controller.start()
    for _ in range(100):                       # 2 s of healthy data
        clock.advance(0.02)
        controller.tick()
    link.unplug()
    unplugged = clock.t
    for _ in range(100):
        clock.advance(0.02)
        controller.tick()
    assert controller.state == PAUSED_STATE
    assert controller.scheduler.pause_reason == "collection_disconnected"
    assert controller.disconnected
    assert not controller.resume(), "no data yet"
    link.replug()
    for _ in range(10):
        clock.advance(0.02)
        controller.tick()
    assert controller.resume()
    run_to_end(controller, clock)
    result = controller.finalize()
    gap = next(e for e in controller.scheduler.events if e["type"] == "gap")
    assert gap["gap_kind"] == "data_gap"
    assert unplugged <= gap["start_monotonic"] <= unplugged + 0.15
    flagged = [iv for iv in phase_intervals(result.intervals) if "data_gap" in iv["flags"]]
    assert flagged and all(not iv["training_eligible"] for iv in flagged)
    assert repetition_intervals(result.intervals)[0]["completion_status"] == "interrupted"


def test_a_dead_logger_counts_as_a_disconnection(tmp_path):
    clock = FakeClock()
    controller, link = _recording(tmp_path, clock)
    controller.start()
    for _ in range(25):
        clock.advance(0.02)
        controller.tick()
    link.alive = False
    for _ in range(10):
        clock.advance(0.02)
        controller.tick()
    assert controller.scheduler.pause_reason == "collection_disconnected"


def test_shutdown_mid_run_cancels_with_a_reason_and_saves(tmp_path):
    clock = FakeClock()
    controller, link = _recording(tmp_path, clock)
    controller.start()
    for _ in range(200):
        clock.advance(0.02)
        controller.tick()
    controller.shutdown()
    assert controller.result.status == CANCELLED
    assert controller.result.reason == "application_shutdown"
    record = json.loads((tmp_path / notes.RUN_RECORD_NAME).read_text())
    assert record["status"] == CANCELLED and record["end_reason"] == "application_shutdown"
    assert record["intervals"][-1]["end_collection_time"] is not None
    assert "close" in link.calls


def test_a_render_failure_fails_the_run(tmp_path):
    clock = FakeClock()
    controller, _ = _recording(tmp_path, clock)
    controller.start()
    clock.advance(3.0)
    controller.tick()
    controller.fail("render_failure: device lost")
    result = controller.finalize()
    assert result.status == FAILED and result.reason == "render_failure: device lost"


def test_reset_starts_a_fresh_run_with_the_same_frozen_configuration(tmp_path):
    clock = FakeClock()
    cfg = config()
    controller = ChoreographyController(ASSETS, cfg, clock=clock, wall=clock.wall)
    cfg.gestures.append("hand_open")            # edited after the run was set up
    controller.start()
    clock.advance(1.0)
    controller.tick()
    with pytest.raises(RuntimeError):
        controller.reset()
    controller.stop()
    controller.finalize()
    first = controller.run_id
    controller.reset()
    assert controller.state == "ready" and controller.run_id != first
    assert "hand_open" not in controller.plan.config.gestures
    assert controller.pose() == ("rest", 0.0)


def test_an_interrupted_recording_can_be_recovered_from_its_journal(tmp_path):
    clock = FakeClock()
    controller, _ = _recording(tmp_path, clock)
    controller.start()
    for _ in range(300):
        clock.advance(0.02)
        controller.tick()
    # The process dies here: no stop, no finalize. Only the journal survives.
    controller._journal.close()
    path = notes.recover_run(tmp_path, resolve_clocks=False)
    record = json.loads(path.read_text())
    assert record["recovered"] is True
    assert record["status"] == FAILED and record["end_reason"] == "recovered_after_interruption"
    last = [iv for iv in record["intervals"] if iv["track"] == "phase"][-1]
    assert last["completion_status"] == "interrupted"
    # Closed at the last heartbeat, within half a second of when the run died.
    assert clock.t - 0.5 - 1e-6 <= last["end_monotonic"] <= clock.t
    assert all(iv["completion_status"] != COMPLETED
               for iv in record["intervals"] if iv["track"] == "repetition"
               and iv["interval_id"] == last["parent_repetition_id"])


# --- no rest between gestures (direct transitions) -------------------------------------

def test_without_rests_each_ramp_out_runs_straight_into_the_next_ramp_in():
    cfg = config(gestures=["hand_closed", "wrist_flexion"], repetitions=2,
                 rest_between_gestures=False, timing_overrides={"hand_closed": {"rest": 4.0}})
    plan = build_plan(cfg, GESTURES)
    sequence = [p.phase for p in plan.phases]
    assert sequence == [WARMUP] + [RAMP_IN, HOLD, RAMP_OUT] * 4, "the warm-up is the only rest"
    for before, after in zip(plan.phases, plan.phases[1:]):
        assert after.start == pytest.approx(before.end), "no gap between phases"
    ramps_in = [p for p in plan.phases if p.phase == RAMP_IN]
    assert [p.transition_from for p in ramps_in] == [None, "hand_closed", "wrist_flexion",
                                                    "hand_closed"]
    assert plan.to_dict()["rest_between_gestures"] is False
    # Everything but the rests is the same as with rests.
    with_rests = build_plan(config(gestures=["hand_closed", "wrist_flexion"], repetitions=2,
                                   timing_overrides={"hand_closed": {"rest": 4.0}}), GESTURES)
    rests = sum(p.duration for p in with_rests.phases if p.phase == REST)
    assert plan.total_duration == pytest.approx(with_rests.total_duration - rests)


def test_without_rests_the_cycle_pause_still_separates_cycles():
    plan = build_plan(config(gestures=["hand_closed", "wrist_flexion"], repetitions=2,
                             rest_between_gestures=False, cycle_pause_enabled=True,
                             cycle_pause=5.0), GESTURES)
    sequence = [p.phase for p in plan.phases]
    assert sequence == ([WARMUP] + [RAMP_IN, HOLD, RAMP_OUT] * 2 + [CYCLE_PAUSE]
                        + [RAMP_IN, HOLD, RAMP_OUT] * 2)
    ramps_in = [p for p in plan.phases if p.phase == RAMP_IN]
    assert [p.transition_from for p in ramps_in] == [None, "hand_closed", None, "hand_closed"]


def test_without_rests_a_deliberate_rest_has_to_remain():
    def problems(**kwargs):
        return config(rest_between_gestures=False, **kwargs).validate(GESTURES)
    assert any("rest class is still collected" in p for p in problems(warmup=0.0))
    assert any("rest class" in p for p in problems(warmup=0.0, cycle_pause_enabled=True,
                                                   cycle_pause=0.0))
    assert problems(warmup=0.0, cycle_pause_enabled=True, cycle_pause=4.0) == []
    assert problems(warmup=2.0) == []
    # With rests on, a zero warm-up is still fine: the rests are the rest class.
    assert config(warmup=0.0).validate(GESTURES) == []


def test_old_presets_keep_their_rests(tmp_path):
    raw = config().to_dict()
    raw.pop("rest_between_gestures")
    raw["schema_version"] = 2
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(raw))
    assert ChoreographyConfig.load(path).rest_between_gestures is True
    saved = config(rest_between_gestures=False).save(tmp_path / "v3.json")
    assert ChoreographyConfig.load(saved).rest_between_gestures is False


def test_direct_transitions_are_labelled_and_the_view_warns_of_the_next_gesture():
    clock = FakeClock()
    cfg = config(gestures=["hand_closed", "wrist_flexion"], repetitions=1,
                 rest_between_gestures=False, intensity_enabled=True, intensity_levels=[0.5])
    controller = ChoreographyController(ASSETS, cfg, clock=clock, wall=clock.wall)
    controller.start()
    seen = {}
    while controller.state == RUNNING:
        clock.advance(0.02)
        view = controller.tick()
        if view.phase:
            seen.setdefault((view.motion_class, view.phase), (view.then_motion,
                                                              view.then_intensity_level))
    assert seen[("hand_closed", HOLD)] == ("wrist_flexion", 0.5)
    assert seen[("hand_closed", RAMP_OUT)] == ("wrist_flexion", 0.5)
    assert seen[("hand_closed", RAMP_IN)] == (None, None)
    assert seen[("wrist_flexion", RAMP_OUT)] == (None, None), "the last gesture has no successor"
    phases = phase_intervals(controller.finalize().intervals)
    out, into = phases[3], phases[4]
    assert (out["motion_class"], out["phase"]) == ("hand_closed", RAMP_OUT)
    assert (into["motion_class"], into["phase"]) == ("wrist_flexion", RAMP_IN)
    assert into["transition_from"] == "hand_closed" and phases[1]["transition_from"] is None
    assert into["start_monotonic"] == out["end_monotonic"]
    assert out["position_end"] == pytest.approx(0.0) and into["position_start"] == 0.0


def test_per_sample_targets_pass_through_zero_at_a_direct_transition():
    from libemg.choreography import targets as tg
    clock = FakeClock()
    cfg = config(gestures=["hand_closed", "wrist_flexion"], repetitions=1,
                 rest_between_gestures=False)
    _, result = _preview_run(clock, cfg)
    for iv in result.intervals:
        iv["start_collection_time"] = iv["start_monotonic"] + WALL_OFFSET
        iv["end_collection_time"] = iv["end_monotonic"] + WALL_OFFSET
    modality_clock = ModalityClock(fs=1000.0, anchor=WALL_OFFSET + 100.0, n_samples=7000)
    notes.assign_sample_indices(result.intervals, {"emg": modality_clock})
    targets = tg.sample_targets(result.intervals, modality_clock, "emg")
    # warm-up 2, ramp-in 0.5, hold 1, ramp-out 0.5 -> the transition is at 4.0 s.
    boundary = 4000
    window = slice(boundary - 50, boundary + 50)
    np.testing.assert_allclose(targets.position[window],
                               np.abs(np.arange(-50, 50)) / 500.0,
                               atol=1e-6)    # epoch-second times resolve to ~2e-7 s
    assert (targets.motion_class_id[boundary - 50:boundary] == ASSETS.class_id("hand_closed")).all()
    assert (targets.motion_class_id[boundary:boundary + 50] == ASSETS.class_id("wrist_flexion")).all()
    assert not (targets.phase_code == tg.PHASE_CODES[REST]).any()
