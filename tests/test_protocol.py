"""Coverage for describing a collection before running it."""
import numpy as np
import pytest

from libemg.protocol import (ACQUISITION_FIELDS, COUNTERBALANCED, FIXED,
                             PROTOCOL_NAME, RANDOM, REGRESSION,
                             CollectionProtocol, Segment, main,
                             ramp_and_hold, random_exploration,
                             render_trajectory, within_and_combined_dof)


def _protocol(**kwargs):
    kwargs.setdefault("name", "test")
    kwargs.setdefault("dof_labels", ["DOF 1", "DOF 2"])
    kwargs.setdefault("seed", 11)
    return CollectionProtocol(**kwargs)


# --- trajectories -------------------------------------------------------

def test_segments_run_end_to_end_without_jumps():
    fps = 24
    segments = [Segment("rest", 1.0), Segment("ramp", 1.0, [1.0, 0.0]),
                Segment("hold", 1.0, [1.0, 0.0]), Segment("rest", 1.0)]
    coordinates = render_trajectory(segments, n_dof=2, fps=fps)

    assert coordinates.shape == (4 * fps, 2)
    # A ramp arrives where it was told to, and the hold stays there.
    assert coordinates[2 * fps - 1, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(coordinates[2 * fps:3 * fps, 0], 1.0)
    # Every segment begins where the one before it ended, so there is nothing
    # the subject is asked to follow instantaneously. The one quick transition
    # is the release into a rest, which takes a quarter of a second by design.
    release_rate = 1.0 / (round(0.25 * fps) - 1)
    assert np.max(np.abs(np.diff(coordinates, axis=0))) <= release_rate + 1e-9
    # And it ends back at rest, so the next repetition starts from zero.
    np.testing.assert_allclose(coordinates[-1], 0.0, atol=1e-9)


def test_a_rest_releases_rather_than_snapping_back_to_zero():
    fps = 24
    coordinates = render_trajectory(
        [Segment("hold", 0.5, [1.0]), Segment("rest", 1.0)], n_dof=1, fps=fps)
    release = coordinates[round(0.5 * fps):round(0.5 * fps) + round(0.25 * fps), 0]

    assert release[0] == pytest.approx(1.0) and release[-1] == pytest.approx(0.0)
    assert np.all(np.diff(release) < 0), "the release is a ramp down, not a step"
    np.testing.assert_allclose(coordinates[round(0.75 * fps):, 0], 0.0)


def test_a_hold_at_a_new_target_is_the_way_to_ask_for_a_step():
    coordinates = render_trajectory(
        [Segment("hold", 1.0, [1.0])], n_dof=1, fps=10)
    np.testing.assert_allclose(coordinates, 1.0)


def test_targets_are_clipped_to_what_the_animation_can_show():
    coordinates = render_trajectory(
        [Segment("hold", 1.0, [5.0, -5.0])], n_dof=2, fps=4)
    assert coordinates.min() == -1.0 and coordinates.max() == 1.0


def test_a_sinusoid_oscillates_and_comes_back_to_where_it_started():
    coordinates = render_trajectory(
        [Segment("sinusoid", 4.0, [0.0, 1.0], period=2.0, amplitude=0.8)],
        n_dof=2, fps=50)

    assert coordinates[:, 0].max() == pytest.approx(0.0)
    assert coordinates[:, 1].max() == pytest.approx(0.8, abs=0.01)
    assert coordinates[:, 1].min() == pytest.approx(-0.8, abs=0.01)
    assert coordinates[0, 1] == pytest.approx(0.0)


def test_random_targets_stay_in_range_and_repeat_for_the_same_seed():
    segments = [Segment("random", 4.0, [1.0, 1.0], count=4)]
    first = render_trajectory(segments, n_dof=2, fps=20, seed=3)
    again = render_trajectory(segments, n_dof=2, fps=20, seed=3)
    other = render_trajectory(segments, n_dof=2, fps=20, seed=4)

    np.testing.assert_array_equal(first, again), "a seeded protocol renders the same"
    assert not np.array_equal(first, other)
    assert np.abs(first).max() <= 1.0
    # Targets are drawn from inside the ball, not only from its corners.
    assert np.linalg.norm(first, axis=1).max() <= 1.45


def test_an_unknown_segment_kind_is_refused_by_name():
    with pytest.raises(ValueError, match="nonsense"):
        render_trajectory([Segment("nonsense", 1.0)], n_dof=1, fps=10)


def test_the_standard_template_visits_each_dof_alone_and_then_together():
    segments = within_and_combined_dof(n_dof=2, hold_seconds=1, ramp_seconds=1,
                                       rest_seconds=1)
    held = [tuple(s.target) for s in segments if s.kind == "hold"]

    assert (1.0, 0.0) in held and (0.0, -1.0) in held, "each DOF alone"
    assert (1.0, 1.0) in held and (-1.0, 1.0) in held, "and the pairs together"
    assert len(held) == 8, "4 single-DOF extremes, 4 combined"


def test_the_ramp_template_visits_several_intensities():
    segments = ramp_and_hold(n_dof=1, levels=(0.25, 0.5, 1.0), hold_seconds=1,
                             ramp_seconds=1, rest_seconds=1)
    assert [s.target[0] for s in segments if s.kind == "hold"] == [0.25, 0.5, 1.0]
    # Every hold is reached by a ramp rather than jumped to.
    kinds = [s.kind for s in segments]
    assert all(kinds[index - 1] == "ramp"
               for index, kind in enumerate(kinds) if kind == "hold")


def test_random_exploration_is_mostly_the_random_segment():
    segments = random_exploration(n_dof=2, seconds=30, count=10)
    assert [s.kind for s in segments] == ["rest", "random", "rest"]


# --- the document -------------------------------------------------------

def test_a_protocol_round_trips_through_a_file(tmp_path):
    protocol = _protocol(kind=REGRESSION, notes="pilot",
                         trajectory=within_and_combined_dof(n_dof=2),
                         factors={"limb_position": ["neutral", "elevated"]},
                         acquisition={"device": "Myo", "sampling_rate_hz": "200"})
    protocol.media.animator = "bar"
    protocol.media.fps = 30

    path = protocol.save(tmp_path)
    reloaded = CollectionProtocol.load(path)

    assert path.name == PROTOCOL_NAME
    assert reloaded.to_dict() == protocol.to_dict()
    assert reloaded.seed == protocol.seed, "the seed is what makes a rerun a rerun"
    assert reloaded.media.animator == "bar" and reloaded.media.fps == 30
    assert reloaded.acquisition["device"] == "Myo"
    # Every ISEK field is present in the file, even the ones left blank, so the
    # form is visible rather than silently absent.
    assert set(reloaded.to_dict()["acquisition"]) == {name for name, _ in ACQUISITION_FIELDS}


def test_loading_refuses_a_protocol_from_a_newer_libemg(tmp_path):
    path = tmp_path / PROTOCOL_NAME
    path.write_text("schema: 99\nname: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="newer LibEMG"):
        CollectionProtocol.load(path)


def test_a_protocol_without_a_seed_still_records_the_one_it_used(tmp_path):
    protocol = CollectionProtocol(name="x")
    assert isinstance(protocol.seed, int)
    assert CollectionProtocol.load(protocol.save(tmp_path)).seed == protocol.seed


# --- ordering -----------------------------------------------------------

def test_prompts_are_shuffled_within_each_rep_by_default():
    protocol = _protocol()
    protocol.timing.num_reps = 4
    names = [f"c{i}" for i in range(6)]

    prompts = protocol.schedule(names)

    assert len(prompts) == 24
    for rep in range(4):
        motions = [p.motion for p in prompts if p.rep == rep]
        assert sorted(motions) == list(range(6)), "every class, once, per rep"
    orders = {tuple(p.motion for p in prompts if p.rep == rep) for rep in range(4)}
    assert len(orders) > 1, "the order must not be the same every rep"


def test_the_same_seed_gives_the_same_session_back():
    names = ["a", "b", "c", "d"]
    first = _protocol(seed=99).schedule(names, num_reps=3)
    again = _protocol(seed=99).schedule(names, num_reps=3)
    other = _protocol(seed=100).schedule(names, num_reps=3)

    assert [(p.motion, p.rep) for p in first] == [(p.motion, p.rep) for p in again]
    assert [p.rest_time for p in first] == [p.rest_time for p in again]
    assert [(p.motion, p.rep) for p in first] != [(p.motion, p.rep) for p in other]


def test_a_fixed_order_is_still_available():
    prompts = _protocol(order=FIXED).schedule(["a", "b", "c"], num_reps=2)
    assert [p.motion for p in prompts] == [0, 1, 2, 0, 1, 2]


def test_counterbalancing_rotates_so_every_class_sits_in_every_position():
    prompts = _protocol(order=COUNTERBALANCED).schedule(["a", "b", "c"], num_reps=3)
    orders = [[p.motion for p in prompts if p.rep == rep] for rep in range(3)]

    assert orders == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    first_positions = {order.index(motion) for motion in range(3) for order in orders}
    assert first_positions == {0, 1, 2}


def test_rest_is_jittered_so_the_cue_cannot_be_anticipated():
    protocol = _protocol()
    protocol.timing.rest_time, protocol.timing.rest_jitter = 3.0, 1.0

    rests = [p.rest_time for p in protocol.schedule(["a", "b", "c"], num_reps=4)]

    assert len(set(rests)) > 1, "a fixed rest teaches the subject when to expect it"
    assert all(3.0 <= rest <= 4.0 for rest in rests)


def test_no_jitter_asked_for_is_no_jitter_given():
    protocol = _protocol()
    protocol.timing.rest_jitter = 0.0
    rests = {p.rest_time for p in protocol.schedule(["a", "b"], num_reps=3)}
    assert rests == {protocol.timing.rest_time}


def test_each_rep_is_collected_under_one_level_of_every_factor():
    protocol = _protocol(factors={"limb_position": ["low", "high"],
                                  "intensity": ["30%", "60%", "90%"]})

    prompts = protocol.schedule(["a", "b"], num_reps=6)

    for rep in range(6):
        levels = {tuple(sorted(p.factors.items())) for p in prompts if p.rep == rep}
        assert len(levels) == 1, "a subject cannot change arm position mid-rep"
    assert protocol.levels_for_rep(0) == {"limb_position": "low", "intensity": "30%"}
    assert protocol.levels_for_rep(1) == {"limb_position": "high", "intensity": "60%"}
    # Over six reps both position levels and all three intensities are covered.
    assert {p.factors["intensity"] for p in prompts} == {"30%", "60%", "90%"}
    assert {p.factors["limb_position"] for p in prompts} == {"low", "high"}


def test_the_prompt_says_which_level_the_subject_should_be_in():
    protocol = _protocol(factors={"limb_position": ["elevated"]})
    prompt = protocol.schedule(["Wrist Flexion"], num_reps=1)[0]
    assert prompt.name == "Wrist Flexion"
    assert prompt.label == "Wrist Flexion (elevated)"


def test_with_no_factors_the_prompt_is_just_the_class_name():
    assert _protocol().schedule(["Open"], num_reps=1)[0].label == "Open"


# --- rendering and reporting -------------------------------------------

def test_describe_names_the_acquisition_details_that_are_missing():
    protocol = _protocol(trajectory=within_and_combined_dof(n_dof=2),
                         factors={"limb_position": ["neutral"]})
    described = protocol.describe()

    assert "regression" in described or "classification" in described
    assert f"seed {protocol.seed}" in described
    assert "limb_position" in described
    assert "acquisition details not filled in: 13 of 13" in described


def test_building_media_writes_the_animation_and_its_labels(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    protocol = _protocol(kind=REGRESSION,
                         trajectory=[Segment("rest", 0.2),
                                     Segment("ramp", 0.2, [1.0, 0.0]),
                                     Segment("hold", 0.2, [1.0, 0.0])])
    protocol.media.fps = 5
    protocol.media.video_format = ".gif"
    said = []

    written = protocol.build_media(tmp_path / "media", progress=said.append)

    names = {path.name for path in written}
    assert names == {"test.gif", "test.txt"}
    # The labels are the trajectory, frame for frame: ground truth for a
    # regression collection, already aligned with what was prompted.
    labels = np.loadtxt(tmp_path / "media" / "test.txt", delimiter=",")
    np.testing.assert_allclose(labels, protocol.coordinates())
    assert any("Rendering" in message for message in said)


def test_building_refuses_a_protocol_with_nothing_to_draw(tmp_path):
    with pytest.raises(ValueError, match="no trajectory"):
        _protocol().build_media(tmp_path)


def test_an_unknown_animator_is_refused_by_name(tmp_path):
    protocol = _protocol(trajectory=[Segment("rest", 0.2)])
    protocol.media.animator = "hologram"
    with pytest.raises(ValueError, match="hologram"):
        protocol.build_media(tmp_path)


def test_the_command_line_writes_shows_and_builds(tmp_path, capsys):
    path = tmp_path / "study.yaml"

    assert main(["template", str(path), "--dofs", "Flex", "Extend"]) == 0
    assert path.exists() and "Wrote" in capsys.readouterr().out

    assert main(["show", str(path)]) == 0
    shown = capsys.readouterr().out
    assert "study" in shown and "limb_position" in shown
