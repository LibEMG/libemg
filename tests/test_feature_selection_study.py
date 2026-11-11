"""Feature selection studies: the traces that explain them, the four schemes, the block and the window.

A selection is only useful if it can be trusted and explained. These tests hold
the selector to that: every score it computed is kept, the held-out repetitions
and the left-out subject are never seen while selecting, a saved study comes
back exactly, and the pipeline block hands the window what it ran.
"""
import json

import numpy as np
import pytest

from libemg.feature_selector import (SCHEMES, FeatureSelector, SelectionStopped,
                                     SelectionStudy, SelectionTrace, SelectionUnit)
from libemg._gui._pipeline import selection_viewer as viewer

SUBJECTS, REPS, CLASSES, PER = 3, 4, 3, 12


@pytest.fixture
def cohort():
    """Three subjects; 'good' separates classes for all, 'own' with a sign that flips by subject."""
    rng = np.random.default_rng(0)
    rows = np.array([(s, r, c) for s in range(SUBJECTS) for r in range(REPS)
                     for c in range(CLASSES) for _ in range(PER)])
    s, r, c = rows.T
    n = len(rows)
    flip = np.where(s == 1, -1, 1)[:, None]
    data = {"good": c[:, None] * 2.0 + rng.normal(0, 1, (n, 2)),
            "own": flip * c[:, None] * 2.0 + rng.normal(0, 1, (n, 2)),
            "noise": rng.normal(0, 1, (n, 3)),
            "weak": c[:, None] * 0.4 + rng.normal(0, 1, (n, 2))}
    return data, c, s, r


# ======================================================================
# a single selection's trace
# ======================================================================
def test_the_trace_keeps_every_score_and_agrees_with_run_selection(cohort):
    data, labels, _, reps = cohort
    selector = FeatureSelector()
    trace = selector.select(data, "CA", labels, {"var": reps}, early_stop=3)
    results, order = selector.run_selection(data, "CA", labels, {"var": reps}, early_stop=3)

    assert trace.selected == order
    np.testing.assert_array_equal(trace.legacy_results(), results)
    assert trace.scores.shape == (3, len(data))
    # Iteration k scores every candidate not yet chosen, and only those.
    for step, chosen in enumerate(trace.selected):
        scored = {name for name, _ in trace.ranking(step)}
        assert scored == set(data) - set(trace.selected[:step])
        assert trace.ranking(step)[0][0] == chosen
    np.testing.assert_allclose(trace.selected_scores,
                               [trace.ranking(i)[0][1] for i in range(3)])


def test_a_filter_trace_is_one_ranked_pass(cohort):
    data, labels, _, _ = cohort
    trace = FeatureSelector().select(data, "FE", labels)
    assert trace.strategy == "filter" and trace.iterations == 1
    assert [name for name, _ in trace.ranking(0)] == trace.selected
    assert np.all(np.diff(trace.selected_scores) <= 0)


def test_undefined_candidates_are_noted_and_skipped_only_when_asked(cohort):
    data, labels, _, _ = cohort
    # A constant feature puts every window at distance zero: IIF is 0/0 for it alone.
    groups = {"constant": np.ones((len(labels), 1)), "good": data["good"]}
    trace = FeatureSelector().select(groups, "IIF", labels, early_stop=1, skip_undefined=True)
    assert trace.selected == ["good"] and np.isnan(trace.scores[0, 0])
    assert trace.notes and trace.notes[0].startswith("constant:")
    with pytest.raises(ValueError, match="undefined"):
        FeatureSelector().select(groups, "IIF", labels)


# ======================================================================
# the four schemes
# ======================================================================
@pytest.mark.parametrize("scheme", SCHEMES)
def test_selection_never_sees_the_test_repetitions(cohort, scheme):
    data, labels, subjects, reps = cohort
    selector = FeatureSelector()
    kwargs = dict(subject_var=subjects, repetition_var=reps, test_repetitions=[3],
                  early_stop=2)
    before = selector.run_study(data, "CA", labels, scheme, **kwargs)
    # Wreck every test window: a selection that looked at them would change.
    wrecked = {name: np.where((reps == 3)[:, None], 1e3 * np.random.default_rng(1).normal(
        size=x.shape), x) for name, x in data.items()}
    after = selector.run_study(wrecked, "CA", labels, scheme, **kwargs)
    for a, b in zip(before.units, after.units):
        assert a.trace.selected == b.trace.selected
        np.testing.assert_array_equal(a.trace.scores, b.trace.scores)


def test_within_subject_selects_per_subject_from_their_training_reps(cohort):
    data, labels, subjects, reps = cohort
    study = FeatureSelector().run_study(data, "CA", labels, "within_subject",
                                        subject_var=subjects, repetition_var=reps,
                                        test_repetitions=[3], early_stop=2)
    assert [u.name for u in study.units] == [f"Subject {s}" for s in range(SUBJECTS)]
    for s, unit in enumerate(study.units):
        assert unit.train_windows == (REPS - 1) * CLASSES * PER
        assert unit.test_windows == CLASSES * PER
        assert unit.test_subjects == [str(s)] and unit.criterion_folds == "leave one repetition out"
        assert unit.test_accuracy.shape == (2,) and np.all(unit.test_accuracy >= 0)
        assert unit.trace.parts is None


def test_shared_features_aggregate_each_subjects_own_score(cohort):
    data, labels, subjects, reps = cohort
    study = FeatureSelector().run_study(data, "SI", labels, "within_subject_shared",
                                        subject_var=subjects, repetition_var=reps,
                                        test_repetitions=[3], early_stop=2)
    unit, = study.units
    trace = unit.trace
    assert trace.parts.shape == (2, len(data), SUBJECTS)
    assert trace.part_names == [f"Subject {s}" for s in range(SUBJECTS)]
    np.testing.assert_allclose(trace.scores, np.nanmean(trace.parts, axis=2))
    # Each subject's part is exactly that subject's own score on its training reps.
    own = (subjects == 2) & (reps != 3)
    expected = FeatureSelector().evaluate(data["good"][own], "SI", labels[own])
    assert trace.parts[0, trace.candidates.index("good"), 2] == pytest.approx(expected)
    assert set(unit.test_accuracy_by_subject) == {"0", "1", "2"}
    median = FeatureSelector().run_study(data, "SI", labels, "within_subject_shared",
                                         subject_var=subjects, repetition_var=reps,
                                         early_stop=1, aggregate="median")
    np.testing.assert_allclose(median.units[0].trace.scores,
                               np.median(median.units[0].trace.parts, axis=2))


def test_subject_independent_pools_training_and_tests_every_subject(cohort):
    data, labels, subjects, reps = cohort
    study = FeatureSelector().run_study(data, "CA", labels, "subject_independent",
                                        subject_var=subjects, repetition_var=reps,
                                        test_repetitions=[3], early_stop=2)
    unit, = study.units
    assert unit.train_windows == SUBJECTS * (REPS - 1) * CLASSES * PER
    assert set(unit.test_accuracy_by_subject) == {"0", "1", "2"}
    np.testing.assert_allclose(unit.test_accuracy,
                               np.mean(list(unit.test_accuracy_by_subject.values()), axis=0))


def test_cross_subject_leaves_each_subject_out_of_training(cohort):
    data, labels, subjects, reps = cohort
    selector = FeatureSelector()
    study = selector.run_study(data, "CA", labels, "cross_subject", subject_var=subjects,
                               repetition_var=reps, early_stop=2)
    assert len(study.units) == SUBJECTS
    for s, unit in enumerate(study.units):
        assert str(s) not in unit.train_subjects and unit.test_subjects == [str(s)]
        assert unit.train_windows == (SUBJECTS - 1) * REPS * CLASSES * PER
        assert unit.test_windows == REPS * CLASSES * PER
        assert unit.criterion_folds == "leave one subject out"
    # The subject whose 'own' feature runs the other way is never helped by it.
    assert "own" not in study.units[1].trace.selected[:1]
    held = selector.run_study(data, "CA", labels, "cross_subject", subject_var=subjects,
                              repetition_var=reps, test_repetitions=[3], early_stop=2)
    assert all(u.test_windows == CLASSES * PER for u in held.units)
    assert all(u.train_windows == (SUBJECTS - 1) * (REPS - 1) * CLASSES * PER for u in held.units)


def test_studies_refuse_what_they_cannot_do(cohort):
    data, labels, subjects, reps = cohort
    selector = FeatureSelector()
    with pytest.raises(ValueError, match="scheme"):
        selector.run_study(data, "CA", labels, "everyone")
    with pytest.raises(ValueError, match="repetition_var"):
        selector.run_study(data, "CA", labels, test_repetitions=[3])
    with pytest.raises(ValueError, match="two subjects"):
        selector.run_study(data, "CA", labels, "cross_subject",
                           subject_var=np.zeros(len(labels)))
    with pytest.raises(ValueError, match="None of the test"):
        selector.run_study(data, "CA", labels, repetition_var=reps, test_repetitions=[9])
    with pytest.raises(SelectionStopped):
        selector.run_study(data, "CA", labels, subject_var=subjects, should_stop=lambda: True)


def test_progress_is_reported_up_to_done(cohort):
    data, labels, subjects, reps = cohort
    seen = []
    FeatureSelector().run_study(data, "SI", labels, "cross_subject", subject_var=subjects,
                                early_stop=2, on_progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0


# ======================================================================
# saving and loading
# ======================================================================
def test_a_saved_study_is_strict_json_and_comes_back_exactly(cohort, tmp_path):
    data, labels, subjects, reps = cohort
    study = FeatureSelector().run_study(data, "SI", labels, "within_subject_shared",
                                        subject_var=subjects, repetition_var=reps,
                                        test_repetitions=[3], settings={"folder": "x"})
    study.units[0].trace.scores[0, 0] = np.inf
    path = study.save(tmp_path / "study.json")

    def refuse(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")
    json.loads(path.read_text(), parse_constant=refuse)
    back = SelectionStudy.load(path)
    assert back.to_dict() == study.to_dict()
    assert np.isposinf(back.units[0].trace.scores[0, 0])
    assert back.settings["folder"] == "x" and back.settings["test_repetitions"] == [3]
    with pytest.raises(ValueError, match="not a saved"):
        SelectionStudy.from_dict({"format": "something else"})


def test_consensus_counts_how_often_each_feature_was_chosen():
    def unit(name, selected):
        trace = SelectionTrace("CA", "max", "sequential", ["a", "b", "c"],
                               np.zeros((len(selected), 3)), selected)
        return SelectionUnit(name, trace)
    study = SelectionStudy("within_subject", "CA", "max", "sequential", ["a", "b", "c"],
                           [unit("1", ["b", "a"]), unit("2", ["b", "c"]), unit("3", ["a"])])
    assert study.rank_matrix().tolist() == [[2, 1, 0], [0, 1, 2], [1, 0, 0]]
    assert [row[:2] for row in study.consensus()] == [("b", 2), ("a", 2), ("c", 1)]
    assert [row[:2] for row in study.consensus(top=1)] == [("b", 2), ("a", 1), ("c", 0)]


# ======================================================================
# what the window draws
# ======================================================================
def _trace():
    scores = np.array([[70.0, 80.0, 75.0, np.nan],
                       [np.nan, 85.0, 90.0, np.nan]])
    parts = np.stack([scores - 1, scores + 1], axis=2)
    parts[0, 2] = [68.0, 82.0]          # subject 1 alone would have taken 'c' first
    return SelectionTrace("CA", "max", "sequential", ["a", "b", "c", "d"], scores, ["b", "c"],
                          parts=parts, part_names=["Subject 0", "Subject 1"],
                          notes=["d: undefined"])


def test_decision_rows_rank_the_alternatives_against_the_choice():
    trace = _trace()
    rows = viewer.decision_rows(trace, 0)
    assert [r["name"] for r in rows] == ["b", "c", "a"]
    assert [r["chosen"] for r in rows] == [True, False, False]
    assert [r["gap"] for r in rows] == [0.0, -5.0, -10.0]
    assert viewer.decision_rows(trace, 1)[0]["name"] == "c"
    assert viewer.part_winners(trace, 0) == ["b", "c"]


def test_the_explanation_names_the_choice_the_runner_up_and_the_dissent():
    text = viewer.explain(_trace(), 0)
    assert "b gave the best score" in text and "runner-up c" in text
    assert "1 of 2 subjects would have chosen b" in text
    assert "With b already chosen" in viewer.explain(_trace(), 1)


def test_every_iteration_is_laid_out_with_the_choice_marked():
    cells = dict(viewer.trace_cells(_trace()))
    assert list(cells) == ["b", "c", "a", "d"]
    assert cells["b"] == [("80", "chosen"), ("in set", "in set")]
    assert cells["c"] == [("75", "scored"), ("90", "chosen")]
    assert cells["d"] == [("-", "none"), ("-", "none")]


def test_metric_names_bars_and_scores_read_well():
    assert viewer.metric_code("Accuracy (CA)") == "CA"
    assert viewer.metric_code("separability") == "SI"
    assert len(viewer.metric_choices()) == len(FeatureSelector().get_metrics())
    assert viewer.format_score(np.inf) == "inf" and viewer.format_score(np.nan) == "-"
    start, end = viewer.bar_limits([91.0, 92.0, 95.0])
    assert 85 < start < 91 and end > 95
    assert viewer.bar_limits([0.1, 0.5])[0] == 0.0
    assert np.isfinite(viewer.finite_bars([1.0, np.inf])).all()


# ======================================================================
# the block, run over stored recordings
# ======================================================================
def _write_cohort(root):
    rng = np.random.default_rng(0)
    for s in range(3):
        for r in range(3):
            for c in range(3):
                folder = root / f"S_{s}"
                folder.mkdir(parents=True, exist_ok=True)
                t = np.arange(400) / 500.0
                signal = (1.0 + c + 0.3 * s) * np.sin(2 * np.pi * (20 + 25 * c) * t)
                block = signal[:, None] * np.linspace(0.6, 1.4, 3)[None, :]
                np.savetxt(folder / f"C_{c}_R_{r}.csv", block + rng.normal(0, 0.5, block.shape),
                           delimiter=",")
    return root


def _selection_document(folder, **params):
    from libemg.pipeline import PipelineDocument
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder), "sampling_rate": 500,
        "regex_filters": "S_|/|0,1,2|subjects\nC_|_R|0,1,2|classes\n_R_|.csv|0,1,2|reps"})
    window = document.add_node("window.enframe", params={"window_size": 100,
                                                         "window_increment": 50})
    block = document.add_node("sink.feature_selection", params=dict(
        {"features": ["MAV", "WL", "ZC", "SSC"], "test_values": "2", "max_features": 3},
        **params))
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", block, "windows")
    document.connect(window, "metadata", block, "labels")
    return document


def test_the_block_runs_a_study_over_stored_data_and_saves_it(tmp_path):
    from libemg.pipeline import compile_pipeline
    folder = _write_cohort(tmp_path / "data")
    path = tmp_path / "study.json"
    pipeline = compile_pipeline(_selection_document(
        folder, scheme="Cross subject", results_path=str(path)))
    progress = []
    pipeline.run(on_progress=progress.append)

    study = pipeline.selection_view
    assert study.scheme == "cross_subject" and len(study.units) == 3
    assert study.candidates == ["MAV", "WL", "ZC", "SSC"]
    assert all(len(u.trace.selected) == 3 for u in study.units)
    # Repetitions are matched as the filenames write them, not by filter position.
    assert study.settings["test_repetitions"] == ["2"]
    assert study.settings["subjects"] == ["0", "1", "2"]
    assert SelectionStudy.load(path).to_dict()["units"] == study.to_dict()["units"]
    assert progress[-1] == 1.0 and len(progress) > 3


def test_the_block_says_what_a_scheme_needs(tmp_path):
    from libemg.pipeline import compile_pipeline
    from libemg._gui._pipeline.compile import CompileError
    folder = _write_cohort(tmp_path / "data")
    pipeline = compile_pipeline(_selection_document(folder, scheme="Cross subject",
                                                    subject_field=""))
    with pytest.raises(CompileError, match="Subject Field"):
        pipeline.run()


# ======================================================================
# the window itself
# ======================================================================
@pytest.fixture
def dpg_context():
    # Items only, no viewport: create_context() needs no display, which is what
    # keeps this test in the default run. Showing a viewport here would also
    # make any later Tk window in the same process crash on Windows.
    import dearpygui.dearpygui as dpg
    dpg.create_context()
    yield dpg
    dpg.destroy_context()


@pytest.mark.parametrize("scheme, metric", [("within_subject", "CA"),
                                            ("within_subject_shared", "SI"),
                                            ("subject_independent", "FE")])
def test_the_window_draws_steps_saves_and_loads(dpg_context, cohort, tmp_path, scheme, metric):
    dpg = dpg_context
    data, labels, subjects, reps = cohort
    study = FeatureSelector().run_study(data, metric, labels, scheme, subject_var=subjects,
                                        repetition_var=reps, test_repetitions=[3])
    window = viewer.SelectionViewer(study).spawn_window()
    assert dpg.does_item_exist(window._tag("decision"))
    if study.strategy == "sequential":
        window._step(1)
        assert window.iteration == 1 and dpg.get_value(window._tag("iteration")) == 2
    if len(study.units) > 1:
        window._unit_chosen(None, study.units[1].name)
        assert window.unit == 1 and window.iteration == 0
    saved = window.save(str(tmp_path / "out"))
    assert saved.endswith(".json")
    empty = viewer.SelectionViewer().spawn_window()
    empty.load(saved)
    assert empty.study.to_dict()["units"] == study.to_dict()["units"]
    window.cleanup()
    empty.cleanup()
    assert not dpg.does_item_exist(window.window_tag)
