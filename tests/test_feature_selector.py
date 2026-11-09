"""Analytic metric checks, selection regressions and real dataset integration."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from libemg.feature_selector import FeatureSelector


@pytest.fixture
def analytic():
    # Two identical repetitions per class, means 0 and 4, unbiased variance 4/3.
    x = np.array([-1, 1, -1, 1, 3, 5, 3, 5], dtype=float)[:, None]
    y = np.repeat([10, 30], 4)
    groups = np.tile([0, 0, 1, 1], 2)
    return x, y, groups


def test_registry_and_legacy_names():
    selector = FeatureSelector()
    expected = {"mwRI", "swRI", "swSI", "MSA", "CD", "MAV", "SI", "mSI", "mwSI", "BD",
                "KLD", "HD", "FDR", "VOR", "FE", "TSM", "DS", "CDM", "PU", "rPU", "NS",
                "rNS", "CE", "rCE", "C", "rC", "CA", "ACA", "UD", "ICF", "IIF", "TT", "RI", "AE"}
    assert set(selector.get_metrics(abbreviations=True)) == expected
    assert len(selector.get_metrics()) == len(expected)
    for code, name in zip(selector.get_metrics(abbreviations=True), selector.get_metrics()):
        assert selector.get_metric_info(code) == selector.get_metric_info(name)
    assert selector.get_metric_info("MSA")["objective"] == "min"
    assert "variant" in selector.get_metric_info("C")["reference_status"]
    assert "variant" in selector.get_metric_info("CDM")["reference_status"]


@pytest.mark.parametrize("code, expected", [
    ("SI", np.sqrt(3)), ("mSI", np.sqrt(3)), ("MSA", np.sqrt(4 / 3)),
    ("BD", 1.5), ("KLD", 6), ("HD", 1 - np.exp(-1.5)), ("FDR", 6),
    ("FE", 1), ("VOR", 0), ("TSM", 4), ("RI", 0), ("CD", 0),
    ("mwRI", np.sqrt(3) / 4), ("swRI", 0), ("mwSI", np.sqrt(2)), ("swSI", 0),
    ("DS", np.inf), ("CA", 100), ("ACA", 100), ("AE", 0), ("UD", 100),
    ("ICF", 0), ("IIF", 0), ("CDM", 0), ("MAV", 2.5), ("TT", 1000),
])
def test_analytic_metric_values(analytic, code, expected):
    x, y, groups = analytic
    score = FeatureSelector(allow_approximations=True).evaluate(x, code, y, {"var": groups}, raw_emg=x,
                                       window_size_ms=200, window_increment_ms=100, recording_var=y)
    assert score == pytest.approx(expected)


def test_gaussian_distances_covariance_term_and_affine_invariance():
    selector = FeatureSelector()
    # Same means, variances 2 and 8: only the determinant term contributes to BD.
    x = np.array([-1, 1, -2, 2])[:, None]
    y = ["left", "left", "right", "right"]
    bd = selector.evaluate(x, "BD", y)
    assert bd == pytest.approx(0.5 * np.log(1.25))
    assert selector.evaluate(x, "HD", y) == pytest.approx(1 - np.exp(-bd))
    assert selector.evaluate(x, "KLD", y) == pytest.approx(
        ((4 - 1 + np.log(0.25)) / 2 + (0.25 - 1 + np.log(4)) / 2) / 2)
    for code in ("SI", "mSI", "BD", "KLD", "HD", "FDR", "TSM"):
        assert selector.evaluate(x * 7 + 5, code, y) == pytest.approx(selector.evaluate(x, code, y))


def test_overlap_endpoints_and_multiclass_efficiency():
    selector = FeatureSelector()
    assert selector.evaluate([0, 1, 1, 2], "FE", [0, 0, 1, 1]) == 0.5
    assert selector.evaluate([0, 2, 1, 3], "VOR", [0, 0, 1, 1]) == pytest.approx(1 / 3)
    assert selector.evaluate([0, 1, 3, 4, 6, 7], "FE", np.repeat([2, 4, 9], 2)) == 1
    assert selector.evaluate([0, 1, 3, 4, 6, 7], "VOR", np.repeat([2, 4, 9], 2)) == 0


def test_repetition_drift_and_std_use_sample_std(analytic):
    x, y, groups = analytic
    x = x.copy()
    x[groups == 1] += 2
    selector = FeatureSelector()
    assert selector.evaluate(x, "CD", y, repetition_var=groups) == 2
    assert selector.evaluate(x, "RI", y, repetition_var=groups) == pytest.approx(1 / np.sqrt(2))
    # No covariance is necessary for drift, even with singleton class/cells.
    assert selector.evaluate([-1, 3, 1, 5], "CD", [10, 30, 10, 30],
                             repetition_var=[0, 0, 1, 1]) == 2
    uneven = np.array([0, 2, 0, 4, 8, 10, 8, 12])[:, None]
    within_trials = [selector.evaluate(uneven[groups == group], "SI", y[groups == group])
                     for group in np.unique(groups)]
    assert selector.evaluate(uneven, "swSI", y, trial_var=groups) == pytest.approx(
        np.std(within_trials, ddof=1))


def test_nearest_neighbors_exclude_self_and_paper_iif():
    selector = FeatureSelector()
    x, labels = [0, 1, 2, 10], [0, 0, 1, 1]
    # Stable nearest indices: [1,0,1,2], distances [1,1,1,8].
    assert selector.evaluate(x, "ICF", labels) == 0.25
    assert selector.evaluate(x, "IIF", labels, iif_mode="paper") == 10
    # Same-class distances [1,1,8,8], other-class [2,1,1,9].
    assert selector.evaluate(x, "IIF", labels) == pytest.approx(18 / 13)
    assert selector.evaluate([0, 0, 1, 1], "ICF", labels) == 0
    assert selector.evaluate([0, 0, 0, 0], "ICF", labels) == 0.5


class FixedPredictions(ClassifierMixin, BaseEstimator):
    def fit(self, x, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, x):
        return x[:, 0].astype(int)


def test_active_error_uses_no_motion_as_label_not_boolean_sum():
    selector = FeatureSelector(classifier=FixedPredictions())
    # Each fold: one correct, one no-motion prediction, one active error.
    labels = np.tile([4, 4, 9], 4)
    predictions = np.tile([4, 99, 4], 4)[:, None]
    groups = np.repeat([0, 1], 6)
    values = selector.evaluate_metrics(["CA", "ACA", "AE"], predictions, labels,
                                        {"var": groups}, nm_class=99)
    assert values["CA"] == pytest.approx(100 / 3)
    assert values["ACA"] == pytest.approx(200 / 3)
    assert values["AE"] == pytest.approx(100 / 3)


def test_usable_data_predicts_future_trials_before_updating():
    selector = FeatureSelector(classifier=FixedPredictions())
    labels = np.tile([0, 0, 1, 1], 3)
    features = np.tile([0, 0, 0, 0], 3)[:, None]
    # Seed contributes four windows; each subsequent trial accepts two.
    assert selector.evaluate(features, "UD", labels, trial_var=np.repeat([0, 1, 2], 4)) == pytest.approx(200 / 3)


@pytest.fixture
def ranked_data():
    rng = np.random.default_rng(7)
    labels = np.repeat([3, 8, 12], 30)
    features = {"noise": rng.normal(size=(90, 2)),
                "signal": labels[:, None] + rng.normal(0, 0.1, (90, 1)),
                "weak": labels[:, None] + rng.normal(0, 4, (90, 1))}
    return features, labels


def test_selection_strategies_results_and_no_mutation(ranked_data, capsys):
    features, labels = ranked_data
    originals = {name: x.copy() for name, x in features.items()}
    selector = FeatureSelector()
    for metric in ("CA", "SI", "FE"):
        results, names = selector.run_selection(features, metric, labels, early_stop=2, verbose=True)
        assert names[0] == "signal"
        assert len(names) == 2
        if metric == "FE":
            assert results.shape == (2,)
            assert results[0] >= results[1]
        else:
            assert results.shape == (3, 2)
            assert np.isnan(results[1, 0]) and np.isnan(results[2]).all()
        for name, x in features.items():
            np.testing.assert_array_equal(x, originals[name])
    assert "iteration" in capsys.readouterr().out
    first = selector.run_selection(features, "CA", labels)
    second = selector.run_selection(features, "accuracy", labels)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_objective_override_and_stable_ties(analytic):
    x, y, groups = analytic
    selector = FeatureSelector()
    data = {"first": x, "second": x * 2}
    assert selector.run_selection(data, "MSA", y, strategy="filter")[1] == ["first", "second"]
    assert selector.run_selection(data, "MSA", y, strategy="filter", objective="max")[1] == ["second", "first"]
    for strategy in ("filter", "sequential"):
        scores, names = selector.run_selection(data, "TT", y, strategy=strategy, training_time_ms=1234)
        assert names == ["first", "second"]
        assert np.nanmin(scores) == np.nanmax(scores) == 1234


def test_prism_analytic_resolution_curves():
    selector = FeatureSelector(prism_resolution=1, allow_approximations=True)
    # B=0: balanced mixed cell (purity=entropy=0); B=1: two pure cells (1).
    x, y = [0, 0.1, 0.9, 1], [0, 0, 1, 1]
    for code in ("PU", "CE"):
        assert selector.evaluate(x, code, y) == pytest.approx(0.5)
        assert selector.evaluate(x, "r" + code, y) == pytest.approx(1 / 3)
    # Documented concentration variant: one cell=1, two equal cells=1/2.
    assert selector.evaluate(x, "C", y) == pytest.approx(0.75)
    assert selector.evaluate(x, "rC", y) == pytest.approx(5 / 6)
    for code in ("PU", "rPU", "NS", "rNS", "CE", "rCE", "C", "rC"):
        assert 0 <= selector.evaluate(x, code, y) <= 1


def test_cdm_mixed_duplicate_cells_and_nonlinear_partition():
    selector = FeatureSelector(cdm_min_samples=2, allow_approximations=True)
    assert selector.evaluate([0, 0, 1, 1], "CDM", [0, 1, 0, 1]) == 0.5
    xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert selector.evaluate(xor, "CDM", [0, 1, 1, 0]) == 0
    assert FeatureSelector(cdm_min_samples=4, allow_approximations=True).evaluate(xor, "CDM", [0, 1, 1, 0]) == 0.5


@pytest.mark.parametrize("code", ["C", "rC", "CDM"])
def test_unverified_variants_require_explicit_opt_in(analytic, code):
    x, y, _ = analytic
    with pytest.raises(ValueError, match="allow_approximations=True"):
        FeatureSelector().evaluate(x, code, y)


@pytest.mark.parametrize("code", ["SI", "mSI", "BD", "KLD", "HD", "MSA", "FDR", "TSM"])
def test_singular_multidimensional_features(analytic, code):
    x, y, _ = analytic
    duplicated = np.hstack((x, x, np.zeros_like(x)))
    assert np.isfinite(FeatureSelector().evaluate(duplicated, code, y))


@pytest.mark.parametrize("kwargs, message", [
    ({"data": {}}, "at least one"),
    ({"data": {"bad": [0, np.nan, 1, 2]}}, "NaN"),
    ({"data": {"a": [0, 1], "b": [0, 1, 2]}}, "same number"),
    ({"class_var": [0, 0, 0, 0]}, "two classes"),
    ({"class_var": [0, 1]}, "one value"),
    ({"metric": "unknown"}, "Unknown metric"),
    ({"early_stop": 0}, "early_stop"),
    ({"strategy": "random"}, "strategy"),
    ({"objective": "minimum"}, "objective"),
    ({"metric": "RI"}, "repetition_var"),
    ({"metric": "mwSI"}, "trial_var"),
    ({"metric": "MAV"}, "raw_emg"),
    ({"metric": "TT"}, "TT requires"),
    ({"metric": "CA", "crossvalidation_var": {"var": [0, 0, 0, 0]}}, "two groups"),
    ({"metric": "CA", "crossvalidation_var": {"crossval_amount": 3}}, "Each class"),
    ({"crossvalidation_var": {"var": [0, 1, 0, 1], "crossval_amount": 2}}, "either"),
])
def test_invalid_inputs(kwargs, message):
    arguments = dict(data={"a": [0, 1, 2, 3]}, metric="FE", class_var=[0, 0, 1, 1])
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=message):
        FeatureSelector().run_selection(**arguments)


def test_missing_class_recording_and_undefined_ratios():
    selector = FeatureSelector()
    with pytest.raises(ValueError, match="every class"):
        selector.evaluate([0, 1, 2, 3], "RI", [0, 0, 1, 1], repetition_var=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="0/0"):
        selector.evaluate([0, 0, 0, 0], "IIF", [0, 0, 1, 1])
    with pytest.raises(ValueError, match="all classes"):
        selector.evaluate([0, 1, 2, 3], "CA", [0, 0, 1, 1], {"var": [0, 0, 1, 1]})


@pytest.fixture(scope="module")
def myo_features(tmp_path_factory):
    # Stage the checked-in real Myo recordings in the dataset module's layout.
    # This exercises its actual loader without requiring a download in CI.
    folder = tmp_path_factory.mktemp("selector_myo")
    source = Path(__file__).parent / "data" / "myo_dataset"
    for trial in range(3):
        destination = folder / f"trial_{trial + 1}"
        destination.mkdir()
        for label in range(3):
            recording = np.loadtxt(source / f"R_{trial}_C_{label}_EMG.csv", delimiter=",")
            for repetition in range(2):
                segment = recording[repetition * 200:(repetition + 1) * 200]
                np.savetxt(destination / f"R_{repetition}_C_{label}.csv", segment, delimiter=",")
    from examples.feature_selection_metrics import load_example
    return load_example(str(folder))


@pytest.mark.parametrize("metric", FeatureSelector().get_metrics(abbreviations=True))
def test_every_metric_with_dataset_module(myo_features, metric):
    features, labels, crossvalidation, options = myo_features
    selector = FeatureSelector(allow_approximations=True)
    score = selector.evaluate(features, metric, labels, crossvalidation, **options)
    assert np.isfinite(score)
    for strategy in ("filter", "sequential"):
        results, order = selector.run_selection(features, metric, labels, crossvalidation,
                                                 strategy=strategy, early_stop=2, **options)
        assert len(order) == len(set(order)) == 2
        assert set(order) <= set(features)
        if strategy == "filter":
            assert results.shape == (2,)
            assert results[0] == pytest.approx(selector.evaluate(features[order[0]], metric, labels,
                                                                crossvalidation, **options))
        else:
            assert results.shape == (3, 2)
            assert np.isnan(results[1, 0]) and np.isnan(results[2]).all()
            selected = np.hstack([features[name] for name in order])
            assert results[1, 1] == pytest.approx(selector.evaluate(selected, metric, labels,
                                                                   crossvalidation, **options))
