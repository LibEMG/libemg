"""Unit tests for FeatureSelector.

feature_selector.py had the lowest coverage in the library (7%) and no tests
at all. It ranks features, so the property that matters is ordering: given
features of known quality, a selector must put the good one first. These tests
build synthetic features whose separability is known by construction, so the
expected ranking is not a guess.

Two selection strategies exist and are tested separately:

* **sequential** (accuracy, activeerror, meansemiprincipalaxis, repeatability,
  separability) -- greedy forward selection, returning an F x F upper-triangle
  matrix of metric values.
* **metric** (featureefficiency) -- scores each feature once, returning a
  length-F vector.
"""

import numpy as np
import pytest

from libemg.feature_selector import FeatureSelector

N_CLASSES = 4
PER_CLASS = 50
N_SAMPLES = N_CLASSES * PER_CLASS
COLUMNS = 2

SEQUENTIAL_METRICS = [
    "accuracy",
    "activeerror",
    "meansemiprincipalaxis",
    "repeatability",
    "separability",
]
METRIC_METRICS = ["featureefficiency"]
ALL_METRICS = SEQUENTIAL_METRICS + METRIC_METRICS

# Metrics whose ranking tracks class separability, so the feature built to be
# most separable must come first. The others measure ellipsoid size
# (meansemiprincipalaxis) or rep-to-rep drift (repeatability), which this
# fixture holds constant across features by design.
SEPARABILITY_RANKING_METRICS = [
    "accuracy",
    "activeerror",
    "separability",
    "featureefficiency",
]


@pytest.fixture
def labels():
    return np.repeat(np.arange(N_CLASSES), PER_CLASS)


@pytest.fixture
def feature_set(labels):
    """Three features whose class separability is known and very different.

    GOOD has class means 8 apart against unit noise, MID 2 apart, JUNK 0 apart
    (pure noise, carrying no class information at all).
    """

    def build(separation, seed):
        rng = np.random.default_rng(seed)
        return np.stack(
            [rng.normal(labels * separation, 1.0) for _ in range(COLUMNS)], axis=1
        )

    return {"GOOD": build(8.0, 1), "MID": build(2.0, 2), "JUNK": build(0.0, 3)}


@pytest.fixture
def selector():
    return FeatureSelector()


def crossval(folds=3):
    return {"crossval_amount": folds}


def leave_one_out(labels, groups=4):
    """Explicit cross-validation groups, which avoids the random shuffle in
    the crossval_amount path."""
    return {"var": np.tile(np.arange(groups), N_SAMPLES // groups)}


# --------------------------------------------------------------------------- #
# The metric registry
# --------------------------------------------------------------------------- #


class TestMetricRegistry:
    def test_get_metrics_lists_the_implemented_metrics(self, selector):
        assert set(selector.get_metrics()) == set(ALL_METRICS)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_every_advertised_metric_has_a_callback(self, selector, metric):
        """run_selection resolves the callback by name, so a listed metric with
        no _get_<name> method fails only at call time."""
        assert hasattr(selector, f"_get_{metric}")

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_every_metric_has_a_type_and_objective(self, selector, metric):
        metric_type, objective = selector._get_metric_type(metric)
        assert metric_type in ("sequential", "metric")
        assert objective in (np.nanargmax, np.nanargmin)

    def test_unknown_metric_is_rejected(self, selector):
        with pytest.raises(AssertionError):
            selector._get_metric_type("not_a_metric")

    def test_metric_types_match_the_two_strategies(self, selector):
        for metric in SEQUENTIAL_METRICS:
            assert selector._get_metric_type(metric)[0] == "sequential"
        for metric in METRIC_METRICS:
            assert selector._get_metric_type(metric)[0] == "metric"


# --------------------------------------------------------------------------- #
# Shapes and contracts
# --------------------------------------------------------------------------- #


class TestSelectionContract:
    @pytest.mark.parametrize("metric", SEQUENTIAL_METRICS)
    def test_sequential_returns_square_matrix_and_full_order(
        self, selector, feature_set, labels, metric
    ):
        results, order = selector.run_selection(
            feature_set, metric, labels, leave_one_out(labels)
        )
        n = len(feature_set)
        assert results.shape == (n, n)
        assert sorted(order) == sorted(feature_set), "every feature should be ranked once"
        assert len(set(order)) == n, "a feature was selected twice"

    @pytest.mark.parametrize("metric", METRIC_METRICS)
    def test_metric_returns_one_score_per_feature(
        self, selector, feature_set, labels, metric
    ):
        results, order = selector.run_selection(
            feature_set, metric, labels, leave_one_out(labels)
        )
        assert np.shape(results) == (len(feature_set),)
        assert sorted(order) == sorted(feature_set)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_results_are_finite(self, selector, feature_set, labels, metric):
        results, _ = selector.run_selection(
            feature_set, metric, labels, leave_one_out(labels)
        )
        finite = np.isfinite(np.asarray(results, dtype=float))
        assert finite.any(), f"{metric} produced no usable values"

    def test_sequential_matrix_is_upper_triangular(self, selector, feature_set, labels):
        """Each round scores one fewer candidate, so the lower triangle of the
        re-sorted matrix is NaN."""
        results, _ = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels)
        )
        n = results.shape[0]
        for row in range(n):
            assert np.isnan(results[row, :row]).all(), (
                f"row {row} should have {row} NaN entries on the left"
            )
            assert np.isfinite(results[row, row:]).all()

    def test_early_stop_limits_the_number_selected(self, selector, feature_set, labels):
        _, order = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels), early_stop=2
        )
        assert len(order) == 2

    def test_early_stop_keeps_the_best_features(self, selector, feature_set, labels):
        _, full = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels)
        )
        _, stopped = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels), early_stop=2
        )
        assert stopped == full[:2], "stopping early should not change the ranking"


# --------------------------------------------------------------------------- #
# Ranking: the thing the class exists to do
# --------------------------------------------------------------------------- #


class TestRanking:
    @pytest.mark.parametrize("metric", SEPARABILITY_RANKING_METRICS)
    def test_most_separable_feature_is_chosen_first(
        self, selector, feature_set, labels, metric
    ):
        _, order = selector.run_selection(
            feature_set, metric, labels, leave_one_out(labels)
        )
        assert order[0] == "GOOD", (
            f"{metric} ranked {order[0]} above GOOD, whose classes are 8 apart "
            "against unit noise"
        )

    @pytest.mark.parametrize("metric", SEPARABILITY_RANKING_METRICS)
    def test_pure_noise_is_ranked_last(self, selector, feature_set, labels, metric):
        _, order = selector.run_selection(
            feature_set, metric, labels, leave_one_out(labels)
        )
        assert order[-1] == "JUNK", (
            f"{metric} ranked {order[-1]} below JUNK, which carries no class "
            "information at all"
        )

    def test_accuracy_improves_as_features_accumulate(
        self, selector, feature_set, labels
    ):
        """Greedy forward selection: the best score of each round should not
        fall as more features are added."""
        results, _ = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels)
        )
        best_per_round = [np.nanmax(results[i, :]) for i in range(results.shape[0])]
        assert best_per_round[0] <= max(best_per_round)
        assert best_per_round[0] > 90, (
            "GOOD alone should already classify this fixture almost perfectly"
        )

    def test_accuracy_is_a_percentage(self, selector, feature_set, labels):
        results, _ = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels)
        )
        finite = results[np.isfinite(results)]
        assert finite.min() >= 0.0 and finite.max() <= 100.0

    def test_activeerror_agrees_with_accuracy(self, selector, feature_set, labels):
        """Both rank by class separation, in opposite directions, so the
        feature each picks first should be the same one.

        Uses the default nm_class=2 -- see TestKnownDefects for why any other
        value does not work.
        """
        _, by_accuracy = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels)
        )
        _, by_error = selector.run_selection(
            feature_set, "activeerror", labels, leave_one_out(labels), nm_class=2
        )
        assert by_accuracy[0] == by_error[0]


# --------------------------------------------------------------------------- #
# Cross-validation plumbing
# --------------------------------------------------------------------------- #


class TestCrossValidation:
    def test_explicit_groups_are_used(self, selector, feature_set, labels):
        results, _ = selector.run_selection(
            feature_set, "accuracy", labels, {"var": np.tile(np.arange(5), N_SAMPLES // 5)}
        )
        assert np.isfinite(results).any()

    def test_a_single_group_is_rejected(self, selector, feature_set, labels):
        """Leave-one-out needs at least two groups to leave one out of."""
        with pytest.raises(AssertionError):
            selector.run_selection(
                feature_set, "accuracy", labels, {"var": np.zeros(N_SAMPLES)}
            )

    def test_k_fold_runs(self, selector, feature_set, labels):
        results, order = selector.run_selection(
            feature_set, "accuracy", labels, crossval(3)
        )
        assert np.isfinite(results).any()
        assert sorted(order) == sorted(feature_set)

    def test_explicit_groups_give_reproducible_results(
        self, selector, feature_set, labels
    ):
        """The 'var' path does no shuffling, so repeated runs must agree."""
        first, first_order = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels)
        )
        second, second_order = selector.run_selection(
            feature_set, "accuracy", labels, leave_one_out(labels)
        )
        np.testing.assert_allclose(first, second, equal_nan=True)
        assert first_order == second_order


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


class TestPrint:
    @pytest.mark.parametrize("metric", SEQUENTIAL_METRICS)
    def test_print_sequential_results(self, selector, feature_set, labels, metric, capsys):
        results, order = selector.run_selection(
            feature_set, metric, labels, leave_one_out(labels)
        )
        selector.print(metric, results, order)
        output = capsys.readouterr().out
        assert output.strip(), f"print produced nothing for {metric}"
        for name in feature_set:
            assert name in output

    def test_print_metric_results(self, selector, feature_set, labels, capsys):
        results, order = selector.run_selection(
            feature_set, "featureefficiency", labels, leave_one_out(labels)
        )
        selector.print("featureefficiency", results, order)
        assert capsys.readouterr().out.strip()


class TestKnownDefects:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "run_selection(verbose=True) calls self._print_wrapper_results and "
            "self._print_filter_results, neither of which exists -- the methods "
            "are named _print_sequential_results and _print_metric_results. The "
            "documented verbose flag raises AttributeError for every metric."
        ),
    )
    @pytest.mark.parametrize("metric", ["accuracy"])
    def test_verbose_flag_prints_instead_of_raising(
        self, selector, feature_set, labels, metric, capsys
    ):
        selector.run_selection(
            feature_set, metric, labels, leave_one_out(labels), verbose=True
        )
        assert capsys.readouterr().out.strip()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "_get_activeerror ends in "
            "`sum(errors + activepredictions == dictionary['nm'])`, comparing a "
            "0/1/2 count of conditions against the no-motion CLASS LABEL. The "
            "intended test is 'both conditions true', i.e. == 2, which is only "
            "correct because nm_class defaults to 2. With nm_class 0 or 1 the "
            "ranking inverts (noise is selected first); with 3 or above the "
            "metric is identically zero. The comparison should be to the "
            "literal 2, not to nm."
        ),
    )
    @pytest.mark.parametrize("nm_class", [0, 1])
    def test_activeerror_works_for_any_no_motion_label(
        self, selector, feature_set, labels, nm_class
    ):
        """With nm_class 0 or 1 the ranking inverts outright."""
        _, order = selector.run_selection(
            feature_set, "activeerror", labels, leave_one_out(labels), nm_class=nm_class
        )
        assert order[0] == "GOOD", (
            f"with nm_class={nm_class}, active error ranked {order[0]} first"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Same root cause as above, different symptom: `errors + "
            "activepredictions` can only be 0, 1 or 2, so comparing it to an "
            "nm_class of 3 or more is never true and the metric is identically "
            "zero for every feature. The selection then just returns the input "
            "order, which looks like a ranking but is not one."
        ),
    )
    def test_activeerror_is_not_degenerate_for_large_no_motion_labels(
        self, selector, feature_set, labels
    ):
        results, _ = selector.run_selection(
            feature_set, "activeerror", labels, leave_one_out(labels), nm_class=3
        )
        finite = results[np.isfinite(results)]
        assert len(np.unique(finite)) > 1, (
            f"active error returned the same value ({finite[0]}) for every "
            "feature, so the ranking carries no information"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The crossval_amount path calls np.random.shuffle without a seed "
            "(feature_selector.py:163), so k-fold selection is not reproducible "
            "run to run. A selection algorithm whose answer changes between "
            "identical calls cannot be reported in a paper. Threading a "
            "random_state through would fix it."
        ),
    )
    def test_k_fold_selection_is_reproducible(self, selector, feature_set, labels):
        first, _ = selector.run_selection(
            {k: v.copy() for k, v in feature_set.items()}, "accuracy", labels.copy(), crossval(3)
        )
        second, _ = selector.run_selection(
            {k: v.copy() for k, v in feature_set.items()}, "accuracy", labels.copy(), crossval(3)
        )
        np.testing.assert_allclose(first, second, equal_nan=True)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The crossval_amount path shuffles the caller's arrays in place "
            "(`data[f] = data[f][shuffle_id,:]`), so the feature dictionary "
            "passed in comes back reordered. Callers reusing that dictionary -- "
            "to compare two metrics, say -- silently get permuted data."
        ),
    )
    def test_k_fold_does_not_mutate_the_caller_s_data(self, selector, feature_set, labels):
        before = {k: v.copy() for k, v in feature_set.items()}
        selector.run_selection(feature_set, "accuracy", labels, crossval(3))
        for name, original in before.items():
            np.testing.assert_array_equal(
                feature_set[name], original, err_msg=f"{name} was modified in place"
            )
