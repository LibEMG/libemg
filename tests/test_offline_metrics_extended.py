"""Coverage for the parts of OfflineMetrics the original suite leaves out.

tests/test_offline_metrics.py checks eight classification metrics against
sklearn. offline_metrics.py is 2000 lines; the other two families had nothing:

* Regression metrics -- checked against sklearn, or hand-computed where
  sklearn has no equivalent (NRMSE, MAPE).
* Separability metrics -- there is no reference implementation to compare
  against, and inventing expected values would only pin in whatever the code
  does today. Instead these are tested as *orderings*: a metric that claims to
  measure class separation must score well-separated synthetic Gaussians
  better than overlapping ones. That is cheap, robust to implementation
  detail, and catches sign flips, axis mix-ups and swapped arguments.

Data model for the separability metrics
---------------------------------------
``features_train`` is a list of feature matrices (one per subject or file).
``reps_train`` is a parallel list of dicts mapping class label -> list of
(start, end) row ranges, one range per repetition. Class 0 is rest and is
excluded from the class loops, so the fixtures below include it.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from libemg.offline_metrics import OfflineMetrics

N_CLASSES = 4
N_REPS = 4
SAMPLES_PER_REP = 50
N_FEATURES = 3


@pytest.fixture(scope="module")
def om():
    return OfflineMetrics()


# --------------------------------------------------------------------------- #
# Regression metrics
# --------------------------------------------------------------------------- #


@pytest.fixture
def regression_pair(seeded_rng):
    truth = seeded_rng.normal(size=(200, 2))
    predictions = truth + seeded_rng.normal(scale=0.3, size=(200, 2))
    return truth, predictions


class TestRegressionMetrics:
    def test_R2_matches_sklearn(self, om, regression_pair):
        truth, predictions = regression_pair
        np.testing.assert_allclose(
            om.get_R2(truth, predictions),
            r2_score(truth, predictions, multioutput="raw_values"),
            rtol=1e-9,
        )

    def test_MSE_matches_sklearn(self, om, regression_pair):
        truth, predictions = regression_pair
        np.testing.assert_allclose(
            om.get_MSE(truth, predictions),
            mean_squared_error(truth, predictions, multioutput="raw_values"),
            rtol=1e-9,
        )

    def test_MAE_matches_sklearn(self, om, regression_pair):
        truth, predictions = regression_pair
        np.testing.assert_allclose(
            om.get_MAE(truth, predictions),
            mean_absolute_error(truth, predictions, multioutput="raw_values"),
            rtol=1e-9,
        )

    def test_RMSE_is_the_root_of_MSE(self, om, regression_pair):
        truth, predictions = regression_pair
        np.testing.assert_allclose(
            om.get_RMSE(truth, predictions),
            np.sqrt(om.get_MSE(truth, predictions)),
            rtol=1e-9,
        )

    def test_NRMSE_is_RMSE_over_the_range(self, om, regression_pair):
        """No sklearn equivalent, so compute it the documented way."""
        truth, predictions = regression_pair
        expected = om.get_RMSE(truth, predictions) / (truth.max(axis=0) - truth.min(axis=0))
        np.testing.assert_allclose(om.get_NRMSE(truth, predictions), expected, rtol=1e-9)

    def test_MAPE_on_hand_computed_values(self, om):
        truth = np.array([[100.0], [200.0], [400.0]])
        predictions = np.array([[110.0], [180.0], [400.0]])
        # errors: 10%, 10%, 0% -> mean 6.666...%, one value per DOF
        np.testing.assert_allclose(om.get_MAPE(truth, predictions), [0.2 / 3], rtol=1e-9)

    def test_perfect_prediction_is_perfect(self, om, regression_pair):
        truth, _ = regression_pair
        np.testing.assert_allclose(om.get_R2(truth, truth), 1.0, atol=1e-12)
        np.testing.assert_allclose(om.get_MSE(truth, truth), 0.0, atol=1e-12)
        np.testing.assert_allclose(om.get_MAE(truth, truth), 0.0, atol=1e-12)
        np.testing.assert_allclose(om.get_RMSE(truth, truth), 0.0, atol=1e-12)

    def test_worse_predictions_score_worse(self, om, seeded_rng):
        truth = seeded_rng.normal(size=(200, 2))
        close = truth + seeded_rng.normal(scale=0.1, size=(200, 2))
        far = truth + seeded_rng.normal(scale=2.0, size=(200, 2))

        assert np.all(om.get_MSE(truth, close) < om.get_MSE(truth, far))
        assert np.all(om.get_MAE(truth, close) < om.get_MAE(truth, far))
        assert np.all(om.get_R2(truth, close) > om.get_R2(truth, far))


# --------------------------------------------------------------------------- #
# extract_offline_metrics
# --------------------------------------------------------------------------- #


class TestExtractOfflineMetrics:
    def test_returns_every_requested_metric(self, om, y_true, y_predictions):
        metrics = ["CA", "RECALL", "PREC", "F1"]
        result = om.extract_offline_metrics(metrics, y_true, y_predictions)
        assert set(result) == set(metrics)

    def test_common_metrics_with_a_null_label(self, om, y_true, y_predictions):
        """With a null label supplied, all three common metrics come back."""
        result = om.extract_common_metrics(y_true, y_predictions, null_label=2)
        assert set(result) == set(om.get_common_metrics())

    def test_common_metrics_without_a_null_label_omit_AER(self, om, y_true, y_predictions):
        """AER needs a null class; without one it is skipped rather than failing."""
        result = om.extract_common_metrics(y_true, y_predictions)
        assert set(result) == set(om.get_common_metrics()) - {"AER"}

    def test_unknown_metric_raises_attribute_error(self, om, y_true, y_predictions):
        with pytest.raises(AttributeError):
            om.extract_offline_metrics(["NOT_A_METRIC"], y_true, y_predictions)

    def test_mismatched_lengths_are_rejected(self, om, y_true, y_predictions):
        with pytest.raises(AssertionError):
            om.extract_offline_metrics(["CA"], y_true, y_predictions[:-1])

    def test_AER_without_null_label_is_skipped_not_fatal(self, om, y_true, y_predictions):
        result = om.extract_offline_metrics(["CA", "AER"], y_true, y_predictions)
        assert "CA" in result
        assert "AER" not in result

    def test_available_metrics_all_have_getters(self, om):
        for metric in om.get_available_metrics():
            assert hasattr(om, f"get_{metric}"), f"get_available_metrics lists {metric}, "
            "but there is no get_{metric} to call"


# --------------------------------------------------------------------------- #
# Separability metrics, as orderings
# --------------------------------------------------------------------------- #


def build_separability_fixture(separation=10.0, scatter=1.0, drift=0.0, seed=0):
    """Synthetic Gaussian classes in the layout OfflineMetrics expects.

    Parameters
    ----------
    separation: distance between adjacent class centres.
    scatter: within-class standard deviation.
    drift: how far a class centre moves between successive repetitions.
    """
    rng = np.random.default_rng(seed)
    rows = []
    ranges = {c: [] for c in range(N_CLASSES)}
    cursor = 0
    for label in range(N_CLASSES):
        for rep in range(N_REPS):
            centre = np.zeros(N_FEATURES)
            centre[0] = label * separation + rep * drift
            rows.append(rng.normal(centre, scatter, size=(SAMPLES_PER_REP, N_FEATURES)))
            ranges[label].append((cursor, cursor + SAMPLES_PER_REP))
            cursor += SAMPLES_PER_REP
    return [np.vstack(rows)], [ranges]


# Metrics whose value should rise as classes move apart.
HIGHER_IS_MORE_SEPARABLE = ["SI", "mSI", "KLD", "HD", "FE", "TSM", "mwSI", "BD"]

# Metrics that measure overlap, so the value should fall as classes move apart.
LOWER_IS_MORE_SEPARABLE = ["VOR", "ICF"]


class TestSeparabilityOrdering:
    @pytest.mark.parametrize("metric", HIGHER_IS_MORE_SEPARABLE)
    def test_separated_classes_score_higher(self, om, metric):
        separated = getattr(om, f"get_{metric}")(*build_separability_fixture(10.0), N_CLASSES)
        overlapping = getattr(om, f"get_{metric}")(*build_separability_fixture(0.2), N_CLASSES)
        assert separated > overlapping, (
            f"{metric} scored well-separated classes ({separated:.4f}) no better than "
            f"overlapping ones ({overlapping:.4f})"
        )

    @pytest.mark.parametrize("metric", LOWER_IS_MORE_SEPARABLE)
    def test_separated_classes_score_lower(self, om, metric):
        separated = getattr(om, f"get_{metric}")(*build_separability_fixture(10.0), N_CLASSES)
        overlapping = getattr(om, f"get_{metric}")(*build_separability_fixture(0.2), N_CLASSES)
        assert separated < overlapping, (
            f"{metric} measures overlap, so separated classes ({separated:.4f}) should "
            f"score below overlapping ones ({overlapping:.4f})"
        )

    @pytest.mark.parametrize("metric", HIGHER_IS_MORE_SEPARABLE + LOWER_IS_MORE_SEPARABLE)
    def test_metric_is_finite(self, om, metric):
        value = getattr(om, f"get_{metric}")(*build_separability_fixture(), N_CLASSES)
        assert np.isfinite(value), f"{metric} returned {value}"

    def test_MSA_grows_with_within_class_scatter(self, om):
        """Mean Semi-Principal Axes sizes the training ellipsoid, so it tracks
        spread, not separation."""
        tight = om.get_MSA(*build_separability_fixture(scatter=0.5), N_CLASSES)
        diffuse = om.get_MSA(*build_separability_fixture(scatter=4.0), N_CLASSES)
        assert tight < diffuse

    def test_CD_grows_with_centroid_drift(self, om):
        """Centroid Drift measures movement between successive repetitions."""
        stable = om.get_CD(*build_separability_fixture(drift=0.0), N_CLASSES)
        drifting = om.get_CD(*build_separability_fixture(drift=6.0), N_CLASSES)
        assert stable < drifting


class TestSeparabilityKnownDefects:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "get_BD loops `for i in range(1, len(reps_train[0][x]))`, which is the "
            "number of repetition segments for class x, where every sibling metric "
            "loops over the number of classes. With one segment per class the "
            "inner list stays empty and min() raises; with more than one it "
            "compares against the wrong set of classes."
        ),
    )
    def test_BD_handles_a_single_segment_per_class(self, om):
        global N_REPS
        original = N_REPS
        try:
            N_REPS = 1
            features, ranges = build_separability_fixture(10.0)
        finally:
            N_REPS = original
        assert np.isfinite(om.get_BD(features, ranges, N_CLASSES))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "get_IIF ends in `num_sum / dnum_sum` with no guard. When classes are "
            "cleanly separable every nearest neighbour shares its label, dnum_sum "
            "stays 0, and the metric raises ZeroDivisionError on exactly the data "
            "it should score best."
        ),
    )
    def test_IIF_on_perfectly_separable_data(self, om):
        assert np.isfinite(om.get_IIF(*build_separability_fixture(50.0), N_CLASSES))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "_get_KL_divergence hardcodes `term3 = 24` where the Gaussian KL "
            "divergence needs the feature dimensionality. The result is off by "
            "(d - 24)/2 for any feature count other than 24, which is why KLD "
            "goes negative on low-dimensional data."
        ),
    )
    def test_KLD_is_non_negative(self, om):
        """KL divergence between two distributions is >= 0 by definition."""
        value = om.get_KLD(*build_separability_fixture(0.2), N_CLASSES)
        assert value >= 0, f"KLD returned {value:.4f}"
