"""Coverage for the parts of OfflineMetrics the original suite leaves out.

tests/test_offline_metrics.py checks the classification metrics against
sklearn. This module covers the two families it does not:

* Regression metrics -- checked against sklearn, or hand-computed where
  sklearn has no equivalent (NRMSE, MAPE).
* The extract_* / get_*_metrics entry points, including how a missing null
  label is handled.

Separability metrics used to live here too, against the OfflineMetrics
implementations added in #144. #144 has been reverted in favour of the
registry in libemg/_feature_selection_metrics.py (#147), which covers a
superset of the same metrics, fixes three defects these tests documented
(KLD's hardcoded dimensionality, IIF's division by zero, BD's loop bounds),
and is verified against closed-form values in tests/test_feature_selector.py.
Those tests are therefore not ported -- the replacement is stronger.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from libemg.offline_metrics import OfflineMetrics



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
