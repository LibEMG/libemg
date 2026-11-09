"""Unit tests for EMGClassifier and EMGRegressor.

emg_predictor.py is the second most-edited non-GUI module in the library and
had no direct test coverage. The tests here split into two kinds:

* Model-zoo contracts -- every advertised model string must fit and predict.
  These catch a model being renamed, a default parameter going stale, or a
  sklearn upgrade breaking one estimator but not the rest.
* Post-processing behaviour -- rejection, majority vote, velocity and deadband
  are pure transforms over prediction arrays. They are tested on hand-built
  inputs, with no model fitting at all, so a failure points at one function.
"""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from libemg.emg_predictor import EMGClassifier, EMGPredictor, EMGRegressor
from libemg.feature_extractor import FeatureExtractor

from tests.conftest import WINDOW_INCREMENT, WINDOW_SIZE

CLASSIFIER_MODELS = ["LDA", "KNN", "SVM", "QDA", "RF", "NB", "GB", "MLP"]

# EMGRegressor('SVM') is SVR(kernel='linear') wrapped per output. On unscaled
# EMG features it takes ~5 minutes to converge on even this small fixture,
# against well under a second for every other model. It is covered, but behind
# --slow so the default run stays fast. (The runtime itself is worth a look:
# the default is effectively unusable without feature scaling.)
REGRESSOR_MODELS = [
    "LR",
    pytest.param("SVM", marks=pytest.mark.slow),
    "RF",
    "GB",
    "MLP",
]


@pytest.fixture(scope="module")
def training_set(request):
    """HTD features and labels from the bundled dataset, split by rep."""
    import tests.conftest as c

    odh = c.load_myo_odh()
    windows, metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    features = FeatureExtractor().extract_feature_group("HTD", windows)
    labels = metadata["classes"].squeeze().astype(int)
    reps = metadata["reps"].squeeze().astype(int)
    train, test = reps != 2, reps == 2
    return {
        "windows": windows,
        "train_features": {k: v[train] for k, v in features.items()},
        "train_labels": labels[train],
        "train_windows": windows[train],
        "test_features": {k: v[test] for k, v in features.items()},
        "test_labels": labels[test],
    }


def fit_classifier(model, training_set, **kwargs):
    classifier = EMGClassifier(model, **kwargs)
    classifier.fit(
        {
            "training_features": training_set["train_features"],
            "training_labels": training_set["train_labels"],
        }
    )
    return classifier


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #


class TestClassifierModels:
    @pytest.mark.parametrize("model", CLASSIFIER_MODELS)
    def test_every_model_fits_and_predicts(self, model, training_set):
        classifier = fit_classifier(model, training_set)
        predictions, probabilities = classifier.run(training_set["test_features"])

        n_test = len(training_set["test_labels"])
        n_classes = len(np.unique(training_set["train_labels"]))

        assert len(predictions) == n_test
        assert np.asarray(probabilities).shape[0] == n_test
        assert set(np.unique(predictions)) <= set(np.unique(training_set["train_labels"])), (
            f"{model} predicted a label it was never trained on"
        )
        assert np.all(np.isfinite(np.asarray(probabilities)))
        assert n_classes > 1

    @pytest.mark.parametrize("model", CLASSIFIER_MODELS)
    def test_probabilities_are_valid(self, model, training_set):
        classifier = fit_classifier(model, training_set)
        _, probabilities = classifier.run(training_set["test_features"])
        probabilities = np.asarray(probabilities)
        assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0 + 1e-9)

    def test_unknown_model_string_is_rejected(self):
        with pytest.raises(AssertionError, match="approved models"):
            EMGClassifier("NOT_A_MODEL")

    def test_custom_model_object_is_accepted(self, training_set):
        """Anything with fit/predict/predict_proba, not just the named strings."""
        classifier = EMGClassifier(LinearDiscriminantAnalysis())
        classifier.fit(
            {
                "training_features": training_set["train_features"],
                "training_labels": training_set["train_labels"],
            }
        )
        predictions, _ = classifier.run(training_set["test_features"])
        assert len(predictions) == len(training_set["test_labels"])

    def test_fit_without_any_data_raises(self):
        with pytest.raises(ValueError, match="Incorrect combination"):
            EMGClassifier("LDA").fit()

    def test_model_without_predict_proba_raises_clearly(self, training_set):
        class NoProba:
            def fit(self, x, y):
                self.classes_ = np.unique(y)

            def predict(self, x):
                return np.zeros(len(x))

        classifier = EMGClassifier(NoProba())
        classifier.fit(
            {
                "training_features": training_set["train_features"],
                "training_labels": training_set["train_labels"],
            }
        )
        with pytest.raises(AttributeError, match="predict_proba"):
            classifier.run(training_set["test_features"])


class TestRegressorModels:
    @pytest.fixture(scope="class")
    def regression_set(self, training_set):
        classes = training_set["train_labels"].astype(float)
        return np.column_stack([classes / 4.0, 1.0 - classes / 4.0])

    @pytest.mark.parametrize("model", REGRESSOR_MODELS)
    def test_every_model_fits_and_predicts(self, model, training_set, regression_set):
        regressor = EMGRegressor(model)
        regressor.fit(
            {
                "training_features": training_set["train_features"],
                "training_labels": regression_set,
            }
        )
        predictions = regressor.run(training_set["test_features"])

        assert predictions.shape == (len(training_set["test_labels"]), 2), (
            f"{model} should emit one value per DOF per window"
        )
        assert np.all(np.isfinite(predictions))

    def test_string_models_are_wrapped_for_multioutput(self, training_set, regression_set):
        """A named model is wrapped in MultiOutputRegressor; a passed-in object
        is used as-is and must handle multi-output itself."""
        from sklearn.multioutput import MultiOutputRegressor

        assert isinstance(EMGRegressor("LR").model, MultiOutputRegressor)

    def test_unknown_model_string_is_rejected(self):
        with pytest.raises(AssertionError, match="approved models"):
            EMGRegressor("NOT_A_MODEL")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_load_roundtrip_preserves_predictions(training_set, tmp_path):
    """A pickled model is the one artifact users keep across library versions."""
    original = fit_classifier("LDA", training_set)
    expected, expected_probabilities = original.run(training_set["test_features"])

    path = tmp_path / "lda.pickle"
    original.save(str(path))
    assert path.exists()

    restored = EMGClassifier.from_file(str(path))
    predictions, probabilities = restored.run(training_set["test_features"])

    np.testing.assert_array_equal(predictions, expected)
    np.testing.assert_allclose(probabilities, expected_probabilities, rtol=0, atol=0)


def test_save_load_preserves_post_processing(training_set, tmp_path):
    """Rejection and majority vote settings must survive the round trip."""
    original = fit_classifier("LDA", training_set)
    original.add_rejection(0.8)
    original.add_majority_vote(7)

    path = tmp_path / "configured.pickle"
    original.save(str(path))
    restored = EMGClassifier.from_file(str(path))

    assert restored.rejection is True
    assert restored.rejection_threshold == 0.8
    assert restored.majority_vote == 7


# --------------------------------------------------------------------------- #
# Post-processing, on hand-built inputs
# --------------------------------------------------------------------------- #


class TestRejection:
    def make(self, threshold=0.9):
        classifier = EMGClassifier("LDA")
        classifier.add_rejection(threshold)
        return classifier

    def test_confident_predictions_pass_through(self):
        assert self.make()._rejection_helper(3, 0.95) == 3

    def test_unconfident_predictions_become_minus_one(self):
        assert self.make()._rejection_helper(3, 0.5) == -1

    def test_threshold_is_exclusive(self):
        """A probability exactly at the threshold is rejected."""
        assert self.make(0.9)._rejection_helper(2, 0.9) == -1

    def test_disabled_by_default(self):
        assert EMGClassifier("LDA").rejection is False

    def test_rejection_appears_in_real_output(self, training_set):
        """With an impossible threshold, everything is rejected."""
        classifier = fit_classifier("LDA", training_set)
        classifier.add_rejection(1.01)
        predictions, _ = classifier.run(training_set["test_features"])
        assert set(np.unique(predictions)) == {-1}


class TestMajorityVote:
    def vote(self, predictions, num_samples):
        classifier = EMGClassifier("LDA")
        classifier.add_majority_vote(num_samples)
        return list(classifier._majority_vote_helper(np.asarray(predictions)))

    def test_smooths_a_single_outlier(self):
        assert self.vote([1, 1, 1, 2, 1, 1, 1], 5) == [1, 1, 1, 1, 1, 1, 1]

    def test_is_causal(self):
        """The vote looks backwards only -- a window must never be decided by
        samples that have not arrived yet."""
        assert self.vote([1, 1, 1, 9, 9, 9, 9], 3)[:3] == [1, 1, 1]

    def test_leading_window_uses_what_is_available(self):
        """Before num_samples predictions exist, vote over the partial window."""
        assert self.vote([4, 4, 4], 5) == [4, 4, 4]

    def test_window_of_one_is_a_no_op(self):
        original = [0, 3, 1, 2, 2]
        assert self.vote(original, 1) == original

    def test_transition_lags_by_the_window(self):
        """A genuine class change takes effect once it wins the vote.

        With a 3-sample window the new class takes over at index 4, the first
        position where two of the three samples in view are the new class.
        """
        assert self.vote([1, 1, 1, 2, 2, 2, 2], 3) == [1, 1, 1, 1, 2, 2, 2]


class TestVelocity:
    def test_thresholds_are_set_per_class(self, training_set):
        classifier = fit_classifier("LDA", training_set)
        classifier.add_velocity(training_set["train_windows"], training_set["train_labels"])

        classes = set(np.unique(training_set["train_labels"]).tolist())
        assert classifier.velocity is True
        assert set(classifier.th_min_dic) == classes
        assert set(classifier.th_max_dic) == classes

    def test_max_threshold_is_above_min(self, training_set):
        classifier = fit_classifier("LDA", training_set)
        classifier.add_velocity(training_set["train_windows"], training_set["train_labels"])
        for label in classifier.th_min_dic:
            assert classifier.th_max_dic[label] >= classifier.th_min_dic[label]

    def test_custom_metric_handle_is_used(self, training_set):
        """A caller-supplied intensity metric replaces the default MAV sum."""
        calls = []

        def metric(windows):
            calls.append(windows.shape)
            return np.ones(windows.shape[0])

        classifier = fit_classifier("LDA", training_set)
        classifier.add_velocity(
            training_set["train_windows"],
            training_set["train_labels"],
            velocity_metric_handle=metric,
        )
        assert calls, "custom velocity metric was never called"
        assert all(v == 1.0 for v in classifier.th_min_dic.values())


class TestDeadband:
    def test_values_inside_the_band_become_zero(self, training_set):
        classes = training_set["train_labels"].astype(float)
        targets = np.column_stack([classes / 4.0, 1.0 - classes / 4.0])

        regressor = EMGRegressor("LR")
        regressor.fit(
            {
                "training_features": training_set["train_features"],
                "training_labels": targets,
            }
        )
        regressor.add_deadband(10.0)  # wider than any plausible prediction
        predictions = regressor.run(training_set["test_features"])
        assert np.all(predictions == 0.0)

    def test_default_is_no_deadband(self, training_set):
        classes = training_set["train_labels"].astype(float)
        targets = np.column_stack([classes / 4.0, 1.0 - classes / 4.0])
        regressor = EMGRegressor("LR")
        regressor.fit(
            {"training_features": training_set["train_features"], "training_labels": targets}
        )
        assert np.any(regressor.run(training_set["test_features"]) != 0.0)


# --------------------------------------------------------------------------- #
# Feature error handling
# --------------------------------------------------------------------------- #


class TestFixFeatureErrors:
    @staticmethod
    def corrupt(features):
        corrupted = {k: v.copy() for k, v in features.items()}
        key = sorted(corrupted)[0]
        corrupted[key][0, 0] = np.nan
        corrupted[key][1, 0] = np.inf
        corrupted[key][2, 0] = -np.inf
        return corrupted

    def test_repairs_nan_and_inf_when_enabled(self, training_set):
        classifier = EMGClassifier("LDA", fix_feature_errors=True, silent=True)
        classifier.fit(
            {
                "training_features": self.corrupt(training_set["train_features"]),
                "training_labels": training_set["train_labels"],
            }
        )
        predictions, _ = classifier.run(self.corrupt(training_set["test_features"]))
        assert len(predictions) == len(training_set["test_labels"])
        assert np.all(np.isfinite(predictions))

    def test_propagates_corruption_when_disabled(self, training_set):
        """Without the flag, bad features are not silently patched up."""
        classifier = EMGClassifier("LDA", fix_feature_errors=False)
        with pytest.raises(ValueError):
            classifier.fit(
                {
                    "training_features": self.corrupt(training_set["train_features"]),
                    "training_labels": training_set["train_labels"],
                }
            )


# --------------------------------------------------------------------------- #
# Feature parameters
# --------------------------------------------------------------------------- #


def test_install_feature_parameters_is_stored():
    predictor = EMGPredictor(LinearDiscriminantAnalysis())
    assert predictor.feature_params == {}
    predictor.install_feature_parameters({"AR_order": 9})
    assert predictor.feature_params == {"AR_order": 9}


def test_format_data_stacks_a_feature_dictionary():
    """A dict of per-feature matrices becomes one (windows, features) array."""
    predictor = EMGPredictor(LinearDiscriminantAnalysis())
    features = {"MAV": np.zeros((10, 8)), "WL": np.ones((10, 8))}
    stacked = predictor._format_data(features)
    assert stacked.shape == (10, 16)


def test_format_data_passes_arrays_through():
    predictor = EMGPredictor(LinearDiscriminantAnalysis())
    array = np.zeros((10, 16))
    assert predictor._format_data(array) is array
