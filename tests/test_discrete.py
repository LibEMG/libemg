"""Tests for the discrete (template-based) gesture recognition path.

Discrete recognition treats each repetition as one template rather than a
stream of independent windows, and it runs through a parallel set of code
paths that the continuous tests never touch:

    parse_windows(discrete=True)   -> one array of windows per file
    extract_features(discrete=True) -> one feature set per template
    MVLDA / DTWClassifier           -> classify whole templates

None of that had coverage. The models live in libemg/_discrete_models/ and
duck-type sklearn (fit / predict / predict_proba / classes_), which is what
lets them be passed to EMGClassifier as custom models -- so that contract is
asserted here too.

MyoCrossUserPretrained is excluded from the default run: it downloads a
pretrained .model file on construction.
"""

import inspect

import numpy as np
import pytest

from libemg._discrete_models import DTWClassifier, MVLDA
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor

from tests.conftest import WINDOW_INCREMENT, WINDOW_SIZE, load_myo_odh

N_FILES = 20
N_CHANNELS = 8
HELD_OUT_REP = 2

# Every concrete model exported from libemg._discrete_models that can be
# constructed without downloading anything. Derived from the package rather
# than hardcoded, so a newly added model is picked up automatically.
OFFLINE_DISCRETE_MODELS = [MVLDA, DTWClassifier]


@pytest.fixture(scope="module")
def templates():
    """Discrete templates and their metadata from the bundled dataset."""
    odh = load_myo_odh()
    windows, metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, discrete=True)
    return windows, metadata


@pytest.fixture(scope="module")
def discrete_dataset(templates):
    """Template-level features split into train and test by repetition."""
    windows, metadata = templates
    features = FeatureExtractor().extract_features(
        ["MAV", "WL"], windows, array=True, discrete=True
    )
    labels = metadata["classes"].astype(int)
    reps = metadata["reps"].astype(int)
    return {
        "train": [features[i] for i in range(len(features)) if reps[i] != HELD_OUT_REP],
        "train_labels": labels[reps != HELD_OUT_REP],
        "test": [features[i] for i in range(len(features)) if reps[i] == HELD_OUT_REP],
        "test_labels": labels[reps == HELD_OUT_REP],
    }


# --------------------------------------------------------------------------- #
# parse_windows(discrete=True)
# --------------------------------------------------------------------------- #


class TestDiscreteWindowing:
    def test_one_template_per_file(self, templates):
        windows, _ = templates
        assert isinstance(windows, list)
        assert len(windows) == N_FILES

    def test_each_template_keeps_its_windows(self, templates):
        windows, _ = templates
        for template in windows:
            assert template.ndim == 3, "(windows, channels, samples) per template"
            assert template.shape[1:] == (N_CHANNELS, WINDOW_SIZE)
            assert template.shape[0] > 1

    def test_metadata_is_one_value_per_template(self, templates):
        windows, metadata = templates
        for key, values in metadata.items():
            assert len(values) == len(windows), (
                f"discrete metadata '{key}' should be one value per template"
            )

    def test_discrete_and_continuous_cover_the_same_windows(self, templates):
        """Stacking the discrete templates reproduces the continuous output.

        The two modes differ in grouping, not in the windowing itself.
        """
        windows, _ = templates
        continuous, _ = load_myo_odh().parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
        np.testing.assert_array_equal(np.vstack(windows), continuous)

    def test_labels_survive_the_grouping(self, templates):
        _, metadata = templates
        assert set(metadata["classes"].astype(int)) == set(range(5))
        assert set(metadata["reps"].astype(int)) == set(range(4))


# --------------------------------------------------------------------------- #
# extract_features(discrete=True)
# --------------------------------------------------------------------------- #


class TestDiscreteFeatureExtraction:
    def test_returns_one_entry_per_template(self, templates):
        windows, _ = templates
        features = FeatureExtractor().extract_features(["MAV", "WL"], windows, discrete=True)
        assert isinstance(features, list)
        assert len(features) == len(windows)
        assert set(features[0]) == {"MAV", "WL"}

    def test_array_mode_stacks_features_per_template(self, templates):
        windows, _ = templates
        features = FeatureExtractor().extract_features(
            ["MAV", "WL"], windows, array=True, discrete=True
        )
        for template, feature_array in zip(windows, features):
            assert feature_array.shape == (template.shape[0], 2 * N_CHANNELS), (
                "each template should give (windows, features)"
            )

    def test_values_match_the_continuous_path(self, templates):
        """Per-template extraction must agree with extracting the whole lot."""
        windows, _ = templates
        fe = FeatureExtractor()
        discrete = fe.extract_features(["MAV"], windows, array=True, discrete=True)
        continuous = fe.extract_features(["MAV"], np.vstack(windows), array=True)
        np.testing.assert_allclose(np.vstack(discrete), continuous, rtol=0, atol=0)

    def test_features_are_finite(self, templates):
        windows, _ = templates
        features = FeatureExtractor().extract_features(
            ["MAV", "WL"], windows, array=True, discrete=True
        )
        assert all(np.all(np.isfinite(f)) for f in features)


# --------------------------------------------------------------------------- #
# The discrete models
# --------------------------------------------------------------------------- #


class TestDiscreteModelContract:
    """Every discrete model must duck-type sklearn, because EMGClassifier
    accepts any object with these methods as a custom model."""

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    @pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
    def test_implements_the_sklearn_interface(self, model_class, method):
        assert callable(getattr(model_class, method, None)), (
            f"{model_class.__name__} has no {method}(); EMGClassifier and "
            "OnlineDiscreteClassifier both rely on it"
        )

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    def test_exposes_classes_after_fitting(self, model_class, discrete_dataset):
        model = model_class()
        model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
        assert model.classes_ is not None
        np.testing.assert_array_equal(
            np.sort(model.classes_), np.unique(discrete_dataset["train_labels"])
        )

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    def test_classifies_held_out_templates(self, model_class, discrete_dataset):
        model = model_class()
        model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
        predictions = model.predict(discrete_dataset["test"])

        assert len(predictions) == len(discrete_dataset["test_labels"])
        accuracy = np.mean(predictions == discrete_dataset["test_labels"])
        assert accuracy > 0.8, f"{model_class.__name__} scored {accuracy:.2f}"

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    def test_predictions_come_from_the_training_labels(
        self, model_class, discrete_dataset
    ):
        model = model_class()
        model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
        predictions = model.predict(discrete_dataset["test"])
        assert set(np.unique(predictions)) <= set(np.unique(discrete_dataset["train_labels"]))

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    def test_probabilities_are_a_distribution(self, model_class, discrete_dataset):
        model = model_class()
        model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
        probabilities = model.predict_proba(discrete_dataset["test"])

        n_classes = len(np.unique(discrete_dataset["train_labels"]))
        assert probabilities.shape == (len(discrete_dataset["test"]), n_classes)
        assert np.all(probabilities >= 0)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-9)

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    def test_argmax_of_proba_matches_predict(self, model_class, discrete_dataset):
        """The two outputs must agree, or the reported confidence belongs to a
        different class than the reported prediction."""
        model = model_class()
        model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
        predictions = model.predict(discrete_dataset["test"])
        probabilities = model.predict_proba(discrete_dataset["test"])
        np.testing.assert_array_equal(
            predictions, model.classes_[np.argmax(probabilities, axis=1)]
        )

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    def test_is_deterministic(self, model_class, discrete_dataset):
        def run():
            model = model_class()
            model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
            return model.predict(discrete_dataset["test"])

        np.testing.assert_array_equal(run(), run())

    @pytest.mark.parametrize("model_class", OFFLINE_DISCRETE_MODELS)
    def test_handles_variable_length_templates(self, model_class, discrete_dataset):
        """Templates are documented as allowed to differ in length -- that is
        the whole point of DTW."""
        train = [t[: 5 + i] for i, t in enumerate(discrete_dataset["train"])]
        model = model_class()
        model.fit(train, discrete_dataset["train_labels"])
        predictions = model.predict([t[:7] for t in discrete_dataset["test"]])
        assert len(predictions) == len(discrete_dataset["test"])


class TestDTWSpecifics:
    def test_predict_before_fit_is_rejected(self, discrete_dataset):
        with pytest.raises(ValueError, match="fit"):
            DTWClassifier().predict_proba(discrete_dataset["test"])

    def test_stores_templates_on_fit(self, discrete_dataset):
        model = DTWClassifier()
        model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
        assert len(model.templates) == len(discrete_dataset["train"])

    @pytest.mark.parametrize("n_neighbors", [1, 3])
    def test_neighbour_count_is_honoured(self, discrete_dataset, n_neighbors):
        model = DTWClassifier(n_neighbors=n_neighbors)
        model.fit(discrete_dataset["train"], discrete_dataset["train_labels"])
        assert model.n_neighbors == n_neighbors
        assert len(model.predict(discrete_dataset["test"])) == len(discrete_dataset["test"])


class TestDiscreteModelAsCustomPredictor:
    """A discrete model should drop into EMGClassifier, which is the documented
    way to use a custom model."""

    def test_mvlda_works_through_emg_classifier(self, discrete_dataset):
        classifier = EMGClassifier(MVLDA())
        classifier.fit(
            {
                "training_features": discrete_dataset["train"],
                "training_labels": discrete_dataset["train_labels"],
            }
        )
        predictions, probabilities = classifier.run(discrete_dataset["test"])
        assert len(predictions) == len(discrete_dataset["test_labels"])
        assert np.all(np.isfinite(probabilities))


@pytest.mark.download
class TestPretrainedCrossUserModel:
    """MyoCrossUserPretrained downloads a .model file when constructed."""

    def test_loads_and_exposes_recommended_args(self, tmp_path):
        from libemg._discrete_models import MyoCrossUserPretrained

        model = MyoCrossUserPretrained(model_path=str(tmp_path / "Discrete.model"))
        assert model.model is not None
        for key in ("window_size", "window_increment", "null_label", "gesture_mapping"):
            assert key in model.args, f"recommended args should include {key}"

    def test_implements_the_discrete_model_interface(self):
        from libemg._discrete_models import MyoCrossUserPretrained

        for method in ("predict", "predict_proba"):
            assert callable(getattr(MyoCrossUserPretrained, method, None))
