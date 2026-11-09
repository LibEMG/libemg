"""Contracts for every place a contributor adds something to libemg.

The suite is only a safety net if it covers code nobody has written yet. A
test that iterates a hardcoded list of models or filters passes happily when
someone adds a seventh one and wires it up wrong.

So every list here is **derived from the library itself** -- from a registry
function where one exists, and by parsing the source where the options live in
a dict literal or an if/elif chain. Add a model, a filter, a feature or a
dataset and these tests pick it up on the next run with no edit.

Where an extension point has no registry at all, that is recorded here too:
an un-enumerable extension point is one nothing can check.

Covered elsewhere, already registry-driven:
  * datasets  -> test_dataset_contracts.py (get_dataset_list)
  * features  -> test_feature_properties.py (get_feature_list)
  * selector metrics -> test_feature_selector.py (get_metrics)
  * discrete models  -> test_discrete.py
"""

import ast
import inspect
import pkgutil

import numpy as np
import pytest

import libemg
import libemg.emg_predictor as emg_predictor
import libemg.filtering as filtering
from libemg.emg_predictor import EMGClassifier, EMGRegressor
from libemg.feature_extractor import FeatureExtractor
from libemg.filtering import Filter
from libemg.offline_metrics import OfflineMetrics

# --------------------------------------------------------------------------- #
# Deriving the registries from source
# --------------------------------------------------------------------------- #


def model_config_names(class_name):
    """Model strings accepted by EMGClassifier / EMGRegressor.

    `model_config` is a dict literal local to each __init__, so it cannot be
    imported -- but it can be read off the syntax tree, which is enough to keep
    this test honest when a model is added.
    """
    tree = ast.parse(inspect.getsource(emg_predictor))
    class_def = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    for node in ast.walk(class_def):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "model_config":
            return [key.value for key in node.value.keys]
    raise AssertionError(f"no model_config dict found in {class_name}")


def filter_names():
    """Filter names accepted by Filter.install_filters.

    These live in an if/elif chain rather than a registry, so they are read
    out of the comparisons in the source.
    """
    tree = ast.parse(inspect.getsource(filtering))
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "install_filters"
    )
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    names.add(comparator.value)
                elif isinstance(comparator, (ast.List, ast.Tuple)):
                    names.update(
                        e.value
                        for e in comparator.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
    return sorted(names)


CLASSIFIER_MODELS = model_config_names("EMGClassifier")
REGRESSOR_MODELS = model_config_names("EMGRegressor")
FILTER_NAMES = filter_names()
FEATURES = FeatureExtractor().get_feature_list()
FEATURE_GROUPS = list(FeatureExtractor().get_feature_groups())
AVAILABLE_METRICS = OfflineMetrics().get_available_metrics()
STREAMER_FUNCTIONS = sorted(
    name
    for name, _ in inspect.getmembers(libemg.streamers, inspect.isfunction)
    if name.endswith("_streamer")
)


def test_registries_were_actually_discovered():
    """If a refactor moves these out of reach, every test below would quietly
    parametrize over nothing and pass. Fail loudly instead."""
    assert len(CLASSIFIER_MODELS) >= 8, CLASSIFIER_MODELS
    assert len(REGRESSOR_MODELS) >= 5, REGRESSOR_MODELS
    assert len(FILTER_NAMES) >= 6, FILTER_NAMES
    assert len(FEATURES) >= 50
    assert len(FEATURE_GROUPS) >= 10
    assert len(AVAILABLE_METRICS) >= 20


# --------------------------------------------------------------------------- #
# Adding a model
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_classification_set():
    """A small, perfectly separable set: enough for any classifier to fit."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 30)
    features = {"MAV": rng.normal(labels[:, None] * 6.0, 1.0, size=(90, 4))}
    return features, labels


class TestNewModelIsWiredCorrectly:
    @pytest.mark.parametrize("model", CLASSIFIER_MODELS)
    def test_classifier_model_fits_and_predicts(self, model, tiny_classification_set):
        """Every string in EMGClassifier's model_config must work end to end."""
        features, labels = tiny_classification_set
        classifier = EMGClassifier(model)
        classifier.fit({"training_features": features, "training_labels": labels})
        predictions, probabilities = classifier.run(features)

        assert len(predictions) == len(labels)
        assert set(np.unique(predictions)) <= set(np.unique(labels))
        assert np.asarray(probabilities).shape[0] == len(labels)

    @pytest.mark.parametrize("model", REGRESSOR_MODELS)
    def test_regressor_model_fits_and_predicts(self, model, tiny_classification_set):
        features, labels = tiny_classification_set
        targets = np.column_stack([labels / 2.0, 1.0 - labels / 2.0])
        regressor = EMGRegressor(model)
        regressor.fit({"training_features": features, "training_labels": targets})
        predictions = regressor.run(features)

        assert predictions.shape == (len(labels), 2)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.parametrize("model", CLASSIFIER_MODELS)
    def test_classifier_model_supports_probabilities(self, model, tiny_classification_set):
        """EMGClassifier.run always asks for predict_proba, so a model added
        without probability support fails at prediction, not construction."""
        features, labels = tiny_classification_set
        classifier = EMGClassifier(model)
        classifier.fit({"training_features": features, "training_labels": labels})
        _, probabilities = classifier.run(features)
        probabilities = np.asarray(probabilities)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1 + 1e-9)

    def test_unknown_model_names_are_rejected(self):
        for cls in (EMGClassifier, EMGRegressor):
            with pytest.raises(AssertionError, match="approved models"):
                cls("DEFINITELY_NOT_A_MODEL")


# --------------------------------------------------------------------------- #
# Adding a filter
# --------------------------------------------------------------------------- #


# install_filters needs different keys per filter; 'standardize' needs an ODH
# and is exercised in test_filtering.py instead.
FILTER_SPECS = {
    "notch": {"cutoff": 60, "bandwidth": 3},
    "lowpass": {"cutoff": 100, "order": 4},
    "highpass": {"cutoff": 20, "order": 2},
    "bandpass": {"cutoff": [20, 450], "order": 4},
    "bandstop": {"cutoff": [20, 450], "order": 4},
}


class TestNewFilterIsWiredCorrectly:
    def test_every_accepted_name_has_a_test_spec(self):
        """A filter name added to install_filters must be given a spec here,
        otherwise the tests below silently skip it."""
        untested = set(FILTER_NAMES) - set(FILTER_SPECS) - {"standardize"}
        assert not untested, (
            f"install_filters accepts {sorted(untested)} but no spec exists in "
            "FILTER_SPECS, so nothing exercises it. Add one."
        )

    @pytest.mark.parametrize("name", sorted(FILTER_SPECS))
    def test_filter_builds_real_coefficients(self, name):
        """Every filter must install usable b/a coefficients.

        This is the check that catches a half-wired new filter: the name is
        accepted, nothing raises, and the filter then does nothing.
        """
        filt = Filter(1000)
        filt.install_filters({"name": name, **FILTER_SPECS[name]})

        installed = filt.filters[-1]
        assert "b" in installed and "a" in installed, (
            f"'{name}' installed without filter coefficients, so it will pass "
            "data through untouched"
        )
        assert np.all(np.isfinite(installed["b"]))
        assert np.all(np.isfinite(installed["a"]))

    @pytest.mark.parametrize("name", sorted(FILTER_SPECS))
    def test_filter_actually_changes_the_signal(self, name):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(2000, 4))
        filt = Filter(1000)
        filt.install_filters({"name": name, **FILTER_SPECS[name]})
        filtered = filt.filter(data)

        assert filtered.shape == data.shape
        assert not np.allclose(filtered, data), f"'{name}' left the signal unchanged"


class TestFilterKnownDefects:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "install_filters dispatches on filter_dictionary['name'] through an "
            "if/elif chain with no else. An unrecognised name falls through all "
            "branches and still gets appended as {'name': ...} with no "
            "coefficients, and filter() then returns the data untouched with no "
            "error. A typo -- 'bandpas', 'high_pass' -- silently disables "
            "filtering instead of failing."
        ),
    )
    def test_unknown_filter_name_is_rejected(self):
        with pytest.raises((ValueError, KeyError, AssertionError)):
            Filter(1000).install_filters({"name": "bandpas", "cutoff": 60, "order": 2})

    @pytest.mark.xfail(
        strict=True,
        reason="Same defect, seen from the caller's side: the data comes back unfiltered.",
    )
    def test_unknown_filter_name_does_not_silently_pass_data_through(self):
        data = np.random.default_rng(0).normal(size=(500, 4))
        filt = Filter(1000)
        filt.install_filters({"name": "bandpas", "cutoff": 60, "order": 2})
        assert not np.array_equal(filt.filter(data), data), (
            "a misspelled filter name left the signal completely unfiltered"
        )


# --------------------------------------------------------------------------- #
# Adding an offline metric
# --------------------------------------------------------------------------- #


class TestNewMetricIsWiredCorrectly:
    @pytest.mark.parametrize("metric", AVAILABLE_METRICS)
    def test_advertised_metric_has_a_getter(self, metric):
        assert hasattr(OfflineMetrics(), f"get_{metric}"), (
            f"get_available_metrics() lists '{metric}' but there is no "
            f"get_{metric}() to call"
        )

    @pytest.mark.parametrize("metric", AVAILABLE_METRICS)
    def test_getter_is_callable_and_documented(self, metric):
        method = getattr(OfflineMetrics(), f"get_{metric}")
        assert callable(method)
        assert method.__doc__, f"get_{metric} has no docstring"

    def test_common_metrics_are_a_subset_of_available(self):
        om = OfflineMetrics()
        assert set(om.get_common_metrics()) <= set(om.get_available_metrics())


# --------------------------------------------------------------------------- #
# Adding a feature
# --------------------------------------------------------------------------- #


class TestNewFeatureIsWiredCorrectly:
    def test_feature_list_and_implementations_agree(self):
        """Duplicated from test_feature_properties on purpose: this is the
        single most common way a new feature is added wrong."""
        fe = FeatureExtractor()
        implemented = {
            name[3:-4]
            for name, _ in inspect.getmembers(fe, inspect.ismethod)
            if name.startswith("get") and name.endswith("feat")
        }
        assert set(FEATURES) == implemented

    @pytest.mark.parametrize("feature", FEATURES)
    def test_feature_has_a_docstring(self, feature):
        method = getattr(FeatureExtractor(), f"get{feature}feat")
        assert method.__doc__, f"get{feature}feat has no docstring"

    def test_every_feature_group_member_exists(self):
        groups = FeatureExtractor().get_feature_groups()
        for group, members in groups.items():
            unknown = set(members) - set(FEATURES)
            if group == "COMB":
                continue  # known defect, see test_feature_properties.py
            assert not unknown, f"group {group} references unknown features {unknown}"


# --------------------------------------------------------------------------- #
# Extension points with no registry
# --------------------------------------------------------------------------- #


class TestUnenumerableExtensionPoints:
    """Places a contributor can add code that nothing can systematically check.

    These are not failures -- they record where the safety net has holes, so
    the gap is visible rather than assumed covered.
    """

    def test_streamer_functions_are_discoverable(self):
        """libemg.streamers exposes one function per device with no registry,
        so the most that can be checked automatically is naming and docs."""
        assert len(STREAMER_FUNCTIONS) >= 8, STREAMER_FUNCTIONS

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param(
                n,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "leap_streamer is public API with no docstring, so it is "
                        "absent from the generated Sphinx reference. Every other "
                        "streamer documents its arguments."
                    ),
                ),
            )
            if n == "leap_streamer"
            else n
            for n in STREAMER_FUNCTIONS
        ],
    )
    def test_streamer_is_documented(self, name):
        """A device entry point with no docstring never reaches the docs."""
        assert getattr(libemg.streamers, name).__doc__, f"{name} has no docstring"

    def test_every_streamer_module_imports(self):
        """Mirrors test_imports.py, scoped to the highest-churn package."""
        import libemg._streamers as streamers_package

        modules = list(pkgutil.iter_modules(streamers_package.__path__))
        assert len(modules) >= 8
