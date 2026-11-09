"""Contract tests for every registered dataset wrapper.

Adding a dataset is the most common outside contribution this project gets,
and a new wrapper is easy to get subtly wrong in ways nothing catches until
someone tries to run a benchmark on it months later.

Every test here constructs datasets but never downloads one, so the whole
module runs in well under a second and is safe on any machine. What is being
checked is that each wrapper's metadata satisfies the assumptions the rest of
the library makes about it -- particularly datasets.evaluate(), which does
arithmetic on `sampling` and iterates over `num_subjects`.

The one test that does touch the network is marked @download and skipped
unless you pass --download.
"""

import inspect

import numpy as np
import pytest

from libemg._datasets.dataset import Dataset
from libemg.datasets import get_dataset_info, get_dataset_list

# Datasets that genuinely mix hardware, and so cannot have a single integer
# sampling rate. They need the multi_rate path through parse_windows rather
# than the scalar arithmetic in evaluate(). Adding a name here is a statement
# that evaluate() is not expected to drive it.
MULTI_RATE_DATASETS = {"EMGEPN100"}


# Which evaluator drives which registry matters, because they call
# prepare_data with different keyword arguments:
#
#   evaluate / evaluate_weaklysupervised -> prepare_data(split=..., subjects=...)
#   evaluate_crossuser                   -> prepare_data(split=...) and optionally
#                                           feature_list/window_size/window_inc
#
# so the contracts below are checked per registry rather than over one merged
# dictionary.

PER_SUBJECT_DATASETS = {}
for _type in ("CLASSIFICATION", "REGRESSION", "WEAKLYSUPERVISED"):
    PER_SUBJECT_DATASETS.update(get_dataset_list(_type))

CROSS_USER_DATASETS = {}
for _type in ("CLASSIFICATION", "REGRESSION"):
    CROSS_USER_DATASETS.update(get_dataset_list(_type, cross_user=True))

DATASETS = {**CROSS_USER_DATASETS, **PER_SUBJECT_DATASETS}
DATASET_NAMES = sorted(DATASETS)

# A name can map to different classes in the per-subject and cross-user
# registries (EMG2POSE -> EMG2POSEUD or EMG2POSECU), so registration is
# checked against every registered class, not against DATASETS.values().
REGISTERED_CLASSES = set(PER_SUBJECT_DATASETS.values()) | set(CROSS_USER_DATASETS.values())

# get_dataset_info looks the name up in get_dataset_list(), which defaults to
# type='CLASSIFICATION'. Anything registered only under another type is
# unreachable through it.
CLASSIFICATION_NAMES = set(get_dataset_list("CLASSIFICATION"))


@pytest.fixture(scope="module", params=DATASET_NAMES)
def dataset(request):
    """An instance of each registered dataset. Construction only, no download."""
    return DATASETS[request.param]()


# --------------------------------------------------------------------------- #
# Construction and metadata
# --------------------------------------------------------------------------- #


def test_registry_is_not_empty():
    assert len(DATASETS) > 20, "the dataset registry looks truncated"


def test_every_dataset_subclasses_dataset(dataset):
    assert isinstance(dataset, Dataset)


@pytest.mark.parametrize("attribute", ["description", "citation", "num_reps"])
def test_descriptive_metadata_is_present(dataset, attribute):
    value = getattr(dataset, attribute, None)
    assert value is not None, f"{type(dataset).__name__} has no {attribute}"
    if isinstance(value, str):
        assert value.strip(), f"{type(dataset).__name__}.{attribute} is blank"
    elif isinstance(value, (int, float, np.integer, np.floating)):
        assert value > 0, f"{type(dataset).__name__}.{attribute} is {value}"
    else:
        assert len(value) > 0, f"{type(dataset).__name__}.{attribute} is empty"


def test_gestures_is_a_mapping(dataset):
    assert isinstance(dataset.gestures, dict)


@pytest.mark.parametrize("name", sorted(CLASSIFICATION_NAMES))
def test_classification_datasets_name_their_gestures(name):
    """A classification dataset without a gesture map cannot be interpreted.

    Regression datasets are exempt: continuous-target sets such as HyserRandom
    have no discrete gestures and legitimately leave this empty.
    """
    gestures = DATASETS[name]().gestures
    assert len(gestures) > 0, f"{name} declares no gestures"


def test_num_channels_is_a_positive_integer(dataset):
    name = type(dataset).__name__
    if name in MULTI_RATE_DATASETS:
        pytest.skip(f"{name} is multi-hardware; channels are per-device")
    assert isinstance(dataset.num_channels, (int, np.integer))
    assert dataset.num_channels > 0


def test_prepare_data_is_overridden(dataset):
    """The base class prepare_data is a no-op returning None."""
    assert type(dataset).prepare_data is not Dataset.prepare_data, (
        f"{type(dataset).__name__} never overrides prepare_data"
    )


# --------------------------------------------------------------------------- #
# What evaluate() assumes
# --------------------------------------------------------------------------- #


def test_sampling_rate_supports_window_arithmetic(dataset):
    """evaluate() computes int(dataset.sampling / 1000 * window_size).

    A dict-valued sampling rate raises TypeError there, which is why multi-rate
    datasets must be declared rather than discovered at run time.
    """
    name = type(dataset).__name__
    if name in MULTI_RATE_DATASETS:
        pytest.skip(f"{name} is declared multi-rate")

    assert isinstance(dataset.sampling, (int, float, np.integer, np.floating)), (
        f"{name}.sampling is {type(dataset.sampling).__name__}; evaluate() needs a "
        "number. If this dataset really does mix sampling rates, add it to "
        "MULTI_RATE_DATASETS and give it a multi_rate-aware prepare_data."
    )
    assert int(dataset.sampling / 1000 * 200) > 0


def test_num_subjects_supports_iteration(dataset):
    """evaluate() does `for s in range(0, dataset.num_subjects)`."""
    name = type(dataset).__name__
    assert isinstance(dataset.num_subjects, (int, np.integer)), (
        f"{name}.num_subjects is {type(dataset.num_subjects).__name__}; "
        "evaluate() calls range() on it"
    )
    assert dataset.num_subjects > 0
    assert len(range(0, dataset.num_subjects)) == dataset.num_subjects


def accepts(prepare_data, keyword):
    parameters = inspect.signature(prepare_data).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return keyword in parameters


@pytest.mark.parametrize("name", sorted(PER_SUBJECT_DATASETS))
@pytest.mark.parametrize("keyword", ["split", "subjects"])
def test_per_subject_prepare_data_signature(name, keyword):
    """evaluate() calls prepare_data(split=True, subjects=[s]) per subject."""
    prepare_data = PER_SUBJECT_DATASETS[name]().prepare_data
    assert accepts(prepare_data, keyword), (
        f"{name}.prepare_data does not accept '{keyword}', which "
        "datasets.evaluate() passes to it"
    )


@pytest.mark.parametrize("name", sorted(CROSS_USER_DATASETS))
def test_cross_user_prepare_data_signature(name):
    """evaluate_crossuser() calls prepare_data(split=True)."""
    prepare_data = CROSS_USER_DATASETS[name]().prepare_data
    assert accepts(prepare_data, "split"), (
        f"{name}.prepare_data does not accept 'split', which "
        "datasets.evaluate_crossuser() passes to it"
    )


# --------------------------------------------------------------------------- #
# get_dataset_info
# --------------------------------------------------------------------------- #


class TestDatasetInfo:
    def test_get_info_does_not_raise(self, dataset, capsys):
        name = type(dataset).__name__
        if name in KNOWN_INFO_FAILURES:
            pytest.xfail(KNOWN_INFO_FAILURES[name])
        dataset.get_info()
        assert capsys.readouterr().out.strip(), "get_info printed nothing"

    @pytest.mark.parametrize("name", sorted(CLASSIFICATION_NAMES))
    def test_get_dataset_info_by_name(self, name, capsys):
        """The documented entry point for inspecting a dataset."""
        if DATASETS[name].__name__ in KNOWN_INFO_FAILURES:
            pytest.xfail(KNOWN_INFO_FAILURES[DATASETS[name].__name__])
        get_dataset_info(name)
        output = capsys.readouterr().out
        assert "Invalid dataset name" not in output

    @pytest.mark.parametrize(
        "name", sorted(set(DATASET_NAMES) - CLASSIFICATION_NAMES)
    )
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "get_dataset_info looks the name up in get_dataset_list(), which "
            "defaults to type='CLASSIFICATION'. Every regression and "
            "weakly-supervised dataset therefore prints 'ERROR: Invalid dataset "
            "name' instead of its information. Searching the ALL registry would "
            "fix it."
        ),
    )
    def test_get_dataset_info_resolves_non_classification_datasets(self, name, capsys):
        get_dataset_info(name)
        assert "Invalid dataset name" not in capsys.readouterr().out


# Dataset.get_info builds its output with string concatenation, so any field it
# touches must already be a string. A list or dict there is a TypeError.
KNOWN_INFO_FAILURES = {
    "EMGEPN100": (
        "recording_device is the list ['myo', 'gForce'], and Dataset.get_info "
        "does `'\\nDevice: ' + self.recording_device`. Either pass a string "
        "('Myo, gForce') or wrap the field in str() inside get_info."
    )
}


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #


class TestRegistryWiring:
    def test_names_are_unique_per_registry(self):
        for dataset_type in ("CLASSIFICATION", "REGRESSION"):
            for cross_user in (False, True):
                registry = get_dataset_list(dataset_type, cross_user)
                assert len(registry) == len(set(registry))

    def test_all_includes_classification_and_regression(self):
        combined = get_dataset_list("ALL")
        for dataset_type in ("CLASSIFICATION", "REGRESSION"):
            for name in get_dataset_list(dataset_type):
                assert name in combined, f"{name} is missing from the ALL registry"

    def test_cross_user_datasets_are_also_listed_normally(self):
        """A cross-user dataset should still be reachable from the main list,
        otherwise get_dataset_info cannot find it."""
        for name in get_dataset_list("CLASSIFICATION", cross_user=True):
            assert name in get_dataset_list("CLASSIFICATION"), (
                f"{name} is registered for cross-user evaluation but not in the "
                "classification list that get_dataset_info searches"
            )

    @staticmethod
    def exported_leaf_dataset_classes():
        """Concrete dataset classes: public, and not a base of another one.

        Intermediate classes such as Ninapro and _Hyser exist to be subclassed
        and are not meant to appear in a registry.
        """
        import libemg._datasets as datasets_package

        exported = set()
        for module_name in dir(datasets_package):
            module = getattr(datasets_package, module_name)
            if not inspect.ismodule(module):
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, Dataset)
                    and obj is not Dataset
                    and obj.__module__.startswith("libemg._datasets")
                    and not obj.__name__.startswith("_")
                ):
                    exported.add(obj)

        bases = {base for obj in exported for base in obj.__mro__[1:] if base in exported}
        return exported - bases

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "GrabMyoCrossUser and HyserMVC are concrete datasets with their own "
            "prepare_data, but appear in no registry, so get_dataset_list never "
            "returns them and neither evaluator can reach them. They can only be "
            "imported directly from libemg._datasets."
        ),
    )
    def test_every_concrete_dataset_class_is_registered(self):
        """A dataset nothing can reach from get_dataset_list is dead code."""
        unreachable = sorted(
            c.__name__ for c in self.exported_leaf_dataset_classes() - REGISTERED_CLASSES
        )
        assert not unreachable, (
            f"dataset classes defined but reachable from no registry: {unreachable}. "
            "Add them to get_dataset_list, or make them private if they are "
            "helpers rather than datasets."
        )


# --------------------------------------------------------------------------- #
# The one test that downloads
# --------------------------------------------------------------------------- #


@pytest.mark.download
def test_evaluate_smoke_on_one_subject_myo(tmp_path):
    """One real end-to-end benchmark run. Nightly, not per-PR."""
    from libemg.datasets import evaluate

    evaluate(
        "LDA",
        window_size=200,
        window_inc=100,
        feature_list=["MAV"],
        included_datasets=["OneSubjectMyo"],
        output_file=str(tmp_path / "out.pkl"),
        metrics=["CA"],
    )
    assert (tmp_path / "out.pkl").exists()
