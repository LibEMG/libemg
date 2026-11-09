"""Shared fixtures and opt-in markers for the libemg test suite.

Running the suite
-----------------
    pytest                  # fast, hermetic: no network, no hardware
    pytest --slow           # adds tests marked @pytest.mark.slow
    pytest --download       # adds tests that download datasets
    pytest --hardware       # adds tests that need a physical device

The ``slow`` marker and its ``--slow`` flag come from the ``pytest-skip-slow``
plugin, which is declared in requirements.txt. ``download`` and ``hardware``
are defined here.

Everything in the default run is built from ``tests/data/``, which is committed
to the repository, so the suite does not depend on the working directory or on
network access.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

# Paths are resolved from this file rather than the process working directory so
# that `pytest` works from anywhere, not just the repository root.
TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
GOLDEN_DIR = DATA_DIR / "golden"

# Window parameters used by the pipeline and golden-value tests. Changing these
# invalidates the committed golden fixtures -- see tests/README.md.
WINDOW_SIZE = 50
WINDOW_INCREMENT = 25


# --------------------------------------------------------------------------- #
# Opt-in markers
# --------------------------------------------------------------------------- #

_OPT_IN_MARKERS = {
    "download": "needs network access to download a dataset",
    "hardware": "needs a physical EMG device to be connected",
}


def pytest_addoption(parser):
    for name, reason in _OPT_IN_MARKERS.items():
        parser.addoption(
            f"--{name}",
            action="store_true",
            default=False,
            help=f"include tests marked {name} ({reason})",
        )


def pytest_collection_modifyitems(config, items):
    for name, reason in _OPT_IN_MARKERS.items():
        if config.getoption(f"--{name}"):
            continue
        skip = pytest.mark.skip(reason=f"needs --{name}: {reason}")
        for item in items:
            if name in item.keywords:
                item.add_marker(skip)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = TESTS_DIR.parent


@pytest.fixture(scope="session", autouse=True)
def run_from_repo_root():
    """Run the whole session from the repository root.

    Compatibility shim for the older test modules, which open fixture files
    through relative paths like 'tests/data/emg_data_myo.csv' and so only pass
    when pytest happens to be invoked from the root. New tests should take the
    ``data_dir`` fixture (or the module-level DATA_DIR) and not rely on this.
    """
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield REPO_ROOT
    finally:
        os.chdir(previous)


@pytest.fixture(scope="session")
def data_dir():
    """Absolute path to tests/data/."""
    return DATA_DIR


@pytest.fixture(scope="session")
def golden_dir():
    """Absolute path to tests/data/golden/, created on first use."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    return GOLDEN_DIR


# --------------------------------------------------------------------------- #
# Raw signals
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def myo_data(data_dir):
    """A single 8-channel Myo recording as a (samples, channels) array."""
    return np.loadtxt(data_dir / "emg_data_myo.csv", delimiter=",")


@pytest.fixture(scope="session")
def zeros_data(data_dir):
    """An all-zero recording. Degenerate input for feature robustness tests."""
    return np.loadtxt(data_dir / "emg_data_zeros.csv", delimiter=",")


@pytest.fixture(scope="session")
def myo_windows(myo_data):
    """Windows from a single Myo recording, shaped (windows, channels, samples)."""
    return get_windows(myo_data, WINDOW_SIZE, WINDOW_INCREMENT)


@pytest.fixture(scope="session")
def zero_windows(zeros_data):
    return get_windows(zeros_data, WINDOW_SIZE, WINDOW_INCREMENT)


# --------------------------------------------------------------------------- #
# The bundled Myo dataset: 5 classes x 4 reps, 8 channels
# --------------------------------------------------------------------------- #

MYO_CLASSES = ["0", "1", "2", "3", "4"]
MYO_REPS = ["0", "1", "2", "3"]


def myo_regex_filters():
    """RegexFilters matching the R_<rep>_C_<class>_EMG.csv naming scheme."""
    return [
        RegexFilter(
            left_bound="_C_", right_bound="_EMG", values=MYO_CLASSES, description="classes"
        ),
        RegexFilter(left_bound="R_", right_bound="_C_", values=MYO_REPS, description="reps"),
    ]


def load_myo_odh():
    """Build an OfflineDataHandler over tests/data/myo_dataset/.

    Not a fixture so that tests needing a pristine handler (isolate_* mutate by
    deep copy, but parse_windows caches) can build their own.
    """
    odh = OfflineDataHandler()
    odh.get_data(
        folder_location=str(DATA_DIR / "myo_dataset") + os.sep,
        regex_filters=myo_regex_filters(),
        delimiter=",",
    )
    return odh


@pytest.fixture(scope="session")
def myo_odh():
    """Session-scoped OfflineDataHandler over the bundled Myo dataset.

    Treat as read-only. Use ``fresh_myo_odh`` if the test mutates it.
    """
    return load_myo_odh()


@pytest.fixture
def fresh_myo_odh():
    """A function-scoped OfflineDataHandler, safe to mutate."""
    return load_myo_odh()


@pytest.fixture(scope="session")
def myo_dataset_windows(myo_odh):
    """(windows, metadata) parsed from the whole bundled dataset."""
    return myo_odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def fe():
    return FeatureExtractor()


@pytest.fixture
def seeded_rng():
    """A deterministic generator. Never use np.random directly in a test."""
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def y_true(data_dir):
    with open(data_dir / "test_labels", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def y_predictions(data_dir):
    with open(data_dir / "predictions", "rb") as f:
        return pickle.load(f)
