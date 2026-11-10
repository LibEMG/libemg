"""Property tests covering every feature in the extractor.

37 of the features have MATLAB ground truth (tests/test_feature_verification.py).
Producing ground truth for the rest is slow, but most feature regressions are
not subtle numerical errors -- they are shape changes, NaN leaks, registry
drift and copy-paste duplication. Those can be caught for all features at once
without any reference values, which is what this module does.

Anything with no MATLAB reference is additionally pinned to a committed
snapshot so it cannot change unnoticed.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

from tests.conftest import DATA_DIR, GOLDEN_DIR, WINDOW_INCREMENT, WINDOW_SIZE

FE = FeatureExtractor()
ALL_FEATURES = FE.get_feature_list()
N_CHANNELS = 8

# WENT computes -sum(p * log(p)) without guarding p == 0, so an all-zero window
# yields 0 * log(0) = nan. Real EMG never produces this, but a disconnected or
# saturated channel would. Strict xfail: when the guard is added, this flips to
# a failure and the entry below should be deleted.
NAN_ON_ZERO_INPUT = {"WENT"}


def extract(feature, windows):
    return FE.extract_features([feature], windows)[feature]


def matlab_verified_features():
    """Features that have a MATLAB reference committed under tests/data/."""
    return {p.stem.replace("matlab_", "") for p in DATA_DIR.glob("matlab_*.csv")}


def unverified_features():
    """Features with no ground truth of any kind. Computed, not hardcoded, so
    that adding a matlab_<NAME>.csv automatically retires it from this set."""
    return sorted(set(ALL_FEATURES) - matlab_verified_features())


# --------------------------------------------------------------------------- #
# Registry consistency
# --------------------------------------------------------------------------- #


def test_every_listed_feature_has_an_implementation():
    """get_feature_list() and the get<NAME>feat methods are in bijection."""
    implemented = {
        name[3:-4]
        for name, _ in inspect.getmembers(FE, inspect.ismethod)
        if name.startswith("get") and name.endswith("feat")
    }
    listed = set(ALL_FEATURES)

    assert listed - implemented == set(), "listed but not implemented"
    assert implemented - listed == set(), "implemented but not listed in get_feature_list()"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "getMEANfeat is defined twice (feature_extractor.py:341 and :1562). The "
        "two bodies are currently identical, so the shadowing is harmless -- "
        "until someone edits the first one. Delete the duplicate and remove "
        "this marker."
    ),
)
def test_no_duplicate_feature_methods():
    """Each get<NAME>feat is defined exactly once.

    Python silently keeps the last definition of a repeated method, so a
    duplicate is invisible at runtime until someone edits the dead copy and
    cannot work out why nothing changed.
    """
    source = Path(inspect.getfile(FeatureExtractor)).read_text()
    tree = ast.parse(source)
    class_def = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "FeatureExtractor"
    )

    seen, duplicates = set(), []
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in seen:
                duplicates.append(f"{node.name} (line {node.lineno})")
            seen.add(node.name)

    assert not duplicates, f"duplicate method definitions: {', '.join(duplicates)}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The COMB group lists 'AR9', which exists nowhere in the extractor. "
        "extract_feature_group('COMB') therefore returns 3 of its 4 features "
        "and reports no error. Either rename it to 'AR' (with AR_order=9 via "
        "feature parameters) or implement AR9, then remove this marker."
    ),
)
def test_feature_groups_resolve_to_real_features():
    """Every name in every feature group exists in get_feature_list()."""
    listed = set(ALL_FEATURES)
    unknown = {
        group: sorted(set(members) - listed)
        for group, members in FE.get_feature_groups().items()
        if set(members) - listed
    }
    assert not unknown, f"feature groups reference unknown features: {unknown}"


def test_feature_groups_are_not_empty():
    for group, members in FE.get_feature_groups().items():
        assert len(members) > 0, f"feature group {group} is empty"


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(
            g,
            marks=pytest.mark.xfail(
                strict=True, reason="COMB lists AR9, which is silently dropped"
            ),
        )
        if g == "COMB"
        else g
        for g in FE.get_feature_groups()
    ],
)
def test_feature_group_extracts_all_its_members(group, myo_windows):
    """Asking for a group returns every feature in it.

    extract_feature_group drops names it cannot resolve without raising, so a
    typo in a group definition costs the user features and tells them nothing.
    """
    expected = set(FE.get_feature_groups()[group])
    extracted = set(FE.extract_feature_group(group, myo_windows))
    assert extracted == expected, f"{group} is missing {sorted(expected - extracted)}"


# --------------------------------------------------------------------------- #
# Per-feature properties
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_feature_shape(feature, myo_windows):
    """Output is (n_windows, k * n_channels) for some whole number k.

    Several features legitimately emit more than one value per channel -- AR
    and CC emit 4, the phasor features 7, DFTR 6, the wavelet features 8 -- so
    the contract is a whole multiple, not exactly one.
    """
    values = np.asarray(extract(feature, myo_windows))
    assert values.ndim == 2, f"{feature} returned a {values.ndim}-D array"
    assert values.shape[0] == myo_windows.shape[0]
    assert values.shape[1] % N_CHANNELS == 0, (
        f"{feature} returned {values.shape[1]} columns, not a multiple of "
        f"{N_CHANNELS} channels"
    )


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_feature_is_finite_on_real_data(feature, myo_windows):
    values = np.asarray(extract(feature, myo_windows))
    assert np.all(np.isfinite(values)), (
        f"{feature} produced {np.count_nonzero(~np.isfinite(values))} non-finite "
        "values on real EMG"
    )


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_feature_is_deterministic(feature, myo_windows):
    """Two calls on identical input agree bit for bit."""
    np.testing.assert_array_equal(
        np.asarray(extract(feature, myo_windows)),
        np.asarray(extract(feature, myo_windows)),
        err_msg=f"{feature} is not deterministic",
    )


@pytest.mark.parametrize(
    "feature",
    [
        pytest.param(
            f,
            marks=pytest.mark.xfail(
                strict=True, reason="0 * log(0) is not guarded; see NAN_ON_ZERO_INPUT"
            ),
        )
        if f in NAN_ON_ZERO_INPUT
        else f
        for f in ALL_FEATURES
    ],
)
def test_feature_is_finite_on_zero_input(feature, zero_windows):
    """A flat channel must not become NaN. Mirrors a disconnected electrode."""
    values = np.asarray(extract(feature, zero_windows))
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_feature_handles_single_channel(feature, myo_data):
    """One channel is a valid recording, not an edge case to crash on."""
    windows = get_windows(myo_data[:, :1], WINDOW_SIZE, WINDOW_INCREMENT)
    values = np.asarray(extract(feature, windows))
    assert values.shape[0] == windows.shape[0]
    assert values.ndim == 2


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_feature_handles_constant_window(feature):
    """A constant non-zero signal. Variance is 0, which is where the division
    guards in spectral and entropy features either hold or do not."""
    windows = np.full((3, N_CHANNELS, WINDOW_SIZE), 7.0)
    values = np.asarray(extract(feature, windows))
    assert values.shape[0] == 3


# --------------------------------------------------------------------------- #
# Snapshot for features with no MATLAB reference
# --------------------------------------------------------------------------- #


def snapshot_features():
    """The arrays pinned by test_unverified_feature_snapshot.

    Imported by tests/regenerate_golden.py -- keep the two in step.
    """
    data = np.loadtxt(DATA_DIR / "emg_data_myo.csv", delimiter=",")
    windows = get_windows(data, WINDOW_SIZE, WINDOW_INCREMENT)
    return {f: np.asarray(extract(f, windows), dtype=np.float64) for f in unverified_features()}


def test_unverified_features_are_known():
    """Guards the assumption behind the snapshot below: most features do have
    MATLAB ground truth, and the handful that do not is a short list."""
    unverified = unverified_features()
    assert len(unverified) < len(ALL_FEATURES) / 2, (
        f"{len(unverified)} of {len(ALL_FEATURES)} features now lack MATLAB ground "
        "truth; the verification suite has fallen behind the extractor"
    )


@pytest.mark.parametrize("feature", unverified_features())
def test_unverified_feature_snapshot(feature):
    """Pin features that have no MATLAB reference to committed values.

    This is not a correctness check -- it says nothing about whether the
    feature is right. It makes a change in its output visible in review
    instead of silent.
    """
    path = GOLDEN_DIR / "feature_snapshot.npz"
    if not path.exists():
        pytest.fail(
            f"missing {path}. Generate it with: python tests/regenerate_golden.py"
        )
    golden = np.load(path)
    if feature not in golden:
        pytest.fail(
            f"{feature} has no entry in the snapshot. It was probably added after "
            "the snapshot was last written; regenerate with "
            "python tests/regenerate_golden.py"
        )

    data = np.loadtxt(DATA_DIR / "emg_data_myo.csv", delimiter=",")
    windows = get_windows(data, WINDOW_SIZE, WINDOW_INCREMENT)
    np.testing.assert_allclose(
        np.asarray(extract(feature, windows), dtype=np.float64),
        golden[feature],
        rtol=1e-9,
        atol=1e-9,
        err_msg=f"{feature} output changed and it has no MATLAB reference to fall back on",
    )
