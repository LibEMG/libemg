"""End-to-end tests for the offline pipeline.

The library's core job is one chain:

    get_data -> parse_windows -> extract_features -> fit -> run -> metrics

A break anywhere in it is a break in libemg, even when every module's own unit
tests still pass. These tests exercise the whole chain over the Myo dataset
committed under tests/data/, so they need no network and run in seconds.

Golden values
-------------
``test_golden_*`` compare against arrays committed in tests/data/golden/. They
are the alarm for *silent numerical drift*: a refactor that preserves every
shape and still changes every number. If one fails, that is a result to
explain, not a fixture to overwrite. Regenerate only once the change is
understood and intended:

    python tests/regenerate_golden.py

A note on the fixture
---------------------
Reps 0, 1 and 2 of the bundled dataset are mutually consistent: holding any one
of them out and training on the rest classifies the held-out rep perfectly.
Rep 3 is not -- training on 0/1/2 and testing on rep 3 scores at chance (~0.22
for 5 classes). Rep 3 looks like it came from a different session. Accuracy
assertions therefore hold out rep 2, while the golden feature snapshot still
covers all four reps so none of the data goes unwatched.
"""

import numpy as np
import pytest

from libemg.emg_predictor import EMGClassifier, EMGRegressor
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics

from tests.conftest import WINDOW_INCREMENT, WINDOW_SIZE, load_myo_odh

# Held out for scoring. See the module docstring for why this is not rep 3.
HELD_OUT_REP = 2
FEATURE_GROUP = "HTD"


def stack_features(feature_dict):
    """Features as one (windows, features) matrix, in a stable column order."""
    return np.hstack([feature_dict[k] for k in sorted(feature_dict)])


def run_pipeline(odh=None, model="LDA"):
    """The full offline chain. Returns everything downstream tests assert on."""
    if odh is None:
        odh = load_myo_odh()

    windows, metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    features = FeatureExtractor().extract_feature_group(FEATURE_GROUP, windows)

    labels = metadata["classes"].squeeze().astype(int)
    reps = metadata["reps"].squeeze().astype(int)
    train, test = reps != HELD_OUT_REP, reps == HELD_OUT_REP

    classifier = EMGClassifier(model)
    classifier.fit(
        {
            "training_features": {k: v[train] for k, v in features.items()},
            "training_labels": labels[train],
        }
    )
    predictions, probabilities = classifier.run({k: v[test] for k, v in features.items()})

    return {
        "windows": windows,
        "features": features,
        "labels": labels,
        "reps": reps,
        "test_mask": test,
        "predictions": np.asarray(predictions),
        "probabilities": np.asarray(probabilities),
        "accuracy": OfflineMetrics().get_CA(labels[test], predictions),
    }


# --------------------------------------------------------------------------- #
# The chain holds together
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pipeline():
    return run_pipeline()


def test_pipeline_shapes_are_consistent(pipeline):
    """Every stage agrees with the next on how many windows there are."""
    n_windows = pipeline["windows"].shape[0]
    assert pipeline["windows"].ndim == 3, "windows should be (windows, channels, samples)"
    assert pipeline["windows"].shape[1:] == (8, WINDOW_SIZE)

    for name, values in pipeline["features"].items():
        assert values.shape[0] == n_windows, f"{name} has the wrong number of rows"

    assert pipeline["labels"].shape == (n_windows,)
    assert pipeline["reps"].shape == (n_windows,)
    assert pipeline["predictions"].shape[0] == pipeline["test_mask"].sum()


def test_pipeline_classifies_held_out_rep(pipeline):
    """The end-to-end accuracy floor. The real value here is ~1.0."""
    assert pipeline["accuracy"] > 0.9, (
        f"end-to-end accuracy fell to {pipeline['accuracy']:.3f}; the pipeline is "
        "producing data the classifier cannot separate"
    )


def test_pipeline_predicts_only_trained_labels(pipeline):
    """Predictions are drawn from the label set, not invented."""
    trained_labels = set(np.unique(pipeline["labels"]).tolist())
    predicted = set(np.unique(pipeline["predictions"]).tolist())
    assert predicted <= trained_labels


def test_pipeline_is_deterministic():
    """Two identical runs agree exactly. Catches unseeded randomness."""
    first, second = run_pipeline(), run_pipeline()
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    np.testing.assert_allclose(first["probabilities"], second["probabilities"], rtol=0, atol=0)


def test_pipeline_regression_path():
    """The same chain through EMGRegressor on synthetic continuous targets."""
    odh = load_myo_odh()
    windows, metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    features = FeatureExtractor().extract_feature_group(FEATURE_GROUP, windows)

    # Two output DOFs derived from the class label, so the mapping is learnable
    # without pulling in a regression dataset that would need downloading.
    classes = metadata["classes"].squeeze().astype(float)
    targets = np.column_stack([classes / 4.0, 1.0 - classes / 4.0])
    reps = metadata["reps"].squeeze().astype(int)
    train, test = reps != HELD_OUT_REP, reps == HELD_OUT_REP

    regressor = EMGRegressor("LR")
    regressor.fit(
        {
            "training_features": {k: v[train] for k, v in features.items()},
            "training_labels": targets[train],
        }
    )
    predictions = regressor.run({k: v[test] for k, v in features.items()})

    assert predictions.shape == (int(test.sum()), 2)
    assert np.all(np.isfinite(predictions))
    r2 = OfflineMetrics().get_R2(targets[test], predictions)
    assert np.mean(r2) > 0.5, f"regression path degraded to R2={np.mean(r2):.3f}"


# --------------------------------------------------------------------------- #
# Golden values: the numbers themselves must not drift
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def golden(golden_dir):
    path = golden_dir / "pipeline.npz"
    if not path.exists():
        pytest.fail(
            f"missing golden fixture {path}. Generate it with:\n"
            "    python tests/regenerate_golden.py"
        )
    return np.load(path)


def test_golden_windows(pipeline, golden):
    """get_windows output is unchanged. Localises drift before features.

    The windows themselves are not committed -- the HTD snapshot below is
    derived from them, so any element change shows up there. What is pinned
    here is the segmentation: how many windows, of what size, in what order.
    """
    np.testing.assert_array_equal(pipeline["windows"].shape, golden["window_shape"])
    np.testing.assert_allclose(
        np.asarray(pipeline["windows"], dtype=np.float64).sum(axis=(1, 2)),
        golden["window_sums"],
        rtol=1e-12,
        err_msg="per-window totals changed: windowing is cutting the signal differently",
    )


def test_golden_features(pipeline, golden):
    """Every HTD feature value matches, to ~6 decimal places."""
    np.testing.assert_allclose(
        stack_features(pipeline["features"]),
        golden["htd_features"],
        rtol=1e-9,
        atol=1e-9,
        err_msg=(
            "HTD feature values changed. If this is intentional, regenerate with "
            "python tests/regenerate_golden.py and say so in the PR."
        ),
    )


def test_golden_predictions(pipeline, golden):
    """The LDA decision on the held-out rep is unchanged."""
    np.testing.assert_array_equal(pipeline["predictions"], golden["predictions"])
    np.testing.assert_allclose(pipeline["accuracy"], golden["accuracy"], rtol=1e-12)
