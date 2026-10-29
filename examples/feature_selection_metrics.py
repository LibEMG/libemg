"""Exercise every feature selector metric on a small OneSubjectMyo subset.

Run from the repository root:
    python examples/feature_selection_metrics.py

The dataset loader downloads its public example data if it is not already
present. C/rC and CDM are explicitly documented variants; see the feature
selection documentation before using their values in research comparisons.
"""

from pathlib import Path
import sys

import numpy as np

# Also allow direct execution from an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libemg.datasets import OneSubjectMyoDataset
from libemg.feature_extractor import FeatureExtractor
from libemg.feature_selector import FeatureSelector


def load_example(dataset_folder="OneSubjectMyoDataset/"):
    """Load three motions, three trials and two channels without overlapping windows."""
    handler = OneSubjectMyoDataset(dataset_folder=dataset_folder).prepare_data(split=False)
    handler = handler.isolate_data("sets", [0, 1, 2])
    handler = handler.isolate_data("classes", [0, 1, 2])
    windows, metadata = handler.parse_windows(50, 50)
    # Keep the complete nonoverlapping recordings for meaningful MAV and TT.
    windows = windows[:, :2]
    features = FeatureExtractor().extract_features(["MAV", "WL", "ZC"], windows)
    labels = metadata["classes"].astype(int).ravel()
    trials = metadata["sets"].astype(int).ravel()
    repetitions = 2 * trials + metadata["reps"].astype(int).ravel()
    recordings = 3 * repetitions + labels
    options = dict(repetition_var=repetitions, trial_var=trials, raw_emg=windows,
                   window_size_ms=250, window_increment_ms=250, recording_var=recordings)
    return features, labels, {"var": trials}, options


def main():
    features, labels, crossvalidation, options = load_example()
    selector = FeatureSelector(allow_approximations=True)
    codes = selector.get_metrics(abbreviations=True)
    scores = selector.evaluate_metrics(codes, features, labels, crossvalidation, **options)
    print(f"{len(labels)} windows; {len(codes)} metrics; features: {list(features)}")
    for metric in codes:
        values, order = selector.run_selection(features, metric, labels, crossvalidation,
                                                early_stop=2, **options)
        assert not np.isnan(scores[metric])
        assert len(order) == 2 and not np.isnan(values[0]).all()
        status = selector.get_metric_info(metric)["reference_status"]
        print(f"{metric:5s} {scores[metric]:12.6g}  {' -> '.join(order):18s} {status}")


if __name__ == "__main__":
    main()
