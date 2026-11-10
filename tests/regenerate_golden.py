"""Regenerate the committed golden fixtures under tests/data/golden/.

    python tests/regenerate_golden.py

Run this ONLY when a golden-value test has failed, you understand why, and the
new numbers are the intended ones. Overwriting a golden fixture to make a red
test green throws away the single thing it exists to detect. Say in the pull
request why the numbers moved.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.conftest import GOLDEN_DIR, WINDOW_INCREMENT, WINDOW_SIZE  # noqa: E402
from tests.test_feature_properties import snapshot_features  # noqa: E402
from tests.test_pipeline_offline import run_pipeline, stack_features  # noqa: E402


def write_pipeline_golden():
    result = run_pipeline()
    path = GOLDEN_DIR / "pipeline.npz"
    np.savez_compressed(
        path,
        window_shape=np.asarray(result["windows"].shape),
        window_sums=np.asarray(result["windows"], dtype=np.float64).sum(axis=(1, 2)),
        htd_features=stack_features(result["features"]),
        predictions=result["predictions"],
        accuracy=np.asarray(result["accuracy"]),
    )
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  windows   {result['windows'].shape}")
    print(f"  accuracy  {result['accuracy']:.6f}")


def write_feature_snapshot():
    values = snapshot_features()
    path = GOLDEN_DIR / "feature_snapshot.npz"
    np.savez_compressed(path, **values)
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  {len(values)} features pinned")


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"window_size={WINDOW_SIZE} window_increment={WINDOW_INCREMENT}\n")
    write_pipeline_golden()
    write_feature_snapshot()
