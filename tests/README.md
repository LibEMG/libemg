# LibEMG test suite

## Running it

```bash
pytest                  # the default run: fast, no network, no hardware
pytest --slow           # adds tests marked slow (minutes, not seconds)
pytest --download       # adds tests that download a dataset
pytest --hardware       # adds tests that need a physical EMG device
pytest --cov=libemg --cov-report=term-missing    # with coverage
```

The default run takes about 20 seconds and touches nothing outside the
repository. That is deliberate: a suite that is slow or flaky gets ignored,
and an ignored suite protects nothing. Anything slow or external lives behind
a flag and runs in the nightly workflow instead.

CI runs the default suite on every push and pull request across Python 3.9,
3.10 and 3.11, with a coverage floor. `.github/workflows/nightly.yaml` runs
the `--slow` and `--download` tests once a day.

## Layout

| File | What it covers |
| --- | --- |
| `conftest.py` | Shared fixtures, the `--download` / `--hardware` flags, path handling |
| `test_pipeline_offline.py` | The full chain end to end, plus golden values |
| `test_data_handler.py` | `OfflineDataHandler`, windowing, metadata, isolation |
| `test_emg_predictor.py` | Every model string, save/load, rejection, majority vote, velocity, deadband |
| `test_feature_properties.py` | Shape, finiteness and determinism for all 51 features; registry consistency |
| `test_feature_verification.py` | 37 features against MATLAB ground truth |
| `test_feature_extractor.py` | Feature extractor API |
| `test_filtering.py` | Each filter type, verified in the frequency domain |
| `test_offline_metrics.py` | Classification metrics against sklearn |
| `test_offline_metrics_extended.py` | Regression metrics, and separability metrics as orderings |
| `test_dataset_contracts.py` | Metadata contracts for all 26 dataset wrappers |
| `test_imports.py` | Every module in the package imports cleanly |
| `test_online_classifier.py`, `test_stream_architecture.py` | Skipped: see "Known gaps" |

## Fixtures

Everything in the default run is built from data committed under `tests/data/`:

- `myo_dataset/` — 20 files, 5 classes x 4 reps, 8 channels, 320 KB. The
  fixture behind most tests.
- `emg_data_myo.csv`, `emg_data_zeros.csv` — single recordings for feature tests.
- `matlab_*.csv` — MATLAB ground truth for 37 features.
- `golden/` — committed outputs, see below.

**A note on `myo_dataset`:** reps 0, 1 and 2 are mutually consistent — hold any
one out, train on the rest, and it classifies perfectly. Rep 3 is not; training
on 0/1/2 and testing on rep 3 scores at chance. It appears to come from a
different session. Tests that need an accuracy floor hold out rep 2, not rep 3.

## Golden values

`test_pipeline_offline.py` and the feature snapshot in
`test_feature_properties.py` compare against arrays committed in
`tests/data/golden/`. They exist to catch *silent numerical drift*: a change
that preserves every shape and still moves every number.

When one fails, that is a result to explain, not a fixture to overwrite. Work
out why the numbers moved first. If the change is intended:

```bash
python tests/regenerate_golden.py
```

and say in the pull request why the values changed. Overwriting a golden
fixture to turn a red test green throws away the only thing it was watching.

## Markers

| Marker | Flag | Meaning |
| --- | --- | --- |
| `slow` | `--slow` | Minutes rather than seconds. From the `pytest-skip-slow` plugin. |
| `download` | `--download` | Fetches a dataset over the network. |
| `hardware` | `--hardware` | Needs a physical EMG device connected. |

Markers are strict (`--strict-markers` in `pytest.ini`): a typo in a marker
name is an error, not a silently ignored annotation.

## Known gaps

**Online and real-time paths are not covered.** `test_online_classifier.py` and
`test_stream_architecture.py` are skipped at module level: they import
`mock_emg_stream`, which was removed from `libemg.streamers` in commit
`afa6034`, and there is no replacement mock streamer. Their timing assertions
(`abs(10 - elapsed) < 0.1`) were also flaky by construction on shared CI
runners. Rewriting them needs a new mock streamer and sample-count assertions
instead of wall-clock ones.

Also uncovered: `feature_selector.py`, `animator.py`, `_gui/`, `environments/`,
and the vendor modules in `_streamers/` beyond the import check.

## Known defects recorded as xfail

Tests marked `xfail(strict=True)` document a real bug in the library. They are
not failures to ignore — strict mode means that **when the bug is fixed the
test turns red**, which is the signal to delete the marker. Run
`pytest -rx` to list them with their reasons.

Current entries:

| Where | Defect |
| --- | --- |
| `feature_extractor.py:1562` | `getMEANfeat` is defined twice; the second shadows the first |
| `feature_extractor.py:34` | The `COMB` group lists `AR9`, which does not exist, so `extract_feature_group('COMB')` silently returns 3 of 4 features |
| `feature_extractor.py` (WENT) | `-sum(p·log p)` is unguarded, so an all-zero window yields NaN |
| `data_handler.py` (`_get_mode_windows`) | A file yielding exactly one window produces 0-d metadata and breaks concatenation |
| `data_handler.py` (`_get_mode_windows`) | A file shorter than the window crashes `apply_along_axis` |
| `data_handler.py:610` | `isolate_channels` validates against the sample count when `channel_last=False` |
| `data_handler.py:558` | The multi-rate path requires array-valued sampling-rate metadata and rejects scalars |
| `offline_metrics.py` (`get_BD`) | Loops over segment count where every sibling metric loops over class count |
| `offline_metrics.py` (`get_IIF`) | Divides by zero on perfectly separable data |
| `offline_metrics.py` (`_get_KL_divergence`) | Hardcodes `term3 = 24` where the feature dimensionality belongs |
| `datasets.py` (`get_dataset_info`) | Searches only the classification registry, so every regression dataset reports "Invalid dataset name" |
| `_datasets/emg_epn100.py` | `recording_device` is a list, which breaks `Dataset.get_info` string concatenation |
| `_datasets/` | `GrabMyoCrossUser` and `HyserMVC` are concrete datasets in no registry |

## Adding tests

- Take fixtures from `conftest.py` rather than loading files directly, and
  never hardcode a relative path.
- Use `seeded_rng`, not `np.random`, so failures reproduce.
- Assert on behaviour, not just that a call returned something. `assert result
  is not None` passes for almost any bug.
- Put the *why* in the assertion message. The person reading it will be
  someone whose unrelated pull request just went red.
- If you add a dataset, `test_dataset_contracts.py` covers it automatically —
  make sure it passes before opening the pull request.
