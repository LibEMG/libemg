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

CI runs the default suite on every push and pull request, with a coverage
floor. `.github/workflows/nightly.yaml` runs the `--slow` and `--download`
tests once a day.

**Supported: Python 3.11, 3.12 and 3.13**, all three in the CI matrix, and
`requires-python = ">=3.11"` in `pyproject.toml` says the same thing — a
release should never promise a version nothing runs against.

- **3.10** — dropped as a support decision, not because anything broke. It was
  in the matrix until the floor moved up.
- **3.9 and below** — libemg uses PEP 604 (`X | None`) in 33 function
  signatures with no `from __future__ import annotations`. Parameter
  annotations are evaluated at definition time, so the library raises
  `TypeError` on import. Adding that one import to the eight affected modules
  would fix it.

Code has to run on the whole range, which rules out anything 3.13-only. The
live example is `pathlib.PurePath.full_match`: `libemg/annotations.py`
translates its dataset globs by hand rather than calling it.

### Setting up a dev environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # editable install + test and docs extras
```

Runtime dependencies live in `pyproject.toml`; `requirements.txt` is just a
convenience wrapper around `pip install -e .[test,docs,torch]`.

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
| `test_offline_metrics_extended.py` | Regression metrics; separability, partition and train/test metrics as orderings |
| `test_feature_selector.py` | Feature ranking, both selection strategies, cross-validation plumbing |
| `test_dataset_contracts.py` | Metadata contracts for all 26 dataset wrappers |
| `test_discrete.py` | Template-based path: `discrete=True` windowing and features, MVLDA, DTW |
| `test_extension_contracts.py` | Registry-driven checks so new models/filters/features/metrics are covered automatically |
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

Markers are strict (`--strict-markers` in `pyproject.toml`): a typo in a marker
name is an error, not a silently ignored annotation.

## Known gaps

**Online and real-time paths are not covered.** `test_online_classifier.py` and
`test_stream_architecture.py` are skipped at module level: they import
`mock_emg_stream`, which was removed from `libemg.streamers` in commit
`afa6034`, and there is no replacement mock streamer. Their timing assertions
(`abs(10 - elapsed) < 0.1`) were also flaky by construction on shared CI
runners. Rewriting them needs a new mock streamer and sample-count assertions
instead of wall-clock ones.

Also uncovered: `animator.py`, `_gui/`, `environments/`, the `evaluate*`
functions in `datasets.py` (they need downloads, so they run nightly), and the
vendor modules in `_streamers/` beyond the import check.

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
| `data_handler.py:346` | `get_data` never sorts the glob result, so files load in filesystem order and results differ between machines |
| `feature_selector.py` (`run_selection`) | `verbose=True` calls `_print_wrapper_results` / `_print_filter_results`, neither of which exists |
| `feature_selector.py:163` | The `crossval_amount` path shuffles with unseeded `np.random`, so k-fold selection is not reproducible |
| `feature_selector.py:166` | The `crossval_amount` path shuffles the caller's feature dictionary in place |
| `feature_selector.py` (`_get_activeerror`) | Compares a 0/1/2 condition count against the no-motion class label; only correct when `nm_class == 2` |
| `filtering.py` (`install_filters`) | An unrecognised filter name falls through the if/elif chain and installs a no-op, so `filter()` silently returns unfiltered data |
| `streamers.py` (`leap_streamer`) | Public API with no docstring, so it never reaches the generated docs |
| `_datasets/emg_epn100.py` | `recording_device` is a list, which breaks `Dataset.get_info` string concatenation |
| `_datasets/` | `GrabMyoCrossUser` and `HyserMVC` are concrete datasets in no registry |

## Extension points

`test_extension_contracts.py` exists so the suite covers code nobody has
written yet. Every list in it is derived from the library — from a registry
function where one exists, and by parsing the source where the options live in
a dict literal or an if/elif chain. Add one of the following and it is checked
on the next run, with no test edit:

| You add | Derived from | What is enforced |
| --- | --- | --- |
| A dataset | `get_dataset_list()` | Metadata types `evaluate()` needs, `prepare_data` signature, `get_info` works, registered exactly once |
| A feature | `get_feature_list()` | Shape, finiteness, determinism, edge inputs, docstring, list/implementation bijection, snapshot or MATLAB reference |
| A feature group | `get_feature_groups()` | Every member resolves, and extraction returns all of them |
| A classifier/regressor model | `model_config` (parsed from source) | Fits, predicts, respects the label set, supports `predict_proba` |
| A filter | `install_filters` (parsed from source) | Installs real b/a coefficients and actually changes the signal |
| An offline metric | `get_available_metrics()` | Has a getter, is callable, is documented |
| A selector metric | `FeatureSelector.get_metrics()` | Has a callback, a type and an objective |
| A discrete model | `libemg._discrete_models` | sklearn duck type, `classes_`, proba sums to 1, `argmax(proba) == predict` |

`test_registries_were_actually_discovered` guards the whole scheme: if a
refactor moves one of those registries out of reach, the derived lists would
silently become empty and every test over them would vacuously pass. That test
fails instead.

Streamers are the one extension point with no registry — they are plain
functions in `libemg/streamers.py`. The most that can be checked is that each
module imports and each public function is documented.

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
