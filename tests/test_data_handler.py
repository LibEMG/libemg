"""Unit tests for OfflineDataHandler and its metadata machinery.

data_handler.py is the most-edited non-GUI module in the library (34 of the
last 200 commits to libemg/) and had no direct test coverage. Almost everything
downstream -- windowing, feature extraction, training -- reads what this module
produces, so a silent change here is a silent change everywhere.

The invariant these tests exist to protect is sample/metadata alignment: for
every file, each metadata array must have exactly one entry per sample of EMG.
Everything else in the pipeline assumes it.
"""

import numpy as np
import pytest

from libemg.data_handler import ColumnFetcher, FilePackager, OfflineDataHandler, RegexFilter
from libemg.utils import get_windows, make_regex

from tests.conftest import MYO_CLASSES, MYO_REPS, myo_regex_filters

N_CHANNELS = 8
SAMPLES_PER_FILE = 600  # every file in the bundled dataset except two
N_FILES = 20


# --------------------------------------------------------------------------- #
# RegexFilter
# --------------------------------------------------------------------------- #


class TestRegexFilter:
    def test_matches_only_intended_files(self):
        rf = RegexFilter(left_bound="_C_", right_bound="_EMG", values=["0", "1"], description="c")
        files = ["R_0_C_0_EMG.csv", "R_0_C_1_EMG.csv", "R_0_C_7_EMG.csv", "notes.txt"]
        assert rf.get_matching_files(files) == ["R_0_C_0_EMG.csv", "R_0_C_1_EMG.csv"]

    def test_metadata_is_the_index_of_the_value(self):
        rf = RegexFilter(
            left_bound="_C_", right_bound="_EMG", values=["4", "2", "0"], description="c"
        )
        # Deliberately unsorted values: the stored metadata is the position in
        # the list the caller passed, not the value itself.
        assert rf.get_metadata("R_0_C_0_EMG.csv") == 2
        assert rf.get_metadata("R_0_C_4_EMG.csv") == 0

    def test_none_values_is_rejected(self):
        with pytest.raises(ValueError):
            RegexFilter(left_bound="_C_", right_bound="_E", values=None, description="c")

    @pytest.mark.parametrize(
        "filename, expected",
        [("Participant1/", "1"), ("Participant9/", "9"), ("Participant12/", "12")],
    )
    def test_multi_digit_values_backtrack_correctly(self, filename, expected):
        """Two-digit values must win over the one-digit prefix that also matches.

        make_regex builds an unanchored alternation, so '(?:1|2|...|12)' could
        match just '1' of 'Participant12'. It backtracks to the full value only
        because of the right bound. Datasets with more than nine participants
        or reps depend on this.
        """
        values = [str(i) for i in range(1, 13)]
        rf = RegexFilter(
            left_bound="Participant", right_bound="/", values=values, description="participant"
        )
        assert rf.get_metadata(filename) == values.index(expected)

    def test_make_regex_is_used_consistently(self):
        rf = RegexFilter(left_bound="a", right_bound="b", values=["x"], description="d")
        assert rf.pattern == make_regex("a", "b", ["x"])


# --------------------------------------------------------------------------- #
# get_data
# --------------------------------------------------------------------------- #


class TestGetData:
    def test_finds_every_file(self, myo_odh):
        assert len(myo_odh.data) == N_FILES

    def test_data_is_samples_by_channels(self, myo_odh):
        for file_data in myo_odh.data:
            assert file_data.ndim == 2
            assert file_data.shape[1] == N_CHANNELS
            assert file_data.shape[0] > N_CHANNELS, "looks transposed"

    def test_metadata_has_one_entry_per_sample(self, myo_odh):
        """The load-bearing invariant of the whole class."""
        for i, file_data in enumerate(myo_odh.data):
            n_samples = file_data.shape[0]
            for key in myo_odh.extra_attributes:
                values = getattr(myo_odh, key)[i]
                assert len(values) == n_samples, (
                    f"file {i}: metadata '{key}' has {len(values)} entries for "
                    f"{n_samples} samples"
                )

    def test_metadata_is_constant_within_a_file(self, myo_odh):
        """Each file is one class and one rep, so its metadata is one value."""
        for i in range(len(myo_odh.data)):
            for key in ("classes", "reps"):
                assert len(np.unique(getattr(myo_odh, key)[i])) == 1

    def test_all_declared_attributes_are_present(self, myo_odh):
        assert set(myo_odh.extra_attributes) == {"classes", "reps"}
        for key in myo_odh.extra_attributes:
            assert len(getattr(myo_odh, key)) == len(myo_odh.data)

    def test_every_class_and_rep_is_represented(self, myo_odh):
        classes = {int(c[0][0]) for c in myo_odh.classes}
        reps = {int(r[0][0]) for r in myo_odh.reps}
        assert classes == set(range(len(MYO_CLASSES)))
        assert reps == set(range(len(MYO_REPS)))

    def test_missing_folder_raises_value_error(self):
        odh = OfflineDataHandler()
        with pytest.raises(ValueError, match="not a directory"):
            odh.get_data(folder_location="does/not/exist/", regex_filters=myo_regex_filters())

    def test_data_column_subsets_channels(self, data_dir):
        odh = OfflineDataHandler()
        odh.get_data(
            folder_location=str(data_dir / "myo_dataset") + "/",
            regex_filters=myo_regex_filters(),
            delimiter=",",
            data_column=[0, 1, 2],
        )
        assert all(d.shape[1] == 3 for d in odh.data)


# --------------------------------------------------------------------------- #
# parse_windows
# --------------------------------------------------------------------------- #


class TestParseWindows:
    @pytest.mark.parametrize(
        "window_size, increment",
        [(50, 25), (100, 50), (37, 13), (200, 200), (300, 7)],
    )
    def test_window_count_matches_the_formula(self, myo_odh, window_size, increment):
        """floor((T - size) / increment) + 1 windows per file, summed."""
        expected = sum(
            (d.shape[0] - window_size) // increment + 1 for d in myo_odh.data
        )
        windows, _ = myo_odh.parse_windows(window_size, increment)
        assert windows.shape[0] == expected

    def test_window_shape(self, myo_odh):
        windows, _ = myo_odh.parse_windows(50, 25)
        assert windows.shape[1:] == (N_CHANNELS, 50)

    def test_metadata_has_one_entry_per_window(self, myo_odh):
        windows, metadata = myo_odh.parse_windows(50, 25)
        for key, values in metadata.items():
            assert values.shape[0] == windows.shape[0], f"metadata '{key}' misaligned"

    def test_windows_match_get_windows_on_the_first_file(self, myo_odh):
        """parse_windows is get_windows applied per file, then stacked. Pin the
        relationship so the two cannot drift apart."""
        windows, _ = myo_odh.parse_windows(50, 25)
        expected_first = get_windows(myo_odh.data[0], 50, 25)
        np.testing.assert_array_equal(windows[: expected_first.shape[0]], expected_first)

    def test_metadata_operations_mean(self, myo_odh):
        """A per-window operation other than the default mode."""
        _, default = myo_odh.parse_windows(50, 25)
        _, averaged = myo_odh.parse_windows(50, 25, metadata_operations={"reps": "mean"})
        # Metadata is constant within a file here, so mean == mode; what is
        # being checked is that the option is wired through at all.
        np.testing.assert_allclose(averaged["reps"], default["reps"])

    def test_unknown_metadata_operation_raises(self, myo_odh):
        with pytest.raises(KeyError):
            myo_odh.parse_windows(50, 25, metadata_operations={"reps": "not_an_operation"})

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "A file yielding exactly one window breaks metadata concatenation. "
            "_get_mode_windows ends in an unconditional .squeeze(), which turns "
            "a single window's metadata into a 0-d array while every other file "
            "produces 1-d, and np.concatenate then refuses to stack them."
        ),
    )
    def test_single_window_per_file(self, myo_odh):
        """window_size just under the shortest file: one window from that file."""
        windows, metadata = myo_odh.parse_windows(596, 1)
        assert windows.shape[0] == metadata["classes"].shape[0]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "A file shorter than the window crashes metadata windowing. "
            "get_windows correctly returns an empty (0, C, L) array, but "
            "_get_mode_windows then calls np.apply_along_axis on it and numpy "
            "raises 'Cannot apply_along_axis when any iteration dimensions are "
            "0'. One short recording in a folder takes down the whole load."
        ),
    )
    def test_file_shorter_than_window_is_skipped_not_fatal(self, data_dir, tmp_path):
        import shutil

        source = data_dir / "myo_dataset"
        for path in sorted(source.glob("*.csv")):
            shutil.copy(path, tmp_path / path.name)
        # Truncate one file to below the window length.
        short = np.loadtxt(tmp_path / "R_0_C_0_EMG.csv", delimiter=",")[:10]
        np.savetxt(tmp_path / "R_0_C_0_EMG.csv", short, delimiter=",")

        odh = OfflineDataHandler()
        odh.get_data(
            folder_location=str(tmp_path) + "/",
            regex_filters=myo_regex_filters(),
            delimiter=",",
        )
        windows, metadata = odh.parse_windows(50, 25)
        assert windows.shape[0] == metadata["classes"].shape[0]

    def test_discrete_keeps_files_separate(self, myo_odh):
        windows, metadata = myo_odh.parse_windows(50, 25, discrete=True)
        assert isinstance(windows, list)
        assert len(windows) == N_FILES
        for key, values in metadata.items():
            assert len(values) == N_FILES, f"discrete '{key}' should be one value per file"


class TestParseWindowsMultiRate:
    """The multi-rate path, added for datasets that mix sampling rates."""

    @staticmethod
    def build_handler(rates, samples=1000):
        odh = OfflineDataHandler()
        odh.sampling_rates = []
        odh.classes = []
        odh.extra_attributes = ["classes", "sampling_rates"]
        for i, fs in enumerate(rates):
            odh.data.append(np.ones((samples, N_CHANNELS), dtype=float) * (i + 1))
            odh.sampling_rates.append(np.ones((samples, 1)) * fs)
            odh.classes.append(np.ones((samples, 1)) * i)
        return odh

    def test_window_size_scales_with_sampling_rate(self):
        """A 100 ms window is 20 samples at 200 Hz and 50 at 500 Hz."""
        odh = self.build_handler([200, 500])
        windows, _ = odh.parse_windows(100, 50, multi_rate=True)
        assert isinstance(windows, list) and len(windows) == 2
        assert windows[0].shape[2] == 20
        assert windows[1].shape[2] == 50

    def test_returns_one_entry_per_file(self):
        odh = self.build_handler([200, 500, 1000])
        windows, metadata = odh.parse_windows(100, 50, multi_rate=True)
        assert len(windows) == 3
        for key, values in metadata.items():
            assert len(values) == 3
            for file_windows, file_metadata in zip(windows, values):
                assert len(file_metadata) == file_windows.shape[0], f"'{key}' misaligned"

    def test_rejects_discrete(self):
        odh = self.build_handler([200])
        with pytest.raises(ValueError, match="not supported"):
            odh.parse_windows(100, 50, multi_rate=True, discrete=True)

    def test_sub_sample_window_raises(self):
        """0 ms of signal is not a window."""
        odh = self.build_handler([200])
        with pytest.raises(ValueError, match="<= 0 samples"):
            odh.parse_windows(0, 0, multi_rate=True)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "_multi_rate_parse_windows_helper reads the rate as "
            "getattr(self, key)[i][0].item(), which assumes an array per file. "
            "Scalar per-file metadata is supported everywhere else in this "
            "module and raises here."
        ),
    )
    def test_accepts_scalar_sampling_rate_metadata(self):
        odh = self.build_handler([200])
        odh.sampling_rates = [200.0]  # scalar, as a MetadataFetcher may produce
        windows, _ = odh.parse_windows(100, 50, multi_rate=True)
        assert windows[0].shape[2] == 20


# --------------------------------------------------------------------------- #
# isolate_data / isolate_channels
# --------------------------------------------------------------------------- #


class TestIsolateData:
    def test_selects_only_requested_values(self, myo_odh):
        isolated = myo_odh.isolate_data("classes", [0])
        assert len(isolated.data) == len(MYO_REPS)
        for values in isolated.classes:
            assert np.all(values == 0)

    def test_multiple_values(self, myo_odh):
        isolated = myo_odh.isolate_data("classes", [0, 1])
        assert len(isolated.data) == 2 * len(MYO_REPS)

    def test_fast_and_slow_agree(self, myo_odh):
        """fast=True and fast=False are two implementations of one operation.

        No ground truth needed: they must produce the same result.
        """
        fast = myo_odh.isolate_data("classes", [1, 3], fast=True)
        slow = myo_odh.isolate_data("classes", [1, 3], fast=False)
        assert len(fast.data) == len(slow.data)
        for a, b in zip(fast.data, slow.data):
            np.testing.assert_array_equal(a, b)
        for key in fast.extra_attributes:
            for a, b in zip(getattr(fast, key), getattr(slow, key)):
                np.testing.assert_array_equal(a, b)

    def test_does_not_mutate_the_original(self, fresh_myo_odh):
        before = len(fresh_myo_odh.data)
        fresh_myo_odh.isolate_data("classes", [0])
        assert len(fresh_myo_odh.data) == before

    def test_isolating_by_rep_then_class_commutes(self, myo_odh):
        by_rep_first = myo_odh.isolate_data("reps", [0]).isolate_data("classes", [2])
        by_class_first = myo_odh.isolate_data("classes", [2]).isolate_data("reps", [0])
        assert len(by_rep_first.data) == len(by_class_first.data) == 1
        np.testing.assert_array_equal(by_rep_first.data[0], by_class_first.data[0])


class TestIsolateChannels:
    def test_selects_requested_channels(self, myo_odh):
        isolated = myo_odh.isolate_channels([0, 2, 4])
        for original, new in zip(myo_odh.data, isolated.data):
            assert new.shape == (original.shape[0], 3)
            np.testing.assert_array_equal(new, original[:, [0, 2, 4]])

    def test_preserves_metadata(self, myo_odh):
        isolated = myo_odh.isolate_channels([0, 1])
        for key in myo_odh.extra_attributes:
            assert len(getattr(isolated, key)) == len(getattr(myo_odh, key))

    @pytest.mark.parametrize("channel", [-1, N_CHANNELS, 99])
    def test_out_of_range_returns_none(self, myo_odh, channel):
        assert myo_odh.isolate_channels([channel]) is None

    def test_does_not_mutate_the_original(self, fresh_myo_odh):
        fresh_myo_odh.isolate_channels([0])
        assert fresh_myo_odh.data[0].shape[1] == N_CHANNELS

    def test_channel_last_false_selects_rows(self, fresh_myo_odh):
        """With [channels, samples] data, channels are the first axis."""
        fresh_myo_odh.data = [d.T for d in fresh_myo_odh.data]
        isolated = fresh_myo_odh.isolate_channels([0, 1, 2], channel_last=False)
        assert isolated.data[0].shape == (3, SAMPLES_PER_FILE)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The bound check is `c >= len(self.data[0][0])`, which is the channel "
            "count only for [samples, channels] data. With channel_last=False it "
            "measures the number of samples, so out-of-range channels sail past "
            "validation and raise IndexError instead of returning None."
        ),
    )
    def test_channel_last_false_validates_against_channel_count(self, fresh_myo_odh):
        fresh_myo_odh.data = [d.T for d in fresh_myo_odh.data]  # (8, 600)
        # 99 is not a channel, and 99 < 600, so the guard lets it through.
        assert fresh_myo_odh.isolate_channels([99], channel_last=False) is None


# --------------------------------------------------------------------------- #
# active_threshold
# --------------------------------------------------------------------------- #


def test_active_threshold_relabels_quiet_windows(myo_odh, seeded_rng):
    """Active windows quieter than no-motion + n*std become no motion.

    Rest is low-amplitude noise rather than exact zeros so that the no-motion
    standard deviation is non-zero -- with an all-zero rest the threshold
    collapses to 0 and nothing can fall below it.
    """
    rest = seeded_rng.normal(scale=1.0, size=(20, N_CHANNELS, 50))
    loud = seeded_rng.normal(scale=100.0, size=(5, N_CHANNELS, 50))
    quiet = seeded_rng.normal(scale=0.01, size=(5, N_CHANNELS, 50))

    active_windows = np.vstack([loud, quiet])
    labels = [1] * 10

    updated = myo_odh.active_threshold(rest, active_windows, list(labels), num_std=3, nm_label=0)

    assert list(updated[:5]) == [1] * 5, "strong contractions should keep their label"
    assert list(updated[5:]) == [0] * 5, "near-silent windows should become no motion"


def test_active_threshold_leaves_strong_contractions_alone(myo_odh, seeded_rng):
    rest = seeded_rng.normal(scale=1.0, size=(20, N_CHANNELS, 50))
    loud = seeded_rng.normal(scale=50.0, size=(8, N_CHANNELS, 50))
    labels = [2] * 8
    assert list(myo_odh.active_threshold(rest, loud, list(labels), nm_label=0)) == labels


# --------------------------------------------------------------------------- #
# MetadataFetchers
# --------------------------------------------------------------------------- #


@pytest.fixture
def packaged_dataset(tmp_path):
    """Two EMG files, each with a sibling labels file of the same length."""
    for rep in range(2):
        np.savetxt(tmp_path / f"R_{rep}_EMG.csv", np.arange(60).reshape(10, 6), delimiter=",")
        np.savetxt(tmp_path / f"R_{rep}_LABELS.csv", np.full((10, 1), rep), delimiter=",")
    return tmp_path


class TestMetadataFetchers:
    def test_file_packager_attaches_sibling_labels(self, packaged_dataset):
        odh = OfflineDataHandler()
        odh.get_data(
            folder_location=str(packaged_dataset) + "/",
            regex_filters=[
                RegexFilter(
                    left_bound="R_", right_bound="_EMG", values=["0", "1"], description="reps"
                )
            ],
            metadata_fetchers=[
                FilePackager(
                    RegexFilter(
                        left_bound="R_",
                        right_bound="_LABELS",
                        values=["0", "1"],
                        description="labels",
                    ),
                    package_function=lambda metadata_file, data_file: (
                        metadata_file.replace("_LABELS", "") == data_file.replace("_EMG", "")
                    ),
                )
            ],
            delimiter=",",
        )

        assert len(odh.data) == 2
        assert "labels" in odh.extra_attributes
        for i, file_data in enumerate(odh.data):
            assert len(odh.labels[i]) == file_data.shape[0], "labels misaligned with samples"

    def test_column_fetcher_pulls_metadata_out_of_the_data_file(self, tmp_path):
        """The last column is a label channel, not EMG."""
        raw = np.hstack([np.random.default_rng(0).normal(size=(10, 4)), np.full((10, 1), 3.0)])
        np.savetxt(tmp_path / "R_0_EMG.csv", raw, delimiter=",")

        odh = OfflineDataHandler()
        odh.get_data(
            folder_location=str(tmp_path) + "/",
            regex_filters=[
                RegexFilter(left_bound="R_", right_bound="_EMG", values=["0"], description="reps")
            ],
            metadata_fetchers=[ColumnFetcher("labels", column_mask=[4])],
            delimiter=",",
            data_column=[0, 1, 2, 3],
        )

        assert odh.data[0].shape == (10, 4), "the label column should not be EMG"
        assert len(odh.labels[0]) == 10
        np.testing.assert_allclose(np.asarray(odh.labels[0]).squeeze(), 3.0)
