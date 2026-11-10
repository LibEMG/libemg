"""Unit tests for the Filter class.

filtering.py had no test coverage. It is small and barely churns, but it sits
directly on the signal path: a filter that quietly stops filtering produces
data that still has the right shape, still trains a model, and is simply
wrong. So these tests assert on the *frequency content* of the output wherever
they can, not merely that a call returned an array.
"""

import numpy as np
import pytest

from libemg.filtering import Filter

FS = 1000
DURATION_S = 2
N_CHANNELS = 4


@pytest.fixture
def time_axis():
    return np.arange(0, DURATION_S, 1 / FS)


def sine(frequencies, time, channels=N_CHANNELS):
    """A multi-channel signal made of the given frequencies."""
    signal = sum(np.sin(2 * np.pi * f * time) for f in frequencies)
    return np.tile(signal[:, None], (1, channels))


def power_at(data, frequency, fs=FS):
    """Magnitude of the FFT bin nearest to `frequency`, averaged over channels."""
    spectrum = np.abs(np.fft.rfft(data, axis=0))
    freqs = np.fft.rfftfreq(data.shape[0], 1 / fs)
    return float(spectrum[np.argmin(np.abs(freqs - frequency))].mean())


# --------------------------------------------------------------------------- #
# Behaviour: the filters change the signal in the way they claim
# --------------------------------------------------------------------------- #


class TestFilterBehaviour:
    def test_notch_removes_only_the_target_frequency(self, time_axis):
        data = sine([60, 150], time_axis)

        filt = Filter(FS)
        filt.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
        filtered = filt.filter(data)

        assert power_at(filtered, 60) < 0.05 * power_at(data, 60), "60 Hz survived the notch"
        assert power_at(filtered, 150) > 0.9 * power_at(data, 150), "150 Hz should be untouched"

    def test_highpass_removes_low_frequency_drift(self, time_axis):
        data = sine([2, 200], time_axis)

        filt = Filter(FS)
        filt.install_filters({"name": "highpass", "cutoff": 20, "order": 2})
        filtered = filt.filter(data)

        assert power_at(filtered, 2) < 0.2 * power_at(data, 2)
        assert power_at(filtered, 200) > 0.8 * power_at(data, 200)

    def test_lowpass_removes_high_frequency_noise(self, time_axis):
        data = sine([10, 400], time_axis)

        filt = Filter(FS)
        filt.install_filters({"name": "lowpass", "cutoff": 100, "order": 4})
        filtered = filt.filter(data)

        assert power_at(filtered, 400) < 0.1 * power_at(data, 400)
        assert power_at(filtered, 10) > 0.8 * power_at(data, 10)

    def test_bandpass_keeps_the_passband_and_drops_both_sides(self, time_axis):
        data = sine([5, 100, 480], time_axis)

        filt = Filter(FS)
        filt.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
        filtered = filt.filter(data)

        assert power_at(filtered, 100) > 0.8 * power_at(data, 100)
        assert power_at(filtered, 5) < 0.2 * power_at(data, 5)
        assert power_at(filtered, 480) < 0.2 * power_at(data, 480)

    def test_bandstop_is_the_inverse_of_bandpass(self, time_axis):
        data = sine([10, 100, 480], time_axis)

        filt = Filter(FS)
        filt.install_filters({"name": "bandstop", "cutoff": [20, 450], "order": 4})
        filtered = filt.filter(data)

        assert power_at(filtered, 100) < 0.2 * power_at(data, 100)
        assert power_at(filtered, 10) > 0.7 * power_at(data, 10)


# --------------------------------------------------------------------------- #
# Installation and validation
# --------------------------------------------------------------------------- #


class TestInstallFilters:
    @pytest.mark.parametrize(
        "spec",
        [
            {"name": "notch", "cutoff": 60, "bandwidth": 3},
            {"name": "lowpass", "cutoff": 100, "order": 4},
            {"name": "highpass", "cutoff": 20, "order": 2},
            {"name": "bandpass", "cutoff": [20, 450], "order": 4},
            {"name": "bandstop", "cutoff": [20, 450], "order": 4},
        ],
    )
    def test_every_filter_type_installs_and_runs(self, spec, time_axis):
        data = sine([50, 200], time_axis)
        filt = Filter(FS)
        filt.install_filters(spec)

        assert len(filt.filters) == 1
        filtered = filt.filter(data)
        assert filtered.shape == data.shape
        assert np.all(np.isfinite(filtered))

    def test_filters_accumulate_in_installation_order(self):
        filt = Filter(FS)
        filt.install_filters({"name": "highpass", "cutoff": 20, "order": 2})
        filt.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
        assert [f["name"] for f in filt.filters] == ["highpass", "notch"]

    @pytest.mark.parametrize(
        "spec",
        [
            {"name": "lowpass", "cutoff": 600, "order": 4},  # above Nyquist (500)
            {"name": "bandpass", "cutoff": [20, 900], "order": 4},
            {"name": "notch", "cutoff": 600, "bandwidth": 3},
        ],
    )
    def test_cutoff_above_nyquist_is_rejected(self, spec):
        """These guards are the only validation the class has."""
        with pytest.raises(AssertionError):
            Filter(FS).install_filters(spec)

    def test_standardize_requires_data(self):
        with pytest.raises(AssertionError):
            Filter(FS).install_filters({"name": "standardize"})

    def test_filtering_with_nothing_installed_returns_input_unchanged(self, time_axis):
        data = sine([50], time_axis)
        np.testing.assert_array_equal(Filter(FS).filter(data), data)


class TestCommonFilters:
    def test_high_sampling_rate_gets_a_bandpass(self):
        filt = Filter(1000)
        filt.install_common_filters()
        assert [f["name"] for f in filt.filters] == ["bandpass", "notch"]

    def test_low_sampling_rate_falls_back_to_highpass(self):
        """A 20-450 Hz bandpass is impossible below a 900 Hz sampling rate."""
        filt = Filter(200)
        filt.install_common_filters()
        assert [f["name"] for f in filt.filters] == ["highpass", "notch"]

    def test_the_fallback_boundary_is_1000_hz(self):
        assert Filter(999).__class__ is Filter  # guard against a silent API change
        low = Filter(999)
        low.install_common_filters()
        high = Filter(1000)
        high.install_common_filters()
        assert low.filters[0]["name"] == "highpass"
        assert high.filters[0]["name"] == "bandpass"


# --------------------------------------------------------------------------- #
# Integration with OfflineDataHandler
# --------------------------------------------------------------------------- #


class TestFilterOnDataHandler:
    def test_filters_in_place_and_preserves_shape(self, fresh_myo_odh):
        before = [d.copy() for d in fresh_myo_odh.data]

        filt = Filter(200)
        filt.install_filters({"name": "highpass", "cutoff": 20, "order": 2})
        filt.filter(fresh_myo_odh)

        assert len(fresh_myo_odh.data) == len(before)
        for original, filtered in zip(before, fresh_myo_odh.data):
            assert filtered.shape == original.shape
            assert np.all(np.isfinite(filtered))
            assert not np.array_equal(filtered, original), "filter had no effect"

    def test_metadata_is_untouched(self, fresh_myo_odh):
        before = [c.copy() for c in fresh_myo_odh.classes]
        filt = Filter(200)
        filt.install_filters({"name": "highpass", "cutoff": 20, "order": 2})
        filt.filter(fresh_myo_odh)
        for original, after in zip(before, fresh_myo_odh.classes):
            np.testing.assert_array_equal(original, after)

    def test_standardize_centres_the_data(self, fresh_myo_odh):
        filt = Filter(200)
        filt.install_filters({"name": "standardize", "data": fresh_myo_odh})
        filt.filter(fresh_myo_odh)

        stacked = np.vstack(fresh_myo_odh.data)
        np.testing.assert_allclose(stacked.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(stacked.std(axis=0), 1.0, atol=1e-6)


class TestFrequencyDomain:
    def test_spectrum_and_frequency_axis_line_up(self, time_axis):
        data = sine([60], time_axis)
        power_spectrum, freqs = Filter(FS).get_frequency_domain(data)

        assert power_spectrum.shape[0] == freqs.shape[0]
        assert power_spectrum.shape[1] == data.shape[1]
        assert np.all(power_spectrum >= 0), "a power spectrum cannot be negative"

    def test_peak_lands_on_the_signal_frequency(self, time_axis):
        """A pure 60 Hz tone should peak at 60 Hz, within one bin."""
        data = sine([60], time_axis)
        power_spectrum, freqs = Filter(FS).get_frequency_domain(data)
        peak = freqs[np.argmax(power_spectrum[:, 0])]
        bin_width = freqs[1] - freqs[0]
        assert abs(peak - 60) <= bin_width
