"""Coverage for the checks that run before a session records anything."""
import numpy as np
import pytest

from libemg.signal_quality import (SignalQualityReport, check_signal,
                                   fatigue_shift, median_frequency)

FS = 1000.0


def _emg(n=4000, centre=90.0, width=70.0, scale=1.0, seed=0):
    """Noise shaped into the EMG band, which is what a healthy channel looks like."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(0, scale, n))
    freqs = np.fft.rfftfreq(n, 1 / FS)
    spectrum = spectrum * np.exp(-((freqs - centre) ** 2) / (2 * width ** 2))
    signal = np.fft.irfft(spectrum, n)
    return signal / (np.std(signal) or 1.0) * scale


def _channels(count=4, seed=0, **kwargs):
    return np.column_stack([_emg(seed=seed + index, **kwargs)
                            for index in range(count)])


def test_a_healthy_set_of_channels_raises_nothing():
    report = check_signal(_channels(), FS)

    assert report.ok and report.problems == []
    assert len(report.channels) == 4
    assert report.samples == 4000
    assert "no problems detected" in report.summary()
    # Every channel gets a median frequency, which is what fatigue is tracked on.
    assert all(60 < channel.median_frequency < 130 for channel in report.channels)


def test_a_disconnected_channel_is_named():
    data = _channels()
    data[:, 1] *= 1e-6

    report = check_signal(data, FS)

    assert any("Channel(s) 1" in problem and "flat" in problem
               for problem in report.problems)
    assert report.channels[1].dead and not report.channels[0].dead


def test_a_device_delivering_nothing_is_reported_rather_than_passed():
    assert "not delivering" in check_signal(np.empty((0, 4)), FS).problems[0]
    flat = check_signal(np.zeros((2000, 4)), FS)
    assert any("Every channel is flat" in problem for problem in flat.problems)


def test_a_clipping_amplifier_is_caught_while_the_gain_can_still_be_turned_down():
    data = _channels()
    data[:, 2] = np.clip(data[:, 2] * 40, -1, 1)

    report = check_signal(data, FS)

    assert any("Channel(s) 2" in problem and "clipping" in problem
               for problem in report.problems)
    assert report.channels[2].clipped_fraction > 0.5
    assert report.channels[0].clipped_fraction == 0.0


def test_one_sample_at_the_peak_is_not_clipping():
    data = _channels(count=3)
    data[100, 0] = 10 * data[:, 0].max()
    assert check_signal(data, FS).channels[0].clipped_fraction == 0.0


def test_mains_pickup_is_caught_at_either_line_frequency():
    t = np.arange(4000) / FS
    for line in (50.0, 60.0):
        data = _channels()
        data[:, 3] = data[:, 3] + 5 * np.sin(2 * np.pi * line * t)

        report = check_signal(data, FS, line_frequency=line)

        assert any("Channel(s) 3" in problem and f"{line:.0f} Hz" in problem
                   for problem in report.problems)
        assert report.channels[3].line_noise_ratio > 0.5
        assert report.channels[0].line_noise_ratio < 0.15
        # A 60 Hz rig is not upset by a clean channel when told it is 50 Hz.
        assert check_signal(_channels(), FS, line_frequency=line).ok


def test_a_channel_far_louder_than_the_rest_is_reported_at_rest():
    data = _channels()
    data[:, 0] *= 50

    assert any("amplitude at rest" in problem
               for problem in check_signal(data, FS).problems)
    # During a contraction one channel being loud is the point, so it is not
    # reported when the window is not a resting one.
    assert not any("amplitude at rest" in problem
                   for problem in check_signal(data, FS, at_rest=False).problems)


def test_a_short_window_is_checked_but_flagged_as_short():
    report = check_signal(_channels(n=500), FS)
    assert any("makes the frequency checks worth trusting" in note
               for note in report.notes)


def test_the_report_survives_a_round_trip_to_a_dict():
    report = check_signal(_channels(), FS, modality="imu")
    as_dict = report.to_dict()
    assert as_dict["modality"] == "imu"
    assert len(as_dict["channels"]) == 4
    assert set(as_dict) == {"modality", "sampling_rate", "samples", "channels",
                            "problems", "notes"}
    assert SignalQualityReport().ok


def test_median_frequency_follows_where_the_power_is():
    low = median_frequency(_channels(count=2, centre=60), FS)
    high = median_frequency(_channels(count=2, centre=140), FS)

    assert np.all(low < high)
    assert np.all(np.abs(low - 60) < 25) and np.all(np.abs(high - 140) < 25)
    # A signal with nothing in the band reports zero rather than a wrong number.
    assert np.all(median_frequency(np.zeros((1000, 2)), FS) == 0)
    assert np.all(median_frequency(np.zeros((4, 2)), FS) == 0)


def test_a_session_that_fatigued_the_muscle_says_so():
    early = _channels(centre=110)
    late = _channels(centre=80, seed=1)

    shift, message = fatigue_shift(early, late, FS)

    assert shift < -0.08
    assert "Median frequency fell" in message and "lengthen the rests" in message


def test_a_session_that_did_not_fatigue_says_nothing():
    shift, message = fatigue_shift(_channels(centre=100),
                                   _channels(centre=100, seed=2), FS)
    assert message == "" and abs(shift) < 0.08


def test_fatigue_needs_something_to_compare():
    assert fatigue_shift(np.zeros((1000, 2)), np.zeros((1000, 2)), FS) == (0.0, "")


def test_a_single_channel_is_accepted_as_a_flat_array():
    report = check_signal(_emg(), FS)
    assert len(report.channels) == 1
    assert median_frequency(_emg(), FS).shape == (1,)
