"""Check a physiological signal is worth recording before a session records it.

Most unusable EMG datasets were unusable from the first repetition, and nothing
about the recording says so: a disconnected channel writes a flat line, a
loose electrode writes motion artifact, a badly grounded rig writes mains hum,
and an over-gained amplifier writes a clipped square wave. All four look like
data. They are found weeks later, in analysis, when the subject has gone home.

So the checks are here, to be run against a few seconds of stream before the
first prompt goes up, while the electrodes can still be re-seated.

Everything is measured relative to the signal itself rather than in microvolts.
A LibEMG stream carries whatever units its device chose, and a threshold in
microvolts is wrong for all but one of them -- whereas "this channel is a
hundredth as active as its neighbours" and "a fifth of this channel's power sits
in a 2 Hz band at the mains frequency" mean the same thing on every device.

What is checked, and why
------------------------
**Dead and flat channels.** A channel that is not moving is not connected. The
usual cause is a lead that came off between donning and starting.

**Clipping.** A signal that sits at its extreme for runs of consecutive samples
has hit the amplifier rail. The peaks are gone, and every amplitude feature
computed from them is wrong in a way that looks plausible.

**Mains interference.** Power line pickup at 50 or 60 Hz sits inside the EMG
band, so it cannot be filtered out without taking signal with it. Measured as
the fraction of in-band power within a narrow band at the line frequency and its
first harmonic.

**Baseline activity.** How much a channel is doing while the subject is at rest.
High baseline is drifting electrodes, movement artifact or a subject who is not
actually relaxed -- and it sets the floor for everything the session can resolve.

**Channel imbalance.** One channel far louder or quieter than the rest is
usually a contact problem rather than anatomy.

**Fatigue.** The median frequency of the EMG power spectrum falls as a muscle
fatigues. Compared between the start and the end of a session it says whether
the last repetitions are the same task as the first ones.

Examples
---------
Before a collection::

    from libemg.signal_quality import check_signal

    report = check_signal(odh.get_data()[0]['emg'], sampling_rate=1000)
    print(report.summary())
    if report.problems:
        ...   # re-seat the electrodes rather than collect

Tracking fatigue across a session::

    from libemg.signal_quality import median_frequency

    start = median_frequency(first_rep, 1000)
    end = median_frequency(last_rep, 1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How still a channel has to be, relative to the busiest channel, before it is
#: called dead. Generous on purpose: a genuinely quiet muscle at rest is normal,
#: a channel three orders of magnitude below its neighbours is a lead off.
DEAD_CHANNEL_RATIO = 0.01
#: Samples in a row at the extreme before it is called clipping rather than a
#: peak that happened to be the largest one.
CLIP_RUN = 3
#: Fraction of samples that may sit in a clipped run before it is reported.
CLIP_TOLERANCE = 0.001
#: Fraction of in-band power at the line frequency before it is reported.
LINE_NOISE_TOLERANCE = 0.15
#: How far a channel's activity may sit from the median channel's, as a ratio,
#: before the set is called imbalanced.
IMBALANCE_RATIO = 6.0
#: Drop in median frequency, as a fraction, taken as meaningful fatigue.
FATIGUE_TOLERANCE = 0.08


@dataclass
class ChannelQuality:
    """What one channel looked like over the window that was checked."""

    index: int
    rms: float = 0.0
    peak: float = 0.0
    dc_offset: float = 0.0
    clipped_fraction: float = 0.0
    line_noise_ratio: float = 0.0
    median_frequency: float = 0.0
    dead: bool = False

    def to_dict(self):
        return {"index": self.index, "rms": self.rms, "peak": self.peak,
                "dc_offset": self.dc_offset,
                "clipped_fraction": self.clipped_fraction,
                "line_noise_ratio": self.line_noise_ratio,
                "median_frequency": self.median_frequency, "dead": self.dead}


@dataclass
class SignalQualityReport:
    """The state of every channel, and what is wrong with any of them.

    ``problems`` is the part to act on: each entry is one sentence naming the
    channels affected and what to do about it. An empty list means nothing was
    detected, which is not the same as a guarantee -- these checks catch the
    failures that are visible in a few seconds of stream, not every way a
    recording can go wrong.
    """

    modality: str = "emg"
    sampling_rate: float = 0.0
    samples: int = 0
    channels: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problems

    def to_dict(self):
        return {"modality": self.modality, "sampling_rate": self.sampling_rate,
                "samples": self.samples,
                "channels": [channel.to_dict() for channel in self.channels],
                "problems": list(self.problems), "notes": list(self.notes)}

    def summary(self):
        """A few lines fit to print in a window or a terminal."""
        seconds = self.samples / self.sampling_rate if self.sampling_rate else 0.0
        lines = [f"{self.modality}: {len(self.channels)} channels, {self.samples} samples "
                 f"({seconds:.1f} s at {self.sampling_rate:.0f} Hz)"]
        for problem in self.problems:
            lines.append(f"  ! {problem}")
        for note in self.notes:
            lines.append(f"  - {note}")
        if not self.problems:
            lines.append("  no problems detected")
        return "\n".join(lines)


def check_signal(data, sampling_rate, modality="emg", line_frequency=60.0,
                 at_rest=True):
    """Look over a window of stream and report anything wrong with it.

    Parameters
    ----------
    data: array_like
        ``samples x channels``. A few seconds is enough; a few hundred samples
        is not enough to say anything about the spectrum and the report says so.
    sampling_rate: float
        Samples per second, needed for every frequency-domain check.
    modality: str (optional), default='emg'
        Named in the report, so a multi-modality rig can be checked one stream
        at a time.
    line_frequency: float (optional), default=60.0
        Mains frequency where the rig is: 60 Hz in North America, 50 Hz in most
        of the rest of the world. Checked at this frequency and its first
        harmonic.
    at_rest: bool (optional), default=True
        Whether the subject was relaxed for this window. Only the resting case
        supports saying anything about baseline activity, so the imbalance and
        baseline checks are skipped when it is False.

    Returns
    ----------
    report: SignalQualityReport
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    report = SignalQualityReport(modality=modality, sampling_rate=float(sampling_rate),
                                 samples=int(data.shape[0]))
    if data.size == 0 or data.shape[0] < 2:
        report.problems.append(
            f"No {modality} samples arrived, so nothing could be checked. The "
            f"device is not delivering.")
        return report

    detrended = data - data.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(detrended ** 2, axis=0))
    loudest = float(rms.max()) if rms.size else 0.0
    spectra = _band_powers(detrended, sampling_rate, line_frequency)

    for index in range(data.shape[1]):
        column = data[:, index]
        report.channels.append(ChannelQuality(
            index=index,
            rms=float(rms[index]),
            peak=float(np.max(np.abs(column - column.mean()))),
            dc_offset=float(column.mean()),
            clipped_fraction=_clipped_fraction(column),
            # Plain floats, not numpy scalars: this report is written to YAML
            # and a numpy scalar has no representer there.
            line_noise_ratio=float(spectra["line"][index]),
            median_frequency=float(spectra["median_frequency"][index]),
            dead=bool(loudest > 0 and rms[index] <= DEAD_CHANNEL_RATIO * loudest)))

    _report_dead(report, rms, loudest)
    _report_clipping(report)
    _report_line_noise(report, line_frequency)
    if at_rest:
        _report_imbalance(report, rms)
    if report.samples < 2 * sampling_rate:
        report.notes.append(
            f"Only {report.samples / max(1.0, sampling_rate):.1f} s was checked. Two "
            f"seconds or more makes the frequency checks worth trusting.")
    return report


def _clipped_fraction(column, run=CLIP_RUN):
    """Fraction of samples sitting in a run at the signal's extreme.

    A single sample at the maximum is just the largest sample. Several in a row
    at the same value, at the extreme, is an amplifier that ran out of range --
    and that is what removes the peaks amplitude features are computed from.
    """
    extreme = np.max(np.abs(column))
    if extreme <= 0:
        return 0.0
    at_rail = np.abs(np.abs(column) - extreme) <= 1e-9 * max(1.0, extreme)
    if not at_rail.any():
        return 0.0
    # Count only rail samples that sit in a run of at least `run` of them.
    padded = np.concatenate(([False], at_rail, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    total = 0
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start >= run:
            total += stop - start
    return float(total) / len(column)


def _band_powers(detrended, sampling_rate, line_frequency):
    """Line-frequency share of in-band power, and median frequency, per channel."""
    n_samples, n_channels = detrended.shape
    zeros = np.zeros(n_channels)
    if n_samples < 8 or sampling_rate <= 0:
        return {"line": zeros, "median_frequency": zeros}
    window = np.hanning(n_samples)[:, None]
    spectrum = np.abs(np.fft.rfft(detrended * window, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    # The EMG band. Below it is movement artifact and electrode drift, above it
    # is mostly amplifier noise, and including either would flatter the ratios.
    in_band = (freqs >= 20.0) & (freqs <= min(500.0, sampling_rate / 2))
    if not in_band.any():
        return {"line": zeros, "median_frequency": zeros}
    band_power = spectrum[in_band].sum(axis=0)
    band_power = np.where(band_power > 0, band_power, np.inf)

    line = np.zeros(n_channels)
    for harmonic in (1, 2):
        centre = line_frequency * harmonic
        if centre >= sampling_rate / 2:
            break
        near = in_band & (np.abs(freqs - centre) <= 2.0)
        if near.any():
            line += spectrum[near].sum(axis=0) / band_power

    # Median frequency: where half the in-band power lies below. The standard
    # fatigue index, and it falls as a muscle tires.
    cumulative = np.cumsum(spectrum[in_band], axis=0)
    half = cumulative[-1] / 2.0
    median = np.zeros(n_channels)
    band_freqs = freqs[in_band]
    for channel in range(n_channels):
        if cumulative[-1, channel] <= 0:
            continue
        median[channel] = float(band_freqs[np.searchsorted(cumulative[:, channel],
                                                           half[channel])])
    return {"line": line, "median_frequency": median}


def _report_dead(report, rms, loudest):
    dead = [channel.index for channel in report.channels if channel.dead]
    if loudest <= 0:
        report.problems.append(
            "Every channel is flat. Either the device is delivering zeros or "
            "nothing is connected to it.")
    elif dead:
        report.problems.append(
            f"Channel(s) {_listed(dead)} are effectively flat next to the others. "
            f"Check those leads are seated before collecting.")


def _report_clipping(report):
    clipped = [channel.index for channel in report.channels
               if channel.clipped_fraction > CLIP_TOLERANCE]
    if clipped:
        worst = max(channel.clipped_fraction for channel in report.channels)
        report.problems.append(
            f"Channel(s) {_listed(clipped)} are clipping ({100 * worst:.1f}% of "
            f"samples at the rail). Reduce the gain -- the peaks amplitude "
            f"features are computed from are already gone.")


def _report_line_noise(report, line_frequency):
    noisy = [channel.index for channel in report.channels
             if channel.line_noise_ratio > LINE_NOISE_TOLERANCE]
    if noisy:
        worst = max(channel.line_noise_ratio for channel in report.channels)
        report.problems.append(
            f"Channel(s) {_listed(noisy)} carry heavy {line_frequency:.0f} Hz mains "
            f"pickup ({100 * worst:.0f}% of in-band power). Check the ground "
            f"electrode and move away from the mains source; this sits inside "
            f"the EMG band and cannot be filtered out cleanly later.")


def _report_imbalance(report, rms):
    active = rms[rms > 0]
    if len(active) < 3:
        return
    middle = float(np.median(active))
    if middle <= 0:
        return
    odd = [channel.index for channel in report.channels
           if channel.rms > 0 and not (1 / IMBALANCE_RATIO <= channel.rms / middle
                                       <= IMBALANCE_RATIO)]
    if odd:
        report.problems.append(
            f"Channel(s) {_listed(odd)} sit far from the others in amplitude at "
            f"rest. Usually a contact problem rather than anatomy.")


def _listed(indices):
    return ", ".join(str(index) for index in indices)


def median_frequency(data, sampling_rate, band=(20.0, 500.0)):
    """Median frequency of the power spectrum, per channel, in Hz.

    The standard fatigue index: as a muscle fatigues, conduction velocity falls
    and the spectrum shifts down. Compared between the start and the end of a
    session it says whether the last repetitions are the same task as the first.

    Returns
    ----------
    numpy.ndarray
        One value per channel. Zero for a channel with no power in the band.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] < 8:
        return np.zeros(data.shape[1])
    detrended = data - data.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(detrended * np.hanning(len(detrended))[:, None],
                                  axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(detrended), d=1.0 / sampling_rate)
    in_band = (freqs >= band[0]) & (freqs <= min(band[1], sampling_rate / 2))
    if not in_band.any():
        return np.zeros(data.shape[1])
    cumulative = np.cumsum(spectrum[in_band], axis=0)
    band_freqs = freqs[in_band]
    out = np.zeros(data.shape[1])
    for channel in range(data.shape[1]):
        if cumulative[-1, channel] <= 0:
            continue
        out[channel] = band_freqs[np.searchsorted(cumulative[:, channel],
                                                  cumulative[-1, channel] / 2.0)]
    return out


def fatigue_shift(early, late, sampling_rate):
    """How far the median frequency moved between two stretches of a session.

    Parameters
    ----------
    early, late: array_like
        ``samples x channels`` from the start and the end of the session. The
        same muscle contracting at the same intended effort, or the comparison
        means nothing.
    sampling_rate: float

    Returns
    ----------
    (shift, message): tuple of float and str
        ``shift`` is the fractional change in mean median frequency, negative
        for the downward shift fatigue produces. ``message`` is empty when the
        shift is within :data:`FATIGUE_TOLERANCE`.
    """
    first = median_frequency(early, sampling_rate)
    last = median_frequency(late, sampling_rate)
    usable = (first > 0) & (last > 0)
    if not usable.any():
        return 0.0, ""
    shift = float(np.mean(last[usable] - first[usable]) / np.mean(first[usable]))
    if shift < -FATIGUE_TOLERANCE:
        return shift, (
            f"Median frequency fell {100 * abs(shift):.0f}% between the start and "
            f"the end of this session, which is the shift a fatiguing muscle "
            f"makes. The later repetitions are not the same task as the earlier "
            f"ones -- lengthen the rests or shorten the session.")
    return shift, ""


__all__ = ["check_signal", "SignalQualityReport", "ChannelQuality",
           "median_frequency", "fatigue_shift", "DEAD_CHANNEL_RATIO",
           "CLIP_RUN", "CLIP_TOLERANCE", "LINE_NOISE_TOLERANCE",
           "IMBALANCE_RATIO", "FATIGUE_TOLERANCE"]
