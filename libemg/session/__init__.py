"""Read, index and re-cut a continuous screen guided training recording.

Screen guided training records a collection in one piece. Every sample the
device delivered between pressing Start and the last repetition ending goes into
``_session_<modality>.csv`` -- the prompts, the rests in front of them, and the
pauses where the collection sat waiting for someone to press Continue. Nothing
is dropped for being outside a repetition.

That only helps if it is possible to say later what part of the recording was
what, so the collection also writes ``_session_annotations.json`` beside it: one
entry per stage, with the wall-clock time the stage ran and the sample indices it
occupies in each modality's file. This module is that file. It is the object the
collection panel fills in as it runs, and the object a script loads afterwards to
cut the same recording again without collecting it again.

Two things follow from keeping the recording whole and describing it separately.

A repetition becomes a *view* on the session rather than a file that had to
exist. A model that wants a longer stretch than one repetition -- a temporal
model reading across the transition into and out of a contraction, or through a
rest into the next prompt -- reads the session file and uses these indices to
know what it is looking at, instead of stitching per-repetition files back
together across the gaps where the recording was stopped.

And the cut is reproducible. ``C_0_R_0_emg.csv`` is nothing more than the session
file sliced at indices recorded here, so cutting it a second way -- with the
rests kept, with a lead-in before each prompt, with only some classes -- costs a
function call rather than another session in front of the screen.

Examples
---------
Cut a finished session again, this time keeping the rests and a quarter second
of lead-in before every prompt::

    from libemg.session import SessionAnnotations

    session = SessionAnnotations.load('data/_session_annotations.json')
    session.cut('data/recut/', kinds=('prompt', 'rest'), lead_in=0.25)

Read a whole modality and ask what each part of it is, without cutting at all::

    arrivals, emg = session.read('emg')
    for span in session.timeline('emg'):
        print(span.kind, span.label, span.start, span.end)

Or from a terminal::

    python -m libemg.session show data/_session_annotations.json
    python -m libemg.session cut data/_session_annotations.json --kinds prompt rest
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# File prefix for the raw, uncut recording a continuous collection writes, and
# the name of the annotation file that says what is in it. The recording is kept
# after the repetitions are cut out of it, so a session can be re-cut without
# being collected again -- which is the whole reason this module exists.
SESSION_PREFIX = "_session_"
ANNOTATIONS_NAME = SESSION_PREFIX + "annotations.json"
SCHEMA_VERSION = 1

# The stages a collection moves through. Only PROMPT is a repetition in the
# usual sense; the other two are recorded because they are in the recording and
# something downstream may want them, not because the collection needs them.
PROMPT = "prompt"
REST = "rest"
PAUSE = "pause"

# How each kind of stage is named when it is cut to a file. Only a prompt gets
# the standard ``C_<motion>_R_<rep>_`` name an OfflineDataHandler looks for; the
# others deliberately carry no ``C_``/``R_`` tokens at all, so pointing a regex
# parser at the folder cannot sweep a rest up as if it were a repetition.
NAME_TEMPLATES = {
    PROMPT: "C_{motion}_R_{rep}_{modality}.csv",
    REST: "rest_{motion}_{rep}_{modality}.csv",
    PAUSE: "pause_{index}_{modality}.csv",
}

# Passes of the trimmed fit that recovers the sample clock, the fraction of
# batches each pass keeps, and the fewest it will fit to. See
# :func:`fit_production_clock`.
LOW_LAG_FIT_ROUNDS = 5
LOW_LAG_FIT_KEEP = 0.30
MIN_FIT_POINTS = 10


def fit_production_clock(arrivals):
    """Recover the device's sample clock from the times batches were read.

    Every row of a session file carries the wall-clock time the logger read the
    batch it arrived in, so a whole burst shares one timestamp and sits later
    than any of its samples were actually produced. Only the *last* sample of a
    batch is informative: it was produced just before that read, so its lag is
    the link's delivery latency and nothing else. The samples ahead of it in the
    same batch are older by an amount that says more about how long the link
    stalled than about the clock.

    Those per-batch points are fitted by least squares for the rate, which the
    lag cannot bias as long as it is not drifting. The offset is then moved down
    to a low percentile of the residual lag: the moments the link was least
    behind are the ones that say where production actually was. Taking the
    offset from the fit itself would sit it at the *mean* lag and start every cut
    that far early, which at a one-second repetition is enough to pull the tail
    of the rest period into the window.

    Note the rate and the offset are estimated together rather than one from the
    other. A rate error of a fraction of a percent tilts the residuals far enough
    over a minute that any envelope taken against a separately measured rate
    describes the end of the session instead of the middle.

    What cannot be recovered is a floor: if the link is never less than some
    latency behind, that much lag is indistinguishable from the clock's origin,
    and every stage is cut that late. It is the same lateness for all of them, so
    it shifts the set rather than scattering it.

    Parameters
    ----------
    arrivals: array_like
        The first column of a session file: one wall-clock read time per sample,
        repeated across the samples that arrived in the same batch.

    Returns
    ----------
    (fs, anchor, lag_spread): tuple of float, or None
        Sample ``i`` was produced at ``anchor + i / fs``. ``lag_spread`` is the
        interquartile range of the delivery latency, in seconds: how much of a
        stage's boundary would have moved had the stream been cut on arrival time
        instead. None when the recording is too short to measure a rate from.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    n = len(arrivals)
    if n < 2:
        return None
    # One point per batch: the index of its last sample, and the time it was
    # read. A batch is a run of equal timestamps -- one read of shared memory.
    ends = np.append(np.flatnonzero(np.diff(arrivals)), n - 1)
    if len(ends) < 3 or arrivals[ends[-1]] <= arrivals[ends[0]]:
        return None
    index, arrival = ends.astype(float), arrivals[ends]
    # Fit to the least delayed batches, re-chosen from all of them each round
    # against the current line. A plain fit through every batch follows the
    # middle of the delay distribution, and that distribution is one-sided and
    # arrives in runs -- a stall delays a stretch of batches together, not one of
    # them -- so it tilts the line. Re-fitting to the low side tracks the
    # deliveries that were barely delayed at all, which are the ones that say
    # where the clock is. Against simulated links this holds the rate to about a
    # tenth of a percent in the worst case, where the plain fit drifts three
    # times as far.
    selected = np.ones(len(index), bool)
    seconds_per_sample = offset = None
    for _ in range(LOW_LAG_FIT_ROUNDS):
        seconds_per_sample, offset = np.polyfit(index[selected], arrival[selected], 1)
        lag = arrival - (offset + index * seconds_per_sample)
        low = lag <= np.quantile(lag, LOW_LAG_FIT_KEEP)
        if low.sum() < MIN_FIT_POINTS:
            break
        selected = low
    if not seconds_per_sample or seconds_per_sample <= 0:
        return None
    fs = 1.0 / seconds_per_sample
    lag = arrival - (offset + index * seconds_per_sample)
    # Not the minimum: one anomalously early read would take the anchor with it,
    # and the whole session's cuts with that.
    anchor = float(offset + np.percentile(lag, 5))
    spread = float(np.percentile(lag, 95) - np.percentile(lag, 5))
    return fs, anchor, spread


@dataclass
class ModalityClock:
    """What one modality's session file turned out to hold.

    ``fs`` and ``anchor`` are the fit :func:`fit_production_clock` recovered:
    sample ``i`` of ``_session_<modality>.csv`` was produced at
    ``anchor + i / fs``. Storing them is what makes the annotation file
    self-contained -- a later cut reads indices rather than repeating the fit.
    """

    fs: float
    anchor: float
    lag_spread: float = 0.0
    n_samples: int = 0

    def index_at(self, moment):
        """The sample index produced at a wall-clock ``moment``, clamped to the file."""
        return int(min(max(0, round((moment - self.anchor) * self.fs)), self.n_samples))

    def to_dict(self):
        return {"fs": self.fs, "anchor": self.anchor,
                "lag_spread": self.lag_spread, "n_samples": self.n_samples}

    @classmethod
    def from_dict(cls, raw):
        return cls(fs=float(raw["fs"]), anchor=float(raw["anchor"]),
                   lag_spread=float(raw.get("lag_spread", 0.0)),
                   n_samples=int(raw.get("n_samples", 0)))


@dataclass
class Stage:
    """One stage of a collection, and where it sits in the recording.

    The wall-clock ``t_start``/``t_end`` are what the stage *was*; ``indices``
    is where that lands in each modality's session file, filled in once the
    sample clock has been measured. Both are kept: the times are what the
    collection observed and can never be recomputed, the indices are what a cut
    actually uses and depend on a fit that may be worth redoing.
    """

    kind: str
    label: str = ""
    motion: int | None = None
    rep: int | None = None
    t_start: float = 0.0
    t_end: float = 0.0
    # What the stage was asked to last, which is not what it lasted. The render
    # loop only checks its clock between frames, and a frame costs a decode and a
    # texture upload, so a prompt always overruns and never by the same amount
    # twice. Cutting to the asked duration from the onset keeps every repetition
    # the same length; cutting to t_end would write that overrun into the data.
    duration: float = 0.0
    # Which take of this (motion, rep) this is, and whether a later take replaced
    # it. A redone repetition does not remove the first attempt from the
    # recording -- it is still in there, and something reading the session
    # continuously will run straight through it -- so it is described rather than
    # quietly dropped.
    attempt: int = 0
    superseded: bool = False
    # What the collection was varying when this stage ran -- limb position,
    # contraction intensity, whatever the protocol declared. Recorded per stage
    # because a factor that is not written down cannot be controlled for, and
    # these are the ones a model is most often blamed for not generalising over.
    factors: dict = field(default_factory=dict)
    indices: dict = field(default_factory=dict)

    @property
    def observed_duration(self):
        return max(0.0, self.t_end - self.t_start)

    def span(self, modality):
        """``(start, end)`` for this stage in ``modality``, or None if unresolved.

        ``end`` is the asked duration from the onset, so every repetition of the
        same prompt is the same length. Where the stage actually stopped is
        ``indices[modality]['end_observed']``.
        """
        located = self.indices.get(modality)
        if not located:
            return None
        return int(located["start"]), int(located["end"])

    def to_dict(self):
        return {"kind": self.kind, "label": self.label, "motion": self.motion,
                "rep": self.rep, "t_start": self.t_start, "t_end": self.t_end,
                "duration": self.duration, "attempt": self.attempt,
                "superseded": self.superseded, "factors": self.factors,
                "indices": self.indices}

    @classmethod
    def from_dict(cls, raw):
        return cls(kind=raw["kind"], label=raw.get("label", ""),
                   motion=raw.get("motion"), rep=raw.get("rep"),
                   t_start=float(raw.get("t_start", 0.0)),
                   t_end=float(raw.get("t_end", 0.0)),
                   duration=float(raw.get("duration", 0.0)),
                   attempt=int(raw.get("attempt", 0)),
                   superseded=bool(raw.get("superseded", False)),
                   factors=dict(raw.get("factors", {})),
                   indices={mod: dict(where) for mod, where in raw.get("indices", {}).items()})


@dataclass
class Span:
    """A stretch of one modality's session file, and what was happening during it."""

    kind: str
    label: str
    motion: int | None
    rep: int | None
    start: int
    end: int
    stage: int | None = None

    def __len__(self):
        return max(0, self.end - self.start)


@dataclass
class CutFile:
    """One file :meth:`SessionAnnotations.cut` wrote."""

    path: Path
    modality: str
    stage: Stage
    got: int
    wanted: int

    @property
    def short(self):
        return self.got < self.wanted


class SessionAnnotations:
    """Where every stage of a continuous collection sits in its recording.

    Built by the collection panel as the collection runs and saved after each
    stage, so a session interrupted halfway still says what it managed to
    collect. Loaded afterwards by anything that wants to cut the recording, read
    it whole, or ask what part of it was what.

    Parameters
    ----------
    folder: str or Path
        Where the session files and this annotation file live.
    prefix: str
        File prefix of the recording, ``_session_`` unless it was collected
        under another name.
    modalities: list
        The modalities the collection recorded.
    class_map: dict
        Motion index (as a string, since it survives a JSON round trip) to the
        class name it was collected under. Carried here so a later cut does not
        need ``collection_details.json`` beside it.
    timestamps: bool
        Whether the collection asked for a timestamp column on the files it cut.
        Used as the default for :meth:`cut`.
    """

    def __init__(self, folder='.', prefix=SESSION_PREFIX, modalities=(),
                 class_map=None, timestamps=False, created='', stages=None,
                 clocks=None):
        self.folder = Path(folder)
        self.prefix = prefix
        self.modalities = list(modalities)
        self.class_map = dict(class_map or {})
        self.timestamps = bool(timestamps)
        self.created = created
        self.stages = list(stages or [])
        self.clocks = dict(clocks or {})

    # -- where things are -------------------------------------------------

    @property
    def path(self):
        """The annotation file itself."""
        return self.folder / ANNOTATIONS_NAME

    def source_path(self, modality):
        """The uncut recording for one modality."""
        return self.folder / f"{self.prefix}{modality}.csv"

    # -- persistence ------------------------------------------------------

    def to_dict(self):
        return {
            "schema": SCHEMA_VERSION,
            "prefix": self.prefix,
            "created": self.created,
            "timestamps": self.timestamps,
            "modalities": list(self.modalities),
            "class_map": self.class_map,
            "files": {mod: f"{self.prefix}{mod}.csv" for mod in self.modalities},
            "clocks": {mod: clock.to_dict() for mod, clock in self.clocks.items()},
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def save(self, path=None):
        """Write the annotation file. Called after every stage, not just at the end."""
        path = Path(path) if path is not None else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and moved into place, so a collection that dies partway
        # through the write leaves the previous stage's annotations intact rather
        # than a truncated file that neither this nor anything else can read.
        scratch = path.with_suffix(path.suffix + ".part")
        with open(scratch, 'w') as handle:
            json.dump(self.to_dict(), handle, indent=2)
        scratch.replace(path)
        return path

    @classmethod
    def load(cls, path):
        """Load an annotation file. ``path`` may be the file or the folder holding it."""
        path = Path(path)
        if path.is_dir():
            path = path / ANNOTATIONS_NAME
        with open(path) as handle:
            raw = json.load(handle)
        schema = int(raw.get("schema", SCHEMA_VERSION))
        if schema > SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written by a newer LibEMG (schema {schema}, this "
                f"understands {SCHEMA_VERSION}).")
        return cls(
            folder=path.parent,
            prefix=raw.get("prefix", SESSION_PREFIX),
            modalities=raw.get("modalities", []),
            class_map=raw.get("class_map", {}),
            timestamps=raw.get("timestamps", False),
            created=raw.get("created", ''),
            stages=[Stage.from_dict(s) for s in raw.get("stages", [])],
            clocks={mod: ModalityClock.from_dict(c)
                    for mod, c in raw.get("clocks", {}).items()},
        )

    # -- reading the recording --------------------------------------------

    def read(self, modality):
        """The whole recording for one modality as ``(arrivals, samples)``.

        ``arrivals`` is the time each sample's *batch* was read, which is not
        when the sample was produced -- see :func:`fit_production_clock`. Use
        :meth:`produced_times` when a per-sample clock is what is wanted.
        """
        source = self.source_path(modality)
        if not source.exists():
            raise FileNotFoundError(f"No {modality} recording at {source}.")
        # A collection that was stopped before its first batch landed leaves the
        # file empty, which loadtxt warns about rather than returning nothing for.
        if source.stat().st_size == 0:
            return np.empty(0), np.empty((0, 0))
        data = np.loadtxt(source, delimiter=",", ndmin=2)
        if data.size == 0:
            return np.empty(0), np.empty((0, 0))
        return data[:, 0], data[:, 1:]

    def produced_times(self, modality, start=0, stop=None):
        """When samples ``start:stop`` of ``modality`` were produced, in wall-clock seconds."""
        clock = self.clocks.get(modality)
        if clock is None:
            raise KeyError(f"No measured clock for {modality}; call resolve() first.")
        stop = clock.n_samples if stop is None else stop
        return clock.anchor + np.arange(start, stop) / clock.fs

    # -- turning times into indices ---------------------------------------

    def measure_clocks(self, force=False):
        """Fit each modality's sample clock from its recording.

        Skips a modality that already has a clock unless ``force``, so loading an
        annotation file and cutting from it does not re-read every session file.

        Returns
        ----------
        unmeasured: list
            Modalities whose recording is missing or too short to fit a rate to.
            Their stages keep their wall-clock times and get no indices.
        """
        unmeasured = []
        for mod in self.modalities:
            if mod in self.clocks and not force:
                continue
            source = self.source_path(mod)
            if not source.exists():
                unmeasured.append(mod)
                continue
            arrivals, samples = self.read(mod)
            fit = fit_production_clock(arrivals)
            if fit is None:
                unmeasured.append(mod)
                continue
            fs, anchor, spread = fit
            self.clocks[mod] = ModalityClock(fs=fs, anchor=anchor, lag_spread=spread,
                                             n_samples=len(samples))
        return unmeasured

    def assign_indices(self):
        """Fill in every stage's sample indices from the measured clocks."""
        for stage in self.stages:
            for mod, clock in self.clocks.items():
                start = clock.index_at(stage.t_start)
                # The asked duration from the onset, so repetitions of the same
                # prompt come out the same length whatever the render loop did.
                end = int(min(start + round(stage.duration * clock.fs), clock.n_samples))
                stage.indices[mod] = {
                    "start": start,
                    "end": end,
                    # Where the stage actually stopped, which is a frame or so
                    # later. Kept so the stages can be laid end to end over the
                    # whole recording with nothing falling between them.
                    "end_observed": clock.index_at(stage.t_end),
                }
        return self

    def resolve(self, force=False):
        """Measure what needs measuring and index every stage. Returns unmeasured modalities."""
        unmeasured = self.measure_clocks(force=force)
        self.assign_indices()
        return unmeasured

    # -- querying ---------------------------------------------------------

    def select(self, kinds=(PROMPT,), include_superseded=False):
        """The stages of the given kinds, in the order they were collected."""
        kinds = tuple(kinds)
        return [s for s in self.stages
                if s.kind in kinds and (include_superseded or not s.superseded)]

    def timeline(self, modality):
        """Every sample of one modality, labelled, with nothing falling between stages.

        Each stage runs to the start of the next one rather than to its own end,
        so the spans tile ``0 .. n_samples`` without a gap: the frame or two a
        prompt overran by belongs to that prompt, not to nowhere. Anything before
        the first stage or after the last is returned as an unlabelled ``pause``,
        since that is what the collection was doing.

        This is the view for reading a session continuously -- through a rest and
        into the next contraction, say -- as opposed to :meth:`select`, which is
        the view for cutting repetitions out of it.
        """
        clock = self.clocks.get(modality)
        if clock is None:
            raise KeyError(f"No measured clock for {modality}; call resolve() first.")
        located = [(i, s) for i, s in enumerate(self.stages) if modality in s.indices]
        spans = []
        for position, (index, stage) in enumerate(located):
            start = int(stage.indices[modality]["start"])
            if position + 1 < len(located):
                end = int(located[position + 1][1].indices[modality]["start"])
            else:
                end = clock.n_samples
            spans.append(Span(kind=stage.kind, label=stage.label, motion=stage.motion,
                              rep=stage.rep, start=start, end=max(start, end),
                              stage=index))
        if not spans:
            return [Span(kind=PAUSE, label="", motion=None, rep=None,
                         start=0, end=clock.n_samples)]
        if spans[0].start > 0:
            spans.insert(0, Span(kind=PAUSE, label="", motion=None, rep=None,
                                 start=0, end=spans[0].start))
        return spans

    def describe(self):
        """A human-readable summary of what the session holds."""
        lines = [f"{self.path}",
                 f"  collected {self.created or 'at an unrecorded time'}"]
        for mod in self.modalities:
            clock = self.clocks.get(mod)
            if clock is None:
                lines.append(f"  {mod}: no measured clock")
                continue
            lines.append(f"  {mod}: {clock.n_samples} samples at a measured "
                         f"{clock.fs:.1f} Hz ({clock.n_samples / clock.fs:.1f} s), "
                         f"delivery latency varied by {1e3 * clock.lag_spread:.0f} ms")
        counts = {}
        for stage in self.stages:
            counts[stage.kind] = counts.get(stage.kind, 0) + 1
        tally = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
        lines.append(f"  stages: {tally or 'none'}")
        superseded = [s for s in self.stages if s.superseded]
        if superseded:
            lines.append(f"  {len(superseded)} stage(s) were redone and are still in the "
                         f"recording; they are not cut unless asked for.")
        for stage in self.stages:
            where = ", ".join(f"{mod} {w['start']}:{w['end']}"
                              for mod, w in sorted(stage.indices.items()))
            name = stage.label or "-"
            tag = " (superseded)" if stage.superseded else ""
            lines.append(f"    {stage.kind:<6} {name:<20} rep {stage.rep}{tag} "
                         f"{where or 'unindexed'}")
        return "\n".join(lines)

    # -- cutting ----------------------------------------------------------

    def cut(self, output_folder=None, kinds=(PROMPT,), timestamps=None,
            include_superseded=False, lead_in=0.0, lead_out=0.0, modalities=None):
        """Write each selected stage out as its own file, by slicing the recording.

        This is the only thing that turns a session into the per-repetition
        layout an ``OfflineDataHandler`` reads, and it is a pure function of the
        recording and the indices in this file -- so it can be run again, with
        different arguments, long after the collection.

        Parameters
        ----------
        output_folder: str or Path, optional
            Where to write. Defaults to the folder the session is in, which is
            where a collection puts its repetitions.
        kinds: tuple
            Which stages to write. Only ``'prompt'`` by default; a rest or a
            pause is written under a name that carries no ``C_``/``R_`` tokens,
            so adding them cannot confuse a regex parser pointed at the folder.
        timestamps: bool, optional
            Prepend the time each sample was *produced*. Defaults to what the
            collection was configured with. Note this is the fitted clock, not
            the arrival times in the session file -- a per-batch arrival time
            repeated down a column is not a clock.
        include_superseded: bool
            Write the earlier takes of a redone repetition too. They would
            otherwise overwrite each other, since a redo is collected under the
            same name.
        lead_in, lead_out: float
            Seconds to extend each stage by at its start and end. The samples are
            there -- the recording was never cut -- so a window that needs
            context either side of a contraction can simply ask for it.
        modalities: list, optional
            Restrict to some modalities. Defaults to all of them.

        Returns
        ----------
        written: list of CutFile
            One per file written. ``CutFile.short`` marks a stage that ran past
            the end of the recording.
        """
        output_folder = Path(output_folder) if output_folder is not None else self.folder
        output_folder.mkdir(parents=True, exist_ok=True)
        timestamps = self.timestamps if timestamps is None else timestamps
        chosen = self.select(kinds=kinds, include_superseded=include_superseded)
        # A pause has no class to be named after, so it is named by where it sits
        # in the collection -- its position among all the stages, not among the
        # ones this call happens to be writing, so the name does not move when
        # the same session is cut with different ``kinds``.
        position = {id(stage): index for index, stage in enumerate(self.stages)}
        written = []
        for mod in (modalities if modalities is not None else self.modalities):
            clock = self.clocks.get(mod)
            if clock is None:
                continue
            source = self.source_path(mod)
            if not source.exists():
                continue
            _, samples = self.read(mod)
            for stage in chosen:
                span = stage.span(mod)
                if span is None:
                    continue
                start, end = span
                start = max(0, start - int(round(lead_in * clock.fs)))
                end = min(len(samples), end + int(round(lead_out * clock.fs)))
                chunk = samples[start:end]
                name = NAME_TEMPLATES[stage.kind].format(
                    motion=stage.motion, rep=stage.rep, modality=mod,
                    index=position[id(stage)])
                if timestamps:
                    produced = clock.anchor + np.arange(start, start + len(chunk)) / clock.fs
                    chunk = np.hstack((produced[:, None], chunk))
                destination = output_folder / name
                np.savetxt(destination, chunk, delimiter=",")
                written.append(CutFile(path=destination, modality=mod, stage=stage,
                                       got=end - start,
                                       wanted=int(round((stage.duration + lead_in + lead_out)
                                                        * clock.fs))))
        return written


def cut_session(path, output_folder=None, kinds=(PROMPT,), refit=False, **kwargs):
    """Load a session's annotations and cut it, in one call.

    The clock is re-measured only for modalities that have none -- a session
    interrupted before it could finish indexing, say -- or when ``refit`` is set.
    """
    session = SessionAnnotations.load(path)
    session.resolve(force=refit)
    return session.cut(output_folder=output_folder, kinds=kinds, **kwargs)


def _cut(arguments):
    session = SessionAnnotations.load(arguments.path)
    unmeasured = session.resolve(force=arguments.refit)
    for mod in unmeasured:
        print(f"LibEMG -> session ({mod} has no recording to measure a clock from, so "
              f"none of its stages were cut.)")
    written = session.cut(output_folder=arguments.output_folder,
                          kinds=tuple(arguments.kinds),
                          timestamps=arguments.timestamps,
                          include_superseded=arguments.include_superseded,
                          lead_in=arguments.lead_in, lead_out=arguments.lead_out)
    if arguments.refit or unmeasured:
        session.save()
    short = [cut for cut in written if cut.short]
    print(f"Wrote {len(written)} files to "
          f"{arguments.output_folder or session.folder}.")
    if short:
        names = ", ".join(cut.path.name for cut in short[:5])
        print(f"{len(short)} ran past the end of the recording and are short: {names}"
              f"{' ...' if len(short) > 5 else ''}")
    return 0


def _show(arguments):
    session = SessionAnnotations.load(arguments.path)
    if arguments.refit:
        session.resolve(force=True)
    print(session.describe())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m libemg.session",
        description="Inspect and re-cut a continuous screen guided training recording.")
    commands = parser.add_subparsers(dest="command", required=True)

    show = commands.add_parser("show", help="Describe what a session holds.")
    show.add_argument("path", help="The annotation file, or the folder holding it.")
    show.add_argument("--refit", action="store_true",
                      help="Re-measure the sample clock from the recording.")

    cut = commands.add_parser("cut", help="Cut a session into per-stage files.")
    cut.add_argument("path", help="The annotation file, or the folder holding it.")
    cut.add_argument("--output-folder", default=None,
                     help="Where to write. Defaults to the session's own folder.")
    cut.add_argument("--kinds", nargs="+", default=[PROMPT],
                     choices=[PROMPT, REST, PAUSE],
                     help="Which stages to write. Repetitions only by default.")
    cut.add_argument("--timestamps", action="store_true", default=None,
                     help="Prepend the time each sample was produced.")
    cut.add_argument("--include-superseded", action="store_true",
                     help="Also write the earlier takes of redone repetitions.")
    cut.add_argument("--lead-in", type=float, default=0.0,
                     help="Seconds of recording to keep before each stage.")
    cut.add_argument("--lead-out", type=float, default=0.0,
                     help="Seconds of recording to keep after each stage.")
    cut.add_argument("--refit", action="store_true",
                     help="Re-measure the sample clock before cutting.")

    arguments = parser.parse_args(argv)
    if arguments.command == "show":
        return _show(arguments)
    return _cut(arguments)


__all__ = ["SessionAnnotations", "Stage", "Span", "ModalityClock", "CutFile",
           "fit_production_clock", "cut_session", "main",
           "SESSION_PREFIX", "ANNOTATIONS_NAME", "SCHEMA_VERSION",
           "PROMPT", "REST", "PAUSE", "NAME_TEMPLATES"]


if __name__ == "__main__":
    sys.exit(main())
