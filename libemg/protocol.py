"""Describe a data collection before running it, and build the media it needs.

A screen guided training session is an experiment, and most of what decides
whether its data is any good is fixed before anyone contracts a muscle: what is
prompted, in what order, for how long, with what rests, under what conditions,
and recorded against what acquisition settings. Until now those decisions lived
in whichever script happened to launch the GUI, in the alphabetical order of a
media folder, and in a lab notebook. This module makes them a file.

A protocol is that file. It says what to collect, generates the media to prompt
it with, produces the order the prompts run in, and is copied into the data
folder so the recording carries its own description.

Why each part is here
---------------------
**Order.** Collecting class 1 first and class 8 last in every repetition
confounds class with everything that drifts over a session: fatigue, electrode
gel drying, skin impedance, the subject settling into the task. Shuffling the
prompts within each repetition breaks that confound, and recording the seed
keeps the session reproducible.

**Jittered rest.** A rest of exactly three seconds teaches the subject when the
prompt is coming, and they pre-empt it. The contraction onset then sits before
the cue rather than after it, and the first part of every repetition is not what
it is labelled. A rest of three seconds plus a random fraction cannot be
anticipated.

**Design factors.** Classifier accuracy falls sharply when the limb is in a
position, or the contraction at an intensity, that was not in the training set --
the position and intensity effects are two of the most reported failure modes in
myoelectric control. Collecting across levels of those factors is the mitigation,
and it is only useful if which level each repetition was collected under is
written down. Levels vary by repetition, because a subject cannot change arm
position halfway through one.

**Ramp cueing.** A static image says what to do but not how hard or how fast,
so the contraction profile is whatever the subject felt like. An animated prompt
that ramps to a level and holds paces them, and the animation's own coordinates
are a per-frame record of what was asked for. LibEMG's own guidance already
recommends ramp contractions for velocity control; this generates them.

**Acquisition metadata.** The ISEK standards for reporting EMG ask for electrode
type, size, inter-electrode distance, placement, skin preparation, amplifier
bandwidth, sampling rate and resolution. None of it can be recovered from the
recording afterwards, and all of it decides whether the data can be compared to
anyone else's. It is a form here so that it is filled in at collection time.

Examples
---------
Build a two-DOF regression protocol and generate its animation::

    from libemg.protocol import CollectionProtocol, within_and_combined_dof

    protocol = CollectionProtocol(name='wrist-2dof', kind='regression',
                                  dof_labels=['Flexion/Extension', 'Pronation/Supination'])
    protocol.trajectory = within_and_combined_dof(hold_seconds=3, ramp_seconds=2)
    protocol.media.animator = 'scatter'
    protocol.save('wrist-2dof.yaml')
    protocol.build_media('animation/')

Or from a terminal::

    python -m libemg.protocol show wrist-2dof.yaml
    python -m libemg.protocol build wrist-2dof.yaml --media-folder animation/
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SCHEMA_VERSION = 1
PROTOCOL_NAME = "collection_protocol.yaml"

#: What a protocol collects. A classification protocol prompts one gesture at a
#: time and labels whole repetitions; a regression protocol prompts a moving
#: target and labels every frame of it.
CLASSIFICATION = "classification"
REGRESSION = "regression"

#: The animators a protocol can be rendered with, by the name used in the file.
#: Keyed here rather than imported at module load because importing the animator
#: pulls in matplotlib, which a protocol that is only being read does not need.
ANIMATORS = {
    "scatter": "ScatterPlotAnimator",
    "arrow": "ArrowPlotAnimator",
    "target": "TargetPlotAnimator",
    "bar": "BarPlotAnimator",
    "single_bar": "SingleDirectionBarPlotAnimator",
}

#: Trajectory primitives. Between them these cover what the myoelectric
#: regression literature prompts with: ramps to a level, steady holds, smooth
#: oscillation, and randomly placed targets for a training set that is not a
#: grid.
SEGMENT_KINDS = ("rest", "ramp", "hold", "sinusoid", "random")

#: Ordering strategies for the prompts within a repetition. See the module
#: docstring on why the default is not ``fixed``.
FIXED, RANDOM, COUNTERBALANCED = "fixed", "random", "counterbalanced"
ORDERINGS = (FIXED, RANDOM, COUNTERBALANCED)


# ======================================================================
# trajectories
# ======================================================================

@dataclass
class Segment:
    """One piece of a target trajectory.

    A trajectory is a list of these, rendered end to end. ``target`` is in
    normalized activation per degree of freedom, where 1 is the full range the
    animation displays -- not a force, since nothing here can know what the
    subject's maximum is.
    """

    kind: str
    seconds: float = 1.0
    target: list = field(default_factory=list)
    # Sinusoid only: how long one cycle takes and how far it swings.
    period: float = 2.0
    amplitude: float = 1.0
    # Random only: how many targets to visit in ``seconds``, and what fraction
    # of the time to each is spent arriving rather than holding.
    count: int = 4
    ramp_fraction: float = 0.5

    def to_dict(self):
        kept = {"kind": self.kind, "seconds": self.seconds}
        if self.kind in ("ramp", "hold", "sinusoid"):
            kept["target"] = list(self.target)
        if self.kind == "sinusoid":
            kept.update(period=self.period, amplitude=self.amplitude)
        if self.kind == "random":
            kept.update(count=self.count, ramp_fraction=self.ramp_fraction)
        return kept

    @classmethod
    def from_dict(cls, raw):
        return cls(kind=raw["kind"], seconds=float(raw.get("seconds", 1.0)),
                   target=list(raw.get("target", [])),
                   period=float(raw.get("period", 2.0)),
                   amplitude=float(raw.get("amplitude", 1.0)),
                   count=int(raw.get("count", 4)),
                   ramp_fraction=float(raw.get("ramp_fraction", 0.5)))


def render_trajectory(segments, n_dof, fps, seed=None):
    """Turn a list of :class:`Segment` into the coordinates an animator draws.

    Every segment starts where the previous one left off, so the trajectory is
    continuous: there is no instantaneous jump for the subject to fail to
    follow, unless a ``hold`` at a new target asks for one deliberately.

    Parameters
    ----------
    segments: list of Segment
    n_dof: int
        How many degrees of freedom the animation shows.
    fps: int
        Frames per second of the animation the coordinates are for.
    seed: int or None
        Seeds the ``random`` segments, so the same protocol renders the same
        animation every time it is built.

    Returns
    ----------
    coordinates: numpy.ndarray
        ``N x n_dof``, one row per frame.
    """
    rng = np.random.default_rng(seed)
    position = np.zeros(n_dof)
    pieces = []
    for segment in segments:
        frames = max(1, int(round(segment.seconds * fps)))
        target = _as_vector(segment.target, n_dof)
        if segment.kind == "rest":
            piece = np.zeros((frames, n_dof))
            # Arrive at rest rather than snapping to it, so a rest after a
            # contraction is a release the subject can follow.
            arrive = min(frames, max(1, int(round(0.25 * fps))))
            piece[:arrive] = np.linspace(position, np.zeros(n_dof), arrive)
        elif segment.kind == "ramp":
            piece = np.linspace(position, target, frames)
        elif segment.kind == "hold":
            piece = np.tile(target, (frames, 1))
        elif segment.kind == "sinusoid":
            # Starts and ends at the current position so it can be dropped
            # between other segments without a discontinuity.
            phase = 2 * np.pi * np.arange(frames) / max(1e-9, segment.period * fps)
            piece = position + segment.amplitude * np.sin(phase)[:, None] * target
        elif segment.kind == "random":
            piece = _random_targets(position, target, frames, segment, rng, n_dof)
        else:
            raise ValueError(f"Unknown segment kind {segment.kind!r}; "
                             f"expected one of {', '.join(SEGMENT_KINDS)}.")
        pieces.append(piece)
        position = piece[-1].copy()
    if not pieces:
        return np.zeros((0, n_dof))
    return np.vstack(pieces)


def _as_vector(target, n_dof):
    """A target as an n_dof vector, padded with zeros and clipped to the display."""
    vector = np.zeros(n_dof)
    for index, value in enumerate(list(target)[:n_dof]):
        vector[index] = float(value)
    return np.clip(vector, -1.0, 1.0)


def _random_targets(position, extent, frames, segment, rng, n_dof):
    """Ramp to, and hold at, randomly placed targets.

    Drawn inside the unit ball rather than on a grid: a training set that only
    ever visits the axes and the corners teaches a regressor nothing about what
    lies between them, which is most of where a user actually operates.
    """
    scale = np.where(np.abs(extent) > 0, np.abs(extent), 1.0)
    per_target = max(1, frames // max(1, segment.count))
    pieces = []
    for _ in range(max(1, segment.count)):
        point = rng.normal(size=n_dof)
        point *= rng.uniform(0.2, 1.0) / max(1e-9, np.linalg.norm(point))
        point = np.clip(point * scale, -1.0, 1.0)
        arriving = max(1, int(round(per_target * segment.ramp_fraction)))
        holding = max(1, per_target - arriving)
        pieces.append(np.linspace(position, point, arriving))
        pieces.append(np.tile(point, (holding, 1)))
        position = point
    return np.vstack(pieces)[:frames]


def within_and_combined_dof(n_dof=2, hold_seconds=3.0, ramp_seconds=2.0,
                            rest_seconds=3.0, include_combined=True):
    """The standard regression prompt: each DOF alone, then pairs together.

    Training only on one DOF at a time leaves a regressor with no example of
    what two simultaneous intents look like, and it is the simultaneous case
    that proportional control exists for. So each DOF is ramped to both extremes
    on its own, and then the pairs are driven together.

    Returns
    ----------
    segments: list of Segment
    """
    segments = [Segment(kind="rest", seconds=rest_seconds)]
    corners = []
    for dof in range(n_dof):
        for sign in (1.0, -1.0):
            target = [0.0] * n_dof
            target[dof] = sign
            corners.append(target)
    if include_combined and n_dof >= 2:
        for first in range(n_dof):
            for second in range(first + 1, n_dof):
                for sign_a in (1.0, -1.0):
                    for sign_b in (1.0, -1.0):
                        target = [0.0] * n_dof
                        target[first], target[second] = sign_a, sign_b
                        corners.append(target)
    for target in corners:
        segments.append(Segment(kind="ramp", seconds=ramp_seconds, target=target))
        segments.append(Segment(kind="hold", seconds=hold_seconds, target=target))
        segments.append(Segment(kind="rest", seconds=rest_seconds))
    return segments


def ramp_and_hold(n_dof=1, levels=(0.25, 0.5, 0.75, 1.0), hold_seconds=3.0,
                  ramp_seconds=2.0, rest_seconds=3.0, dof=0):
    """Ramp to a series of levels along one DOF, holding at each.

    This is the classification-side prompt worth generating: it paces the
    contraction instead of leaving its speed and depth to the subject, and it
    visits several intensities, which is what keeps a classifier from only
    working at the one effort level it was trained at.
    """
    segments = [Segment(kind="rest", seconds=rest_seconds)]
    for level in levels:
        target = [0.0] * n_dof
        target[dof] = float(level)
        segments.append(Segment(kind="ramp", seconds=ramp_seconds, target=target))
        segments.append(Segment(kind="hold", seconds=hold_seconds, target=target))
        segments.append(Segment(kind="rest", seconds=rest_seconds))
    return segments


def random_exploration(n_dof=2, seconds=60.0, count=20, rest_seconds=3.0):
    """Visit randomly placed targets for a fixed time.

    A training set drawn from the whole space rather than from the corners of
    it, which is what a regressor asked to operate between the axes needs to
    have seen.
    """
    return [Segment(kind="rest", seconds=rest_seconds),
            Segment(kind="random", seconds=seconds, count=count,
                    target=[1.0] * n_dof),
            Segment(kind="rest", seconds=rest_seconds)]


TRAJECTORY_TEMPLATES = {
    "Within and combined DOF": within_and_combined_dof,
    "Ramp and hold (intensity levels)": ramp_and_hold,
    "Random exploration": random_exploration,
}


# ======================================================================
# the protocol document
# ======================================================================

@dataclass
class MediaSpec:
    """How the prompts are drawn, if the protocol generates them."""

    animator: str = "scatter"
    fps: int = 24
    tpd: float = 2.0
    show_direction: bool = True
    show_countdown: bool = True
    show_boundary: bool = True
    normalize_distance: bool = False
    figsize: tuple = (6, 6)
    dpi: int = 80
    video_format: str = ".mp4"
    #: Compass point to image path, drawn around a cartesian animation so the
    #: subject sees what each direction means rather than reading axis labels.
    axis_images: dict = field(default_factory=dict)

    def to_dict(self):
        return {"animator": self.animator, "fps": self.fps, "tpd": self.tpd,
                "show_direction": self.show_direction,
                "show_countdown": self.show_countdown,
                "show_boundary": self.show_boundary,
                "normalize_distance": self.normalize_distance,
                "figsize": list(self.figsize), "dpi": self.dpi,
                "video_format": self.video_format,
                "axis_images": dict(self.axis_images)}

    @classmethod
    def from_dict(cls, raw):
        raw = raw or {}
        return cls(animator=raw.get("animator", "scatter"),
                   fps=int(raw.get("fps", 24)), tpd=float(raw.get("tpd", 2.0)),
                   show_direction=bool(raw.get("show_direction", True)),
                   show_countdown=bool(raw.get("show_countdown", True)),
                   show_boundary=bool(raw.get("show_boundary", True)),
                   normalize_distance=bool(raw.get("normalize_distance", False)),
                   figsize=tuple(raw.get("figsize", (6, 6))),
                   dpi=int(raw.get("dpi", 80)),
                   video_format=raw.get("video_format", ".mp4"),
                   axis_images=dict(raw.get("axis_images", {})))


@dataclass
class Timing:
    """How long each part of the collection lasts.

    ``rest_jitter`` is the one that needs explaining: it is added to
    ``rest_time`` as a uniform random amount, so the subject cannot learn when
    the prompt is coming and start the contraction before the cue.
    """

    num_reps: int = 3
    rep_time: float = 5.0
    rest_time: float = 3.0
    rest_jitter: float = 1.0
    #: How long after the cue a contraction can be assumed to have started.
    #: Recorded rather than applied: the recording is continuous and the
    #: annotations mark the cue, so trimming is a decision for analysis, not one
    #: to bake into the files.
    reaction_time: float = 0.2

    def to_dict(self):
        return {"num_reps": self.num_reps, "rep_time": self.rep_time,
                "rest_time": self.rest_time, "rest_jitter": self.rest_jitter,
                "reaction_time": self.reaction_time}

    @classmethod
    def from_dict(cls, raw):
        raw = raw or {}
        return cls(num_reps=int(raw.get("num_reps", 3)),
                   rep_time=float(raw.get("rep_time", 5.0)),
                   rest_time=float(raw.get("rest_time", 3.0)),
                   rest_jitter=float(raw.get("rest_jitter", 1.0)),
                   reaction_time=float(raw.get("reaction_time", 0.2)))


#: The acquisition settings the ISEK standards for reporting EMG data ask for.
#: Held as a list of (field, prompt) so the form, the file and the documentation
#: cannot disagree about what is being asked.
ACQUISITION_FIELDS = (
    ("device", "Device / amplifier"),
    ("electrode_type", "Electrode type (e.g. Ag/AgCl gelled, dry bar)"),
    ("electrode_size_mm", "Electrode size (mm)"),
    ("inter_electrode_distance_mm", "Inter-electrode distance (mm)"),
    ("electrode_placement", "Placement (muscle, landmark, fibre direction)"),
    ("num_channels", "Number of channels"),
    ("skin_preparation", "Skin preparation"),
    ("ground_reference", "Ground / reference location"),
    ("sampling_rate_hz", "Sampling rate (Hz)"),
    ("bandwidth_hz", "Amplifier bandwidth (Hz)"),
    ("notch_hz", "Notch filter (Hz, blank if none)"),
    ("gain", "Gain"),
    ("adc_bits", "A/D resolution (bits)"),
)


@dataclass
class Prompt:
    """One thing to prompt, once: which class, which repetition, under what."""

    motion: int
    rep: int
    name: str
    label: str
    rep_time: float
    rest_time: float
    factors: dict = field(default_factory=dict)


class CollectionProtocol:
    """What to collect, how to prompt it, in what order, and under what settings.

    Parameters
    ----------
    name: str
        What this protocol is called. Used for the media filename.
    kind: str
        ``'classification'`` or ``'regression'``.
    dof_labels: list
        One label per degree of freedom (regression) or per bar (bar plots).
    gestures: list
        Classification only: the class names, in the order they index from. Left
        empty, the collection falls back to whatever is in the media folder.
    """

    def __init__(self, name="collection", kind=CLASSIFICATION, dof_labels=None,
                 gestures=None, trajectory=None, media=None, timing=None,
                 order=RANDOM, seed=None, factors=None, acquisition=None,
                 notes=""):
        self.name = name
        self.kind = kind
        self.dof_labels = list(dof_labels or [])
        self.gestures = list(gestures or [])
        self.trajectory = list(trajectory or [])
        self.media = media or MediaSpec()
        self.timing = timing or Timing()
        self.order = order
        # Kept rather than generated per run: the ordering is only reproducible
        # if the seed that produced it is written down with the data.
        self.seed = random.randrange(1 << 30) if seed is None else int(seed)
        self.factors = dict(factors or {})
        self.acquisition = dict(acquisition or {})
        self.notes = notes

    # -- persistence ------------------------------------------------------

    def to_dict(self):
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "kind": self.kind,
            "notes": self.notes,
            "dof_labels": list(self.dof_labels),
            "gestures": list(self.gestures),
            "media": self.media.to_dict(),
            "trajectory": [segment.to_dict() for segment in self.trajectory],
            "timing": self.timing.to_dict(),
            "order": {"mode": self.order, "seed": self.seed},
            "factors": {name: list(levels) for name, levels in self.factors.items()},
            "acquisition": {name: self.acquisition.get(name, "")
                            for name, _ in ACQUISITION_FIELDS},
        }

    def save(self, path):
        """Write the protocol. ``path`` may be the file or the folder to put it in."""
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path = path / PROTOCOL_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False,
                           default_flow_style=False)
        return path

    @classmethod
    def load(cls, path):
        """Load a protocol from a file, or from the folder holding one."""
        path = Path(path)
        if path.is_dir():
            path = path / PROTOCOL_NAME
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        schema = int(raw.get("schema", SCHEMA_VERSION))
        if schema > SCHEMA_VERSION:
            raise ValueError(f"{path} was written by a newer LibEMG "
                             f"(schema {schema}, this understands {SCHEMA_VERSION}).")
        order = raw.get("order", {}) or {}
        return cls(name=raw.get("name", "collection"),
                   kind=raw.get("kind", CLASSIFICATION),
                   dof_labels=raw.get("dof_labels", []),
                   gestures=raw.get("gestures", []),
                   trajectory=[Segment.from_dict(s) for s in raw.get("trajectory", [])],
                   media=MediaSpec.from_dict(raw.get("media")),
                   timing=Timing.from_dict(raw.get("timing")),
                   order=order.get("mode", RANDOM), seed=order.get("seed"),
                   factors=raw.get("factors", {}),
                   acquisition=raw.get("acquisition", {}),
                   notes=raw.get("notes", ""))

    # -- what it prompts --------------------------------------------------

    @property
    def n_dof(self):
        return max(1, len(self.dof_labels))

    def coordinates(self):
        """The target trajectory, one row per frame of the animation."""
        return render_trajectory(self.trajectory, self.n_dof, self.media.fps,
                                 seed=self.seed)

    def duration(self):
        """How long the generated animation runs, in seconds."""
        return len(self.coordinates()) / max(1, self.media.fps)

    def build_media(self, folder, verbose=False, progress=None):
        """Render the protocol's prompts into ``folder`` and return what it wrote.

        The animation's own coordinates are written beside it as a ``.txt``,
        which is what screen guided training picks up as per-frame labels -- so
        a regression collection comes out of this with its ground truth already
        aligned to its prompts.

        Parameters
        ----------
        folder: str or Path
            The media folder a collection will read.
        verbose: bool
            Print frame progress, for a long render at a terminal.
        progress: callable or None
            Called as ``progress(message)`` as the render proceeds, for a GUI
            that wants to say what is happening.

        Returns
        ----------
        written: list of Path
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        coordinates = self.coordinates()
        if not len(coordinates):
            raise ValueError("This protocol has no trajectory to render. Add at "
                             "least one segment before building its media.")
        if progress:
            progress(f"Rendering {len(coordinates)} frames "
                     f"({len(coordinates) / self.media.fps:.0f} s) ...")
        output = folder / f"{self.name}{self.media.video_format}"
        animator = self._animator(output)
        animator.save_plot_video(coordinates, title=self.name,
                                 save_coordinates=True, verbose=verbose)
        written = [output, output.with_suffix(".txt")]
        if progress:
            progress(f"Wrote {output.name} and its labels.")
        return [path for path in written if path.exists()]

    def _animator(self, output):
        """Build the animator this protocol asks for.

        Imported here rather than at module scope: the animator pulls in
        matplotlib, and reading or editing a protocol should not have to.
        """
        from PIL import Image
        import libemg.animator as animator_module

        if self.media.animator not in ANIMATORS:
            raise ValueError(f"Unknown animator {self.media.animator!r}; "
                             f"expected one of {', '.join(sorted(ANIMATORS))}.")
        cls = getattr(animator_module, ANIMATORS[self.media.animator])
        common = dict(output_filepath=str(output), fps=self.media.fps,
                      show_direction=self.media.show_direction,
                      show_countdown=self.media.show_countdown,
                      show_boundary=self.media.show_boundary,
                      figsize=tuple(self.media.figsize), dpi=self.media.dpi,
                      tpd=self.media.tpd)
        if self.media.animator in ("bar", "single_bar"):
            return cls(bar_labels=list(self.dof_labels) or ["DOF 1"], **common)
        images = {point: Image.open(path)
                  for point, path in self.media.axis_images.items() if path}
        return cls(normalize_distance=self.media.normalize_distance,
                   axis_images=images or None, **common)

    # -- what order it runs in --------------------------------------------

    def schedule(self, names, num_reps=None):
        """The prompts to run, in order, with their timings and factor levels.

        Parameters
        ----------
        names: list
            The class names available, indexed by class number -- normally what
            the media folder turned out to hold.
        num_reps: int or None
            Overrides the protocol's own repetition count.

        Returns
        ----------
        prompts: list of Prompt
        """
        num_reps = self.timing.num_reps if num_reps is None else num_reps
        rng = random.Random(self.seed)
        prompts = []
        for rep in range(num_reps):
            levels = self.levels_for_rep(rep)
            for motion in self._order_for_rep(rep, len(names), rng):
                suffix = " - ".join(str(value) for value in levels.values())
                prompts.append(Prompt(
                    motion=motion, rep=rep, name=names[motion],
                    label=f"{names[motion]} ({suffix})" if suffix else names[motion],
                    rep_time=self.timing.rep_time,
                    rest_time=self.rest_for(rng),
                    factors=dict(levels)))
        return prompts

    def _order_for_rep(self, rep, n_classes, rng):
        order = list(range(n_classes))
        if self.order == RANDOM:
            rng.shuffle(order)
        elif self.order == COUNTERBALANCED:
            # Rotate by the repetition. Over a full set of repetitions every
            # class has sat in every position, which removes the order effect
            # without the run-to-run variance a shuffle leaves behind.
            offset = rep % max(1, n_classes)
            order = order[offset:] + order[:offset]
        return order

    def rest_for(self, rng):
        """A rest duration, jittered so the next prompt cannot be anticipated."""
        return self.timing.rest_time + rng.uniform(0.0, max(0.0, self.timing.rest_jitter))

    def levels_for_rep(self, rep):
        """Which level of each design factor this repetition is collected under.

        One level per factor per repetition, cycled, because the factors this is
        for -- limb position, contraction intensity -- are things a subject holds
        for a whole repetition and cannot change partway through.
        """
        return {name: levels[rep % len(levels)]
                for name, levels in self.factors.items() if levels}

    def describe(self):
        """A human-readable summary of what this protocol will do."""
        lines = [f"{self.name} ({self.kind})"]
        if self.notes:
            lines.append(f"  {self.notes}")
        if self.dof_labels:
            lines.append(f"  DOFs: {', '.join(self.dof_labels)}")
        if self.trajectory:
            lines.append(f"  trajectory: {len(self.trajectory)} segments, "
                         f"{self.duration():.0f} s at {self.media.fps} fps, "
                         f"drawn as a {self.media.animator} plot")
        lines.append(f"  {self.timing.num_reps} reps of {self.timing.rep_time:.1f} s, "
                     f"resting {self.timing.rest_time:.1f}"
                     f"-{self.timing.rest_time + self.timing.rest_jitter:.1f} s")
        lines.append(f"  order: {self.order} (seed {self.seed})")
        for name, levels in self.factors.items():
            lines.append(f"  factor {name}: {', '.join(str(l) for l in levels)}")
        missing = [label for name, label in ACQUISITION_FIELDS
                   if not str(self.acquisition.get(name, "")).strip()]
        if missing:
            lines.append(f"  acquisition details not filled in: {len(missing)} of "
                         f"{len(ACQUISITION_FIELDS)} ({missing[0]}, ...)")
        return "\n".join(lines)


# ======================================================================
# command line
# ======================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m libemg.protocol",
        description="Describe a data collection, and build the media it prompts with.")
    commands = parser.add_subparsers(dest="command", required=True)

    show = commands.add_parser("show", help="Describe a saved protocol.")
    show.add_argument("path", help="The protocol file, or the folder holding it.")

    build = commands.add_parser("build", help="Render a protocol's media.")
    build.add_argument("path", help="The protocol file, or the folder holding it.")
    build.add_argument("--media-folder", default="media/",
                       help="Where to write the animation and its labels.")
    build.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    template = commands.add_parser(
        "template", help="Write a starting protocol to edit.")
    template.add_argument("path", help="Where to write it.")
    template.add_argument("--kind", default=REGRESSION,
                          choices=[CLASSIFICATION, REGRESSION])
    template.add_argument("--dofs", nargs="+", default=["DOF 1", "DOF 2"])

    arguments = parser.parse_args(argv)
    if arguments.command == "template":
        protocol = CollectionProtocol(
            name=Path(arguments.path).stem, kind=arguments.kind,
            dof_labels=arguments.dofs,
            trajectory=within_and_combined_dof(n_dof=len(arguments.dofs)),
            factors={"limb_position": ["neutral", "elevated", "across body"]})
        print(f"Wrote {protocol.save(arguments.path)}")
        return 0
    protocol = CollectionProtocol.load(arguments.path)
    if arguments.command == "show":
        print(protocol.describe())
        return 0
    written = protocol.build_media(arguments.media_folder,
                                   verbose=not arguments.quiet,
                                   progress=None if arguments.quiet else print)
    print("Wrote " + ", ".join(str(path) for path in written))
    return 0


__all__ = ["CollectionProtocol", "Segment", "MediaSpec", "Timing", "Prompt",
           "render_trajectory", "within_and_combined_dof", "ramp_and_hold",
           "random_exploration", "TRAJECTORY_TEMPLATES", "ACQUISITION_FIELDS",
           "ANIMATORS", "SEGMENT_KINDS", "ORDERINGS", "FIXED", "RANDOM",
           "COUNTERBALANCED", "CLASSIFICATION", "REGRESSION", "PROTOCOL_NAME",
           "SCHEMA_VERSION", "main"]


if __name__ == "__main__":
    sys.exit(main())
