"""Per-sample regression targets for a choreography recording.

A recording's run record (``_session_choreography.json``) says, interval by
interval, what was shown. This module turns that into one target row per
recorded sample, so a regressor can be trained on the continuous recording:

``_choreography_targets_<modality>.csv``
    Written beside ``_session_<modality>.csv`` when a recording is saved. Row
    ``i`` describes sample ``i`` of the session file (same row count). Columns
    are named in a ``#`` header line, so ``np.loadtxt(path, delimiter=",")``
    reads it as plain numbers:

    ===================  =====================================================
    sample_index         row of ``_session_<modality>.csv``
    collection_time      when the sample was produced (session clock fit)
    motion_class_id      versioned class ID; 0 (rest) while resting
    phase_code           see :data:`PHASE_CODES`; -1 outside the run
    position             0 (rest) .. 1 (gesture): linear over the ramps
    intensity_level      the repetition's target fraction of max effort *
    intensity            position * intensity_level; 0 while resting *
    repetition_index     0-based run order of the repetition; -1 when resting
    training_eligible    1 where the interval was displayed cleanly
    ===================  =====================================================

    (* only when the run asked for intensity.)

:func:`load_targets` reads a recording and its targets back as aligned arrays;
:func:`cut_repetitions` writes one data file and one target file per
repetition (ramp-in start to ramp-out end) for file-per-repetition loaders such
as ``OfflineDataHandler`` with a ``FilePackager``.

As with every label here, the targets describe the **instruction displayed**,
not measured movement or force.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .annotations import PHASE_TRACK, REPETITION_TRACK, RUN_RECORD_NAME
from .schedule import COMPLETED, CYCLE_PAUSE, HOLD, PAUSED, RAMP_IN, RAMP_OUT, REST, WARMUP

TARGETS_TEMPLATE = "_choreography_targets_{modality}.csv"
REPETITION_DATA_TEMPLATE = "C_{motion}_R_{rep}_{modality}.csv"
REPETITION_TARGETS_TEMPLATE = "C_{motion}_R_{rep}_{modality}_targets.csv"

#: ``phase_code`` values. -1 marks samples outside every interval (before the
#: run started or after it ended).
PHASE_CODES = {WARMUP: 0, RAMP_IN: 1, HOLD: 2, RAMP_OUT: 3, REST: 4, CYCLE_PAUSE: 5, PAUSED: 6}
OUTSIDE = -1

BASE_COLUMNS = ("sample_index", "collection_time", "motion_class_id", "phase_code", "position")
INTENSITY_COLUMNS = ("intensity_level", "intensity")
TAIL_COLUMNS = ("repetition_index", "training_eligible")


def columns(intensity):
    return BASE_COLUMNS + (INTENSITY_COLUMNS if intensity else ()) + TAIL_COLUMNS


def targets_description(intensity):
    """What the run record says about the target files."""
    return {
        "file_template": TARGETS_TEMPLATE,
        "columns": list(columns(intensity)),
        "phase_codes": dict(PHASE_CODES, outside=OUTSIDE),
        "position": "0 = rest, 1 = gesture target; linear in time over ramp-in (0->1) and "
                    "ramp-out (1->0), 1 over the hold, 0 while resting, frozen while paused",
        "intensity": ("target fraction of maximum effort: position * the repetition's "
                      "intensity_level, 0 while resting") if intensity else None,
        "semantics": "targets describe the displayed instruction, not measured movement or force",
    }


@dataclass
class Targets:
    """Per-sample targets for one modality (all arrays the same length)."""

    modality: str
    sample_index: np.ndarray
    collection_time: np.ndarray
    motion_class_id: np.ndarray
    phase_code: np.ndarray
    position: np.ndarray
    repetition_index: np.ndarray
    training_eligible: np.ndarray
    intensity_level: np.ndarray | None = None
    intensity: np.ndarray | None = None
    samples: np.ndarray | None = None       # the recording itself, when loaded with it
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.sample_index)

    def table(self):
        """The rows written to disk, as an ``(n, columns)`` float array."""
        names = columns(self.intensity is not None)
        return np.column_stack([getattr(self, name) for name in names]).astype(float)

    def select(self, mask):
        """A copy restricted to ``mask`` (a boolean array or index)."""
        picked = {name: (None if value is None else value[mask])
                  for name, value in self.__dict__.items()
                  if isinstance(value, np.ndarray) or value is None}
        return Targets(modality=self.modality, extra=dict(self.extra), **picked)


def sample_targets(intervals, clock, modality, repetition_order=None, intensity=None):
    """One target row per sample of ``modality``, from a run's intervals.

    Parameters
    ----------
    intervals: list of dict
        From :func:`~libemg.choreography.annotations.build_intervals`, with
        sample indices assigned (``sample_indices[modality]``).
    clock: libemg.session.ModalityClock
        The modality's fitted clock (sample ``i`` produced at ``anchor + i / fs``).
    repetition_order: dict, optional
        Repetition ID to its 0-based run order.
    intensity: bool, optional
        Include the intensity columns. Inferred from the intervals when omitted.

    Returns
    ----------
    Targets
    """
    n = int(clock.n_samples)
    index = np.arange(n)
    produced = clock.anchor + index / clock.fs
    phases = [iv for iv in intervals if iv["track"] == PHASE_TRACK]
    if intensity is None:
        intensity = any(iv.get("intensity_start") is not None for iv in phases)
    out = Targets(
        modality=modality, sample_index=index, collection_time=produced,
        motion_class_id=np.zeros(n, dtype=int), phase_code=np.full(n, OUTSIDE, dtype=int),
        position=np.zeros(n), repetition_index=np.full(n, -1, dtype=int),
        training_eligible=np.zeros(n, dtype=int),
        intensity_level=np.zeros(n) if intensity else None,
        intensity=np.zeros(n) if intensity else None)
    order = repetition_order or {}
    for interval in phases:
        bounds = interval.get("sample_indices", {}).get(modality)
        if not bounds or bounds[1] <= bounds[0]:
            continue
        a, b = int(bounds[0]), int(bounds[1])
        span = slice(a, b)
        # A pause during a rest is resting; a pause mid-gesture shows the gesture.
        shown = interval.get("paused_phase") if interval["phase"] == PAUSED else interval["phase"]
        active = shown not in (WARMUP, REST, CYCLE_PAUSE)
        out.motion_class_id[span] = (interval.get("motion_class_id") or 0) if active else 0
        out.phase_code[span] = PHASE_CODES.get(interval["phase"], OUTSIDE)
        out.training_eligible[span] = int(bool(interval.get("training_eligible")))
        repetition = interval.get("parent_repetition_id")
        if active and repetition in order:
            out.repetition_index[span] = order[repetition]
        t0, t1 = interval["start_collection_time"], interval["end_collection_time"]
        p0 = interval.get("position_start") or 0.0
        p1 = interval.get("position_end")
        p1 = p0 if p1 is None else p1
        if t1 > t0 and p0 != p1:
            fraction = np.clip((produced[span] - t0) / (t1 - t0), 0.0, 1.0)
            out.position[span] = p0 + (p1 - p0) * fraction
        else:
            out.position[span] = p0
        if intensity:
            level = interval.get("intensity_level") or 0.0
            out.intensity_level[span] = level
            out.intensity[span] = out.position[span] * level
    return out


def write_target_files(folder, intervals, clocks, repetition_order=None, intensity=None):
    """Write ``_choreography_targets_<modality>.csv`` for every modality with a clock.

    Returns
    ----------
    dict
        Modality to the file name written.
    """
    folder = Path(folder)
    written = {}
    for modality, clock in sorted(clocks.items()):
        targets = sample_targets(intervals, clock, modality, repetition_order, intensity)
        name = TARGETS_TEMPLATE.format(modality=modality)
        _save(folder / name, targets)
        written[modality] = name
    return written


def _save(path, targets):
    names = columns(targets.intensity is not None)
    formats = ["%d", "%.6f", "%d", "%d", "%.6f"]
    if targets.intensity is not None:
        formats += ["%.6f", "%.6f"]
    formats += ["%d", "%d"]
    np.savetxt(path, targets.table(), delimiter=",", fmt=formats, header=",".join(names),
               comments="# ")


def read_target_file(path, modality=None):
    """Read one target file back into :class:`Targets`."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        names = handle.readline().lstrip("#").strip().split(",")
    table = np.loadtxt(path, delimiter=",", ndmin=2)
    if table.size == 0:
        table = np.zeros((0, len(names)))
    data = {name: table[:, i] for i, name in enumerate(names)}
    as_int = ("sample_index", "motion_class_id", "phase_code", "repetition_index",
              "training_eligible")
    for name in as_int:
        data[name] = data[name].astype(int)
    return Targets(modality=modality or path.stem.rsplit("_", 1)[-1],
                   **{k: v for k, v in data.items() if k in Targets.__dataclass_fields__})


def _record(folder):
    return json.loads((Path(folder) / RUN_RECORD_NAME).read_text(encoding="utf-8"))


def load_targets(folder, modality=None, with_samples=True):
    """A recording's samples and per-sample targets, aligned row for row.

    Parameters
    ----------
    folder: str or Path
        The recording's output folder.
    modality: str, optional
        Defaults to the run record's sample-index modality (``emg`` when present).
    with_samples: bool
        Also read ``_session_<modality>.csv`` into ``Targets.samples``.

    Returns
    ----------
    Targets
        ``targets.samples[i]`` is the sample whose targets are row ``i``. Use
        ``targets.select(targets.training_eligible == 1)`` to drop paused or
        interrupted stretches.
    """
    folder = Path(folder)
    record = _record(folder)
    modality = modality or record.get("sample_index_modality")
    path = folder / TARGETS_TEMPLATE.format(modality=modality)
    if path.exists():
        targets = read_target_file(path, modality)
    else:
        # A recording saved before target files existed, or recovered without
        # them: rebuild from the record.
        from libemg.session import ModalityClock
        clock = ModalityClock.from_dict(record["sample_clocks"][modality])
        targets = sample_targets(record["intervals"], clock, modality, _order(record))
    if with_samples:
        from libemg.session import SessionAnnotations
        _, samples = SessionAnnotations.load(folder).read(modality)
        count = min(len(samples), len(targets))
        targets = targets.select(slice(0, count))
        targets.samples = np.asarray(samples)[:count]
    targets.extra = {"run_id": record.get("run_id"), "class_map": record.get("class_map"),
                     "phase_codes": dict(PHASE_CODES, outside=OUTSIDE)}
    return targets


def _order(record):
    return {r["repetition_id"]: r["order_index"]
            for r in record.get("schedule", {}).get("repetitions", [])}


def cut_repetitions(folder, output_folder, modalities=None, include_incomplete=False,
                    lead_in=0.0, lead_out=0.0):
    """One data file and one target file per repetition, ramp-in start to ramp-out end.

    Files are ``C_<class>_R_<rep>_<modality>.csv`` (the session's sample rows,
    without the arrival-time column) and ``C_<class>_R_<rep>_<modality>_targets.csv``
    (the same rows of the target file). Both have the same number of rows, so a
    ``FilePackager`` pairs them without resampling.

    ``output_folder`` must not be inside the recording folder: the offline
    loaders search folders recursively and would read these repetitions a
    second time alongside the hold files already cut there.

    Parameters
    ----------
    include_incomplete: bool
        Also write repetitions that were stopped, failed or interrupted.
    lead_in, lead_out: float
        Seconds of the neighbouring rest to include either side.

    Returns
    ----------
    list of Path
    """
    from libemg.session import SessionAnnotations

    folder, output_folder = Path(folder).resolve(), Path(output_folder).resolve()
    if output_folder == folder or folder in output_folder.parents:
        raise ValueError("Write the repetitions outside the recording folder; the offline loaders "
                         "search it recursively and would read them twice.")
    record = _record(folder)
    session = SessionAnnotations.load(folder)
    reps = [iv for iv in record["intervals"] if iv["track"] == REPETITION_TRACK
            and (include_incomplete or iv["completion_status"] == COMPLETED)]
    output_folder.mkdir(parents=True, exist_ok=True)
    written = []
    for modality in (modalities or sorted(record.get("sample_clocks", {}))):
        clock = session.clocks.get(modality)
        if clock is None:
            continue
        targets = load_targets(folder, modality, with_samples=True)
        for rep in reps:
            bounds = rep.get("sample_indices", {}).get(modality)
            if not bounds:
                continue
            start = max(0, int(bounds[0]) - int(round(lead_in * clock.fs)))
            end = min(len(targets), int(bounds[1]) + int(round(lead_out * clock.fs)))
            if end <= start:
                continue
            piece = targets.select(slice(start, end))
            names = dict(motion=rep["motion_class_id"], rep=rep["gesture_repetition_index"],
                         modality=modality)
            data_path = output_folder / REPETITION_DATA_TEMPLATE.format(**names)
            np.savetxt(data_path, piece.samples, delimiter=",")
            target_path = output_folder / REPETITION_TARGETS_TEMPLATE.format(**names)
            _save(target_path, piece)
            written += [data_path, target_path]
    return written


__all__ = ["Targets", "sample_targets", "write_target_files", "read_target_file", "load_targets",
           "cut_repetitions", "targets_description", "columns", "PHASE_CODES", "OUTSIDE",
           "TARGETS_TEMPLATE", "REPETITION_DATA_TEMPLATE", "REPETITION_TARGETS_TEMPLATE"]
