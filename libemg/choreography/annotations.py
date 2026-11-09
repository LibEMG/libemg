"""Annotation intervals for a choreography run, built from the scheduler's events.

The scheduler emits an ordered stream of events (see
:class:`~libemg.choreography.schedule.ChoreographyScheduler`). Everything a
session says about *when* each phase was on screen is derived from that stream
by one pure function, :func:`build_intervals`, which is run both live and when
recovering an interrupted run from its journal -- so the two cannot disagree.

Interval conventions
--------------------
* Every interval is **half-open, [start, end)**. ``end_*`` is the first instant
  (and ``end_sample_index`` the first sample) *after* the interval. Adjacent
  intervals share a boundary value and never share a sample.
* ``*_collection_time`` is in the recording's clock (``time.time()`` seconds as
  the logger stamped it); see :mod:`libemg.choreography.clock`.
* Sample indices index rows of ``_session_<modality>.csv``. Sample ``i`` was
  produced at ``anchor + i / fs`` (the session's fitted clock) and belongs to an
  interval when ``start <= anchor + i / fs < end``.
* Two tracks. ``phase`` holds warm-up, ramp-in, hold, ramp-out, rest, cycle
  pause and paused segments; they tile the run and never overlap.
  ``repetition`` holds one parent region per repetition (ramp-in start to
  ramp-out end), which overlaps its children on purpose.
* Labels describe the **instruction displayed**, not verified participant
  movement.

Regression targets
------------------
Every phase interval carries ``position_start`` / ``position_end``: where the
displayed hand was between rest (0) and the gesture (1) at each end. Position
is linear in time within an interval, so the value at any instant inside it is
the straight line between the two. Ramp-in runs 0 -> 1, the hold is 1, ramp-out
runs 1 -> 0, resting phases are 0, and a paused interval holds the frozen value.
With intensity on, ``intensity_level`` is the repetition's target (fraction of
maximum effort) and ``intensity_start`` / ``intensity_end`` are ``position *
intensity_level`` (0 while resting). :mod:`libemg.choreography.targets` writes
these out per sample.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from libemg.session import PAUSE, PROMPT, RAMP_IN as RAMP_IN_KIND, RAMP_OUT as RAMP_OUT_KIND
from libemg.session import REST as REST_KIND
from libemg.session import Stage

from .assets import REST_ID
from .schedule import (ACTIVE_PHASES, CANCELLED, COMPLETED, CYCLE_PAUSE, FAILED, HOLD, PAUSED,
                       RAMP_IN, RAMP_OUT, REST, RESTING_PHASES, SEGMENT_COMPLETED, SEGMENT_PAUSED,
                       WARMUP, derived_label, position_for)

# 2: position_* and intensity_* targets on every interval.
INTERVAL_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 2
RUN_RECORD_NAME = "_session_choreography.json"
JOURNAL_NAME = "_session_choreography_events.jsonl"

PHASE_TRACK = "phase"
REPETITION_TRACK = "repetition"
REPETITION_PHASE = "repetition"

INTERVAL_CONVENTION = ("half-open [start, end): end_collection_time and end_sample_index are the "
                       "first instant and the first sample after the interval")
LABEL_SEMANTICS = ("Labels describe the instruction the participant was shown, not verified "
                   "participant movement.")

# Interval completion statuses beyond the segment statuses the scheduler emits.
STATUS_RESUMED = "resumed"
STATUS_INTERRUPTED = "interrupted"
STATUS_EMPTY = "empty"

_RUN_END_EVENTS = {"run_completed": COMPLETED, "run_cancelled": CANCELLED, "run_failed": FAILED}
# Segment statuses whose recording is a faithful display of the instruction.
_ELIGIBLE = (SEGMENT_COMPLETED, SEGMENT_PAUSED)

# How each phase is filed among libemg.session's stage kinds. Only the hold is a
# PROMPT, so the C_<class>_R_<rep>_ files the existing pipeline cuts are steady
# holds -- what they have always meant. The ramps get kinds of their own.
STAGE_KIND = {WARMUP: REST_KIND, REST: REST_KIND, CYCLE_PAUSE: REST_KIND,
              RAMP_IN: RAMP_IN_KIND, HOLD: PROMPT, RAMP_OUT: RAMP_OUT_KIND, PAUSED: PAUSE}


def _new_interval(event, track, phase, collection_session_id, class_ids):
    motion = event.get("motion_class", REST_ID)
    return {
        "schema_version": INTERVAL_SCHEMA_VERSION,
        "run_id": event.get("run_id"),
        "collection_session_id": collection_session_id,
        "interval_id": event["event_id"],
        "track": track,
        "parent_repetition_id": event.get("repetition_id"),
        "motion_class": motion,
        "motion_class_id": class_ids.get(motion),
        "phase": phase,
        "derived_label": derived_label(motion, phase),
        "cycle_index": event.get("cycle_index"),
        "gesture_repetition_index": event.get("gesture_repetition_index"),
        "segment_index": event.get("segment_index", 0),
        "plan_index": event.get("plan_index"),
        "next_repetition_id": event.get("next_repetition_id"),
        "planned_duration": event.get("planned_duration"),
        "planned_start": event.get("planned_start"),
        "start_schedule_time": event.get("schedule_time"),
        "intensity_level": event.get("intensity_level"),
        "transition_from": event.get("transition_from"),
        "position_start": None,
        "position_end": None,
        "intensity_start": None,
        "intensity_end": None,
        "start_monotonic": event["monotonic"],
        "end_monotonic": None,
        "start_collection_time": event.get("collection_time"),
        "end_collection_time": None,
        "start_sample_index": None,
        "end_sample_index": None,
        "sample_indices": {},
        "completion_status": None,
        "end_reason": None,
        "training_eligible": False,
        "flags": [],
        "start_event_id": event["event_id"],
        "end_event_id": None,
    }


def _close(interval, event, status, reason=None):
    interval["end_monotonic"] = event["monotonic"]
    interval["end_collection_time"] = event.get("collection_time")
    interval["completion_status"] = status
    interval["end_reason"] = reason
    interval["end_event_id"] = event.get("event_id")


def build_intervals(events, collection_session_id=None, class_ids=None, parent_regions=True,
                    last_alive=None):
    """Phase and repetition intervals from a run's events.

    Robust to a stream that stops part-way (a crash): whatever is still open
    when the events run out is closed at the last event's time and marked
    ``interrupted``, never ``completed``. A repeated or unmatched end event is
    ignored rather than producing a second interval.

    Parameters
    ----------
    events: list of dict
        In any order; they are sorted by ``sequence``.
    class_ids: dict, optional
        Motion ID to numeric class ID, copied into each interval.
    parent_regions: bool
        Add a ``repetition``-track interval per repetition.
    last_alive: dict, optional
        ``{"monotonic", "collection_time"}`` of the last moment the run was
        known to be alive (the journal's last heartbeat). Intervals left open by
        a truncated stream are closed there when it is later than the last event.

    Returns
    ----------
    intervals: list of dict
        Phase intervals in time order, then repetition intervals in time order.
    """
    class_ids = dict(class_ids or {})
    events = sorted(events, key=lambda e: e["sequence"])
    seen_ids = set()
    phase_intervals, gaps = [], []
    open_phase = open_pause = None
    run_status = run_reason = None
    for event in events:
        if event["event_id"] in seen_ids:
            continue
        seen_ids.add(event["event_id"])
        kind = event["type"]
        if kind == "phase_started":
            if open_phase is not None:
                _close(open_phase, event, STATUS_INTERRUPTED, "missing_phase_end")
                phase_intervals.append(open_phase)
            open_phase = _new_interval(event, PHASE_TRACK, event["phase"],
                                       collection_session_id, class_ids)
        elif kind == "phase_ended":
            if (open_phase is not None and open_phase["plan_index"] == event.get("plan_index")
                    and open_phase["segment_index"] == event.get("segment_index", 0)):
                _close(open_phase, event, event.get("status", SEGMENT_COMPLETED), event.get("reason"))
                phase_intervals.append(open_phase)
                open_phase = None
        elif kind == "paused":
            if open_pause is None:
                open_pause = _new_interval(event, PHASE_TRACK, PAUSED, collection_session_id, class_ids)
                open_pause["pause_reason"] = event.get("reason")
                open_pause["paused_phase"] = event.get("phase")
                open_pause["paused_phase_duration"] = event.get("planned_duration")
                open_pause["planned_duration"] = None
        elif kind == "resumed":
            if open_pause is not None:
                _close(open_pause, event, STATUS_RESUMED, open_pause.get("pause_reason"))
                phase_intervals.append(open_pause)
                open_pause = None
        elif kind == "gap":
            gaps.append(event)
        elif kind in _RUN_END_EVENTS:
            run_status, run_reason = _RUN_END_EVENTS[kind], event.get("reason")
            for still_open in (open_phase, open_pause):
                if still_open is not None:
                    _close(still_open, event, run_status, run_reason or "run_ended")
                    phase_intervals.append(still_open)
            open_phase = open_pause = None
    if events and (open_phase is not None or open_pause is not None):
        last = max(events, key=lambda e: e["monotonic"])
        if last_alive is not None and last_alive.get("monotonic", float("-inf")) > last["monotonic"]:
            last = {"monotonic": last_alive["monotonic"],
                    "collection_time": last_alive.get("collection_time")}
        for still_open in (open_phase, open_pause):
            if still_open is not None:
                # The best timestamp left: the last thing the run recorded.
                _close(still_open, last, STATUS_INTERRUPTED, "events_ended_without_run_end")
                still_open["end_event_id"] = None
                phase_intervals.append(still_open)

    phase_intervals = [iv for iv in phase_intervals if iv["end_monotonic"] > iv["start_monotonic"]]
    for interval in phase_intervals:
        for gap in gaps:
            if (interval["start_monotonic"] < gap["end_monotonic"]
                    and gap["start_monotonic"] < interval["end_monotonic"]):
                flag = gap.get("gap_kind", "gap")
                if flag not in interval["flags"]:
                    interval["flags"].append(flag)
        interval["training_eligible"] = (interval["phase"] != PAUSED
                                         and interval["completion_status"] in _ELIGIBLE
                                         and not interval["flags"])
        _set_targets(interval)
    phase_intervals.sort(key=lambda iv: (iv["start_monotonic"], iv["end_monotonic"]))
    parents = _repetition_intervals(phase_intervals) if parent_regions else []
    return phase_intervals + parents


def _progress(interval, extra=0.0):
    """How far (0..1) through its planned phase an interval is, ``extra`` seconds after it starts.

    For a pause, how far through the phase it interrupted.
    """
    duration = interval.get("planned_duration") or interval.get("paused_phase_duration")
    start = interval.get("planned_start")
    at = interval.get("start_schedule_time")
    if not duration or start is None or at is None:
        return 0.0
    return (at - start + extra) / duration


def _set_targets(interval):
    """Fill ``position_*`` and ``intensity_*`` for one phase interval."""
    level = interval.get("intensity_level")
    if interval["phase"] == PAUSED:
        # The hand is frozen where the pause caught it.
        underlying = interval.get("paused_phase")
        start = end = position_for(underlying, _progress(interval))
        if underlying in RESTING_PHASES:
            # A rest's level is the next repetition's; nothing is being asked for yet.
            interval["next_intensity_level"], interval["intensity_level"] = level, None
    elif interval["phase"] in RESTING_PHASES:
        start = end = 0.0
        interval["next_intensity_level"], interval["intensity_level"] = level, None
    else:
        start = position_for(interval["phase"], _progress(interval))
        end = position_for(interval["phase"], _progress(
            interval, interval["end_monotonic"] - interval["start_monotonic"]))
    interval["position_start"], interval["position_end"] = start, end
    if level is not None:
        own = interval.get("intensity_level") or 0.0
        interval["intensity_start"], interval["intensity_end"] = start * own, end * own


def _repetition_intervals(phase_intervals):
    grouped = {}
    for interval in phase_intervals:
        repetition = interval["parent_repetition_id"]
        if repetition is None:
            continue
        grouped.setdefault(repetition, []).append(interval)
    parents = []
    for repetition_id, children in grouped.items():
        active = [c for c in children if c["phase"] in ACTIVE_PHASES]
        if not active:
            continue
        pauses = [c for c in children if c["phase"] == PAUSED]
        statuses = {c["completion_status"] for c in active}
        flagged = any(c["flags"] for c in children)
        ramp_out = [c for c in active if c["phase"] == RAMP_OUT]
        finished = bool(ramp_out) and ramp_out[-1]["completion_status"] == SEGMENT_COMPLETED
        if FAILED in statuses:
            status = FAILED
        elif CANCELLED in statuses:
            status = CANCELLED
        elif STATUS_INTERRUPTED in statuses or flagged or not finished:
            status = STATUS_INTERRUPTED
        else:
            status = COMPLETED
        first, last = active[0], max(children, key=lambda c: c["end_monotonic"])
        parent = dict(first)
        parent.update({
            "interval_id": repetition_id, "track": REPETITION_TRACK, "phase": REPETITION_PHASE,
            "derived_label": f"{first['motion_class']}/{REPETITION_PHASE}",
            "parent_repetition_id": None, "repetition_id": repetition_id,
            "segment_index": None, "plan_index": None, "planned_duration": None,
            "planned_start": None, "start_schedule_time": None,
            # A repetition rises and falls; the shape is in its children.
            "position_start": None, "position_end": None,
            "intensity_start": None, "intensity_end": None,
            "end_monotonic": last["end_monotonic"],
            "end_collection_time": last["end_collection_time"],
            "end_event_id": last["end_event_id"],
            "completion_status": status, "end_reason": last["end_reason"],
            "training_eligible": status == COMPLETED,
            "flags": sorted({f for c in children for f in c["flags"]}),
            "pause_count": len(pauses),
            "child_interval_ids": [c["interval_id"] for c in children],
            "sample_indices": {},
        })
        parents.append(parent)
    parents.sort(key=lambda iv: iv["start_monotonic"])
    return parents


# -- sample indices ------------------------------------------------------------------

def sample_bounds(clock, start, end):
    """Half-open sample range ``[first, stop)`` produced within ``[start, end)``.

    ``clock`` is a :class:`libemg.session.ModalityClock`. Sample ``i`` was
    produced at ``anchor + i / fs``; it is inside when ``start <= that < end``,
    so a sample exactly on a boundary belongs to the later interval.
    """
    def first_at_or_after(moment):
        position = (moment - clock.anchor) * clock.fs
        return int(min(max(math.ceil(position - 1e-9), 0), clock.n_samples))
    return first_at_or_after(start), first_at_or_after(end)


def assign_sample_indices(intervals, clocks, primary=None):
    """Fill ``sample_indices`` (every modality) and the primary modality's bounds.

    Parameters
    ----------
    clocks: dict
        Modality name to :class:`libemg.session.ModalityClock`.
    primary: str, optional
        Whose indices go in ``start_sample_index`` / ``end_sample_index``.
        ``emg`` when there is one, otherwise the first modality.
    """
    if primary is None and clocks:
        primary = "emg" if "emg" in clocks else sorted(clocks)[0]
    for interval in intervals:
        start, end = interval.get("start_collection_time"), interval.get("end_collection_time")
        if start is None or end is None:
            continue
        interval["sample_indices"] = {
            mod: list(sample_bounds(clock, start, end)) for mod, clock in clocks.items()}
        if primary in interval["sample_indices"]:
            interval["start_sample_index"], interval["end_sample_index"] = \
                interval["sample_indices"][primary]
        interval["sample_index_modality"] = primary
    return primary


# -- the existing session format ------------------------------------------------------

def stages_from_intervals(intervals, class_ids, repetitions):
    """Phase intervals as :class:`libemg.session.Stage` entries for ``_session_annotations.json``.

    Kinds follow the existing format: a hold is a ``prompt`` (so ``C_<id>_R_<rep>``
    files are steady holds), warm-up/rest/cycle pause are ``rest`` stages filed
    under the repetition they lead into (as the image-folder mode files its
    rests), a pause is a ``pause``, and the ramps are ``ramp_in``/``ramp_out``.
    ``phase``, ``interval_id`` and ``status`` say which choreography interval a
    stage is.

    When a pause splits a phase into segments, the segments share a name when
    cut, so all but one are marked ``superseded``: the longest segment that
    displayed the instruction cleanly is the one cut, and every segment stays in
    the annotations.

    Parameters
    ----------
    repetitions: dict
        Repetition ID to ``(motion_id, gesture_repetition_index)``.
    """
    stages = []
    for interval in intervals:
        if interval["track"] != PHASE_TRACK or interval.get("start_collection_time") is None:
            continue
        phase = interval["phase"]
        kind = STAGE_KIND[phase]
        if phase in RESTING_PHASES:
            motion_id, rep = repetitions.get(interval.get("next_repetition_id"), (None, None))
            label, motion = motion_id or "", class_ids.get(motion_id)
        elif phase == PAUSED:
            label, motion, rep = "", None, interval.get("gesture_repetition_index")
        else:
            label, motion = interval["motion_class"], class_ids.get(interval["motion_class"])
            rep = interval["gesture_repetition_index"]
        start, end = interval["start_collection_time"], interval["end_collection_time"]
        level = interval.get("intensity_level")
        if level is None:
            level = interval.get("next_intensity_level")
        stages.append(Stage(kind=kind, label=label, motion=motion, rep=rep, t_start=start,
                            t_end=end, duration=max(0.0, end - start),
                            attempt=interval.get("segment_index") or 0,
                            factors={} if level is None else {"intensity": level},
                            phase=phase, interval_id=interval["interval_id"],
                            status=interval["completion_status"]))
    groups = {}
    for stage in stages:
        if stage.kind != PAUSE:
            groups.setdefault((stage.kind, stage.motion, stage.rep), []).append(stage)
    for group in groups.values():
        if len(group) < 2:
            continue
        best = max(group, key=lambda s: (s.status in _ELIGIBLE, s.duration))
        for stage in group:
            stage.superseded = stage is not best
    return stages


# -- on disk ---------------------------------------------------------------------------

class RunJournal:
    """Append-only record of a run's events, flushed as they happen.

    The first line is a header (configuration, plan, asset, class map); every
    line after it is one event. Lifecycle events are also fsync'd, so a run that
    dies -- power, crash, killed process -- still leaves what is needed to
    rebuild its intervals with :func:`recover_run`.
    """

    _DURABLE = {"run_started", "paused", "resumed", "gap", *_RUN_END_EVENTS}

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "w", encoding="utf-8")

    def header(self, header):
        self._write({"record": "header", **header}, durable=True)

    def event(self, event):
        self._write({"record": "event", **event}, durable=event["type"] in self._DURABLE)

    def footer(self, footer):
        self._write({"record": "footer", **footer}, durable=True)

    def heartbeat(self, beat):
        self._write({"record": "heartbeat", **beat})

    def _write(self, record, durable=False):
        if self._handle is None:
            return
        self._handle.write(json.dumps(record, default=_jsonable) + "\n")
        self._handle.flush()
        if durable:
            try:
                os.fsync(self._handle.fileno())
            except OSError:
                pass

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def read_journal(path, heartbeat=False):
    """``(header, events, footer)`` from a journal, skipping a torn last line.

    With ``heartbeat=True`` a fourth item is returned: the last heartbeat, or None.
    """
    header, footer, events, beat = {}, None, [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue        # a line cut off by the crash being recovered from
            kind = record.pop("record", "event")
            if kind == "header":
                header = record
            elif kind == "footer":
                footer = record
            elif kind == "heartbeat":
                beat = record
            else:
                events.append(record)
    return (header, events, footer, beat) if heartbeat else (header, events, footer)


def write_run_record(path, record):
    """Write the run record atomically (whole file, then moved into place)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".part")
    with open(scratch, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=_jsonable)
    scratch.replace(path)
    return path


def run_record(header, events, intervals, status, reason, clocks=None, primary=None, extra=None):
    """The ``_session_choreography.json`` document."""
    phases = [iv for iv in intervals if iv["track"] == PHASE_TRACK]
    reps = [iv for iv in intervals if iv["track"] == REPETITION_TRACK]
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": "libemg.choreography.run",
        **header,
        "status": status,
        "end_reason": reason,
        "interval_convention": INTERVAL_CONVENTION,
        "label_semantics": LABEL_SEMANTICS,
        "sample_index_modality": primary,
        "sample_clocks": {mod: clock.to_dict() for mod, clock in (clocks or {}).items()},
        "summary": {
            "phase_intervals": len(phases),
            "repetitions_started": len(reps),
            "repetitions_completed": sum(1 for r in reps if r["completion_status"] == COMPLETED),
            "repetitions_requested": header.get("schedule", {}).get("total_repetitions"),
            "pauses": sum(1 for iv in phases if iv["phase"] == PAUSED),
        },
        "intervals": intervals,
        "events": sorted(events, key=lambda e: e["sequence"]),
    }
    if extra:
        record.update(extra)
    return record


def recover_run(folder, resolve_clocks=True):
    """Rebuild a run's record from its journal after the application died mid-run.

    Uses the session annotations' fitted clocks for sample indices when they
    can be measured from the recording. A run whose journal never reached a
    run-end event is recorded as ``failed`` with reason
    ``recovered_after_interruption``; its open intervals are ``interrupted``.

    Returns
    ----------
    Path
        The run record written.
    """
    from libemg.session import SessionAnnotations

    folder = Path(folder)
    header, events, footer, beat = read_journal(folder / JOURNAL_NAME, heartbeat=True)
    ended = [e for e in events if e["type"] in _RUN_END_EVENTS]
    status = _RUN_END_EVENTS[ended[-1]["type"]] if ended else FAILED
    reason = ended[-1].get("reason") if ended else "recovered_after_interruption"
    class_ids = {v: int(k) for k, v in header.get("class_map", {}).get("classes", {}).items()}
    intervals = build_intervals(events, header.get("collection_session_id"), class_ids,
                                header.get("config", {}).get("parent_regions", True),
                                last_alive=beat)
    clocks, primary = {}, None
    session = None
    if resolve_clocks and (folder / "_session_annotations.json").exists():
        session = SessionAnnotations.load(folder)
        repetitions = {r["repetition_id"]: (r["motion_id"], r["gesture_repetition_index"])
                       for r in header.get("schedule", {}).get("repetitions", [])}
        session.stages = stages_from_intervals(intervals, class_ids, repetitions)
        session.resolve()
        session.save()
        clocks = session.clocks
        primary = assign_sample_indices(intervals, clocks)
    written = {}
    if clocks:
        from .targets import write_target_files
        order = {r["repetition_id"]: r["order_index"]
                 for r in header.get("schedule", {}).get("repetitions", [])}
        written = write_target_files(folder, intervals, clocks, order,
                                     intensity=header.get("intensity", {}).get("enabled"))
    return write_run_record(folder / RUN_RECORD_NAME,
                            run_record(header, events, intervals, status, reason, clocks, primary,
                                       extra={"recovered": True, "journal_footer": footer,
                                              "target_files": written}))


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


__all__ = ["build_intervals", "assign_sample_indices", "sample_bounds", "stages_from_intervals",
           "RunJournal", "read_journal", "write_run_record", "run_record", "recover_run",
           "RUN_RECORD_NAME", "JOURNAL_NAME", "PHASE_TRACK", "REPETITION_TRACK",
           "INTERVAL_SCHEMA_VERSION", "RUN_SCHEMA_VERSION", "INTERVAL_CONVENTION",
           "LABEL_SEMANTICS", "STAGE_KIND"]
