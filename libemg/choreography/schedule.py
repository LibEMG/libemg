"""The ordered plan for a choreography run, and the scheduler that walks it.

Planning and running are kept apart.

:func:`build_plan` turns a :class:`~libemg.choreography.config.ChoreographyConfig`
into every phase of the run, in order, with its nominal start and length --
before anything starts. That is what the panel's duration estimate is, what the
session stores as "the schedule", and what a seeded shuffle reproduces exactly.

:class:`ChoreographyScheduler` then plays that plan against a clock it is
*given* (``now`` on every call; nothing here reads a clock). It is the single
authority for what is on screen: the animation pose, the phase label, the
progress readout and every annotation boundary all come from it. A phase
boundary is placed at the instant the plan puts it -- ``start + paused time +
planned offset`` -- not at whichever GUI frame happened to notice it, so a slow
frame moves when the screen *updates*, never where the annotation falls.

Repetition structure
--------------------
One repetition is ``rest -> ramp_in -> hold -> ramp_out -> rest``, the rests
shared with the neighbouring repetitions. A cycle holds one repetition of every
selected gesture that still has repetitions left; a gesture drops out of later
cycles once its count is met. The run opens with the warm-up (relaxed hand),
puts ``rest`` between gestures, puts the cycle pause -- when enabled, instead of
the rest -- between cycles, and ends on the last ramp-out with no trailing rest.
Phases of zero length are left out of the plan altogether.

With ``rest_between_gestures`` off there is no rest between gestures (or between
cycles, unless the cycle pause is on): each ramp-out runs straight into the next
ramp-in, a direct *transition*. Such a ramp-in records the gesture it came from
in ``transition_from``.

Regression targets
------------------
Alongside the class, every instant has a **position**: how far the displayed
hand is from rest toward the gesture, 0 to 1. It rises linearly over the
ramp-in, is 1 through the hold, falls linearly over the ramp-out, and is 0
while resting (:func:`position_for`). When intensity is on, each repetition
also has a target **intensity level** (a fraction of maximum effort), and the
instantaneous **intensity** target is ``position * level``.
"""

from __future__ import annotations

import bisect
import random
import uuid
from dataclasses import dataclass, field

from .assets import REST_ID
from .config import SHUFFLE, ChoreographyConfig

# Phases, as recorded in every annotation.
WARMUP = "warmup"
RAMP_IN = "ramp_in"
HOLD = "hold"
RAMP_OUT = "ramp_out"
REST = "rest"
CYCLE_PAUSE = "cycle_pause"
PAUSED = "paused"

ACTIVE_PHASES = (RAMP_IN, HOLD, RAMP_OUT)
RESTING_PHASES = (WARMUP, REST, CYCLE_PAUSE)
PHASES = (WARMUP, RAMP_IN, HOLD, RAMP_OUT, REST, CYCLE_PAUSE, PAUSED)

PHASE_NAMES = {WARMUP: "Warm-up", RAMP_IN: "Ramping in", HOLD: "Hold", RAMP_OUT: "Ramping out",
               REST: "Rest", CYCLE_PAUSE: "Cycle pause", PAUSED: "Paused"}

# Run states.
READY = "ready"
RUNNING = "running"
PAUSED_STATE = "paused"
COMPLETED = "completed"
CANCELLED = "cancelled"
FAILED = "failed"
TERMINAL = (COMPLETED, CANCELLED, FAILED)

# How a phase segment ended.
SEGMENT_COMPLETED = "completed"       # ran its full planned length
SEGMENT_PAUSED = "paused"             # closed by a user pause; continues after resume
SEGMENT_INTERRUPTED = "interrupted"   # closed by a stall or a lost stream
SEGMENT_CANCELLED = "cancelled"
SEGMENT_FAILED = "failed"


def position_for(phase, progress):
    """Displayed position, 0 (rest) to 1 (gesture), ``progress`` (0..1) through ``phase``.

    Linear in time over the ramps; 1 over the hold; 0 for every resting phase.
    """
    progress = min(max(float(progress), 0.0), 1.0)
    if phase == RAMP_IN:
        return progress
    if phase == HOLD:
        return 1.0
    if phase == RAMP_OUT:
        return 1.0 - progress
    return 0.0


def derived_label(motion_class, phase):
    """``hand_closed/ramp_in``, ``rest/warmup``, or plain ``rest`` for a rest."""
    if phase == REST:
        return REST_ID
    if phase == PAUSED:
        return PAUSED
    return f"{motion_class}/{phase}"


@dataclass(frozen=True)
class PlannedRepetition:
    repetition_id: str
    motion_id: str
    cycle_index: int
    gesture_repetition_index: int     # 0-based count of this gesture so far
    order_index: int                  # 0-based position among all repetitions
    position_in_cycle: int
    intensity: float | None = None    # target fraction of maximum effort, if asked for


@dataclass(frozen=True)
class PlannedPhase:
    index: int
    phase: str
    motion_class: str                 # the motion shown; ``rest`` while resting
    start: float                      # seconds into the plan
    duration: float
    cycle_index: int | None = None
    gesture_repetition_index: int | None = None
    repetition_id: str | None = None          # for ramp_in / hold / ramp_out
    next_repetition_id: str | None = None     # for the rests: the repetition they lead into
    next_motion: str | None = None
    # Target intensity of the repetition this phase belongs to -- for a rest,
    # the one it leads into. None when intensity is off.
    intensity: float | None = None
    # For a ramp-in that follows a ramp-out with no rest between: the gesture
    # the hand is coming from.
    transition_from: str | None = None

    @property
    def end(self):
        return self.start + self.duration

    @property
    def label(self):
        return derived_label(self.motion_class, self.phase)

    def to_dict(self):
        return {"index": self.index, "phase": self.phase, "motion_class": self.motion_class,
                "derived_label": self.label, "start": self.start, "duration": self.duration,
                "cycle_index": self.cycle_index,
                "gesture_repetition_index": self.gesture_repetition_index,
                "repetition_id": self.repetition_id,
                "next_repetition_id": self.next_repetition_id, "next_motion": self.next_motion,
                "intensity_level": self.intensity, "transition_from": self.transition_from}


@dataclass
class ChoreographyPlan:
    """Every phase of a run, built before it starts."""

    config: ChoreographyConfig
    phases: list
    repetitions: list
    cycles: list                      # list of lists of motion IDs, in run order
    requested: dict                   # motion ID -> repetition count
    _starts: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._starts = [p.start for p in self.phases]

    @property
    def total_duration(self):
        return self.phases[-1].end if self.phases else 0.0

    @property
    def total_repetitions(self):
        return len(self.repetitions)

    @property
    def seed(self):
        return self.config.shuffle_seed if self.config.order_mode == SHUFFLE else None

    def phase_index_at(self, t):
        """The phase running ``t`` seconds into the plan (the last one at or past the end)."""
        if not self.phases:
            return -1
        return max(0, min(len(self.phases) - 1, bisect.bisect_right(self._starts, t) - 1))

    def repetition(self, repetition_id):
        for repetition in self.repetitions:
            if repetition.repetition_id == repetition_id:
                return repetition
        raise KeyError(repetition_id)

    def to_dict(self):
        return {
            "order_mode": self.config.order_mode,
            "shuffle_seed": self.config.shuffle_seed,
            "seed_applied": self.seed is not None,
            "cycles": [list(c) for c in self.cycles],
            "requested_repetitions": dict(self.requested),
            "rest_between_gestures": bool(self.config.rest_between_gestures),
            "intensity_enabled": bool(self.config.intensity_enabled),
            "intensity_levels": (list(self.config.intensity_levels)
                                 if self.config.intensity_enabled else None),
            "intensity_order": (self.config.intensity_order
                                if self.config.intensity_enabled else None),
            "total_repetitions": self.total_repetitions,
            "total_duration": self.total_duration,
            "repetitions": [r.__dict__.copy() for r in self.repetitions],
            "phases": [p.to_dict() for p in self.phases],
        }


def plan_cycles(config):
    """Which gesture runs in which cycle, in order. A seeded shuffle reorders each cycle."""
    remaining = config.requested_repetitions()
    rng = random.Random(config.shuffle_seed)
    cycles = []
    while any(count > 0 for count in remaining.values()):
        cycle = [g for g in config.gestures if remaining[g] > 0]
        if config.order_mode == SHUFFLE:
            rng.shuffle(cycle)
        for g in cycle:
            remaining[g] -= 1
        cycles.append(cycle)
    return cycles


def build_plan(config, known_gestures=None):
    """Lay out every phase of a run.

    Raises
    ----------
    ValueError
        With every problem :meth:`ChoreographyConfig.validate` found.
    """
    problems = config.validate(known_gestures)
    if problems:
        raise ValueError(" ".join(problems))
    config = config.snapshot()
    cycles = plan_cycles(config)
    phases, repetitions = [], []
    done = {g: 0 for g in config.gestures}
    intensities = {g: config.intensities_for(g) for g in config.gestures}
    t = 0.0

    def add(phase, motion, duration, **fields):
        nonlocal t
        if duration <= 0:
            return
        phases.append(PlannedPhase(index=len(phases), phase=phase, motion_class=motion,
                                   start=t, duration=float(duration), **fields))
        t = t + float(duration)

    previous = None
    for cycle_index, cycle in enumerate(cycles):
        for position, motion in enumerate(cycle):
            repetition = PlannedRepetition(
                repetition_id=f"rep-{len(repetitions):04d}", motion_id=motion,
                cycle_index=cycle_index, gesture_repetition_index=done[motion],
                order_index=len(repetitions), position_in_cycle=position,
                intensity=intensities[motion][done[motion]])
            done[motion] += 1
            lead = {"next_repetition_id": repetition.repetition_id, "next_motion": motion,
                    "cycle_index": cycle_index, "intensity": repetition.intensity}
            if previous is None:
                add(WARMUP, REST_ID, config.warmup, **lead)
            elif position == 0 and config.cycle_pause_enabled:
                add(CYCLE_PAUSE, REST_ID, config.cycle_pause, **lead)
            elif config.rest_between_gestures:
                add(REST, REST_ID, config.timing_for(previous)["rest"], **lead)
            timing = config.timing_for(motion)
            own = {"cycle_index": cycle_index,
                   "gesture_repetition_index": repetition.gesture_repetition_index,
                   "repetition_id": repetition.repetition_id, "intensity": repetition.intensity}
            # Straight out of the last gesture (no rest, or a rest of zero length).
            came_from = previous if phases and phases[-1].phase == RAMP_OUT else None
            add(RAMP_IN, motion, timing["ramp_in"], transition_from=came_from, **own)
            add(HOLD, motion, timing["hold"], **own)
            add(RAMP_OUT, motion, timing["ramp_out"], **own)
            repetitions.append(repetition)
            previous = motion
    return ChoreographyPlan(config=config, phases=phases, repetitions=repetitions,
                            cycles=cycles, requested=config.requested_repetitions())


# -- running it -------------------------------------------------------------------

@dataclass
class PhaseView:
    """What the screen should show right now. Built by :meth:`ChoreographyScheduler.view`."""

    state: str
    phase: str | None                 # the scheduled phase (frozen while paused)
    display_phase: str | None         # ``paused`` while paused, else ``phase``
    motion_class: str
    next_motion: str | None
    progress: float                   # 0..1 through the current phase
    remaining: float                  # seconds left in the current phase
    elapsed: float                    # seconds into the plan
    total: float
    cycle_index: int | None
    cycle_count: int
    gesture_repetition_index: int | None
    gesture_repetition_count: int | None
    repetitions_done: int
    repetitions_total: int
    pause_reason: str | None = None
    position: float = 0.0             # 0 rest .. 1 gesture; see position_for
    intensity_level: float | None = None    # this (or the next) repetition's target
    intensity: float | None = None    # position * intensity_level, when intensity is on
    # During a hold or ramp-out that runs straight into the next gesture (no
    # rest between): that gesture, and its intensity level.
    then_motion: str | None = None
    then_intensity_level: float | None = None

    @property
    def phase_name(self):
        return PHASE_NAMES.get(self.display_phase, "Ready" if self.state == READY else "")


class ChoreographyScheduler:
    """Plays a :class:`ChoreographyPlan` against a clock passed in on every call.

    Every change is reported as an event dict through ``sink`` (and kept in
    :attr:`events`): ``run_started``, ``phase_started``, ``phase_ended``,
    ``paused``, ``resumed`` and one of ``run_completed`` / ``run_cancelled`` /
    ``run_failed``. Each has a unique ``event_id``, a strictly increasing
    ``sequence``, and ``monotonic`` -- the instant it describes on the clock
    the scheduler is driven with.

    A pause can split a phase into segments. ``phase_ended`` closes the segment
    with a ``status`` saying why (completed, paused, interrupted, cancelled,
    failed); ``phase_started`` after ``resumed`` opens the next segment of the
    same phase, with ``segment_index`` one higher.
    """

    def __init__(self, plan, run_id=None, sink=None):
        self.plan = plan
        self.run_id = run_id or uuid.uuid4().hex
        self.sink = sink
        self.events = []
        self.state = READY
        self.pause_reason = None
        self._t0 = None
        self._paused_total = 0.0
        self._pause_at = None
        self._index = -1
        self._segment = 0
        self._sequence = 0
        self._end_elapsed = 0.0
        self._last_time = None
        self._completed_repetitions = set()

    # -- clock arithmetic ---------------------------------------------------------

    def elapsed(self, now):
        """Seconds into the plan at ``now``: wall time since start, less every pause."""
        if self.state == READY:
            return 0.0
        if self.state in TERMINAL:
            return self._end_elapsed
        if self.state == PAUSED_STATE:
            return self._pause_at - self._t0 - self._paused_total
        return now - self._t0 - self._paused_total

    def _at(self, plan_time):
        """The clock instant a plan time falls on, given the pauses so far."""
        return self._t0 + self._paused_total + plan_time

    @property
    def last_time(self):
        """The latest instant the scheduler has been advanced to."""
        return self._last_time

    @property
    def current_phase(self):
        return self.plan.phases[self._index] if 0 <= self._index < len(self.plan.phases) else None

    @property
    def completed_repetitions(self):
        return len(self._completed_repetitions)

    # -- events ---------------------------------------------------------------------

    def _emit(self, kind, at, **payload):
        self._sequence += 1
        event = {"event_id": uuid.uuid4().hex, "sequence": self._sequence, "type": kind,
                 "run_id": self.run_id, "monotonic": float(at),
                 "schedule_time": float(at - self._t0 - self._paused_total) if self._t0 is not None else 0.0,
                 **payload}
        self.events.append(event)
        if self.sink is not None:
            self.sink(event)
        return event

    def record(self, kind, at, **payload):
        """Add an event of the caller's own (a gap, a diagnostic) to the same ordered stream."""
        return self._emit(kind, at, **payload)

    def _phase_payload(self, phase):
        return {"plan_index": phase.index, "phase": phase.phase,
                "motion_class": phase.motion_class, "derived_label": phase.label,
                "cycle_index": phase.cycle_index,
                "gesture_repetition_index": phase.gesture_repetition_index,
                "repetition_id": phase.repetition_id,
                "next_repetition_id": phase.next_repetition_id,
                "next_motion": phase.next_motion,
                "intensity_level": phase.intensity,
                "transition_from": phase.transition_from,
                "segment_index": self._segment, "planned_start": phase.start,
                "planned_duration": phase.duration}

    def _begin(self, index, at):
        self._index = index
        self._segment = 0
        self._emit("phase_started", at, **self._phase_payload(self.plan.phases[index]))

    def _end_current(self, at, status, reason=None):
        phase = self.current_phase
        payload = self._phase_payload(phase)
        self._emit("phase_ended", at, status=status, reason=reason, **payload)
        if phase.phase == RAMP_OUT and status == SEGMENT_COMPLETED:
            self._completed_repetitions.add(phase.repetition_id)

    def _check_time(self, now):
        if self._last_time is not None and now < self._last_time:
            raise ValueError(f"The clock went backwards ({now} < {self._last_time}).")
        self._last_time = now

    # -- control ------------------------------------------------------------------

    def start(self, now, **payload):
        if self.state != READY:
            raise RuntimeError(f"Cannot start a run that is {self.state}.")
        self._check_time(now)
        self._t0 = float(now)
        self.state = RUNNING
        self._emit("run_started", now, total_duration=self.plan.total_duration,
                   total_repetitions=self.plan.total_repetitions, **payload)
        if not self.plan.phases:
            self._finish(COMPLETED, now, None)
        else:
            self._begin(0, now)

    def tick(self, now):
        """Advance to ``now``, emitting every boundary passed since the last call."""
        self._check_time(now)
        if self.state != RUNNING:
            return
        elapsed = now - self._t0 - self._paused_total
        while self.state == RUNNING:
            phase = self.current_phase
            if elapsed < phase.end:
                break
            boundary = self._at(phase.end)
            self._end_current(boundary, SEGMENT_COMPLETED)
            if phase.index + 1 < len(self.plan.phases):
                self._begin(phase.index + 1, boundary)
            else:
                self._end_elapsed = phase.end
                self._finish(COMPLETED, boundary, None)

    def pause(self, now, reason="user", status=SEGMENT_PAUSED):
        """Freeze the schedule at ``now``. Returns False if there was nothing to pause.

        ``now`` may be earlier than the present -- a stall is paused at the last
        moment the screen was known to be up to date -- but never earlier than
        the last instant the scheduler was advanced to.
        """
        if self.state != RUNNING:
            return False
        self.tick(now)
        if self.state != RUNNING:
            return False
        self._end_current(now, status, reason)
        self._pause_at = float(now)
        self.state = PAUSED_STATE
        self.pause_reason = reason
        self._emit("paused", now, reason=reason, status=status, **self._phase_payload(self.current_phase))
        return True

    def resume(self, now):
        if self.state != PAUSED_STATE:
            return False
        self._check_time(now)
        paused_for = now - self._pause_at
        self._paused_total += paused_for
        self._pause_at = None
        self.state = RUNNING
        reason, self.pause_reason = self.pause_reason, None
        self._emit("resumed", now, paused_seconds=paused_for, reason=reason)
        self._segment += 1
        self._emit("phase_started", now, **self._phase_payload(self.current_phase))
        return True

    def stop(self, now, status=CANCELLED, reason="user_stop"):
        """End the run early. Returns the state it ended in.

        A run that reaches its end by ``now`` completes instead: stopping after
        the last phase does not turn a finished run into a cancelled one.
        """
        if self.state in TERMINAL:
            return self.state
        if self.state == READY:
            self._check_time(now)
            self._t0 = float(now)
            self._finish(status, now, reason)
            return self.state
        if self.state == RUNNING:
            self.tick(now)
            if self.state in TERMINAL:
                return self.state
            self._end_elapsed = now - self._t0 - self._paused_total
            self._end_current(now, SEGMENT_FAILED if status == FAILED else SEGMENT_CANCELLED, reason)
        else:
            self._check_time(now)
            self._end_elapsed = self._pause_at - self._t0 - self._paused_total
        self._finish(status, now, reason)
        return self.state

    def _finish(self, status, at, reason):
        self.state = status
        self._emit(f"run_{status}", at, reason=reason,
                   repetitions_completed=len(self._completed_repetitions),
                   total_repetitions=self.plan.total_repetitions)

    # -- reading it ------------------------------------------------------------------

    def _direct_successor(self, phase):
        """The ramp-in that follows ``phase``'s repetition with no rest between, if any."""
        phases = self.plan.phases
        index = phase.index
        while index + 1 < len(phases) and phases[index + 1].repetition_id == phase.repetition_id:
            index += 1
        following = phases[index + 1] if index + 1 < len(phases) else None
        return following if following is not None and following.phase == RAMP_IN else None

    def view(self, now):
        """What to show at ``now``. Call :meth:`tick` first; this does not advance."""
        plan = self.plan
        cycles = len(plan.cycles)
        base = dict(state=self.state, next_motion=None, elapsed=self.elapsed(now),
                    total=plan.total_duration, cycle_count=cycles,
                    repetitions_done=len(self._completed_repetitions),
                    repetitions_total=plan.total_repetitions, pause_reason=self.pause_reason)
        phase = self.current_phase
        if self.state == READY or self.state in TERMINAL or phase is None:
            return PhaseView(phase=None, display_phase=None, motion_class=REST_ID, progress=0.0,
                             remaining=0.0, cycle_index=None, gesture_repetition_index=None,
                             gesture_repetition_count=None, **base)
        within = min(max(base["elapsed"] - phase.start, 0.0), phase.duration)
        repetitions = (plan.requested.get(phase.motion_class)
                       if phase.motion_class != REST_ID else None)
        position = position_for(phase.phase, within / phase.duration)
        then = self._direct_successor(phase) if phase.phase in (HOLD, RAMP_OUT) else None
        return PhaseView(
            then_motion=then.motion_class if then else None,
            then_intensity_level=then.intensity if then else None,
            position=position, intensity_level=phase.intensity,
            intensity=None if phase.intensity is None else position * phase.intensity,
            phase=phase.phase,
            display_phase=PAUSED if self.state == PAUSED_STATE else phase.phase,
            motion_class=phase.motion_class, progress=within / phase.duration,
            remaining=phase.duration - within, cycle_index=phase.cycle_index,
            gesture_repetition_index=phase.gesture_repetition_index,
            gesture_repetition_count=repetitions,
            **{**base, "next_motion": phase.next_motion})


__all__ = ["build_plan", "plan_cycles", "ChoreographyPlan", "ChoreographyScheduler",
           "PlannedPhase", "PlannedRepetition", "PhaseView", "derived_label", "position_for",
           "WARMUP", "RAMP_IN", "HOLD", "RAMP_OUT", "REST", "CYCLE_PAUSE", "PAUSED",
           "ACTIVE_PHASES", "RESTING_PHASES", "PHASES", "PHASE_NAMES",
           "READY", "RUNNING", "PAUSED_STATE", "COMPLETED", "CANCELLED", "FAILED", "TERMINAL",
           "SEGMENT_COMPLETED", "SEGMENT_PAUSED", "SEGMENT_INTERRUPTED", "SEGMENT_CANCELLED",
           "SEGMENT_FAILED"]
