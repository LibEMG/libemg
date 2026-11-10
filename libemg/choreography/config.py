"""What a choreography run is asked to do: which gestures, how often, for how long.

A :class:`ChoreographyConfig` is plain data. The configuration panel edits one,
presets save and load one, and a run freezes a copy (:meth:`snapshot`) the
moment it starts, so editing the panel mid-run cannot change the run that is
already going -- or what the session says it was collected under.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 2: intensity targets (intensity_enabled, intensity_levels, intensity_order).
# 3: rest_between_gestures (off: direct transitions from one gesture to the next).
# Older presets load unchanged, with intensity off and rests between gestures.
CONFIG_SCHEMA_VERSION = 3

FIXED = "fixed"
SHUFFLE = "shuffle"
ORDER_MODES = (FIXED, SHUFFLE)

#: How intensity levels are handed out to a gesture's repetitions. ``in_turn``
#: gives repetition ``r`` the level ``levels[r % len(levels)]``; ``shuffle``
#: deals the same balanced set in a seeded random order, per gesture.
IN_TURN = "in_turn"
INTENSITY_ORDERS = (IN_TURN, SHUFFLE)

#: The per-gesture timings a gesture may override. Warm-up and the cycle pause
#: belong to the run, not to any one gesture.
TIMING_KEYS = ("ramp_in", "hold", "ramp_out", "rest")
#: Phases whose length has to be above zero: a gesture that is never ramped into
#: is a jump cut, which the participant cannot follow.
NONZERO_KEYS = ("ramp_in", "ramp_out")


def new_seed():
    """A fresh shuffle seed, small enough to read out and type back in."""
    return random.SystemRandom().randrange(1, 2**31 - 1)


@dataclass
class ChoreographyConfig:
    """The settings for one choreography run.

    Attributes
    ----------
    gestures: list of str
        Selected motion IDs, in the order the panel lists them. A fixed order
        runs them in exactly this order every cycle; a shuffle starts from it.
    order_mode: str
        ``'fixed'`` or ``'shuffle'``.
    shuffle_seed: int
        Seed for the shuffle. Always explicit, so the order can be reproduced.
    warmup, ramp_in, hold, ramp_out, rest, cycle_pause: float
        Seconds. ``rest`` is the gap after each gesture's ramp-out, within a
        cycle and (unless the cycle pause is on) between cycles.
    repetitions: int
        How many times each gesture is collected, unless overridden.
    repetition_overrides: dict
        Motion ID to its own repetition count.
    timing_overrides: dict
        Motion ID to a dict of any of ``ramp_in``, ``hold``, ``ramp_out``,
        ``rest``. The ``rest`` override is the rest that follows that gesture.
    cycle_pause_enabled: bool
        Use ``cycle_pause`` instead of ``rest`` between cycles (never both).
    rest_between_gestures: bool
        Wait at rest for ``rest`` seconds after each ramp-out. Off, every
        ramp-out goes straight into the next gesture's ramp-in: the hand
        passes through rest without stopping there, so the recording holds
        direct transitions from one active gesture to the next. Rest is then
        collected only in the warm-up and the cycle pauses, one of which has
        to be kept, and the ``rest`` timing and rest overrides are unused.
    parent_regions: bool
        Also annotate each repetition as a whole, ramp-in to ramp-out, on its
        own track.
    intensity_enabled: bool
        Ask for each repetition at a target intensity (a fraction of maximum
        effort), shown as a bar beside the hand and recorded as a label.
    intensity_levels: list of float
        The target intensities, each above 0 and at most 1.
    intensity_order: str
        ``'in_turn'`` or ``'shuffle'``: how the levels are dealt out to each
        gesture's repetitions. A shuffle is seeded from ``shuffle_seed``.
    """

    gestures: list = field(default_factory=list)
    order_mode: str = FIXED
    shuffle_seed: int = field(default_factory=new_seed)
    warmup: float = 3.0
    ramp_in: float = 0.7
    hold: float = 3.0
    ramp_out: float = 0.7
    rest: float = 2.0
    repetitions: int = 3
    repetition_overrides: dict = field(default_factory=dict)
    cycle_pause_enabled: bool = False
    rest_between_gestures: bool = True
    cycle_pause: float = 10.0
    timing_overrides: dict = field(default_factory=dict)
    parent_regions: bool = True
    intensity_enabled: bool = False
    intensity_levels: list = field(default_factory=lambda: [1.0])
    intensity_order: str = IN_TURN
    schema_version: int = CONFIG_SCHEMA_VERSION

    # -- reading it -----------------------------------------------------------

    def repetitions_for(self, motion_id):
        return int(self.repetition_overrides.get(motion_id, self.repetitions))

    def timing_for(self, motion_id):
        """``{ramp_in, hold, ramp_out, rest}`` for one gesture, overrides applied."""
        own = self.timing_overrides.get(motion_id, {})
        return {key: float(own.get(key, getattr(self, key))) for key in TIMING_KEYS}

    def requested_repetitions(self):
        """Motion ID to repetition count, for every selected gesture."""
        return {g: self.repetitions_for(g) for g in self.gestures}

    def intensities_for(self, motion_id):
        """The target intensity of each of a gesture's repetitions, in order.

        None for every repetition when intensity is off. The levels are dealt
        out as evenly as the count allows: with three levels and five
        repetitions, two levels are used twice and one once.
        """
        count = self.repetitions_for(motion_id)
        if not self.intensity_enabled:
            return [None] * count
        levels = [float(level) for level in self.intensity_levels]
        dealt = [levels[r % len(levels)] for r in range(count)]
        if self.intensity_order == SHUFFLE:
            # Seeded per gesture, so adding a gesture does not reshuffle the others.
            random.Random(f"{self.shuffle_seed}:intensity:{motion_id}").shuffle(dealt)
        return dealt

    # -- checking it ----------------------------------------------------------

    def validate(self, known_gestures=None):
        """Everything wrong with these settings, as sentences. Empty means it can run.

        Parameters
        ----------
        known_gestures: iterable of str, optional
            The selectable motion IDs. A gesture outside it is refused, which
            is what catches ``rest`` or a preset made for a different asset.
        """
        problems = []
        if not self.gestures:
            problems.append("Select at least one gesture.")
        if len(set(self.gestures)) != len(self.gestures):
            problems.append("A gesture is listed more than once.")
        if known_gestures is not None:
            known = set(known_gestures)
            unknown = [g for g in self.gestures if g not in known]
            if unknown:
                problems.append(f"Not selectable motions: {', '.join(unknown)}.")
        if self.order_mode not in ORDER_MODES:
            problems.append(f"Order must be one of {', '.join(ORDER_MODES)}.")
        if not _is_int(self.shuffle_seed):
            problems.append("The shuffle seed must be a whole number.")

        def check_duration(name, value, nonzero=False):
            if not _is_number(value) or not math.isfinite(float(value)):
                problems.append(f"{_readable(name)} must be a finite number of seconds.")
            elif float(value) < 0:
                problems.append(f"{_readable(name)} cannot be negative.")
            elif nonzero and float(value) == 0:
                problems.append(f"{_readable(name)} must be longer than zero.")

        for name in ("warmup", *TIMING_KEYS, "cycle_pause"):
            check_duration(name, getattr(self, name), nonzero=name in NONZERO_KEYS)
        if not _is_positive_int(self.repetitions):
            problems.append("Repetitions must be a whole number of at least 1.")
        for motion_id, count in self.repetition_overrides.items():
            if not _is_positive_int(count):
                problems.append(f"Repetitions for {motion_id} must be a whole number of at least 1.")
        if not self.rest_between_gestures and not (
                _positive(self.warmup)
                or (self.cycle_pause_enabled and _positive(self.cycle_pause))):
            problems.append("With no rest between gestures, keep a warm-up or a pause between "
                            "cycles, so the rest class is still collected deliberately.")
        if self.intensity_enabled:
            levels = self.intensity_levels
            if not isinstance(levels, (list, tuple)) or not levels:
                problems.append("Give at least one intensity level.")
            elif not all(_is_number(v) and math.isfinite(float(v)) and 0 < float(v) <= 1
                         for v in levels):
                problems.append("Intensity levels must be above 0% and at most 100%.")
            if self.intensity_order not in INTENSITY_ORDERS:
                problems.append(f"Intensity order must be one of {', '.join(INTENSITY_ORDERS)}.")
        for motion_id, own in self.timing_overrides.items():
            for key, value in own.items():
                if key not in TIMING_KEYS:
                    problems.append(f"{motion_id}: {key} is not a timing that can be overridden.")
                    continue
                before = len(problems)
                check_duration(key, value, nonzero=key in NONZERO_KEYS)
                for index in range(before, len(problems)):
                    problems[index] = f"{motion_id}: {problems[index]}"
        return problems

    # -- moving it around -----------------------------------------------------

    def snapshot(self):
        """A frozen deep copy, for a run to hold while the panel keeps editing."""
        return copy.deepcopy(self)

    def to_dict(self):
        raw = asdict(self)
        raw["gestures"] = list(self.gestures)
        return raw

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        schema = int(raw.pop("schema_version", CONFIG_SCHEMA_VERSION))
        if schema > CONFIG_SCHEMA_VERSION:
            raise ValueError(f"This preset was written by a newer LibEMG (schema {schema}).")
        allowed = set(cls.__dataclass_fields__) - {"schema_version"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown choreography settings: {', '.join(sorted(unknown))}.")
        config = cls(**{k: v for k, v in raw.items() if k in allowed})
        config.gestures = [str(g) for g in config.gestures]
        config.repetition_overrides = {str(k): v for k, v in config.repetition_overrides.items()}
        config.timing_overrides = {str(k): dict(v) for k, v in config.timing_overrides.items()}
        config.intensity_levels = list(config.intensity_levels)
        return config

    def save(self, path):
        """Write this configuration as a JSON preset."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": "libemg.choreography.preset", **self.to_dict()},
                                   indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("kind", None)
        return cls.from_dict(raw)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive(value):
    return _is_number(value) and math.isfinite(float(value)) and float(value) > 0


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value):
    return _is_int(value) and value >= 1


def _readable(name):
    return {"warmup": "Warm-up", "ramp_in": "Ramp-in", "hold": "Hold", "ramp_out": "Ramp-out",
            "rest": "Rest", "cycle_pause": "Cycle pause"}.get(name, name)


__all__ = ["ChoreographyConfig", "FIXED", "SHUFFLE", "ORDER_MODES", "IN_TURN",
           "INTENSITY_ORDERS", "TIMING_KEYS",
           "CONFIG_SCHEMA_VERSION", "new_seed"]
