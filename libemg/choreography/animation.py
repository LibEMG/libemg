"""Which frame of which clip to show for a scheduler phase.

Every motion clip in the asset runs the same 2.5 s contract::

    0.0 rest   0.3 rest   1.0 target   1.5 end of hold   2.2 back at rest   2.5 rest

The scheduler's phases are mapped onto the matching *interval* of that clip, each
on its own:

=========  ===========================
ramp_in    0.3 -> 1.0 s
hold       1.0 -> 1.5 s
ramp_out   1.5 -> 2.2 s
rest       0.0 s (held)
=========  ===========================

So a 4 s hold plays the clip's 0.5 s hold interval over 4 s while the ramps keep
whatever length they were given -- lengthening one phase never stretches
another, which is what simply changing the playback speed of the whole clip
would do.

Linear motion
-------------
The authored clips are eased: a ramp starts and ends slowly, so a fifth of the
way through the ramp time the hand has covered only about a tenth of the
distance. The annotation's ``position`` target is linear in time (see
:func:`~libemg.choreography.schedule.position_for`), so the display is re-timed
to match it: :class:`MotionProfile` measures, per clip, how far the skinned
hand has moved from rest toward the target at each instant of each ramp, and
inverts that curve. With it applied the hand covers a fraction of its rest ->
target displacement equal to the ``position`` label at every instant; the hold
and the rest are untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .assets import (CLIP_HOLD_END, CLIP_RAMP_IN_START, CLIP_REST, CLIP_RETURNED,
                     CLIP_TARGET, REST_ID)
from .schedule import HOLD, RAMP_IN, RAMP_OUT

SOURCE_INTERVALS = {
    RAMP_IN: (CLIP_RAMP_IN_START, CLIP_TARGET),
    HOLD: (CLIP_TARGET, CLIP_HOLD_END),
    RAMP_OUT: (CLIP_HOLD_END, CLIP_RETURNED),
}


def clip_time(phase, progress):
    """Seconds into the motion clip for ``progress`` (0..1) through ``phase``."""
    interval = SOURCE_INTERVALS.get(phase)
    if interval is None:
        return CLIP_REST
    progress = min(max(float(progress), 0.0), 1.0)
    start, end = interval
    return start + progress * (end - start)


def pose_for(view):
    """``(clip name, clip time)`` to render for a :class:`~libemg.choreography.schedule.PhaseView`.

    Resting phases, a run that has not started, and a run that has ended all show
    the relaxed hand. A paused run shows exactly the frame it was paused on.
    """
    if view.phase in SOURCE_INTERVALS and view.motion_class != REST_ID:
        return view.motion_class, clip_time(view.phase, view.progress)
    return REST_ID, CLIP_REST


class MotionProfile:
    """Re-times each clip's ramps so the displayed hand moves linearly in time.

    For every clip, ``fraction(t)`` is the projection of the skinned hand's
    displacement from rest onto its full rest -> target displacement (0 at
    rest, 1 at the target), sampled through each ramp. :meth:`clip_time` maps a
    nominal (linear) clip time to the clip time at which that fraction equals
    the linear progress.
    """

    #: Samples per ramp, every ``VERTEX_STRIDE``-th vertex: enough to hold the
    #: inverse to well under a percent, cheap enough to measure at start-up.
    SAMPLES = 33
    VERTEX_STRIDE = 6
    METRIC = ("fraction of the rest->target vertex displacement covered (each vertex's "
              "displacement from rest projected onto its rest->target displacement, summed)")

    def __init__(self, tables):
        self.tables = tables            # clip -> {RAMP_IN: (fractions, times), RAMP_OUT: ...}

    @classmethod
    def measure(cls, model, clips=None, rest_clip=REST_ID, samples=None, stride=None):
        samples = samples or cls.SAMPLES
        stride = stride or cls.VERTEX_STRIDE

        def cut(array):
            return None if array is None else array[::stride]

        subsets = [SimpleNamespace(positions=cut(prim.positions), normals=cut(prim.normals),
                                   weights=cut(prim.weights), joints=cut(prim.joints),
                                   skin=prim.skin, node=prim.node)
                   for prim in model.primitives]

        def vertices(clip, t):
            world, joints = model.pose(clip, t)
            return np.vstack([model.skin_vertices(sub, world, joints)[0] for sub in subsets]
                             ).astype(np.float64)

        rest = vertices(rest_clip if rest_clip in model.clips else None, CLIP_REST)
        tables = {}
        for clip in (clips if clips is not None else list(model.clips)):
            if clip == rest_clip or clip not in model.clips:
                continue
            displacement = vertices(clip, CLIP_TARGET) - rest
            total = float((displacement * displacement).sum())
            if total <= 1e-12:
                continue
            table = {}
            for phase in (RAMP_IN, RAMP_OUT):
                start, end = SOURCE_INTERVALS[phase]
                times = np.linspace(start, end, samples)
                fractions = np.array([float(((vertices(clip, t) - rest) * displacement).sum())
                                      / total for t in times])
                fractions = np.clip(fractions, 0.0, 1.0)
                if phase == RAMP_OUT:
                    # Falling 1 -> 0: stored rising, for interpolation.
                    times, fractions = times[::-1], fractions[::-1]
                table[phase] = (np.maximum.accumulate(fractions), times)
            tables[clip] = table
        return cls(tables)

    def clip_time(self, clip, nominal):
        """The clip time to show for nominal (linear) clip time ``nominal``."""
        table = self.tables.get(clip)
        if table is None:
            return nominal
        for phase in (RAMP_IN, RAMP_OUT):
            start, end = SOURCE_INTERVALS[phase]
            if start <= nominal <= end:
                progress = (nominal - start) / (end - start)
                wanted = progress if phase == RAMP_IN else 1.0 - progress
                fractions, times = table[phase]
                if fractions[-1] - fractions[0] <= 1e-9:
                    return nominal
                return float(np.interp(wanted, fractions, times))
        return nominal

    def easing(self):
        """Per clip, the largest gap (0..1) between the authored ramp and a linear one."""
        out = {}
        for clip, table in self.tables.items():
            worst = 0.0
            for phase, (fractions, times) in table.items():
                start, end = SOURCE_INTERVALS[phase]
                linear = (times - start) / (end - start)
                if phase == RAMP_OUT:
                    linear = 1.0 - linear
                worst = max(worst, float(np.abs(fractions - linear).max()))
            out[clip] = round(worst, 4)
        return out

    def record(self):
        """What the run record says about how the hand was moved."""
        return {"motion_profile": "linear", "metric": self.METRIC,
                "authored_easing_corrected": self.easing(),
                "samples_per_ramp": self.SAMPLES, "vertex_stride": self.VERTEX_STRIDE}


__all__ = ["clip_time", "pose_for", "SOURCE_INTERVALS", "MotionProfile"]
