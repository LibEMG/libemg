"""3D hand choreography: an animated hand that shows the participant what to do.

The pieces, from the bottom up:

:mod:`~libemg.choreography.assets`
    The packaged hand model, its manifest and the versioned class-ID map.
:mod:`~libemg.choreography.config`
    What a run is asked to do, its validation and presets.
:mod:`~libemg.choreography.schedule`
    The ordered plan built from a configuration, and the scheduler that walks
    it against a monotonic clock -- the single source of truth for what is on
    screen, which phase it is, and where every annotation boundary falls.
:mod:`~libemg.choreography.annotations`
    Turns the scheduler's event stream into phase and repetition intervals,
    maps them onto the recording's clock and sample indices, and writes them.
:mod:`~libemg.choreography.targets`
    Per-sample regression targets (class, phase, position, intensity) written
    beside the recording, and readers and a per-repetition cutter for them.
:mod:`~libemg.choreography.gltf` and :mod:`~libemg.choreography.renderer`
    Skeletal animation, skinning and an offscreen renderer for Dear PyGui.
:mod:`~libemg.choreography.controller`
    Ties one run together: scheduler, recording, stall and disconnection
    handling. No GUI code, so it is tested with a fake clock.

The GUI lives in :mod:`libemg._gui._choreography_panel`.
"""

from .assets import ChoreographyAssets, Gesture, load_assets
from .config import FIXED, IN_TURN, SHUFFLE, ChoreographyConfig
from .targets import Targets, cut_repetitions, load_targets

__all__ = ["ChoreographyAssets", "Gesture", "load_assets", "ChoreographyConfig", "FIXED", "SHUFFLE",
           "IN_TURN", "Targets", "load_targets", "cut_repetitions"]
