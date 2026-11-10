"""Running LibEMG's pygame environments inside the DearPyGui window.

An environment draws offscreen in its own process and writes its frames into
memory the GUI's texture is backed by, so the game appears in a panel rather
than in a window of its own. See
:mod:`~libemg._gui._environments.frame_bridge` for how frames and input cross
the boundary, and :mod:`~libemg._gui._environments.registry` for how each
environment's setup screen is generated from its own configuration.
"""

from libemg._gui._environments.embedded import EmbeddedEnvironment, EnvironmentRunner
from libemg._gui._environments.factories import ControllerSpec, build_factory
from libemg._gui._environments.frame_bridge import FORWARDED_KEYS, FrameBridge
from libemg._gui._environments.registry import (EnvironmentSpec, Setting,
                                                build_registry, default_registry)

__all__ = ["EmbeddedEnvironment", "EnvironmentRunner", "ControllerSpec",
           "build_factory", "FrameBridge", "FORWARDED_KEYS",
           "EnvironmentSpec", "Setting", "build_registry", "default_registry"]
