"""Visual pipeline construction for LibEMG.

Three layers, deliberately separated so that only the last one needs a display:

- :mod:`~libemg._gui._pipeline.registry` describes the blocks that can be
  placed, generated from the library so it does not rot as LibEMG gains
  features and devices.
- :mod:`~libemg._gui._pipeline.document` is the pipeline itself as plain data,
  which is what gets saved and loaded.
- :mod:`~libemg._gui._pipeline.compile` turns a document into something that
  runs.

A document can be built, saved, loaded, validated and run without DearPyGui
ever being imported. The editor is a view onto it, not the thing itself.
"""

from libemg._gui._pipeline.document import (Link, Node, PipelineDocument, Probe,
                                            SCHEMA_VERSION)
from libemg._gui._pipeline.registry import (NodeSpec, ParamSpec, PortSpec,
                                            PortType, build_registry,
                                            default_registry)

__all__ = ["Link", "Node", "PipelineDocument", "Probe", "SCHEMA_VERSION",
           "NodeSpec", "ParamSpec", "PortSpec", "PortType", "build_registry",
           "default_registry"]
