"""Runtime resources that ship inside the libemg package.

Anything the application needs at run time -- a 3D model, a manifest that
describes it -- lives under this package rather than at a path on the machine it
was authored on, so an installed copy of libemg finds the same files a source
checkout does. Read them through :func:`resource_path` or
:func:`read_resource_bytes`, never by building a path from ``__file__`` in
another module.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """The on-disk path of a packaged resource.

    Parameters
    ----------
    *parts: str
        Path components below ``libemg/resources``, e.g.
        ``resource_path("choreography", "right_hand.glb")``.

    Returns
    ----------
    Path
        The resource's location. libemg is always installed as a directory
        (never imported from a zip), so this is a real file.
    """
    target = resources.files(__name__).joinpath(*parts)
    return Path(str(target))


def read_resource_bytes(*parts: str) -> bytes:
    """The contents of a packaged resource."""
    return resources.files(__name__).joinpath(*parts).read_bytes()


__all__ = ["resource_path", "read_resource_bytes"]
