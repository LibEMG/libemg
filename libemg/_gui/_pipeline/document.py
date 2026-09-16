"""A pipeline as plain data.

This is what gets saved, loaded, diffed and validated. It imports neither
DearPyGui nor the LibEMG runtime, so a pipeline can be built and checked in a
test with no display and nothing running.

Surviving a version change
--------------------------
The registry is generated from the library, so it legitimately differs between
LibEMG versions. A file written against a newer version will name blocks this
one has never heard of. Refusing to open it, or opening it and quietly dropping
what it did not recognise, both lose the user's work. Instead an unknown node
loads as *unresolved*: it keeps its id, its parameters and its position, the
editor marks it, and saving puts it back exactly as it came in. The pipeline
will not run until it is dealt with, but nothing is destroyed by looking at it.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from libemg._gui._pipeline.registry import (MODEL, SINK, SOURCE, WINDOW, NodeSpec,
                                            PortType, default_registry)

#: Bumped when the saved shape changes. Migrations are keyed off it.
SCHEMA_VERSION = 1


@dataclass
class Node:
    """One placed block.

    Attributes
    ----------
    id: str
        Unique within the document.
    spec_id: str
        Which :class:`~libemg._gui._pipeline.registry.NodeSpec` this is.
    params: dict
        Configured values, keyed by parameter name.
    position: tuple
        Canvas position, so layout survives a save.
    unresolved: bool
        Set when the registry has no such spec. The node is preserved verbatim
        and the document will not compile until it is removed or the right
        LibEMG version is installed.
    """

    id: str
    spec_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    position: tuple = (0, 0)
    unresolved: bool = False

    def to_dict(self):
        return {"id": self.id, "spec_id": self.spec_id, "params": dict(self.params),
                "position": list(self.position)}

    @classmethod
    def from_dict(cls, raw, registry):
        spec_id = raw.get("spec_id", "")
        node = cls(id=raw.get("id", ""), spec_id=spec_id,
                   params=dict(raw.get("params", {})),
                   position=tuple(raw.get("position", (0, 0))),
                   unresolved=spec_id not in registry)
        if not node.unresolved:
            node.params = _coerce(registry[spec_id], node.params)
        return node


@dataclass
class Link:
    """A connection between one node's output and another's input."""

    from_node: str
    from_port: str
    to_node: str
    to_port: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        return cls(raw["from_node"], raw["from_port"], raw["to_node"], raw["to_port"])

    def key(self):
        return (self.from_node, self.from_port, self.to_node, self.to_port)


@dataclass
class Probe:
    """A request to watch one output port while the pipeline runs.

    A probe is not a node. Making it one would clutter the canvas and force the
    user to wire up something they only want to look at, so it attaches to a
    port instead.

    Attributes
    ----------
    node: str
        The node whose output is watched.
    port: str
        Which output.
    hz: float
        Maximum updates per second. A probe that cannot keep up with its source
        is still a probe; it just shows less.
    """

    node: str
    port: str
    hz: float = 30.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        return cls(raw["node"], raw["port"], float(raw.get("hz", 30.0)))

    def key(self):
        return (self.node, self.port)


def _coerce(spec, params):
    """Bring a node's stored parameters into the shapes its spec declares."""
    out = spec.defaults()
    for name, value in params.items():
        param = spec.param(name)
        # A parameter the spec no longer declares is kept as-is rather than
        # dropped, so downgrading LibEMG and upgrading again does not lose it.
        out[name] = param.coerce(value) if param is not None else value
    return out


class ValidationError(Exception):
    """Raised when a document cannot be run, with every reason found."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(f"- {p}" for p in self.problems))


class PipelineDocument:
    """A pipeline, as a thing you can edit, save, load and check.

    Parameters
    ----------
    registry: dict or None (optional), default=None
        The block descriptions to validate against. Defaults to the generated
        registry.

    Examples
    ---------
    >>> doc = PipelineDocument()
    >>> source = doc.add_node('source.myo_streamer')
    >>> window = doc.add_node('window.enframe')
    >>> doc.connect(source, 'emg', window, 'input')
    >>> doc.save('pipeline.json')
    """

    def __init__(self, registry=None):
        self.registry = registry if registry is not None else default_registry()
        self.nodes: Dict[str, Node] = {}
        self.links: List[Link] = []
        self.probes: List[Probe] = []
        self.canvas: Dict[str, Any] = {}
        self._counter = 0

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------
    def add_node(self, spec_id, params=None, position=(0, 0), node_id=None):
        """Place a block.

        Returns
        ----------
        str
            The new node's id.
        """
        if spec_id not in self.registry:
            raise KeyError(f"No such block: '{spec_id}'.")
        spec = self.registry[spec_id]
        node_id = node_id or self._next_id(spec_id)
        if node_id in self.nodes:
            raise ValueError(f"A node called '{node_id}' already exists.")
        node = Node(id=node_id, spec_id=spec_id, position=tuple(position))
        node.params = _coerce(spec, params or {})
        self.nodes[node_id] = node
        return node_id

    def _next_id(self, spec_id):
        stem = spec_id.split(".")[-1]
        while True:
            self._counter += 1
            candidate = f"{stem}_{self._counter}"
            if candidate not in self.nodes:
                return candidate

    def remove_node(self, node_id):
        """Remove a block, and anything attached to it."""
        self.nodes.pop(node_id, None)
        self.links = [l for l in self.links
                      if l.from_node != node_id and l.to_node != node_id]
        self.probes = [p for p in self.probes if p.node != node_id]

    def spec(self, node_id):
        """The :class:`NodeSpec` for a node, or None if it is unresolved."""
        node = self.nodes[node_id]
        return self.registry.get(node.spec_id)

    def set_param(self, node_id, name, value):
        """Set one parameter, coerced to its declared kind."""
        node = self.nodes[node_id]
        spec = self.registry.get(node.spec_id)
        param = spec.param(name) if spec else None
        node.params[name] = param.coerce(value) if param else value

    def connect(self, from_node, from_port, to_node, to_port):
        """Link an output to an input.

        Raises
        ----------
        ValueError
            If the link is not one the pipeline could run: mismatched types,
            an input that already has a link and does not accept several, a
            duplicate, or a cycle.
        """
        problem = self.why_not_connect(from_node, from_port, to_node, to_port)
        if problem:
            raise ValueError(problem)
        self.links.append(Link(from_node, from_port, to_node, to_port))

    def why_not_connect(self, from_node, from_port, to_node, to_port):
        """Why a link would be refused, or None if it would be accepted.

        Separate from :meth:`connect` because the editor needs to explain a
        refusal as the user drags, rather than raise at them.
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            return "One end of that link is not on the canvas."
        if from_node == to_node:
            return "A block cannot feed itself."
        source, target = self.registry.get(self.nodes[from_node].spec_id), \
            self.registry.get(self.nodes[to_node].spec_id)
        if source is None or target is None:
            return "One end of that link is a block this version does not recognise."
        out_port = source.port(from_port, "output")
        in_port = target.port(to_port, "input")
        if out_port is None:
            return f"'{source.title}' has no output called '{from_port}'."
        if in_port is None:
            return f"'{target.title}' has no input called '{to_port}'."
        if not PortType.accepts(out_port.type, in_port.type):
            return (f"'{out_port.label}' carries {out_port.type} and "
                    f"'{in_port.label}' expects {in_port.type}.")
        if any(l.key() == (from_node, from_port, to_node, to_port) for l in self.links):
            return "Those are already connected."
        if not in_port.multiple and self.links_into(to_node, to_port):
            return (f"'{in_port.label}' already has a connection, and two sources "
                    "into one input would interleave with no defined order.")
        if self._would_cycle(from_node, to_node):
            return "That would make a loop, and each stage would wait for the other."
        return None

    def disconnect(self, from_node, from_port, to_node, to_port):
        """Remove a link if it exists."""
        key = (from_node, from_port, to_node, to_port)
        self.links = [l for l in self.links if l.key() != key]

    def links_into(self, node_id, port=None):
        """Links arriving at a node, optionally at one port."""
        return [l for l in self.links
                if l.to_node == node_id and (port is None or l.to_port == port)]

    def links_out_of(self, node_id, port=None):
        """Links leaving a node, optionally from one port."""
        return [l for l in self.links
                if l.from_node == node_id and (port is None or l.from_port == port)]

    def _would_cycle(self, from_node, to_node):
        """Whether adding from_node -> to_node closes a loop."""
        seen, stack = set(), [from_node]
        while stack:
            current = stack.pop()
            if current == to_node:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(l.from_node for l in self.links_into(current))
        return False

    # ------------------------------------------------------------------
    # probes
    # ------------------------------------------------------------------
    def add_probe(self, node_id, port, hz=30.0):
        """Watch an output port. Replaces any probe already on that port.

        Raises
        ----------
        ValueError
            If there is no such output, or if it is one that never becomes an
            item of its own and so could not be watched.
        """
        spec = self.registry.get(self.nodes[node_id].spec_id)
        if spec is None or spec.port(port, "output") is None:
            raise ValueError(f"'{node_id}' has no output called '{port}' to probe.")
        if spec.category == WINDOW:
            # A window is folded into the stage it feeds rather than becoming a
            # stage of its own, so it publishes nothing to watch. Accepting the
            # probe and quietly dropping it at compile time would leave an
            # empty plot with no explanation, and would also let it count as a
            # reader and mask a genuinely dangling branch.
            raise ValueError(
                f"A window cannot be probed. It is folded into the block it feeds "
                f"rather than producing anything of its own, so there is nothing to "
                f"watch. Probe the block before it to see the samples going in, or "
                f"the block after it to see what comes out.")
        self.remove_probe(node_id, port)
        self.probes.append(Probe(node_id, port, hz))

    def remove_probe(self, node_id, port):
        self.probes = [p for p in self.probes if p.key() != (node_id, port)]

    def has_probe(self, node_id, port):
        return any(p.key() == (node_id, port) for p in self.probes)

    def probe_render(self, probe):
        """How a probe should be drawn, from the type of the port it watches."""
        spec = self.registry.get(self.nodes[probe.node].spec_id)
        port = spec.port(probe.port, "output") if spec else None
        return PortType.RENDER.get(port.type if port else None, "timeseries")

    # ------------------------------------------------------------------
    # classification and validation
    # ------------------------------------------------------------------
    def sources(self):
        """Node ids whose spec is a source."""
        return [n for n, node in self.nodes.items()
                if (self.registry.get(node.spec_id) or NodeSpec("", "", "")).category == SOURCE]

    def mode(self):
        """Whether this pipeline runs online, offline, or cannot be decided.

        Inferred rather than configured. A pipeline fed by devices is online, a
        pipeline fed by recordings is offline, and one fed by both is neither.

        Returns
        ----------
        str
            ``'online'``, ``'offline'``, ``'empty'`` or ``'mixed'``.
        """
        sources = self.sources()
        if not sources:
            return "empty"
        offline = {s for s in sources if self.registry[self.nodes[s].spec_id].offline_only}
        if not offline:
            return "online"
        if len(offline) == len(sources):
            return "offline"
        return "mixed"

    def validate(self):
        """Every reason this pipeline could not run.

        Returns
        ----------
        list
            Human-readable problems. Empty means it is ready.
        """
        problems = []
        unresolved = [n.id for n in self.nodes.values() if n.unresolved]
        if unresolved:
            problems.append(
                f"These blocks are not recognised by this version of LibEMG: "
                f"{', '.join(sorted(unresolved))}. They were kept so nothing is lost, "
                "but the pipeline cannot run until they are removed.")

        mode = self.mode()
        if mode == "empty":
            problems.append("There is no source, so nothing would ever run.")
        elif mode == "mixed":
            live = [s for s in self.sources()
                    if not self.registry[self.nodes[s].spec_id].offline_only]
            stored = [s for s in self.sources() if s not in live]
            problems.append(
                f"This mixes live sources ({', '.join(sorted(live))}) with stored data "
                f"({', '.join(sorted(stored))}). A run is either one or the other.")

        for node_id, node in self.nodes.items():
            spec = self.registry.get(node.spec_id)
            if spec is None:
                continue
            for port in spec.inputs:
                if port.optional:
                    continue
                if not self.links_into(node_id, port.name):
                    problems.append(f"'{spec.title}' ({node_id}) has nothing connected "
                                    f"to its {port.label} input.")
            if spec.category == MODEL:
                # A model reads features or raw windows, never both and never
                # neither. Its ports are optional individually so the generic
                # rule above does not demand both, so the real requirement is
                # stated here.
                connected = [p.label for p in spec.inputs
                             if self.links_into(node_id, p.name)]
                if not connected:
                    problems.append(
                        f"'{spec.title}' ({node_id}) has no input. Connect features "
                        "to it, or connect a window directly for a model that takes "
                        "raw windows.")
                elif len(connected) > 1:
                    problems.append(
                        f"'{spec.title}' ({node_id}) has both {' and '.join(connected)} "
                        "connected. A model reads one or the other, not both.")
            if mode == "offline" and spec.online_only:
                problems.append(f"'{spec.title}' ({node_id}) only works on a live stream.")
            if mode == "online" and spec.offline_only:
                problems.append(f"'{spec.title}' ({node_id}) only works on stored data.")

        # A source feeding nothing, or a model whose output goes nowhere, is
        # almost always a half-finished edit rather than an intention. A sink
        # is the exception: its output is the result the user reads at the end
        # of a run, so having nothing downstream of it is the normal case.
        for node_id, node in self.nodes.items():
            spec = self.registry.get(node.spec_id)
            if spec is None or not spec.outputs or spec.category == SINK:
                continue
            if not self.links_out_of(node_id) and not any(
                    p.node == node_id for p in self.probes):
                problems.append(f"'{spec.title}' ({node_id}) produces something that "
                                "nothing reads and no probe watches.")

        # Something has to make the result visible. A model or a sink does, and
        # so does a probe: watching a stage live is a legitimate pipeline on
        # its own while exploring, and refusing it would mean a pipeline could
        # not be built up a block at a time.
        visible = any(
            (self.registry.get(n.spec_id) or NodeSpec("", "", "")).category in (SINK, MODEL)
            for n in self.nodes.values()) or bool(self.probes)
        if mode in ("online", "offline") and not visible:
            problems.append("There is no model, output or probe, so the pipeline "
                            "would compute nothing anybody could see.")
        return problems

    def check(self):
        """Validate, raising :class:`ValidationError` if anything is wrong."""
        problems = self.validate()
        if problems:
            raise ValidationError(problems)
        return True

    def order(self):
        """Node ids in an order where every node follows its inputs.

        Returns
        ----------
        list
            A topological order. Cycles cannot occur because
            :meth:`connect` refuses them.
        """
        remaining = dict(self.nodes)
        resolved, out = set(), []
        while remaining:
            ready = [n for n in remaining
                     if all(l.from_node in resolved for l in self.links_into(n))]
            if not ready:
                # Only reachable if links were built by hand around connect().
                out.extend(sorted(remaining))
                break
            for node_id in sorted(ready):
                out.append(node_id)
                resolved.add(node_id)
                remaining.pop(node_id)
        return out

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "links": [l.to_dict() for l in self.links],
            "probes": [p.to_dict() for p in self.probes],
            "canvas": dict(self.canvas),
        }

    def save(self, path):
        """Write the pipeline to a file."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def from_dict(cls, raw, registry=None):
        """Rebuild from saved data, preserving anything not recognised."""
        raw = migrate(raw)
        document = cls(registry=registry)
        for entry in raw.get("nodes", []):
            node = Node.from_dict(entry, document.registry)
            document.nodes[node.id] = node
        known = set(document.nodes)
        for entry in raw.get("links", []):
            link = Link.from_dict(entry)
            # A link to a node that is not in the file is meaningless, and
            # keeping it would make the document lie about its own shape.
            if link.from_node in known and link.to_node in known:
                document.links.append(link)
        for entry in raw.get("probes", []):
            probe = Probe.from_dict(entry)
            if probe.node in known:
                document.probes.append(probe)
        document.canvas = dict(raw.get("canvas", {}))
        return document

    @classmethod
    def load(cls, path, registry=None):
        """Read a pipeline from a file."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), registry=registry)


def migrate(raw):
    """Bring saved data up to the current schema.

    One step per version, applied in order, so a file several versions old is
    carried forward rather than rejected.
    """
    version = int(raw.get("schema_version", 1))
    steps = {}
    while version in steps:
        raw = steps[version](raw)
        version += 1
        raw["schema_version"] = version
    # Stamped on the way out whatever happened, so a caller can always read the
    # version back. Previously a file with no version, or one already current,
    # was returned untouched and reading the key afterwards raised.
    raw = dict(raw)
    raw["schema_version"] = min(version, SCHEMA_VERSION)
    return raw
