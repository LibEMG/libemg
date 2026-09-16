"""What blocks a pipeline can be built from.

The registry is a description, not an implementation. Each :class:`NodeSpec`
says what a block is called, what it consumes, what it produces, and what can
be configured on it. Nothing here imports DearPyGui or constructs anything that
runs; the editor reads this to decide what to draw, and the compiler reads it to
decide what to build.

Generated, not hand-listed
--------------------------
Wherever LibEMG already knows the answer, the registry asks it. Feature names
come from the feature extractor, metric names from the offline metrics, model
names from the predictors' own model tables, and each streamer's parameters
from its signature. A hand-maintained copy of those lists would be wrong the
first time somebody added a feature, and wrong silently.

That is also why a parameter carries its *kind* rather than a widget name. The
kind is what decides whether the editor draws a dropdown, a number field, a
checkbox, a file picker or a multiple-selection list, so the rule for "a
dropdown where the choices are known, a text field where a continuous value is
viable" is written once here instead of once per block.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


# ----------------------------------------------------------------------
# Port types
#
# These become the `category` on a DearPyGui node attribute, and DearPyGui
# refuses to link two attributes whose categories differ. Typing is therefore
# enforced as the user drags, by the toolkit, rather than by validation that
# runs afterwards and has to explain itself.
# ----------------------------------------------------------------------
class PortType:
    """The kinds of thing that can flow along a link."""

    SAMPLES = "samples"          # an N-by-C continuous stream
    WINDOWS = "windows"          # enframed windows
    FEATURES = "features"        # an N-by-F feature matrix
    PREDICTION = "prediction"    # class index, probabilities, velocity
    CONTINUOUS = "continuous"    # a regression output vector
    LABELS = "labels"            # ground truth, offline only
    METRICS = "metrics"          # a summary result table

    ALL = (SAMPLES, WINDOWS, FEATURES, PREDICTION, CONTINUOUS, LABELS, METRICS)

    #: The category DearPyGui enforces during a drag. Usually the type itself.
    #: A classifier's decision and a regressor's vector share one, because a
    #: sink treats them the same: it sends, writes or scores whatever the model
    #: produced. Keeping them as separate *types* still matters, because a
    #: probe draws class probabilities as bars and a velocity vector as a
    #: trace, and that choice comes from the type.
    CATEGORY = {PREDICTION: "decision", CONTINUOUS: "decision"}

    @classmethod
    def category_of(cls, port_type):
        """The drag category a port type belongs to."""
        return cls.CATEGORY.get(port_type, port_type)

    @classmethod
    def accepts(cls, source_type, sink_type):
        """Whether something of one type may flow into a port of another."""
        return cls.category_of(source_type) == cls.category_of(sink_type)

    #: How a probe should draw each type. Deriving this from the port rather
    #: than asking the user is what keeps probing to a single click.
    RENDER = {
        SAMPLES: "timeseries",
        CONTINUOUS: "timeseries",
        WINDOWS: "window_overlay",
        FEATURES: "bars",
        PREDICTION: "probabilities",
        LABELS: "timeseries",
        METRICS: "table",
    }


# Parameter kinds. The editor maps each to one widget.
ENUM = "enum"
INT = "int"
FLOAT = "float"
BOOL = "bool"
STR = "str"
PATH = "path"
FOLDER = "folder"
MULTI_SELECT = "multi_select"


@dataclass(frozen=True)
class ParamSpec:
    """One configurable value on a block.

    Attributes
    ----------
    name: str
        Key this is stored under in a node's params.
    kind: str
        One of the module-level kinds. Decides the widget.
    label: str
        What the editor shows. Defaults to a prettified ``name``.
    default: Any
        Starting value.
    choices: sequence or None
        Allowed values, for :data:`ENUM` and :data:`MULTI_SELECT`.
    minimum, maximum: float or None
        Bounds, for :data:`INT` and :data:`FLOAT`.
    help: str
        One line explaining the parameter, shown as a tooltip.
    advanced: bool
        Hidden behind a disclosure in the editor. For parameters that have a
        sensible default and rarely need touching.
    """

    name: str
    kind: str
    label: str = ""
    default: Any = None
    choices: Optional[Sequence[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    help: str = ""
    advanced: bool = False

    def __post_init__(self):
        if self.kind in (ENUM, MULTI_SELECT) and not self.choices:
            raise ValueError(f"Parameter '{self.name}' is a {self.kind} with no choices.")
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").strip().title())

    def coerce(self, value):
        """Bring ``value`` into this parameter's type and bounds.

        The editor's widgets return strings often enough, and a loaded file can
        carry anything, so a spec is the one place that knows what a valid
        value looks like.
        """
        if value is None:
            return self.default
        try:
            if self.kind == INT:
                value = int(value)
            elif self.kind == FLOAT:
                value = float(value)
            elif self.kind == BOOL:
                value = bool(value) if not isinstance(value, str) \
                    else value.strip().lower() in ("1", "true", "yes", "on")
            elif self.kind in (STR, PATH, FOLDER):
                value = str(value)
            elif self.kind == MULTI_SELECT:
                value = [v for v in value if self.choices is None or v in self.choices]
        except (TypeError, ValueError):
            return self.default
        if self.kind in (INT, FLOAT):
            if self.minimum is not None:
                value = max(value, type(value)(self.minimum))
            if self.maximum is not None:
                value = min(value, type(value)(self.maximum))
        if self.kind == ENUM and self.choices and value not in self.choices:
            return self.default
        return value


@dataclass(frozen=True)
class PortSpec:
    """One input or output on a block.

    Attributes
    ----------
    name: str
        Identifies the port within its node.
    type: str
        A :class:`PortType`. Two ports link only if these match.
    label: str
        What the editor shows.
    multiple: bool
        Whether more than one link may attach. Outputs default to allowing
        many, since fanning one stage out to several consumers is the point;
        inputs default to one, because two writers into one input would
        interleave with no defined order.
    optional: bool
        Whether the pipeline is still valid with this input unconnected.
    """

    name: str
    type: str
    label: str = ""
    multiple: bool = False
    optional: bool = False

    def __post_init__(self):
        if self.type not in PortType.ALL:
            raise ValueError(f"Port '{self.name}' has unknown type '{self.type}'.")
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").strip().title())


# Node categories, which drive grouping in the editor's add menu and the
# online-or-offline classification in the compiler.
SOURCE = "source"
TRANSFORM = "transform"
WINDOW = "window"
FEATURES = "features"
MODEL = "model"
SINK = "sink"


@dataclass(frozen=True)
class NodeSpec:
    """A block that can be placed on the canvas.

    Attributes
    ----------
    id: str
        Stable identifier, stored in saved files. Renaming one breaks old
        files, which is what the unresolved-node handling in
        :mod:`~libemg._gui._pipeline.document` exists to survive.
    category: str
        One of the module-level categories.
    title: str
        Shown on the node's title bar.
    inputs, outputs: sequence of PortSpec
        What it consumes and produces.
    params: sequence of ParamSpec
        What can be configured.
    help: str
        A sentence describing what the block does.
    offline_only, online_only: bool
        Whether the block only makes sense in one mode. The compiler uses this
        to explain a pipeline that mixes the two.
    """

    id: str
    category: str
    title: str
    inputs: Sequence[PortSpec] = field(default_factory=tuple)
    outputs: Sequence[PortSpec] = field(default_factory=tuple)
    params: Sequence[ParamSpec] = field(default_factory=tuple)
    help: str = ""
    offline_only: bool = False
    online_only: bool = False

    def param(self, name):
        """The named :class:`ParamSpec`, or None."""
        for spec in self.params:
            if spec.name == name:
                return spec
        return None

    def port(self, name, direction="input"):
        """The named :class:`PortSpec` on the given side, or None."""
        for spec in (self.inputs if direction == "input" else self.outputs):
            if spec.name == name:
                return spec
        return None

    def defaults(self):
        """A fresh params dict for a newly placed node."""
        return {spec.name: spec.default for spec in self.params}


# ----------------------------------------------------------------------
# Generation helpers
# ----------------------------------------------------------------------
_SKIP_STREAMER_ARGS = {"self", "shared_memory_items", "args", "kwargs"}


def _streamer_functions():
    """The public streamer entry points, with their signatures.

    Anything private or clearly not a device entry point is left out. A new
    streamer added to the module appears here without the registry changing.
    """
    from libemg import streamers

    found = {}
    for name, function in inspect.getmembers(streamers, inspect.isfunction):
        if name.startswith("_"):
            continue
        if function.__module__ != streamers.__name__:
            continue
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            continue
        # Accepting shared_memory_items is the contract a pipeline source has
        # to meet, because that is how its samples reach the rest of the graph.
        # Testing for it rather than keeping an exclusion list is what keeps a
        # streamer that talks over a socket, such as the UDP mock, from being
        # offered as a block that could never be connected to anything.
        if "shared_memory_items" not in parameters:
            continue
        found[name] = function
    return found


def _params_from_signature(function, prefix=""):
    """Turn a function's keyword arguments into parameter specs.

    The same trick the data-collection panel already uses to build its
    arguments, applied to a whole signature so a device's options appear on its
    node without anybody listing them here.
    """
    specs = []
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return specs
    for name, parameter in signature.parameters.items():
        if name in _SKIP_STREAMER_ARGS:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        default = None if parameter.default is inspect.Parameter.empty else parameter.default
        if isinstance(default, bool):
            kind = BOOL
        elif isinstance(default, int):
            kind = INT
        elif isinstance(default, float):
            kind = FLOAT
        elif isinstance(default, (list, tuple)):
            # A sequence default is almost always a pair of cutoffs or a
            # channel list, which reads and edits far better as text than as a
            # row of spinners.
            kind, default = STR, ",".join(str(v) for v in default)
        else:
            kind, default = STR, "" if default is None else str(default)
        specs.append(ParamSpec(name=prefix + name, kind=kind, default=default,
                               advanced=True,
                               help=f"Passed through to {function.__name__}."))
    return specs


def _feature_choices():
    from libemg.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    return list(extractor.get_feature_list()), list(extractor.get_feature_groups().keys())


def _metric_choices():
    from libemg.offline_metrics import OfflineMetrics
    return list(OfflineMetrics().get_available_metrics())


def _model_choices():
    from libemg.emg_predictor import CLASSIFIER_MODELS, REGRESSOR_MODELS
    return list(CLASSIFIER_MODELS), list(REGRESSOR_MODELS)


#: Filter names understood by Filter.install_filters. Taken from the branches
#: it dispatches on, which is the closest thing that module has to a list.
FILTER_NAMES = ("bandpass", "lowpass", "highpass", "bandstop", "notch", "standardize")


def build_registry():
    """Describe every block that can be placed, generated where possible.

    Returns
    ----------
    dict
        Mapping from spec id to :class:`NodeSpec`.
    """
    features, groups = _feature_choices()
    metrics = _metric_choices()
    classifiers, regressors = _model_choices()
    specs = {}

    def add(spec):
        specs[spec.id] = spec

    # ---------------- sources ----------------
    for name, function in _streamer_functions().items():
        add(NodeSpec(
            id=f"source.{name}",
            category=SOURCE,
            title=name.replace("_", " ").title(),
            outputs=[PortSpec("emg", PortType.SAMPLES, "EMG", multiple=True)],
            params=_params_from_signature(function),
            help=(function.__doc__ or "").strip().split("\n")[0][:160],
            online_only=True,
        ))

    add(NodeSpec(
        id="source.synthetic_streamer",
        category=SOURCE,
        title="Synthetic Source",
        outputs=[PortSpec("emg", PortType.SAMPLES, "EMG", multiple=True)],
        params=[
            ParamSpec("sampling_rate", INT, default=1000, minimum=1,
                      label="Sampling Rate", help="Hz."),
            ParamSpec("num_channels", INT, default=8, minimum=1, maximum=256,
                      label="Channels"),
            ParamSpec("pattern", ENUM, choices=["noise", "sine", "bursts"],
                      default="bursts",
                      help="Bursts alternate quiet and active, so a classifier "
                           "has something to separate."),
            ParamSpec("amplitude", FLOAT, default=1.0, minimum=0.0, advanced=True),
        ],
        help="Produces samples without a device, for building and testing a "
             "pipeline before any hardware is attached.",
        online_only=True,
    ))

    add(NodeSpec(
        id="source.offline",
        category=SOURCE,
        title="Stored Data",
        outputs=[PortSpec("emg", PortType.SAMPLES, "EMG", multiple=True),
                 PortSpec("labels", PortType.LABELS, "Labels", multiple=True)],
        params=[
            ParamSpec("folder", FOLDER, default="", help="Folder to search for recordings."),
            ParamSpec("regex_filters", STR, default="",
                      label="Regex Filters",
                      help="One filter per line, as left|right|values|description."),
            ParamSpec("delimiter", STR, default=",", advanced=True),
            ParamSpec("label_key", STR, default="classes", label="Label Key",
                      help="Which metadata field supplies the labels output."),
            ParamSpec("sampling_rate", INT, default=1000, minimum=1,
                      label="Sampling Rate", help="Hz. Used by downstream filters."),
            ParamSpec("replay", ENUM, choices=["as fast as possible", "real time"],
                      default="as fast as possible", advanced=True,
                      help="Pace the replay to simulate a live session, or run flat out."),
        ],
        help="Reads recordings from disk and replays them through the pipeline.",
        offline_only=True,
    ))

    # ---------------- transforms ----------------
    add(NodeSpec(
        id="transform.filter",
        category=TRANSFORM,
        title="Filter",
        inputs=[PortSpec("input", PortType.SAMPLES)],
        outputs=[PortSpec("output", PortType.SAMPLES, multiple=True)],
        params=[
            ParamSpec("name", ENUM, choices=list(FILTER_NAMES), default="bandpass",
                      label="Type"),
            ParamSpec("cutoff", STR, default="20,450",
                      help="One value, or two separated by a comma for a band."),
            ParamSpec("order", INT, default=4, minimum=1, maximum=20),
            ParamSpec("bandwidth", FLOAT, default=3.0, advanced=True,
                      help="Notch only."),
            ParamSpec("sampling_rate", INT, default=1000, minimum=1,
                      label="Sampling Rate", help="Hz."),
        ],
        help="Conditions the signal once, for every consumer downstream.",
    ))

    add(NodeSpec(
        id="transform.channel_mask",
        category=TRANSFORM,
        title="Channel Mask",
        inputs=[PortSpec("input", PortType.SAMPLES)],
        outputs=[PortSpec("output", PortType.SAMPLES, multiple=True)],
        params=[ParamSpec("channels", STR, default="",
                          help="Channel indices to keep, comma separated. Empty keeps all.")],
        help="Narrows the stream to a subset of channels.",
    ))

    # ---------------- window ----------------
    add(NodeSpec(
        id="window.enframe",
        category=WINDOW,
        title="Window",
        inputs=[PortSpec("input", PortType.SAMPLES)],
        outputs=[PortSpec("output", PortType.WINDOWS, multiple=True)],
        params=[
            ParamSpec("window_size", INT, default=200, minimum=2, label="Window Size",
                      help="Samples per window."),
            ParamSpec("window_increment", INT, default=50, minimum=1,
                      label="Window Increment",
                      help="New samples that constitute a window boundary."),
        ],
        # Deliberately no mode parameter. Whether this triggers downstream work
        # on a live stream or enframes a recording in batches follows from what
        # it is connected to, and a user who set it wrongly would get a
        # pipeline that silently did nothing.
        help="Cuts the stream into windows. Triggers downstream work online, "
             "and enframes in batches offline.",
    ))

    # ---------------- features ----------------
    add(NodeSpec(
        id="features.extract",
        category=FEATURES,
        title="Features",
        inputs=[PortSpec("input", PortType.WINDOWS)],
        outputs=[PortSpec("output", PortType.FEATURES, multiple=True)],
        params=[
            ParamSpec("feature_group", ENUM, choices=["(custom)"] + groups,
                      default="(custom)", label="Feature Group",
                      help="Pick a published group, or choose features individually."),
            ParamSpec("features", MULTI_SELECT, choices=features,
                      default=["MAV", "ZC", "SSC", "WL"],
                      help="Used when the group is set to custom."),
        ],
        help="Extracts features once, so several models can share them.",
    ))

    # ---------------- models ----------------
    add(NodeSpec(
        id="model.classifier",
        category=MODEL,
        title="Classifier",
        # Either input, not both. A statistical model takes features; a deep
        # model takes the windows themselves. Both are marked optional so the
        # generic "nothing connected" rule does not demand both, and a
        # model-specific rule in the document requires exactly one.
        inputs=[PortSpec("input", PortType.FEATURES, "Features", optional=True),
                PortSpec("windows", PortType.WINDOWS, "Windows", optional=True)],
        outputs=[PortSpec("output", PortType.PREDICTION, multiple=True)],
        params=[
            ParamSpec("model", ENUM, choices=classifiers, default="LDA"),
            ParamSpec("model_path", PATH, default="", label="Fitted Model",
                      help="A saved predictor to run. Required online."),
            ParamSpec("rejection_threshold", FLOAT, default=0.0, minimum=0.0, maximum=1.0,
                      label="Rejection Threshold",
                      help="Confidence below which a prediction is rejected. 0 disables."),
            ParamSpec("majority_vote", INT, default=0, minimum=0,
                      label="Majority Vote", help="Decisions to vote over. 0 disables."),
            ParamSpec("velocity", BOOL, default=False,
                      help="Also output a proportional velocity."),
        ],
        help="Predicts a class from features.",
    ))

    add(NodeSpec(
        id="model.regressor",
        category=MODEL,
        title="Regressor",
        inputs=[PortSpec("input", PortType.FEATURES, "Features", optional=True),
                PortSpec("windows", PortType.WINDOWS, "Windows", optional=True)],
        outputs=[PortSpec("output", PortType.CONTINUOUS, multiple=True)],
        params=[
            ParamSpec("model", ENUM, choices=regressors, default="LR"),
            ParamSpec("model_path", PATH, default="", label="Fitted Model",
                      help="A saved predictor to run. Required online."),
            ParamSpec("deadband_threshold", FLOAT, default=0.0, minimum=0.0,
                      label="Deadband", help="Outputs below this are zeroed."),
        ],
        help="Predicts continuous outputs from features.",
    ))

    # ---------------- sinks ----------------
    add(NodeSpec(
        id="sink.socket",
        category=SINK,
        title="Socket Output",
        inputs=[PortSpec("input", PortType.PREDICTION)],
        params=[
            ParamSpec("ip", STR, default="127.0.0.1"),
            ParamSpec("port", INT, default=12346, minimum=1, maximum=65535),
            ParamSpec("protocol", ENUM, choices=["UDP", "TCP"], default="UDP"),
        ],
        help="Sends each output over a socket, for an environment to read.",
        online_only=True,
    ))

    add(NodeSpec(
        id="sink.file",
        category=SINK,
        title="File Output",
        inputs=[PortSpec("input", PortType.PREDICTION)],
        params=[ParamSpec("file_path", PATH, default="output.log", label="File")],
        help="Appends each output to a file.",
    ))

    add(NodeSpec(
        id="sink.console",
        category=SINK,
        title="Console Output",
        inputs=[PortSpec("input", PortType.PREDICTION)],
        params=[],
        help="Prints each output. Useful while building a pipeline up.",
    ))

    add(NodeSpec(
        id="sink.metrics",
        category=SINK,
        title="Offline Metrics",
        inputs=[PortSpec("input", PortType.PREDICTION),
                PortSpec("labels", PortType.LABELS)],
        outputs=[PortSpec("output", PortType.METRICS, multiple=True)],
        params=[
            ParamSpec("metrics", MULTI_SELECT, choices=metrics,
                      default=["CA", "CONF_MAT"]),
            ParamSpec("null_label", INT, default=-1, advanced=True,
                      label="Null Label", help="Class treated as no motion."),
        ],
        help="Scores predictions against ground truth once a run finishes.",
        offline_only=True,
    ))

    return specs


_DEFAULT = None


def default_registry(refresh=False):
    """The registry, built once per process.

    Parameters
    ----------
    refresh: bool (optional), default=False
        Rebuild even if one was already made. Intended for tests.

    Returns
    ----------
    dict
        Mapping from spec id to :class:`NodeSpec`.
    """
    global _DEFAULT
    if _DEFAULT is None or refresh:
        _DEFAULT = build_registry()
    return _DEFAULT
