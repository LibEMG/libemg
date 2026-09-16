"""What environments can be launched, and what each one can be set up with.

As with the pipeline editor, the setup screens are generated rather than
hand-written. An environment's settings are already declared: Fitts and
Curricular Fitts have configuration dataclasses with typed fields and
documented defaults, and EMG Hero declares its settings as constructor
arguments. Reading those is what keeps a setup screen correct when somebody
adds a setting, instead of correct until somebody adds a setting.

The per-field help text comes from the same docstrings the API documentation is
built from, so what a user reads beside a control is what the author wrote
about it.
"""

import dataclasses
import inspect
import re
import typing

# Field kinds. The panel maps each to one control.
INT = "int"
FLOAT = "float"
BOOL = "bool"
STR = "str"
ENUM = "enum"
COLOR = "color"
PATH = "path"


class Setting:
    """One configurable value on an environment.

    Attributes
    ----------
    name: str
        Keyword it is passed as.
    kind: str
        One of the module-level kinds, deciding the control drawn.
    label: str
        What the setup screen shows.
    default: Any
        Starting value, taken from the environment's own default.
    choices: list or None
        Allowed values, for an enumeration.
    minimum, maximum: float or None
        Bounds.
    help: str
        Taken from the environment's own docstring.
    optional: bool
        Whether None is a meaningful value, as it is for a timeout that can be
        switched off.
    """

    def __init__(self, name, kind, default=None, label=None, choices=None,
                 minimum=None, maximum=None, help="", optional=False,
                 required=False):
        self.name = name
        self.kind = kind
        self.default = default
        self.label = label or name.replace("_", " ").strip().title()
        self.choices = choices
        self.minimum = minimum
        self.maximum = maximum
        self.help = help
        self.optional = optional
        #: The environment declares no default for this, so it is the user's
        #: to choose. A usable starting value is offered anyway, and the label
        #: says so, because a control showing zero looks like a setting rather
        #: than a blank.
        self.required = required
        if required:
            self.label += "  (required)"

    def coerce(self, value):
        """Bring a value from a control into this setting's type.

        An optional number is the interesting case. A timeout or a time limit
        that can be switched off has no number meaning "off", so a numeric
        control has to spell that somehow; zero is the only value a spinner
        can offer that is not a real duration, so zero means off. Without this
        a timeout left at zero would fail every trial the instant it began.
        """
        if value is None or value == "":
            return None if self.optional else self.default
        if self.optional and self.kind in (INT, FLOAT):
            try:
                if float(value) <= 0:
                    return None
            except (TypeError, ValueError):
                return None
        try:
            if self.kind == INT:
                value = int(float(value))
            elif self.kind == FLOAT:
                value = float(value)
            elif self.kind == BOOL:
                value = bool(value) if not isinstance(value, str) \
                    else value.strip().lower() in ("1", "true", "yes", "on")
            elif self.kind == COLOR:
                value = tuple(int(max(0, min(255, round(float(c))))) for c in value[:3])
            elif self.kind in (STR, PATH):
                value = str(value)
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

    def __repr__(self):
        return f"Setting({self.name!r}, {self.kind!r}, default={self.default!r})"


class EnvironmentSpec:
    """An environment that can be launched from the menu.

    Attributes
    ----------
    id: str
        Stable identifier.
    title: str
        Shown in the menu and on the setup screen.
    help: str
        A sentence describing the task.
    settings: list of Setting
        What can be configured.
    controllers: list of str
        Which controller kinds make sense for it.
    """

    def __init__(self, id, title, factory_name, settings, help="",
                 controllers=("Keyboard", "Classifier", "Regressor"),
                 size_from=("width", "height"), default_size=(1000, 700)):
        self.id = id
        self.title = title
        self.factory_name = factory_name
        self.settings = settings
        self.help = help
        self.controllers = list(controllers)
        self.size_from = size_from
        self.default_size = default_size

    def defaults(self):
        return {s.name: s.default for s in self.settings}

    def setting(self, name):
        for s in self.settings:
            if s.name == name:
                return s
        return None

    def frame_size(self, values):
        """The frame this environment will draw.

        Read from the settings where the environment takes its size from them,
        and fixed where the environment decides for itself. The texture has to
        be made before the first frame arrives, so this has to agree with what
        the environment will actually set up.
        """
        if not self.size_from:
            return tuple(int(v) for v in self.default_size)
        width_key, height_key = self.size_from
        width = values.get(width_key) or self.default_size[0]
        height = values.get(height_key) or self.default_size[1]
        return int(width), int(height)


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
def _docstring_help(owner):
    """Map parameter name to its description, from a numpydoc docstring.

    The environments already document every setting where it is declared, so
    the setup screen shows the author's own words rather than a second
    description that could drift from the first.
    """
    text = inspect.getdoc(owner) or ""
    if "Parameters" not in text:
        return {}
    body = text.split("Parameters", 1)[1]
    body = re.split(r"\n\s*-{3,}\s*\n", body, maxsplit=1)
    body = body[1] if len(body) > 1 else body[0]
    found, current, buffer = {}, None, []
    for line in body.splitlines():
        header = re.match(r"^(\w+)\s*[:(]", line)
        if header and not line.startswith((" ", "\t")):
            if current:
                found[current] = " ".join(buffer).strip()
            current, buffer = header.group(1), []
        elif current:
            buffer.append(line.strip())
    if current:
        found[current] = " ".join(buffer).strip()
    return {k: v for k, v in found.items() if v}


def _kind_for(name, annotation, default):
    """Decide what control a field wants, from its type and its name."""
    text = str(annotation)
    optional = "None" in text or default is None
    if name.endswith("color") or name.startswith("color"):
        return COLOR, optional
    if "bool" in text or isinstance(default, bool):
        return BOOL, optional
    if "int" in text and "float" not in text:
        return INT, optional
    if "float" in text:
        return FLOAT, optional
    if isinstance(default, bool):
        return BOOL, optional
    if isinstance(default, int):
        return INT, optional
    if isinstance(default, float):
        return FLOAT, optional
    return STR, optional


def _settings_from_dataclass(config_class, skip=()):
    """Turn a configuration dataclass into settings."""
    help_text = _docstring_help(config_class)
    settings = []
    for field in dataclasses.fields(config_class):
        if field.name in skip:
            continue
        default = field.default
        if default is dataclasses.MISSING:
            default = (field.default_factory()
                       if field.default_factory is not dataclasses.MISSING else None)
        if callable(default) and not isinstance(default, (tuple, list)):
            # A behaviour, not a value. Nothing sensible to draw for it, and
            # the environment's own default is the right one.
            continue
        if isinstance(default, tuple) and len(default) == 3 and \
                all(isinstance(v, int) for v in default):
            kind, optional = COLOR, False
        elif isinstance(default, (tuple, list)):
            continue
        else:
            kind, optional = _kind_for(field.name, field.type, default)
        minimum = 0 if kind in (INT, FLOAT) and not field.name.startswith("color") else None
        required = field.default is dataclasses.MISSING and             field.default_factory is dataclasses.MISSING
        if required:
            # A field with no default looks optional to the type sniffing
            # above, because it has no default to inspect. It is the opposite:
            # the environment cannot run without it, so None is not a value it
            # may take.
            optional = False
            if kind in (INT, FLOAT) and default is None:
                # Offered so the task is runnable straight away; the label
                # marks it as the user's to set.
                default = 1
        settings.append(Setting(field.name, kind, default=default,
                                minimum=minimum, optional=optional,
                                required=required,
                                help=help_text.get(field.name, "")))
    return settings


def _settings_from_signature(owner, skip=()):
    """Turn a constructor's keyword arguments into settings."""
    # A class may document its parameters on itself or on its constructor, and
    # both are common. Looking in only one place left every EMG Hero setting
    # with no help beside it, because that class documents its arguments where
    # they are declared.
    help_text = _docstring_help(owner)
    if not help_text:
        help_text = _docstring_help(owner.__init__)
    settings = []
    signature = inspect.signature(owner.__init__)
    for name, parameter in signature.parameters.items():
        if name in skip or name == "self":
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        default = None if parameter.default is inspect.Parameter.empty else parameter.default
        if callable(default) or isinstance(default, (list, tuple, dict)):
            continue
        kind, optional = _kind_for(name, parameter.annotation, default)
        settings.append(Setting(name, kind, default=default,
                                minimum=0 if kind in (INT, FLOAT) else None,
                                optional=optional, help=help_text.get(name, "")))
    return settings


def build_registry():
    """Describe every environment that can be launched.

    Returns
    ----------
    dict
        Mapping from spec id to :class:`EnvironmentSpec`.
    """
    from libemg.environments.curricular_fitts import CurricularFittsConfig
    from libemg.environments.emg_hero import EMGHero
    from libemg.environments.fitts import Fitts, FittsConfig, ISOFitts

    specs = {}

    fitts_settings = _settings_from_dataclass(FittsConfig)
    mapping = next((s for s in fitts_settings if s.name == "mapping"), None)
    if mapping is not None:
        # A free text box would let somebody type a mapping that is rejected
        # only once the environment starts. Note that bare 'polar' is not one
        # of them: the environment accepts 'polar+' and 'polar-', which say
        # which way up maps, and raises on anything else.
        mapping.kind = ENUM
        mapping.choices = ["cartesian", "polar+", "polar-"]

    specs["fitts"] = EnvironmentSpec(
        id="fitts", title="Fitts' Law", factory_name="fitts",
        settings=fitts_settings,
        help="A cursor and a single target. The classic test of how quickly and "
             "accurately a control scheme can acquire a target.",
        default_size=(1250, 750))

    iso_settings = list(fitts_settings) + [
        Setting("num_targets", INT, default=8, minimum=2, maximum=24,
                label="Number Of Targets",
                help="Targets arranged around the ring."),
        Setting("target_distance_radius", INT, default=275, minimum=10,
                label="Ring Radius",
                help="Distance in pixels from the centre to each target."),
    ]
    specs["iso_fitts"] = EnvironmentSpec(
        id="iso_fitts", title="ISO Fitts' Law", factory_name="iso_fitts",
        settings=iso_settings,
        help="Targets arranged in a ring, acquired in the standard ISO 9241-9 "
             "order. The usual way to report a throughput figure.",
        default_size=(1250, 750))

    specs["curricular_fitts"] = EnvironmentSpec(
        id="curricular_fitts", title="Curricular Fitts",
        factory_name="curricular_fitts",
        settings=_settings_from_dataclass(
            CurricularFittsConfig, skip=("controller_fields", "controller_map",
                                         "feedback_handle")),
        help="A Fitts task whose difficulty adapts as the user improves. The task "
             "used for user-in-the-loop adaptation. Needs a two degree of freedom "
             "controller, so a regressor rather than a keyboard.",
        # Only a regressor. This task moves its cursor on two axes at once and
        # indexes both, so a controller that yields a single value per frame,
        # which is what the keyboard gives, fails on the first frame it moves.
        controllers=("Regressor",),
        default_size=(1000, 1080))

    specs["emg_hero"] = EnvironmentSpec(
        id="emg_hero", title="EMG Hero", factory_name="emg_hero",
        settings=_settings_from_signature(
            EMGHero, skip=("controller", "prediction_map", "img_files")),
        help="Notes fall down the screen and are hit with the matching gesture. "
             "A rhythm game for practising discrete control.",
        # EMG Hero sets its own display size and offers no width or height
        # setting, so the frame size is fixed rather than read from settings.
        size_from=(), default_size=(525, 700))

    return specs


_DEFAULT = None


def default_registry(refresh=False):
    """The environment registry, built once per process."""
    global _DEFAULT
    if _DEFAULT is None or refresh:
        _DEFAULT = build_registry()
    return _DEFAULT
