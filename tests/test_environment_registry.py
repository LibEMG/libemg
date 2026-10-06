"""Coverage for the generated setup screens behind the environment menu.

The registry's claim is that nothing about an environment's settings is written
down twice: the controls, their types, their defaults and their help text are
all read out of the environment's own configuration class, so a setting added
to :class:`~libemg.environments.fitts.FittsConfig` appears in the panel without
anybody editing the panel.

A test that listed the expected settings by hand would pass just as happily
against a hardcoded copy, which is the failure being guarded against. So the
expectations here are derived live, from ``dataclasses.fields`` and from
``FittsConfig.__doc__``, and a test fails when the two drift apart rather than
when a list was not kept up to date.

The other half is what crosses a process boundary. A factory is built in the
GUI and called in the child that runs the environment, so every one of them is
asserted to survive ``pickle``.
"""
import os

# Set before pygame is imported, which happens as soon as an environment module
# is: the environments are drawn offscreen and never want a window here.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import dataclasses
import pickle

import pytest

from libemg._gui._environments.factories import (FACTORIES, ControllerSpec,
                                                 build_factory,
                                                 keyboard_prediction_map)
from libemg._gui._environments.registry import (BOOL, COLOR, ENUM, FLOAT, INT,
                                                STR, EnvironmentSpec, Setting,
                                                build_registry, default_registry)
from libemg.environments.controllers import KeyboardController
from libemg.environments.emg_hero import EMGHero
from libemg.environments.fitts import Fitts, FittsConfig

CONTROLLER_KINDS = ("keyboard", "classifier", "regressor")


@pytest.fixture
def registry():
    """A freshly built registry, so a test never sees another test's edits."""
    return build_registry()


def _is_colour(value):
    return (isinstance(value, tuple) and len(value) == 3
            and all(isinstance(channel, int) for channel in value))


def _expected_fitts_settings():
    """What the FittsConfig dataclass says its settings should be, read live.

    Derived rather than listed, so replacing the generation with a hardcoded
    copy of today's fields fails the moment the dataclass changes.
    """
    expected = {}
    for field in dataclasses.fields(FittsConfig):
        has_default = not (field.default is dataclasses.MISSING
                           and field.default_factory is dataclasses.MISSING)
        default = field.default
        if default is dataclasses.MISSING:
            default = (field.default_factory()
                       if field.default_factory is not dataclasses.MISSING else None)
        if callable(default) and not isinstance(default, (tuple, list)):
            continue        # a behaviour, not a value: nothing to draw
        if isinstance(default, (tuple, list)) and not _is_colour(default):
            continue        # a sequence with no control to put it in
        expected[field.name] = (has_default, default)
    return expected


# ======================================================================
# 13. what the menu offers
# ======================================================================
def test_the_registry_offers_every_environment_with_something_to_configure(registry):
    assert set(registry) == {"fitts", "iso_fitts", "curricular_fitts", "emg_hero"}

    for key, spec in registry.items():
        assert isinstance(spec, EnvironmentSpec)
        assert spec.id == key, "the key and the id have to agree"
        assert spec.title and spec.title != spec.id
        assert len(spec.help.split()) >= 5, f"{key} needs a sentence, not a label"
        assert spec.settings, f"{key} offers nothing to configure"
        assert all(isinstance(setting, Setting) for setting in spec.settings)
        names = [setting.name for setting in spec.settings]
        assert len(names) == len(set(names)), f"{key} declares a setting twice"
        assert spec.controllers
        assert spec.factory_name in FACTORIES

    ids = [spec.id for spec in registry.values()]
    assert len(ids) == len(set(ids))


def test_the_process_wide_registry_is_built_once_and_can_be_refreshed():
    first = default_registry()

    assert default_registry() is first
    assert default_registry(refresh=True) is not first
    assert set(default_registry()) == set(build_registry())


# ======================================================================
# 14. settings are generated from the dataclass, not copied from it
# ======================================================================
def test_fitts_settings_are_generated_from_the_config_dataclass(registry):
    settings = {setting.name: setting for setting in registry["fitts"].settings}
    expected = _expected_fitts_settings()

    assert set(settings) == set(expected), \
        "the panel and the dataclass disagree about what Fitts can be told"

    for name, (has_default, default) in expected.items():
        setting = settings[name]
        if has_default:
            assert setting.default == default, \
                f"{name} was offered {setting.default!r}, not the config's {default!r}"
            assert setting.required is False
        else:
            # No default to copy, so the panel invents a usable one and says so.
            assert setting.required is True, f"{name} has no default and must say so"


def test_iso_fitts_adds_its_ring_to_everything_fitts_offers(registry):
    fitts = [setting.name for setting in registry["fitts"].settings]
    iso = [setting.name for setting in registry["iso_fitts"].settings]

    assert iso[:len(fitts)] == fitts
    assert set(iso) - set(fitts) == {"num_targets", "target_distance_radius"}
    ring = registry["iso_fitts"].setting("target_distance_radius")
    assert (ring.kind, ring.default) == (INT, 275)


# ======================================================================
# 15. kinds, inferred from the declared types
# ======================================================================
@pytest.mark.parametrize("name, kind", [("num_trials", INT),
                                        ("width", INT),
                                        ("dwell_time", FLOAT),
                                        ("timeout", FLOAT),
                                        ("proportional_control", BOOL),
                                        ("mapping", ENUM),
                                        ("cursor_color", COLOR),
                                        ("background_color", COLOR),
                                        ("save_file", STR)])
def test_a_settings_kind_follows_its_declared_type(registry, name, kind):
    setting = registry["fitts"].setting(name)

    assert setting is not None, f"{name} is not offered at all"
    assert setting.kind == kind


def test_an_optional_number_is_marked_optional_and_a_required_one_is_not(registry):
    fitts = registry["fitts"]

    assert fitts.setting("timeout").optional is True, "float | None can be switched off"
    assert fitts.setting("game_time").optional is True
    assert fitts.setting("num_trials").optional is False, \
        "a field with no default is the user's to set, not one that may be None"
    assert fitts.setting("dwell_time").optional is False


# ======================================================================
# 16. the mappings Fitts actually accepts
# ======================================================================
def test_mapping_offers_only_the_mappings_fitts_accepts(registry):
    mapping = registry["fitts"].setting("mapping")

    assert mapping.kind == ENUM
    assert mapping.choices == ["cartesian", "polar+", "polar-"]
    assert mapping.default == "cartesian"

    # Why bare 'polar' is not on the list: the environment refuses it, and a
    # free text box would let somebody find that out only once it started.
    with pytest.raises(ValueError, match="mapping"):
        Fitts(KeyboardController(), FittsConfig(num_trials=1, mapping="polar"))

    for choice in mapping.choices:
        Fitts(KeyboardController(), FittsConfig(num_trials=1, mapping=choice))


# ======================================================================
# 17. help text, from the environment's own docstring
# ======================================================================
def test_help_text_is_the_config_authors_own_words(registry):
    setting = registry["fitts"].setting("num_trials")

    assert setting.help, "a setting with a documented field must show its help"
    assert setting.help in " ".join(FittsConfig.__doc__.split()), \
        "the help shown beside a control has to come from the config's docstring"
    assert "Number of trials" in setting.help

    documented = [s for s in registry["fitts"].settings if s.help]
    assert len(documented) == len(registry["fitts"].settings), \
        "every Fitts setting is documented in the dataclass, so every one shows help"


# ======================================================================
# 18. coercing what a control hands back
# ======================================================================
def test_coerce_brings_a_control_value_into_the_settings_type():
    assert Setting("n", INT, default=3).coerce("7") == 7
    assert Setting("n", INT, default=3).coerce(7.9) == 7
    assert Setting("n", FLOAT, default=1.0).coerce("2.5") == 2.5
    assert Setting("n", BOOL, default=False).coerce("true") is True
    assert Setting("n", BOOL, default=False).coerce("no") is False
    assert Setting("n", BOOL, default=False).coerce(1) is True
    assert Setting("n", STR, default="").coerce(12) == "12"
    assert Setting("n", COLOR, default=(0, 0, 0)).coerce([300.4, -5, 7.6]) == (255, 0, 8)
    assert Setting("n", ENUM, default="a", choices=["a", "b"]).coerce("b") == "b"
    assert Setting("n", ENUM, default="a", choices=["a", "b"]).coerce("c") == "a", \
        "a value outside the list falls back rather than being passed on"
    assert Setting("n", INT, default=3).coerce("not a number") == 3
    assert Setting("n", INT, default=3, minimum=2, maximum=5).coerce(99) == 5
    assert Setting("n", INT, default=3, minimum=2, maximum=5).coerce(-1) == 2


def test_zero_switches_off_an_optional_timeout_but_not_a_required_count(registry):
    timeout = registry["fitts"].setting("timeout")
    num_trials = registry["fitts"].setting("num_trials")

    # There is no number meaning "no timeout", so zero has to mean it. Without
    # this a spinner left at zero would fail every trial the instant it began.
    assert timeout.optional is True
    assert timeout.coerce(0) is None
    assert timeout.coerce("") is None
    assert timeout.coerce(-4) is None
    assert timeout.coerce(2.5) == 2.5

    # num_trials is not optional, so zero is a number like any other and the
    # same rule must not be applied to it.
    assert num_trials.optional is False
    assert num_trials.coerce(0) == 0
    assert num_trials.coerce("12") == 12
    assert num_trials.coerce("") == num_trials.default


# ======================================================================
# 19. a field the environment declares no default for
# ======================================================================
def test_a_required_field_gets_a_usable_default_and_a_label_that_says_so(registry):
    num_trials = registry["fitts"].setting("num_trials")

    assert num_trials.required is True
    assert num_trials.label.endswith("(required)")
    assert "Num Trials" in num_trials.label
    assert isinstance(num_trials.default, int) and num_trials.default > 0, \
        "a control showing zero looks like a setting rather than a blank"
    # Usable means the task runs with it, not merely that it is not None.
    FittsConfig(**{name: value
                   for name, value in registry["fitts"].defaults().items()})


# ======================================================================
# 20. the frame an environment will draw
# ======================================================================
def test_frame_size_reads_the_configured_size_and_falls_back_to_the_default(registry):
    spec = registry["fitts"]

    assert spec.frame_size({"width": 640, "height": 480}) == (640, 480)
    assert spec.frame_size({"width": "800", "height": "600"}) == (800, 600)
    assert spec.frame_size({}) == spec.default_size == (1250, 750)
    assert spec.frame_size({"width": 0, "height": None}) == spec.default_size, \
        "a size nobody set falls back rather than asking for a zero-wide texture"
    assert spec.frame_size(spec.defaults()) == (1250, 750), \
        "the configured defaults are what the panel starts from"


def test_an_environment_that_fixes_its_own_size_ignores_the_settings(registry):
    hero = registry["emg_hero"]

    # EMG Hero calls set_mode with its own numbers and offers no width or
    # height setting, so the texture has to be made at the size it will draw.
    assert hero.size_from == ()
    assert hero.frame_size({}) == hero.default_size == (525, 700)
    assert hero.frame_size({"width": 1500, "height": 750}) == (525, 700)


# ======================================================================
# 21. an environment with no config dataclass
# ======================================================================
def test_emg_hero_settings_come_from_its_constructor(registry):
    import inspect

    settings = {setting.name: setting for setting in registry["emg_hero"].settings}
    parameters = inspect.signature(EMGHero.__init__).parameters

    assert set(settings) <= set(parameters)
    for skipped in ("self", "controller", "prediction_map", "img_files"):
        assert skipped not in settings, f"{skipped} is not a user setting"

    for name, setting in settings.items():
        declared = parameters[name].default
        assert setting.default == (None if declared is inspect.Parameter.empty
                                   else declared), name

    assert settings["test_time"].kind == INT
    assert settings["min_speed"].kind == FLOAT
    assert settings["save_file"].kind == STR
    assert settings["save_file"].optional is True


def test_emg_hero_settings_show_the_constructors_own_help(registry):
    # EMG Hero documents its parameters on __init__ rather than on the class,
    # which is just as common. Reading only the class docstring left every one
    # of its settings offered with nothing beside it.
    settings = {setting.name: setting for setting in registry["emg_hero"].settings}

    assert settings["test_time"].help.startswith("Amount of time")
    assert all(setting.help for setting in settings.values())


# ======================================================================
# 22. a factory has to survive being sent to another process
# ======================================================================
@pytest.mark.parametrize("environment", ["fitts", "iso_fitts", "curricular_fitts",
                                         "emg_hero"])
@pytest.mark.parametrize("kind", CONTROLLER_KINDS)
def test_every_factory_is_picklable(registry, environment, kind):
    spec = registry[environment]
    values = {name: spec.setting(name).coerce(value)
              for name, value in spec.defaults().items()}

    factory = build_factory(spec, ControllerSpec(kind=kind), values)

    assert isinstance(factory, FACTORIES[spec.factory_name])
    # It is built here and called in the process that runs the environment, so
    # this is the assertion the whole design of factories.py exists to satisfy.
    restored = pickle.loads(pickle.dumps(factory))
    assert isinstance(restored, type(factory))
    assert restored.settings == factory.settings
    assert restored.controller.kind == kind


def test_an_environment_with_no_factory_is_refused(registry):
    spec = registry["fitts"]
    spec.factory_name = "nothing_builds_this"

    with pytest.raises(KeyError):
        build_factory(spec, ControllerSpec(), {})


# ======================================================================
# 23. driving a Fitts task from the keyboard
# ======================================================================
def test_the_keyboard_map_covers_exactly_the_directions_fitts_requires():
    import pygame

    mapping = keyboard_prediction_map()

    assert {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
            -1} <= set(mapping)
    assert mapping[pygame.K_UP] == "N"
    assert mapping[pygame.K_DOWN] == "S"
    assert mapping[pygame.K_RIGHT] == "E"
    assert mapping[pygame.K_LEFT] == "W"
    assert mapping[-1] == "NM", "nothing held has to mean no motion"

    # Fitts asserts that a prediction map covers every direction its own
    # default map produces, so this is the set it checks against. Anything
    # else the keyboard can report has to be in here too, mapped to no motion:
    # a Fitts task looks a prediction up with no fallback, so a key with no
    # entry ends the task with a KeyError.
    assert set(mapping.values()) == {"N", "S", "E", "W", "NM"}
    for key in set(mapping) - {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                               pygame.K_RIGHT}:
        assert mapping[key] == "NM", key
    for key in KeyboardController().keys:
        assert key in mapping, f"the keyboard reports {key} and nothing catches it"
    Fitts(KeyboardController(), FittsConfig(num_trials=1), prediction_map=mapping)


def test_a_keyboard_factory_supplies_the_map_and_a_socket_one_does_not(registry):
    spec = registry["fitts"]
    values = spec.defaults()

    keyboard = build_factory(spec, ControllerSpec(kind="keyboard"), values)
    classifier = build_factory(spec, ControllerSpec(kind="classifier"), values)

    # Without a map of its own, a keyboard-driven Fitts task fails on the first
    # frame with a KeyError for a pygame key code.
    assert set(keyboard._prediction_map().values()) == {"N", "S", "E", "W", "NM"}
    assert classifier._prediction_map() is None, \
        "a classifier sends class indices, which the environment's default map covers"
