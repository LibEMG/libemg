"""Builds an environment inside the process that will run it.

Nothing here is constructed in the GUI. An environment holds pygame objects and
a controller holds an open socket, and neither survives being sent to another
process. So what crosses the boundary is a small description of what to build,
and the building happens on the other side.

That is also why these are module-level classes rather than closures. A lambda
capturing a configuration cannot be pickled, and the failure it produces names
an anonymous function rather than anything a user could act on.
"""


class ControllerSpec:
    """What controller an environment should be driven by.

    Parameters
    ----------
    kind: str
        ``'keyboard'``, ``'classifier'`` or ``'regressor'``.
    ip: str (optional), default='127.0.0.1'
        Address a socket controller listens on.
    port: int (optional), default=12346
        Port a socket controller listens on.
    num_classes: int (optional), default=5
        Classes a classifier controller should expect.
    output_format: str (optional), default='predictions'
        What the classifier is sending.
    class_map: dict or None (optional), default=None
        For a classifier: action name (see ``EnvironmentSpec.actions``) to the
        class that triggers it. None keeps the environment's own default map.
    """

    def __init__(self, kind="keyboard", ip="127.0.0.1", port=12346,
                 num_classes=5, output_format="predictions", class_map=None):
        self.kind = kind
        self.ip = ip
        self.port = port
        self.num_classes = num_classes
        self.output_format = output_format
        self.class_map = dict(class_map) if class_map is not None else None

    def build(self):
        from libemg.environments.controllers import (ClassifierController,
                                                     KeyboardController,
                                                     RegressorController)
        if self.kind == "classifier":
            return ClassifierController(output_format=self.output_format,
                                        num_classes=self.num_classes,
                                        ip=self.ip, port=self.port)
        if self.kind == "regressor":
            return RegressorController(ip=self.ip, port=self.port)
        return KeyboardController()


def emg_hero_keyboard_map():
    """Map keys to the lane commands EMG Hero understands.

    EMG Hero does not speak in directions. Its prediction map has to carry the
    values 0, 1, 2, 3 and -1, which it asserts on construction, so the
    direction map a Fitts task wants is rejected outright. The four number keys
    are the natural fit for four lanes, with nothing held meaning no command.

    Returns
    ----------
    dict
        Key code to lane command, including the idle command.
    """
    import pygame
    return {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        -1: -1,
    }


def keyboard_prediction_map():
    """Map arrow keys to the directions a Fitts task understands.

    A Fitts task turns a prediction into a direction through a prediction map,
    and the default one maps class indices 0 to 4. The keyboard controller does
    not produce class indices; it produces pygame key codes, and -1 when
    nothing is held. So choosing keyboard control without also supplying a map
    fails on the first frame with a KeyError for a key code.

    Building the map here is what makes picking Keyboard in the menu simply
    work, which is what somebody launching a task from a menu expects.

    Returns
    ----------
    dict
        Key code to direction, including no-motion for nothing held.
    """
    import pygame
    mapping = {
        pygame.K_UP: "N",
        pygame.K_DOWN: "S",
        pygame.K_RIGHT: "E",
        pygame.K_LEFT: "W",
        -1: "NM",
    }
    # Every other key the GUI forwards maps to no motion. The keyboard
    # controller reports whichever of its keys is held, and a Fitts task looks
    # the result up with no fallback, so a digit pressed while playing would
    # otherwise end the task with a KeyError rather than being ignored.
    for key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                pygame.K_SPACE, pygame.K_ESCAPE,
                pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d):
        mapping.setdefault(key, "NM")
    return mapping


def class_prediction_map(actions, class_map, idle_command):
    """The prediction map a classifier's class map describes.

    Every class without an action of its own takes the idle one -- holds the
    cursor still, plays nothing -- rather than ending the task with a KeyError,
    so a model with more classes than the task has actions can drive it.

    Parameters
    ----------
    actions: list of (name, command)
        What the task can be told, as ``EnvironmentSpec.actions`` lists it.
    class_map: dict
        Action name to class. A class below 0 leaves the action unmapped.
    idle_command: Any
        The command for every other class.

    Returns
    ----------
    PredictionMap
    """
    from libemg.environments._base import PredictionMap
    commands = dict(actions)
    return PredictionMap.from_actions(
        {commands[name]: cls for name, cls in class_map.items() if name in commands},
        default=idle_command)


class _Factory:
    """Common shape: hold settings, build on the other side.

    Parameters
    ----------
    controller: ControllerSpec
        How the task is driven, and for a classifier which class means what.
    settings: dict
        Configured values, already coerced.
    actions: list of (name, command)
        The task's actions, from its spec.
    idle_command: Any
        What a class with no action of its own does.
    recording: RecordingConfig or None
        What to record and where; None records nothing.
    """

    def __init__(self, controller, settings, actions=(), idle_command=None, recording=None):
        self.controller = controller
        self.settings = dict(settings)
        self.actions = [tuple(action) for action in actions]
        self.idle_command = idle_command
        self.recording = recording

    def __call__(self):
        raise NotImplementedError

    def _prediction_map(self):
        """A map suited to the controller, or None to keep the default."""
        if self.controller.kind == "keyboard":
            return keyboard_prediction_map()
        if self.controller.kind == "classifier" and self.controller.class_map and self.actions:
            return class_prediction_map(self.actions, self.controller.class_map, self.idle_command)
        return None

    def _split(self, config_class):
        """Separate settings the config accepts from the rest."""
        import dataclasses
        names = {f.name for f in dataclasses.fields(config_class)}
        accepted = {k: v for k, v in self.settings.items() if k in names}
        remainder = {k: v for k, v in self.settings.items() if k not in names}
        return accepted, remainder


class FittsFactory(_Factory):
    """Builds a Fitts task."""

    def __call__(self):
        from libemg.environments.fitts import Fitts, FittsConfig
        accepted, _ = self._split(FittsConfig)
        return Fitts(self.controller.build(), FittsConfig(**accepted),
                     prediction_map=self._prediction_map(), recording=self.recording)


class ISOFittsFactory(_Factory):
    """Builds an ISO Fitts task, whose ring settings sit outside the config."""

    def __call__(self):
        from libemg.environments.fitts import FittsConfig, ISOFitts
        accepted, extra = self._split(FittsConfig)
        return ISOFitts(self.controller.build(), FittsConfig(**accepted),
                        prediction_map=self._prediction_map(),
                        num_targets=int(extra.get("num_targets", 8)),
                        target_distance_radius=int(
                            extra.get("target_distance_radius", 275)),
                        recording=self.recording)


class CurricularFittsFactory(_Factory):
    """Builds a curricular Fitts task."""

    def __call__(self):
        from libemg.environments.curricular_fitts import (CurricularFitts,
                                                          CurricularFittsConfig)
        accepted, extra = self._split(CurricularFittsConfig)
        return CurricularFitts(self.controller.build(),
                               CurricularFittsConfig(**accepted),
                               save_file=extra.get("save_file"),
                               recording=self.recording)


class EMGHeroFactory(_Factory):
    """Builds the rhythm game, which takes its settings as arguments."""

    def __call__(self):
        import inspect
        from libemg.environments.emg_hero import EMGHero
        names = set(inspect.signature(EMGHero.__init__).parameters)
        accepted = {k: v for k, v in self.settings.items()
                    if k in names and k not in ("prediction_map", "recording")}
        # Its own keyboard map, not the Fitts one: see emg_hero_keyboard_map.
        mapping = (emg_hero_keyboard_map() if self.controller.kind == "keyboard"
                   else self._prediction_map())
        return EMGHero(self.controller.build(), prediction_map=mapping,
                       recording=self.recording, **accepted)


#: Factory per environment id, matching EnvironmentSpec.factory_name.
FACTORIES = {
    "fitts": FittsFactory,
    "iso_fitts": ISOFittsFactory,
    "curricular_fitts": CurricularFittsFactory,
    "emg_hero": EMGHeroFactory,
}


def build_factory(spec, controller_spec, settings, recording=None):
    """Make a picklable factory for an environment.

    Parameters
    ----------
    spec: EnvironmentSpec
        Which environment.
    controller_spec: ControllerSpec
        How it should be driven.
    settings: dict
        Configured values, already coerced.
    recording: RecordingConfig or None (optional), default=None
        What to record of the run, and where. None records nothing.

    Returns
    ----------
    callable
        Call it in the child process to get the environment.
    """
    factory_class = FACTORIES.get(spec.factory_name)
    if factory_class is None:
        raise KeyError(f"No factory for environment '{spec.id}'.")
    actions = [(action.name, action.command) for action in spec.actions]
    idle = spec.action(spec.idle_action) if spec.idle_action else None
    return factory_class(controller_spec, settings, actions=actions,
                         idle_command=idle.command if idle is not None else None,
                         recording=recording)
