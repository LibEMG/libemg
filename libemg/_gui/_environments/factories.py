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
    """

    def __init__(self, kind="keyboard", ip="127.0.0.1", port=12346,
                 num_classes=5, output_format="predictions"):
        self.kind = kind
        self.ip = ip
        self.port = port
        self.num_classes = num_classes
        self.output_format = output_format

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


class _Factory:
    """Common shape: hold settings, build on the other side."""

    def __init__(self, controller, settings):
        self.controller = controller
        self.settings = dict(settings)

    def __call__(self):
        raise NotImplementedError

    def _prediction_map(self):
        """A map suited to the controller, or None to keep the default."""
        return keyboard_prediction_map() if self.controller.kind == "keyboard" else None

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
                     prediction_map=self._prediction_map())


class ISOFittsFactory(_Factory):
    """Builds an ISO Fitts task, whose ring settings sit outside the config."""

    def __call__(self):
        from libemg.environments.fitts import FittsConfig, ISOFitts
        accepted, extra = self._split(FittsConfig)
        return ISOFitts(self.controller.build(), FittsConfig(**accepted),
                        prediction_map=self._prediction_map(),
                        num_targets=int(extra.get("num_targets", 8)),
                        target_distance_radius=int(
                            extra.get("target_distance_radius", 275)))


class CurricularFittsFactory(_Factory):
    """Builds a curricular Fitts task."""

    def __call__(self):
        from libemg.environments.curricular_fitts import (CurricularFitts,
                                                          CurricularFittsConfig)
        accepted, extra = self._split(CurricularFittsConfig)
        return CurricularFitts(self.controller.build(),
                               CurricularFittsConfig(**accepted),
                               save_file=extra.get("save_file"))


class EMGHeroFactory(_Factory):
    """Builds the rhythm game, which takes its settings as arguments."""

    def __call__(self):
        import inspect
        from libemg.environments.emg_hero import EMGHero
        names = set(inspect.signature(EMGHero.__init__).parameters)
        accepted = {k: v for k, v in self.settings.items() if k in names}
        # Its own map, not the Fitts one: see emg_hero_keyboard_map.
        mapping = emg_hero_keyboard_map() if self.controller.kind == "keyboard" else None
        return EMGHero(self.controller.build(), prediction_map=mapping, **accepted)


#: Factory per environment id, matching EnvironmentSpec.factory_name.
FACTORIES = {
    "fitts": FittsFactory,
    "iso_fitts": ISOFittsFactory,
    "curricular_fitts": CurricularFittsFactory,
    "emg_hero": EMGHeroFactory,
}


def build_factory(spec, controller_spec, settings):
    """Make a picklable factory for an environment.

    Parameters
    ----------
    spec: EnvironmentSpec
        Which environment.
    controller_spec: ControllerSpec
        How it should be driven.
    settings: dict
        Configured values, already coerced.

    Returns
    ----------
    callable
        Call it in the child process to get the environment.
    """
    factory_class = FACTORIES.get(spec.factory_name)
    if factory_class is None:
        raise KeyError(f"No factory for environment '{spec.id}'.")
    return factory_class(controller_spec, settings)
