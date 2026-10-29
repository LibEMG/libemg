import json
import time
from abc import ABC, abstractmethod
from multiprocessing import Process
from pathlib import Path
import pickle
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"   # hide pygame welcome message
import pygame

from libemg.environments.controllers import Controller
from libemg.environments.recording import FrameLog, RecordingConfig


class PredictionMap(dict):
    """A prediction map that sends every class it does not list to one action.

    Environments look each received prediction up in their map. A plain dict
    raises KeyError for a class nobody mapped, ending the task on the first
    frame a model with more classes than the task has actions outputs one of
    them. This answers ``default`` instead, so a 16-class model can drive a
    five-action task and its other eleven classes simply do nothing.

    Parameters
    ----------
    mapping : dict
        Received prediction to action.
    default
        The action for anything not in ``mapping``.

    Examples
    --------
    >>> PredictionMap.from_actions({'N': 8, 'S': 7, 'E': 1, 'W': 2, 'NM': 0}, default='NM')[11]
    'NM'
    """

    def __init__(self, mapping, default):
        super().__init__(mapping)
        self.default = default

    def __missing__(self, key):
        return self.default

    def __reduce__(self):
        return (PredictionMap, (dict(self), self.default))

    @classmethod
    def from_actions(cls, classes, default):
        """Build from action -> class, which is how a person thinks of the mapping.

        Parameters
        ----------
        classes : dict
            Action to the class that triggers it. A class below 0, or None,
            leaves the action unmapped.
        default
            The action for every other class.
        """
        mapping = {}
        for action, class_index in classes.items():
            if class_index is None or int(class_index) < 0:
                continue
            mapping.setdefault(int(class_index), action)
        return cls(mapping, default)


class Environment(ABC):
    def __init__(self, controller: Controller, fps: int, log_dictionary: dict, save_file: str | None = None,
                 recording: RecordingConfig | None = None):
        """Abstract environment interface for pygame environments.

        Parameters
        ----------
        controller : Controller
            Controller instance that defines how control actions are parsed.
        fps : int
            Frames per second (Hz).
        log_dictionary : dict
            Dictionary containing metrics to log.
        save_file : str | None, optional
            Name of save file (e.g., log.pkl). Supports .json and .pkl file formats. If None, no results are saved. Defaults to None.
        recording : RecordingConfig | None, optional
            Record every frame -- model outputs, game state and CIIL labels -- into recording.folder, in the form
            libemg.environments.recording.assemble joins with the EMG for training. If None, nothing is recorded.
            Defaults to None.
        """
        # Assumes this is a pygame environment
        self.controller = controller
        self.done = False   # flag to determine when loop should be exited
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.log_dictionary = log_dictionary
        self.save_file = save_file
        self.recording = recording
        self.frame_log = FrameLog() if recording is not None else None
        self.process = Process(target=self.run, daemon=True)

    @property
    def recording_frames(self) -> bool:
        """Whether frames are being recorded. Check it before building a frame's fields."""
        return self.frame_log is not None

    def record_frame(self, trial: int, model: dict | None = None, game: dict | None = None,
                     adaptation: dict | None = None):
        """Record this frame. Does nothing unless the environment was given a recording.

        Parameters
        ----------
        trial : int
            The trial the frame belongs to.
        model, game, adaptation : dict | None
            Field name to value, by group. See libemg.environments.recording for what each group holds.
        """
        if self.frame_log is not None:
            self.frame_log.add(time.time(), trial, model=model, game=game, adaptation=adaptation)

    def recording_details(self) -> dict:
        """What the manifest should say about this environment beyond its frames. Override to add to it."""
        return {"controller": type(self.controller).__name__}

    def save_recording(self):
        """Write the recorded frames. Called by save_results, so a finished or stopped task saves what it has."""
        if self.frame_log is not None:
            self.frame_log.save(self.recording, type(self).__name__, self.recording_details())

    @abstractmethod
    def game_setup(self):
        # setup things like font in here.
        ...

    def run(self):
        """Run environment in main loop. Blocks all further execution. Results are saved after task is completed."""        
        pygame.init()
        pygame.font.init()
        pygame.mixer.init()

        self.game_setup()
        while not self.done:
            self._run_loop()
            pygame.display.update()
            self.clock.tick(self.fps)

        self.save_results()
        
        pygame.display.quit()
        pygame.mixer.quit()
        pygame.font.quit()
        pygame.quit()

    @abstractmethod
    def _run_loop(self):
        ...

    def run_helper(self, block=True):
        if block:
            self.process.start()
            self.process.join()
        else:
            self.process.start()
    

    def save_results(self):
        self.save_recording()
        if self.save_file is None or self.log_dictionary is None:
            # Don't log anything
            return

        file = Path(self.save_file).absolute()
        file.parent.mkdir(parents=True, exist_ok=True) # create parent directories if they don't exist

        if file.suffix == '.pkl':
            with open(self.save_file, 'wb') as f:
                pickle.dump(self.log_dictionary, f)
        elif file.suffix == '.json':
            with open(self.save_file, 'w') as f:
                json.dump(self.log_dictionary, f)
        else:
            raise ValueError(f"Unexpected file format '{file.suffix}'. Choose from '.pkl' or '.json'.")
