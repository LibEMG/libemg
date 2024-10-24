import json
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"   # hide pygame welcome message
import pygame

from libemg.environments.controllers import Controller

class Environment(ABC):
    def __init__(self, controller: Controller, fps: int, log_dictionary: dict, save_file: str | None = None):
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
        """        # Assumes this is a pygame environment
        self.controller = controller
        self.done = False   # flag to determine when loop should be exited
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.log_dictionary = log_dictionary
        self.save_file = save_file
        pygame.init()
        pygame.font.init()
        pygame.mixer.init() 

    def run(self):
        """Run environment in main loop. Blocks all further execution. Results are saved after task is completed."""        
        while not self.done:
            self._run_helper()
            pygame.display.update()
            self.clock.tick(self.fps)

        self.save_results()
        pygame.quit()

    @abstractmethod
    def _run_helper(self):
        # If there's a use case, we could a block = True flag so we'd put this in a different thread if False
        ...

    def save_results(self):
        if self.save_file is None:
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
