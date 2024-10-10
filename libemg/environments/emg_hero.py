import time
from typing import Sequence

import pygame
import numpy as np

from libemg.environments.controllers import Controller
from libemg.environments._base import Environment


class _Note:
    def __init__(self, type):
        self.type = type 
        assert self.type in [0,1,2,3]
        y_poses = [75, 200, 325, 450]
        colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 165, 0)]
        # Based on the type, set up the note 
        self.x_pos = y_poses[self.type]          
        self.y_pos = 0
        self.color = colors[self.type]
        self.length = 35 * (5 * np.random.random()) # Random integer between 1 and 5 

    def move_note(self, speed=5):
        self.y_pos += speed
        if self.y_pos > 1000:
            return -1
        return 0


class EMGHero(Environment):
    def __init__(self, controller: Controller, prediction_map: dict | None = None, test_time: int = 120, min_speed: float = 2.5, max_speed: float = 7.5, min_time: float = 0.6, max_time: float = 2.2,
                 img_files: Sequence | None = None, save_file: str | None = None, fps: int = 60):
        """Guitar Hero style game that tests user's ability to elicit contractions at specific times. Game speed progressively gets quicker over the course of the task.
        Simultaneous contractions, such as with regression, are not currently supported.

        Parameters
        ----------
        controller : Controller
            Interface to parse predictions which determine the notes being played.
        prediction_map : dict | None, optional
            Maps received control commands to notes being played. If None, a standard map for classifiers is created where 0, 1, 2, 3, 4 are mapped to 0, 1, -1, 2, and 3, respectively.
            For custom mappings, pass in a dictionary where keys represent received control signals (from the Controller) and values map to actions in the environment.
            Accepted actions are: -1 (play nothing), 0 (first note), 1 (second note), 2 (third note), 3 (fourth note). All of these actions must be represented by a single key in the dictionary.
            Defaults to None.
        test_time : int, optional
            Amount of time test will take (in seconds). Defaults to 120.
        min_speed : float, optional
            Minimum game speed. Defaults to 2.5.
        max_speed : float, optional
            Maximum game speed. Defaults to 7.5.
        min_time : float, optional
            Minimum time between notes (in seconds).
        max_time : float, optional
            Maximum time between notes (in seconds).
        img_files : Sequence | None, optional
            List of image filenames to put at the bottom of the display to show users which notes are represented by which gestures. If None, no images are shown. Defaults to None.
        save_file : str | None, optional
            Path to save file for logging metrics. If None, no results are logged. Defaults to None.
        fps : int, optional
            Frames per second (in Hz). Defaults to 60.
        """
        if prediction_map is None:
            prediction_map = {
                0: 0,
                1: 1,
                2: -1,
                3: 2,
                4: 3
            }

        assert set(np.unique(list(prediction_map.values()))) == set([0, 1, -1, 2, 3]), f"Did not find all commands (0, 1, 2, 3, -1) represented as values in prediction_map. Got: {prediction_map}."

        if img_files is None:
            img_files = []

        log_dictionary = {
            "times": [],
            "notes": [],
            "button_pressed": []
        }
        super().__init__(controller, fps=fps, log_dictionary=log_dictionary, save_file=save_file)
        self.prediction_map = prediction_map
        self.test_time = test_time
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_time = min_time
        self.max_time = max_time

        pygame.display.set_caption('Testing Environment')
        self.font = pygame.font.SysFont('Comic Sans MS', 30)
        self.screen = pygame.display.set_mode([525, 700])

        self.imgs = []
        if len(img_files) > 0:
            assert len(img_files) == 4, f"Expected 4 image files, but got {len(img_files)}."
            for i in img_files:
                self.imgs.append(pygame.transform.smoothscale(pygame.image.load(i), (100,100)))

        self.last_note = time.time()
        self.start_time = time.time() + self.test_time

        self.notes = []
        self.key_pressed = -1

    def _run_helper(self):
        # Run until the user asks to quit
        gen_time = ((self.start_time - time.time())/self.test_time) * (self.max_time - self.min_time) + self.min_time
        if time.time() - self.last_note > gen_time: # Generation
            new_note = np.random.randint(0,4)
            self.notes.append(_Note(new_note))
            self.last_note = time.time()

        if self.start_time - time.time() <= 0:
            # Time's up
            self.done = True

        # Fill the background with white
        self.screen.fill((255, 255, 255))

        # Update time remaining 
        text = self.font.render('{0:.1f}'.format(self.start_time - time.time()), True, (0,0,0), (255,255,255))
        textRect = text.get_rect()
        textRect.center = (470, 25)
        self.screen.blit(text, textRect)

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
        
        predictions = self.controller.get_data('predictions')

        if predictions is not None:
            # Received data
            assert len(predictions) == 1, f"Expected a single prediction, but got {len(predictions)}. Controllers that produce multiple predictions, like RegressionController, are not currently supported."
            self.key_pressed = self.prediction_map[predictions[0]]

        # Draw notes on bottom of screen 
        pygame.draw.circle(self.screen, (255, 0, 0), (75, 500), 35, width=8 - (self.key_pressed==0) * 8)
        pygame.draw.circle(self.screen, (0, 255, 0), (200, 500), 35, width=8  - (self.key_pressed==1) * 8)
        pygame.draw.circle(self.screen, (0, 0, 255), (325, 500), 35, width=8  - (self.key_pressed==2) * 8)
        pygame.draw.circle(self.screen, (255, 165, 0), (450, 500), 35, width=8  - (self.key_pressed==3) * 8)

        # Move and deal with notes coming down 
        for n in self.notes:
            speed = (1 - (self.start_time - time.time())/self.test_time) * (self.max_speed - self.min_speed) + self.min_speed
            if n.move_note(speed=speed) == -1:
                self.notes.remove(n)
            # Check to see if the shape is over top of the note 
            w = 0
            if n.type == self.key_pressed and n.y_pos >= 500 and n.y_pos - 60 - n.length <= 500:
                w = 5
            pygame.draw.circle(self.screen, n.color, (n.x_pos, n.y_pos), 35, width=w)
            pygame.draw.rect(self.screen, n.color, (n.x_pos - 20, n.y_pos - 30 - n.length, 40, n.length), width=w)
            pygame.draw.circle(self.screen, n.color, (n.x_pos, n.y_pos - 60 - n.length), 35, width=w)

        pygame.draw.rect(self.screen, (255,255,255), (0, 550, 1000, 300))

        # Draw images on screen 
        if len(self.imgs) > 0:
            self.screen.blit(self.imgs[0], (25,550))
            self.screen.blit(self.imgs[1], (150,550))
            self.screen.blit(self.imgs[2], (275,550))
            self.screen.blit(self.imgs[3], (400,550))

        # Log everything 
        self.log_dictionary['times'].append(time.time())
        self.log_dictionary['notes'].append([[n.type, n.y_pos, n.length] for n in self.notes])
        self.log_dictionary['button_pressed'].append(self.key_pressed)
        