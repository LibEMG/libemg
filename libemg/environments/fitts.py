import math
import time
from copy import deepcopy

import pygame
import numpy as np

from libemg.environments.controllers import Controller
from libemg.environments._base import Environment


OUTSIDE_TARGET = pygame.USEREVENT + 1
INSIDE_TARGET = pygame.USEREVENT + 2


class Fitts(Environment):
    def __init__(self, controller: Controller, prediction_map: dict | None = None, num_trials: int = 15, dwell_time: float = 3.0, timeout: float = 30.0, 
                 velocity: float = 25.0, save_file: str | None = None, width: int = 1250, height: int = 750, fps: int = 60, proportional_control: bool = True,
                 target_radius: int = 40, game_time: float | None = None, mapping: str = 'cartesian'):
        """Fitts style task. Targets are generated at random and the user is asked to acquire targets as quickly as possible.

        Parameters
        ----------
        controller : Controller
            Interface to parse predictions which determine the direction of the cursor.
        prediction_map : dict | None, optional
            Maps received control commands to cursor movement - only used if a non-continuous controller is used (e.g., classifier). If a continuous controller is used (e.g., regressor),
            then 2 DoFs are expected when parsing predictions and this parameter is not used. If None, a standard map for classifiers is created where 0, 1, 2, 3, 4 are mapped to
            down, up, no motion, right, and left, respectively. For custom mappings, pass in a dictionary where keys represent received control signals (from the Controller) and 
            values map to actions in the environment. Accepted actions are: 'S' (down), 'N' (up), 'NM' (no motion), 'E' (right), and 'W' (left). All of these actions must be 
            represented by a single key in the dictionary. Defaults to None.
        num_trials : int, optional
            Number of trials user must complete. Defaults to 15.
        dwell_time : float, optional
            Time (in seconds) user must dwell in target to complete trial. Defaults to 3.
        timeout : float, optional
            Time limit (in seconds) that signifies a failed trial. Defaults to 30.
        velocity : float, optional
            Velocity scalar that controls the max speed of the cursor. Defaults to 25.
        save_file : str | None, optional
            Name of save file (e.g., log.pkl). Supports .json and .pkl file formats. If None, no results are saved. Defaults to None.
        width : int, optional
            Width of display (in pixels). Defaults to 1250.
        height : int, optional
            Height of display (in pixels). Defaults to 750.
        fps : int, optional
            Frames per second (in Hz). Defaults to 60.
        proportional_control : bool, optional
            True if proportional control should be used, otherwise False. This value is ignored for Controllers that have proportional control built in, like regressors. Defaults to False.
        target_radius : int, optional
            Radius (in pixels) of each individual target. Defaults to 40.
        game_time : float, optional
            Time (in seconds) that the task should run. If None, no time limit is set and the task ends when the number of targets are acquired.
            If a value is passed, the task is stopped when either the time limit has been reached or the number of trials has been acquired. Defaults to None.
        mapping : str, optional
            Space to map predictions to. Setting this to 'cartesian' uses the standard Fitts' style input space, where predictions map to the x and y position of the cursor.
            Setting this mapping to polar will instead map vertical and horizontal predictions to the radius and angle of a semi-circle, respectively (similar to spinning a wheel).
            Pass in 'polar-right' or 'polar-left' to map to a semi-circle facing right or left, respectively. Defaults to 'cartesian'.
        """
        # logging information
        log_dictionary = {
            'time_stamp':        [],
            'trial_number':      [],
            'goal_target' :      [],
            'global_clock' :     [],
            'cursor_position':   [],
            'class_label':       [],
            'current_direction': []
        }
        default_prediction_map = {
            0: 'S',
            1: 'N',
            2: 'NM',
            3: 'E',
            4: 'W'
        }
        super().__init__(controller, fps=fps, log_dictionary=log_dictionary, save_file=save_file)
        self.mapping = mapping

        if self.mapping == 'cartesian':
            self.render_as_polar = False
        elif self.mapping in ['polar-left', 'polar-right']:
            self.render_as_polar = True
        else:
            raise ValueError(f"Unexpected value for mapping. Got: {self.mapping}.")

        if prediction_map is None:
            prediction_map = default_prediction_map
        assert set(np.unique(list(prediction_map.values()))) == set(list(default_prediction_map.values())), f"Did not find all commands {list(default_prediction_map.values())} represented as values in prediction_map. Got: {prediction_map}."

        self.draw_left = 'left' in self.mapping and self.render_as_polar
        self.prediction_map = prediction_map

        self.font = pygame.font.SysFont('helvetica', 40)
        self.screen = pygame.display.set_mode([width, height])

        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.YELLOW = (255,255,0)
        self.BLUE   = (0,102,204)
        self.small_rad = target_radius

        self.VEL = velocity
        self.dwell_time = dwell_time
        self.max_trial = num_trials
        self.width = width
        self.height = height
        self.max_radius = int(min(self.width, self.height) * 0.5)   # only create targets in a centered circle (based on size of screen)
        self.trial = -1

        # interface objects
        self.cursor = pygame.Rect(self.width//2 - 7, self.height//2 - 7, 14, 14)
        self._get_new_goal_target()
        self.current_direction = [0,0]

        self.timeout_timer = None
        self.timeout = timeout   # (seconds)
        self.trial_duration = 0
        self.proportional_control = proportional_control
        self._info = ['predictions', 'timestamp']
        if self.proportional_control:
            self._info.append('pc')
        self.start_time = time.time()
        self.game_time = game_time
        self.dwell_timer = None

    def _draw(self):
        self.screen.fill(self.BLACK)
        self._draw_targets()
        self._draw_cursor()
        self._draw_timer()
    
    def _draw_targets(self):
        self._draw_circle(self.goal_target, self.RED)
            
    def _draw_cursor(self):
        self._draw_circle(self.cursor, self.YELLOW)

    def _draw_timer(self):
        if self.dwell_timer is not None:
            toc = time.perf_counter()
            duration = round((toc-self.dwell_timer),2)
            time_str = str(duration)
            draw_text = self.font.render(time_str, 1, self.BLUE)
            self.screen.blit(draw_text, (10, 10))

    def _update_game(self):
        self._draw()
        self._run_game_process()
        self._move()
    
    def _run_game_process(self):
        self._check_collisions()
        self._check_events()

    def _check_collisions(self):
        # print(math.sqrt((self.goal_target.centerx - self.cursor.centerx)**2 + (self.goal_target.centery - self.cursor.centery)**2))
        if math.sqrt((self.goal_target.centerx - self.cursor.centerx)**2 + (self.goal_target.centery - self.cursor.centery)**2) < (self.goal_target[2]/2 + self.cursor[2]/2):
            pygame.event.post(pygame.event.Event(INSIDE_TARGET))
            self.Event_Flag = True
        else:
            pygame.event.post(pygame.event.Event(OUTSIDE_TARGET))
            self.Event_Flag = False

    def _check_events(self):
        # closing window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                return

        if self.game_time is not None and (time.time() - self.start_time) >= self.game_time:
            self.done = True
            return
            
        data = self.controller.get_data(self._info)
        
        self.current_direction = [0., 0.]
        if data is not None:
            # Move cursor
            predictions = data[0]
            timestamp = data[1]
            if len(data) == 3:
                pc = data[2]
            else:
                pc = [1. for _ in predictions]

            if len(predictions) == 1 and len(pc) == 1:
                # Output is a class/action, not a set of DOFs
                prediction = predictions[0]
                pc = pc[0]
                direction = self.prediction_map[prediction]

                if direction == 'N':
                    predictions = [0, 1]
                elif direction == 'E':
                    predictions = [1, 0]
                elif direction == 'S':
                    predictions = [0, -1]
                elif direction == 'W':
                    predictions = [-1, 0]
                elif direction == 'NM':
                    predictions = [0, 0]
                else:
                    raise ValueError(f"Expected prediction map to have keys 'N', 'E', 'S', 'W', and 'NM', but found key: {direction}.")
                
                pc = [pc, pc]

            self.current_direction[0] += self.VEL * float(predictions[0]) * pc[0]
            self.current_direction[1] -= self.VEL * float(predictions[1]) * pc[1]    # -ve b/c pygame origin pixel is at top left of screen

            self._log(str(predictions), timestamp)
        
        ## CHECKING FOR COLLISION BETWEEN CURSOR AND RECTANGLES
        if event.type == OUTSIDE_TARGET:
            if self.Event_Flag == False:
                self.dwell_timer = None
                self.duration = 0
        elif event.type == INSIDE_TARGET:
            if self.dwell_timer is None:
                self.dwell_timer = time.perf_counter()
            else:
                toc = time.perf_counter()
                self.duration = round((toc - self.dwell_timer), 2)
            if self.duration >= self.dwell_time:
                self._get_new_goal_target()
                self.dwell_timer = None

        if self.timeout_timer is None:
            self.timeout_timer = time.perf_counter()
        else:
            toc = time.perf_counter()
            self.trial_duration = round((toc - self.timeout_timer), 2)

        if self.trial_duration >= self.timeout:
            # Timeout
            self._get_new_goal_target()
            self.timeout_timer = None

    def _move(self):
        # Making sure its within the bounds of the screen
        right = self.cursor.left + self.cursor[2]
        bottom = self.cursor.top + self.cursor[2]
        if self.cursor.left + self.current_direction[0] > 0 and right + self.current_direction[0] < self.width:
            self.cursor.left += self.current_direction[0]
        if self.cursor.top + self.current_direction[1] > 0 and bottom + self.current_direction[1] < self.height:
            self.cursor.top += self.current_direction[1]
    
    def _get_new_goal_target(self):
        self.timeout_timer = None
        self.trial_duration = 0

        while True:
            target_radius = np.random.randint(self.cursor[2], self.small_rad)
            target_position_radius = np.random.randint(0, self.max_radius - target_radius)
            target_angle = np.random.uniform(0, 2 * math.pi)
            # Convert to cartesian (relative to pygame origin, not center of screen)
            x = self.width // 2 + target_position_radius * math.cos(target_angle)
            y = self.height // 2 - target_position_radius * math.sin(target_angle)  # subtract b/c y is inverted in pygame
            # Continue until we create a target that isn't on the cursor
            if math.dist((x, y), self.cursor.center) > (target_radius + self.cursor[2] // 2):
                break

        left = x - target_radius
        top = y - target_radius
        self.goal_target = pygame.Rect(left, top, target_radius * 2, target_radius * 2)

        self.trial += 1
        if self.trial == self.max_trial:
            self.done = True

    def _log(self, label, timestamp):
        self.log_dictionary['time_stamp'].append(timestamp)
        self.log_dictionary['trial_number'].append(self.trial)
        self.log_dictionary['goal_target'].append((self.goal_target.centerx, self.goal_target.centery, self.goal_target[2]))
        self.log_dictionary['global_clock'].append(time.perf_counter())
        self.log_dictionary['cursor_position'].append((self.cursor.centerx, self.cursor.centery, self.cursor[2]))
        self.log_dictionary['class_label'].append(label) 
        self.log_dictionary['current_direction'].append(self.current_direction)

    def _map_to_polar_space(self, x, y):
        radius = np.interp(x, (0, self.width), (0, self.max_radius))
        theta = np.interp(y, (0, self.height), (0, math.pi))

        # theta is the angle from the bottom of the circle, so sin gives the x component and cos gives the x component
        radius_multiplier = -1 if self.draw_left else 1 # subtract radius so points are drawn on the left of the center of the screen
        polar_x = radius_multiplier * radius * np.sin(theta) + self.width // 2
        polar_y = self.height // 2 -radius * np.cos(theta)  # subtract b/c y is inverted in pygame

        return polar_x, polar_y

    def _draw_circle(self, rect, color, fill = True):
        # Keep the underlying circle (e.g., target or cursor) coordinates the same, but render as polar to keep downstream calculations the same
        polygon_width = 0 if fill else 2
        target_radius = rect.width // 2

        if not self.render_as_polar:
            pygame.draw.circle(self.screen, color, rect.center, target_radius, width=polygon_width)
            return

        points = []
        for circle_theta in np.linspace(0, 2 * math.pi, num=100):
            # Create points to make a circle in Cartesian space
            x = rect.centerx + target_radius * np.cos(circle_theta)
            y = rect.centery + target_radius * np.sin(circle_theta)

            # Remap to polar equivalents
            polar_x, polar_y = self._map_to_polar_space(x, y)
            points.append((polar_x, polar_y))

        pygame.draw.polygon(self.screen, color, points, width=polygon_width)

    def _run_helper(self):
        # updated frequently for graphics & gameplay
        self._update_game()
        pygame.display.set_caption(str(self.clock.get_fps()))


class ISOFitts(Fitts):
    def __init__(self, controller: Controller, prediction_map: dict | None = None, num_targets: int = 30, num_trials: int = 15, dwell_time: float = 3.0, timeout: float = 30.0, 
                 velocity: float = 25.0, save_file: str | None = None, width: int = 1250, height: int = 750, fps: int = 60, proportional_control: bool = True,
                 target_radius: int = 40, target_distance_radius: int = 275, game_time: float | None = None, mapping: str = 'cartesian'):
        """ISO Fitts style task. Targets are generated in a circle and the user is asked to acquire targets as quickly as possible.

        Parameters
        ----------
        controller : Controller
            Interface to parse predictions which determine the direction of the cursor.
        prediction_map : dict | None, optional
            Maps received control commands to cursor movement - only used if a non-continuous controller is used (e.g., classifier). If a continuous controller is used (e.g., regressor),
            then 2 DoFs are expected when parsing predictions and this parameter is not used. If None, a standard map for classifiers is created where 0, 1, 2, 3, 4 are mapped to
            down, up, no motion, right, and left, respectively. For custom mappings, pass in a dictionary where keys represent received control signals (from the Controller) and 
            values map to actions in the environment. Accepted actions are: 'S' (down), 'N' (up), 'NM' (no motion), 'E' (right), and 'W' (left). All of these actions must be 
            represented by a single key in the dictionary. Defaults to None.
        num_targets : int, optional
            Number of targets in task. Defaults to 30.
        num_trials : int, optional
            Number of trials user must complete. Defaults to 15.
        dwell_time : float, optional
            Time (in seconds) user must dwell in target to complete trial. Defaults to 3.
        timeout : float, optional
            Time limit (in seconds) that signifies a failed trial. Defaults to 30.
        velocity : float, optional
            Velocity scalar that controls the max speed of the cursor. Defaults to 25.
        save_file : str | None, optional
            Name of save file (e.g., log.pkl). Supports .json and .pkl file formats. If None, no results are saved. Defaults to None.
        width : int, optional
            Width of display (in pixels). Defaults to 1250.
        height : int, optional
            Height of display (in pixels). Defaults to 750.
        fps : int, optional
            Frames per second (in Hz). Defaults to 60.
        proportional_control : bool, optional
            True if proportional control should be used, otherwise False. This value is ignored for Controllers that have proportional control built in, like regressors. Defaults to False.
        target_radius : int, optional
            Radius (in pixels) of each individual target. Defaults to 40.
        target_distance_radius : int, optional
            Radius (in pixels) of target of targets in Iso Fitts' environment. Defaults to 275.
        game_time : float, optional
            Time (in seconds) that the task should run. If None, no time limit is set and the task ends when the number of targets are acquired.
            If a value is passed, the task is stopped when either the time limit has been reached or the number of trials has been acquired. Defaults to None.
        mapping : str, optional
            Space to map predictions to. Setting this to 'cartesian' uses the standard Fitts' style input space, where predictions map to the x and y position of the cursor.
            Setting this mapping to polar will instead map vertical and horizontal predictions to the radius and angle of a semi-circle, respectively (similar to spinning a wheel).
            Pass in 'polar-right' or 'polar-left' to map to a semi-circle facing right or left, respectively. Defaults to 'cartesian'.
        """
        self.goal_target_idx = -1
        self.num_of_targets = num_targets
        self.big_rad = target_distance_radius

        # interface objects
        self.targets = []
        angle = 0
        angle_increment = 360 // self.num_of_targets
        while angle < 360:
            self.targets.append(pygame.Rect((width//2 - target_radius) + math.cos(math.radians(angle)) * target_distance_radius, (height//2 - target_radius) + math.sin(math.radians(angle)) * target_distance_radius, target_radius * 2, target_radius * 2))
            angle += angle_increment

        super().__init__(controller, prediction_map=prediction_map, num_trials=num_trials, dwell_time=dwell_time, timeout=timeout, velocity=velocity,
                         save_file=save_file, width=width, height=height, fps=fps, proportional_control=proportional_control, target_radius=target_radius, game_time=game_time,
                         mapping=mapping)

    def _draw_targets(self):
        for target in self.targets:
            self._draw_circle(target, self.RED, fill=False) # draw target outlines
        
        self._draw_circle(self.goal_target, self.RED, fill=True)    # fill in goal target

    def _get_new_goal_target(self):
        super()._get_new_goal_target()
        if self.goal_target_idx == -1:
            self.goal_target_idx = 0
            self.next_target_in = self.num_of_targets//2
            self.target_jump = 0
        else:
            self.goal_target_idx =  (self.goal_target_idx + self.next_target_in )% self.num_of_targets
            if self.target_jump == 0:
                self.next_target_in = self.num_of_targets//2 + 1
                self.target_jump = 1
            else:
                self.next_target_in = self.num_of_targets // 2
                self.target_jump = 0
        self.goal_target = self.targets[self.goal_target_idx]
