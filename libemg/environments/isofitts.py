import math
import time

import pygame
import numpy as np

from libemg.environments.controllers import Controller
from libemg.environments._base import Environment

class IsoFitts(Environment):
    def __init__(self, controller: Controller, prediction_map: dict | None = None, num_targets: int = 30, num_trials: int = 15, dwell_time: float = 3.0, timeout: float = 30.0, 
                 velocity: float = 25.0, save_file: str | None = None, width: int = 1250, height: int = 750, fps: int = 60, proportional_control: bool = True,
                 target_radius: int = 40, target_distance_radius: int = 275, game_time: float | None = None):
        """Iso Fitts style task. Targets are generated in a target and the user is asked to acquire targets as quickly as possible.

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
        super().__init__(controller, fps=fps, log_dictionary=log_dictionary, save_file=save_file)
        if prediction_map is None:
            prediction_map = {
                0: 'S',
                1: 'N',
                2: 'NM',
                3: 'E',
                4: 'W'
            }
        assert set(np.unique(list(prediction_map.values()))) == set(['N', 'E', 'S', 'W', 'NM']), f"Did not find all commands ('N', 'E', 'S', 'W', and 'NM') represented as values in prediction_map. Got: {prediction_map}."
        self.prediction_map = prediction_map

        self.font = pygame.font.SysFont('helvetica', 40)
        self.screen = pygame.display.set_mode([width, height])
        

        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.YELLOW = (255,255,0)
        self.BLUE   = (0,102,204)
        self.small_rad = target_radius
        self.big_rad = target_distance_radius
        self.pos_factor1 = self.big_rad/2
        self.pos_factor2 = (self.big_rad * math.sqrt(3))//2

        self.VEL = velocity
        self.dwell_time = dwell_time
        self.num_of_targets = num_targets 
        self.max_trial = num_trials
        self.width = width
        self.height = height
        self.trial = 0

        # interface objects
        self.targets = []
        self.cursor = pygame.Rect(self.width//2 - 7, self.height//2 - 7, 14, 14)
        self.goal_target = -1
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

    def _draw(self):
        self.screen.fill(self.BLACK)
        self._draw_targets()
        self._draw_cursor()
        self._draw_timer()
    
    def _draw_targets(self):
        if not len(self.targets):
            angle = 0
            angle_increment = 360 // self.num_of_targets
            while angle < 360:
                self.targets.append(pygame.Rect((self.width//2 - self.small_rad) + math.cos(math.radians(angle)) * self.big_rad, (self.height//2 - self.small_rad) + math.sin(math.radians(angle)) * self.big_rad, self.small_rad * 2, self.small_rad * 2))
                angle += angle_increment

        for target in self.targets:
            pygame.draw.circle(self.screen, self.RED, (target.x + self.small_rad, target.y + self.small_rad), self.small_rad, 2)
        
        goal_target = self.targets[self.goal_target]
        pygame.draw.circle(self.screen, self.RED, (goal_target.x + self.small_rad, goal_target.y + self.small_rad), self.small_rad)
            
    def _draw_cursor(self):
        pygame.draw.circle(self.screen, self.YELLOW, (self.cursor.left + 7, self.cursor.top + 7), 7)

    def _draw_timer(self):
        if hasattr(self, 'dwell_timer'):
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
        target = self.targets[self.goal_target]
        if math.sqrt((target.centerx - self.cursor.centerx)**2 + (target.centery - self.cursor.centery)**2) < (target[2]/2 + self.cursor[2]/2):
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.goal_target))
            self.Event_Flag = True
        else:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.num_of_targets))
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
        if event.type >= pygame.USEREVENT and event.type < pygame.USEREVENT + self.num_of_targets:
            if self.dwell_timer is None:
                self.dwell_timer = time.perf_counter()
            else:
                toc = time.perf_counter()
                self.duration = round((toc - self.dwell_timer), 2)
            if self.duration >= self.dwell_time:
                self._get_new_goal_target()
                self.dwell_timer = None
                if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                    self.trial += 1
                else:
                    self.done = True
        elif event.type == pygame.USEREVENT + self.num_of_targets:
            if self.Event_Flag == False:
                self.dwell_timer = None
                self.duration = 0
        if self.timeout_timer is None:
            self.timeout_timer = time.perf_counter()
        else:
            toc = time.perf_counter()
            self.trial_duration = round((toc - self.timeout_timer), 2)
        if self.trial_duration >= self.timeout:
            # Timeout
            self._get_new_goal_target()
            self.timeout_timer = None
            if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                self.trial += 1
            else:
                self.done = True

    def _move(self):
        # Making sure its within the bounds of the screen
        if self.cursor.left + self.current_direction[0] > 0 and self.cursor.left + self.current_direction[0] < self.width:
            self.cursor.left += self.current_direction[0]
        if self.cursor.top + self.current_direction[1] > 0 and self.cursor.top + self.current_direction[1] < self.height:
            self.cursor.top += self.current_direction[1]
    
    def _get_new_goal_target(self):
        if self.goal_target == -1:
            self.goal_target = 0
            self.next_target_in = self.num_of_targets//2
            self.target_jump = 0
        else:
            self.goal_target =  (self.goal_target + self.next_target_in )% self.num_of_targets
            if self.target_jump == 0:
                self.next_target_in = self.num_of_targets//2 + 1
                self.target_jump = 1
            else:
                self.next_target_in = self.num_of_targets // 2
                self.target_jump = 0
        self.timeout_timer = None
        self.trial_duration = 0

    def _log(self, label, timestamp):
        target = self.targets[self.goal_target]
        self.log_dictionary['time_stamp'].append(timestamp)
        self.log_dictionary['trial_number'].append(self.trial)
        self.log_dictionary['goal_target'].append((target.centerx, target.centery, target[2]))
        self.log_dictionary['global_clock'].append(time.perf_counter())
        self.log_dictionary['cursor_position'].append((self.cursor.centerx, self.cursor.centery, self.cursor[2]))
        self.log_dictionary['class_label'].append(label) 
        self.log_dictionary['current_direction'].append(self.current_direction)

    def _run_helper(self):
        # updated frequently for graphics & gameplay
        self._update_game()
        pygame.display.set_caption(str(self.clock.get_fps()))


class RotationalIsoFitts(IsoFitts):
    # def _draw_targets(self):
    #     if not len(self.targets):
    #         self.angle = 0
    #         self.angle_increment = 360 // self.num_of_targets
    #         while self.angle < 360:
    #             self.targets.append(pygame.Rect((self.width//2 - self.small_rad) + math.cos(math.radians(self.angle)) * self.big_rad, (self.height//2 - self.small_rad) + math.sin(math.radians(self.angle)) * self.big_rad, self.small_rad * 2, self.small_rad * 2))
                
    #             self.angle += self.angle_increment

    #     for target in self.targets:
    #         pygame.draw.circle(self.screen, self.RED, (target.x + self.small_rad, target.y + self.small_rad), self.small_rad, 2)
        
    #     goal_target = self.targets[self.goal_target]
    #     pygame.draw.circle(self.screen, self.RED, (goal_target.x + self.small_rad, goal_target.y + self.small_rad), self.small_rad)

    def _draw_cursor(self):
        screen_width = self.screen.get_size()[0]
        angle = float(np.interp(self.cursor.left, [0, screen_width], [-math.pi / 2, math.pi / 2]))   # question: should it be 180 or 360??
        arrow_length = 40
        head_width = 15
        arrow_x = int(arrow_length * np.cos(angle))
        arrow_y = int(arrow_length * np.sin(angle))
        start_position = (screen_width // 2, self.cursor.top)
        end_position = (start_position[0] + arrow_x, start_position[1] + arrow_y)

        # bottom_left = (end_position[0] - head_width * np.cos(angle + math.pi / 4), end_position[1] - head_width * np.sin(angle + math.pi / 4))
        # bottom_right = (end_position[0] - head_width * np.cos(angle - math.pi / 4), end_position[1] - head_width * np.sin(angle - math.pi / 4))

        pygame.draw.line(self.screen, self.YELLOW, start_position, end_position, width=5)
        # pygame.draw.line(self.screen, self.YELLOW, bottom_left, end_position, width=line_width)
        # pygame.draw.line(self.screen, self.YELLOW, bottom_right, end_position, width=line_width)
        points = [
            (end_position[0] - head_width * np.cos(angle + math.pi / 4), end_position[1] - head_width * np.sin(angle + math.pi / 4)),
            (end_position[0] - head_width * np.cos(angle - math.pi / 4), end_position[1] - head_width * np.sin(angle - math.pi / 4)),
            (end_position[0] + 5 * np.cos(angle), end_position[1] + 5 * np.sin(angle))
        ]
        pygame.draw.polygon(self.screen, self.YELLOW, points)
        # TODO: This still doesn't look fantastic, but it's a start


    # def _check_collisions(self):
    #     ...
