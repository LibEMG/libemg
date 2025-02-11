import math
import time
from dataclasses import dataclass

import pygame
import numpy as np

from libemg.environments.controllers import Controller
from libemg.environments._base import Environment


OUTSIDE_TARGET = pygame.USEREVENT + 1
INSIDE_TARGET = pygame.USEREVENT + 2


@dataclass(frozen=True)
class FittsConfig:
    """Dataclass to customize parameters (e.g., target size and color) for a real-time Fitts' Law style task.

    Parameters
    ----------
    num_trials : int
        Number of trials user must complete.
    dwell_time : float, optional
        Time (in seconds) user must dwell in target to complete trial. Defaults to 1.0.
    timeout : float | None, optional
        Time limit (in seconds) that signifies a failed trial. If None, no timeout is used. Defaults to None.
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
    cursor_radius : int, optional
        Radius (in pixels) of cursor. Defaults to 14.
    game_time : float, optional
        Time (in seconds) that the task should run. If None, no time limit is set and the task ends when the number of targets are acquired.
        If a value is passed, the task is stopped when either the time limit has been reached or the number of trials has been acquired. Defaults to None.
    mapping : str, optional
        Space to map predictions to. Setting this to 'cartesian' uses the standard Fitts' style input space, where predictions map to the x and y position of the cursor.
        Setting this mapping to polar will instead map horizontal and vertical predictions to the radius and angle of a semi-circle, respectively (similar to spinning a wheel).
        Pass in 'polar+' or 'polar-' to map up or down to counter-clockwise changes in angle, respectively. Defaults to 'cartesian'.
    cursor_color : tuple, optional
        Color of cursor. Pass in a tuple of the format (red, green, blue), where each color value is in the range 0-255. Defaults to orange (255, 95, 31).
    cursor_in_target_color : tuple, optional
        Color of cursor when in target. Pass in a tuple of the format (red, green, blue), where each color value is in the range 0-255. Defaults to blue (0, 102, 204).
    target_color : tuple, optional
        Color of targets. Pass in a tuple of the format (red, green, blue), where each color value is in the range 0-255. Defaults to white (255, 255, 255).
    background_color : tuple, optional
        Color of background. Pass in a tuple of the format (red, green, blue), where each color value is in the range 0-255. Defaults to black (0, 0, 0).
    timer_color : tuple, optional
        Color of dwell timer. Pass in a tuple of the format (red, green, blue), where each color value is in the range 0-255. Defaults to blue (0, 102, 204).
    """
    num_trials: int
    dwell_time: float = 1.0
    timeout: float | None = None
    velocity: float = 25.0
    save_file: str | None = None
    width: int = 1250
    height: int = 750
    fps: int = 60
    proportional_control: bool = True
    target_radius: int = 40
    cursor_radius: int = 7
    game_time: float | None = None
    mapping: str = 'cartesian'
    cursor_color: tuple[int, int, int] = (255, 95, 31)
    cursor_in_target_color: tuple[int, int, int] = (0, 102, 204)
    target_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (0, 0, 0)
    timer_color: tuple[int, int, int] = (0, 102, 204)


class Fitts(Environment):
    def __init__(self, controller: Controller, config: FittsConfig, prediction_map: dict | None = None):
        """Fitts style task. Targets are generated at random and the user is asked to acquire targets as quickly as possible.

        Parameters
        ----------
        controller : Controller
            Interface to parse predictions which determine the direction of the cursor.
        config : FittsConfig
            Configuration class that determines environment parameters (e.g., target size).
        prediction_map : dict | None, optional
            Maps received control commands to cursor movement - only used if a non-continuous controller is used (e.g., classifier). If a continuous controller is used (e.g., regressor),
            then 2 DoFs are expected when parsing predictions and this parameter is not used. If None, a standard map for classifiers is created where 0, 1, 2, 3, 4 are mapped to
            down, up, no motion, right, and left, respectively. For custom mappings, pass in a dictionary where keys represent received control signals (from the Controller) and 
            values map to actions in the environment. Accepted actions are: 'S' (down), 'N' (up), 'NM' (no motion), 'E' (right), and 'W' (left). All of these actions must be 
            represented by a single key in the dictionary. Defaults to None.
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
        self.config = config
        super().__init__(controller, fps=self.config.fps, log_dictionary=log_dictionary, save_file=self.config.save_file)

        if self.config.mapping == 'cartesian':
            self.render_as_polar = False
        elif self.config.mapping in ['polar+', 'polar-']:
            self.render_as_polar = True
        else:
            raise ValueError(f"Unexpected value for mapping. Got: {self.config.mapping}.")

        if prediction_map is None:
            prediction_map = default_prediction_map
        assert set(np.unique(list(prediction_map.values()))) == set(list(default_prediction_map.values())), f"Did not find all commands {list(default_prediction_map.values())} represented as values in prediction_map. Got: {prediction_map}."

        self.prediction_map = prediction_map

        self.font = pygame.font.SysFont('helvetica', 40)
        self.screen = pygame.display.set_mode([self.config.width, self.config.height])

        # gameplay parameters
        self.trial = -1

        if '+' in self.config.mapping:
            theta_bounds = (math.pi, 0)
        else:
            theta_bounds = (0, math.pi)
        
        self.theta_bounds = theta_bounds
        self.polar_origin = (self.config.width // 2, self.config.height // 2)
        self.polar_origin = (self.config.width // 2, self.config.height)
        self.polygon_angles = np.linspace(0, 2 * math.pi, num=100)  # could change how many points are calculated based on desired FPS (having 1000 caused frame rate issues)

        # interface objects
        self.cursor = pygame.Rect(self.config.width // 2 - self.config.cursor_radius, self.config.height // 2 - self.config.cursor_radius,
                                  self.config.cursor_radius * 2, self.config.cursor_radius * 2)
        self._get_new_goal_target()
        self.current_direction = [0,0]

        self.timeout_timer = None
        self.trial_duration = 0
        self._info = ['predictions', 'timestamp']
        if self.config.proportional_control:
            self._info.append('pc')
        self.start_time = time.time()
        self.dwell_timer = None

    def _draw(self):
        self.screen.fill(self.config.background_color)
        self._draw_targets()
        self._draw_cursor()
        self._draw_timer()
    
    def _draw_targets(self):
        self._draw_circle(self.goal_target, self.config.target_color)
    
    def _draw_cursor(self):
        color = self.config.cursor_in_target_color if self.dwell_timer is not None else self.config.cursor_color
        self._draw_circle(self.cursor, color)

    def _draw_timer(self):
        if self.dwell_timer is not None:
            toc = time.perf_counter()
            duration = round((toc-self.dwell_timer),2)
            time_str = str(duration)
            draw_text = self.font.render(time_str, 1, self.config.timer_color)
            self.screen.blit(draw_text, (10, 10))

    def _update_game(self):
        self._draw()
        self._run_game_process()
        self._move()
    
    def _run_game_process(self):
        self._check_collisions()
        self._check_events()

    def _check_collisions(self):
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

        if self.config.game_time is not None and (time.time() - self.start_time) >= self.config.game_time:
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

            self.current_direction[0] += self.config.velocity * float(predictions[0]) * pc[0]
            self.current_direction[1] -= self.config.velocity * float(predictions[1]) * pc[1]    # -ve b/c pygame origin pixel is at top left of screen

            self._log(str(predictions), timestamp)
        
        if len(self.log_dictionary['time_stamp']) == 0:
            # No data has been received, so don't start counting
            print('Waiting for Fitts to receive data...')
            return

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
            if self.duration >= self.config.dwell_time:
                self._get_new_goal_target()

        if self.timeout_timer is None:
            self.timeout_timer = time.perf_counter()
        else:
            toc = time.perf_counter()
            self.trial_duration = round((toc - self.timeout_timer), 2)

        if self.config.timeout is not None and self.trial_duration >= self.config.timeout:
            # Timeout
            self._get_new_goal_target()

    def _move(self):
        self.cursor.left += self.current_direction[0]
        self.cursor.top += self.current_direction[1]

        # Ensure cursor stays in the bounds of the screen
        self.cursor.left = max(0, self.cursor.left)
        self.cursor.left = min(self.config.width - self.cursor.width, self.cursor.left)
        self.cursor.top = max(0, self.cursor.top)
        self.cursor.top = min(self.config.height - self.cursor.height, self.cursor.top)
    
    def _get_new_goal_target(self):
        self.dwell_timer = None
        self.timeout_timer = None
        self.trial_duration = 0

        max_radius = int(min(self.config.width, self.config.height) * 0.5)   # only create targets in a centered circle (based on size of screen)
        while True:
            target_radius = np.random.randint(self.cursor[2], self.config.target_radius)
            target_position_radius = np.random.randint(0, max_radius - target_radius)
            target_angle = np.random.uniform(0, 2 * math.pi)
            # Convert to cartesian (relative to pygame origin, not center of screen)
            x = self.config.width // 2 + target_position_radius * math.cos(target_angle)
            y = self.config.height // 2 - target_position_radius * math.sin(target_angle)  # subtract b/c y is inverted in pygame
            # Continue until we create a target that isn't on the cursor
            if math.dist((x, y), self.cursor.center) > (target_radius + self.cursor[2] // 2):
                break

        left = x - target_radius
        top = y - target_radius
        self.goal_target = pygame.Rect(left, top, target_radius * 2, target_radius * 2)

        self.trial += 1
        if self.trial == self.config.num_trials:
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
        radius = np.interp(x, (0, self.config.width), (0, min(self.config.width // 2, self.config.height)))  # limit radius based on screen dimensions so circles stay on screen
        theta = np.interp(y, (0, self.config.height), self.theta_bounds)

        # theta is the angle from the right side of the screen and goes counter-clockwise (same as unit circle)
        polar_x = radius * np.cos(theta) + self.polar_origin[0]
        polar_y = self.polar_origin[1] - radius * np.sin(theta) # subtract b/c y is inverted in pygame

        return polar_x, polar_y

    def _draw_circle(self, rect, color, fill = True, draw_radius = False):
        # Keep the underlying circle (e.g., target or cursor) coordinates the same, but render as polar to keep downstream calculations the same
        polygon_width = 0 if fill else 2
        target_radius = rect.width // 2

        if not self.render_as_polar:
            pygame.draw.circle(self.screen, color, rect.center, target_radius, width=polygon_width)
            return

        points = []
        for circle_theta in self.polygon_angles:
            # Create points to make a circle in Cartesian space
            x = rect.centerx + target_radius * np.cos(circle_theta)
            y = rect.centery + target_radius * np.sin(circle_theta)

            # Remap to polar equivalents
            polar_x, polar_y = self._map_to_polar_space(x, y)
            points.append((polar_x, polar_y))

        pygame.draw.polygon(self.screen, color, points, width=polygon_width)

        if draw_radius:
            # NOTE: This option is there, but I'm not sure that it should be used. You don't have runways in real life (or other Fitts tasks), so it might not be fair to add.
            # If we are going to add it, we'd need to have them for the angle (not just the radius)
            # Also the calculation of the radius should probably be a field because recalculating and rounding every time will cause instability when just changing the angle.
            semi_circle_x, semi_circle_y = self._map_to_polar_space(rect.centerx, rect.centery)
            semi_circle_radius = int(np.linalg.norm(np.array([semi_circle_x - self.polar_origin[0], semi_circle_y - self.polar_origin[1]])))
            pygame.draw.circle(self.screen, color, self.polar_origin, semi_circle_radius, width=2, draw_top_right=True, draw_top_left=True)
            pygame.draw.line(self.screen, color, (self.polar_origin[0] - semi_circle_radius, self.polar_origin[1]), (self.polar_origin[0] + semi_circle_radius, self.polar_origin[1]))

    def _run_helper(self):
        # updated frequently for graphics & gameplay
        self._update_game()
        pygame.display.set_caption(str(self.clock.get_fps()))


class ISOFitts(Fitts):
    def __init__(self, controller: Controller, config: FittsConfig, prediction_map: dict | None = None, num_targets: int = 8, target_distance_radius: int = 275):
        """ISO Fitts style task. Targets are generated in a circle and the user is asked to acquire targets as quickly as possible.

        Parameters
        ----------
        controller : Controller
            Interface to parse predictions which determine the direction of the cursor.
        config : FittsConfig
            Configuration class that determines environment parameters (e.g., target size).
        prediction_map : dict | None, optional
            Maps received control commands to cursor movement - only used if a non-continuous controller is used (e.g., classifier). If a continuous controller is used (e.g., regressor),
            then 2 DoFs are expected when parsing predictions and this parameter is not used. If None, a standard map for classifiers is created where 0, 1, 2, 3, 4 are mapped to
            down, up, no motion, right, and left, respectively. For custom mappings, pass in a dictionary where keys represent received control signals (from the Controller) and 
            values map to actions in the environment. Accepted actions are: 'S' (down), 'N' (up), 'NM' (no motion), 'E' (right), and 'W' (left). All of these actions must be 
            represented by a single key in the dictionary. Defaults to None.
        num_targets : int, optional
            Number of targets in task. Defaults to 8.
        target_distance_radius : int, optional
            Radius (in pixels) of target of targets in Iso Fitts' environment. Defaults to 275.
        """
        width_is_too_small = target_distance_radius > config.width // 2
        height_is_too_small = target_distance_radius > config.height // 2
        if width_is_too_small and height_is_too_small:
            error_info = f"width and height"
        elif width_is_too_small:
            error_info = f"width"
        elif height_is_too_small:
            error_info = f"height"
        else:
            error_info = None
        
        if error_info is not None:
            raise ValueError(f"Radius between ISO Fitts targets is larger than screen size will allow. "
                             f"Target distance radius must be less than half the screen dimensions. "
                             f"Please increase screen {error_info} or reduce target distance radius.")
        assert target_distance_radius < config.width // 2 and target_distance_radius < config.height // 2, f"Radius between ISO Fitts targets is larger than screen size will allow. Please increase screen size or reduce target distance radius."
        self.goal_target_idx = -1
        self.num_of_targets = num_targets
        self.big_rad = target_distance_radius

        # interface objects
        self.targets = []
        angle = 0
        angle_increment = 360 // self.num_of_targets
        while angle < 360:
            self.targets.append(pygame.Rect(
                (config.width // 2 - config.target_radius) + math.cos(math.radians(angle)) * self.big_rad,
                (config.height // 2 - config.target_radius) + math.sin(math.radians(angle)) * self.big_rad,
                config.target_radius * 2, config.target_radius * 2
            ))
            angle += angle_increment

        super().__init__(controller, config, prediction_map=prediction_map)

    def _draw_targets(self):
        for target in self.targets:
            self._draw_circle(target, self.config.target_color, fill=False) # draw target outlines
        
        self._draw_circle(self.goal_target, self.config.target_color, fill=True)    # fill in goal target

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
