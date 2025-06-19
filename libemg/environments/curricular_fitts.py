from libemg.output_writer import OutputWriter
from libemg.environments.controllers import Controller
from libemg.environments._base import Environment
from libemg.adaptation._base import produce_tciil_feedback
from dataclasses import dataclass
from multiprocessing import Process
from typing import Sequence
import time
import pickle
import pygame
import random
import numpy as np

@dataclass
class CurricularFittsConfig:
    """Configuration for the Curricular Fitts environment.

    Parameters
    ----------
    num_trials : int
        Number of fitts law target to spawn before the game is done.
    width : int
        Width of the main panel in pixels.
    height : int
        Height of the main panel in pixels.
    fps : int
        Frames per second (Hz).
    block_height : int
        Height of the info-block in pixels.
    block_width : int
        Width of the info-block in pixels.
    default_target_radius : int
        Default radius of the target in pixels.
    default_timeout : float
        Default timeout for the target in seconds.
    default_speed : float
        Default speed of the target in pixels per second.
    cursor_radius : int
        Radius of the cursor in pixels.
    brownian_motion : bool
        Whether the target should move (randomly).
    cursor_speed_multiplier : int
        The number multiplied by the controller output to determine the cursor speed.
    color_bg : tuple[int, int, int]
        Background color of the main panel in RGB format.
    color_block_border : tuple[int, int, int]
        Border color of the info-block in RGB format.
    color_block_fill : tuple[int, int, int]
        Fill color of the info-block in RGB format.
    color_cursor : tuple[int, int, int]
        Color of the cursor in RGB format.
    color_target : tuple[int, int, int]
        Color of the target in RGB format.
    """
    # game parameters
    num_trials: int = 100

    # main panel parameters
    width: int = 1000
    height: int = 1080
    fps: int = 60

    # info-block
    block_height: int = 80
    block_width: int = 1000

    # cursor / target defaults
    default_target_radius: int = 25
    default_timeout: float = 10.0
    default_speed: float = 1.0
    cursor_radius: int = 5
    brownian_motion: bool = False
    cursor_speed_multiplier: int = 20
    target_countdown: float = 0.5 # seconds

    # color scheme - darkula by default
    color_bg: tuple[int, int, int] = (0,0,0)
    color_block_border: tuple[int, int, int] = (10, 0, 71)
    color_block_fill: tuple[int, int, int] = (0, 70, 135)
    color_cursor: tuple[int, int, int] = (0, 255, 210)
    color_target: tuple[int, int, int] = (255, 68, 153)
    color_target_good: tuple[int, int, int]  = (0, 255, 0)  # Color for successful target acquisition

    # controller related parameters
    controller_fields : tuple[str, str] = ('predictions', 'timestamp') # for regression; add 'pc' for classification
    controller_map :  tuple[int, int] = (1,1)
    
    # feedback configuration
    feedback_handle: callable = produce_tciil_feedback

class Target:
    def __init__(self, 
                 screen,
                 config: CurricularFittsConfig, 
                 radius : int = 10, 
                 speed: float = 1):
        self.screen = screen
        self.config = config
        self.color = [config.color_target, config.color_target_good]
        self.contact = 0  # 0: not in contact, 1: in contact
        self.radius = radius
        self.randomize_location()
        self.speed = speed
        self.direction = [random.uniform(-1, 1), random.uniform(-1, 1)]

    def randomize_location(self):
        x = random.randint(self.radius, self.config.width - self.radius)
        y = random.randint(self.config.block_height + self.radius, self.config.height - self.radius)  # Spawn below the information block
        self.position =  [x, y]

    def update(self):
        if self.config.brownian_motion:
            # Update target position with Brownian motion
            self.position[0] += self.direction[0] * self.speed
            self.position[1] += self.direction[1] * self.speed
            
            # Randomly change direction occasionally
            if random.random() < 0.01:  # Adjust frequency of direction change
                self.direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
            
            # Keep the target within the bounds of the play area
            if self.position[0] < self.radius or self.position[0] > self.config.width - self.radius:
                self.direction[0] *= -1
            if self.position[1] < self.config.block_height + self.radius or self.position[1] > self.config.height - self.radius:
                self.direction[1] *= -1

    def draw(self):
        if self.contact:
            pygame.draw.circle(self.screen, self.color[1], (int(self.position[0]), int(self.position[1])), self.radius)
        else:
            pygame.draw.circle(self.screen, self.color[0], (int(self.position[0]), int(self.position[1])), self.radius)

    def reset(self):
        self.randomize_location()  # Spawn in a new location

class BaseTargetGenerator:
    def __init__(self,
                  N,
                  F,
                  P):
        self.N = N  # Number of trials or reversals
        self.F = F  # Factor for increase on miss
        self.P = P  # Factor for decrease on hit
        self.yields = []  # Store yields for logging

    def save_yields(self, filename="target_yields.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.yields, f)

class ConstantTargetGenerator(BaseTargetGenerator):
    def __init__(self,
                 config: CurricularFittsConfig):
        super().__init__(N=config.num_trials,
                         F=0, # nothing changes
                         P=0  # nothing changes
        )
        self.config = config
        self.current_radius = config.default_target_radius
        self.current_timeout = config.default_timeout
        self.counter = 0

    def generate(self, 
                 result: bool):
        # result = 1 : Passed
        # result = 0 : Failed (via timeout or miss)
        if self.counter < self.N:
            self.counter += 1
            self.yields.append((self.current_radius, 0, self.current_timeout))
            return self.current_radius, 0, self.current_timeout

class RadiusTargetGenerator(BaseTargetGenerator):
    def __init__(self,
                 config: CurricularFittsConfig,
                 F,
                 P):
        super().__init__(N=config.num_trials,
                         F=F,
                         P=P)
        self.config = config
        self.current_radius = config.default_target_radius
        self.last_result      = -1
        self.counter          = 0
        self.reversal_counter = 0

    def generate(self, result):
        # a success
        if result == 1:
            factor = 1 - (self.P / 100)
            if int(self.current_radius * factor) == self.current_radius and self.P != 0:
                # if its just marginally smaller, reduce by a whole pixel.
                self.current_radius = self.current_radius - 1
            else:
                self.current_radius *= factor
        # a fail
        elif result == 0:
            factor = 1 + (self.F / 100)
            if int(self.current_radius * factor) == self.current_radius and self.F != 0:
                self.current_radius = self.current_radius + 1
            else:
                self.current_radius *= factor
        # first spawn
        else:
            pass
        self.current_radius = int(max([self.config.cursor_radius, self.current_radius]))# don't allow it smaller than cursor radius
        
        # check reversals
        if self.last_result != -1:
            if result != self.last_result:
                self.reversal_counter += 1
        
        self.counter += 1
        self.last_result = result

        self.yields.append((self.current_radius, 0, self.config.default_timeout))  # Assume constant radius and speed of zero
        return self.current_radius, 0, self.config.default_timeout

class SpeedTargetGenerator(BaseTargetGenerator):
    def __init__(self, 
                 config : CurricularFittsConfig,
                 F,
                 P):
        super().__init__(N=config.num_trials, F=F, P=P)
        self.config = config
        self.current_speed = config.default_speed
        self.last_result      = -1
        self.counter          = 0
        self.reversal_counter = 0

    def generate(self, result):
        # a success
        if result == 1:
            self.current_speed *= (1 + self.P / 100)
        # a fail
        elif result == 0:
            self.current_speed *= (1 - self.F / 100)
        # first spawn
        else:
            pass
        
        # check reversals
        if self.last_result != -1:
            if result != self.last_result:
                self.reversal_counter += 1
        
        self.counter += 1
        self.last_result = result

        self.yields.append((self.config.default_target_radius, self.current_speed, self.config.default_timeout))  # Assume constant radius and speed
        return self.config.default_target_radius, self.current_speed, self.config.default_timeout

class TimeoutTargetGenerator(BaseTargetGenerator):
    def __init__(self, 
                 config : CurricularFittsConfig,
                 F,
                 P):
        super().__init__(N=config.num_trials,
                         F=F,
                         P=P)
        self.config = config
        self.current_timeout = config.default_timeout
        self.last_result      = -1
        self.counter          = 0
        self.reversal_counter = 0

    def generate(self, result):
        # a success
        if result == 1:
            self.current_timeout *= (1 - self.P / 100)
        # a fail
        elif result == 0:
            self.current_timeout *= (1 + self.F / 100)
        # first spawn
        else:
            pass
        
        # check reversals
        if self.last_result != -1:
            if result != self.last_result:
                self.reversal_counter += 1
        
        self.counter += 1
        self.last_result = result

        self.yields.append((self.config.default_target_radius, 0, self.current_timeout))  # Assume constant radius and speed of 0
        return self.config.default_target_radius, 0, self.current_timeout

class Log:
    def __init__(self):
        self.entries = {
            "trial_number": [],
            "target_position": [],
            "cursor_position": [],
            "target_size": [],
            "feedback": [],
            "timestamp": []
        }

    def record(self, trial_number, target_position, cursor_position, target_size, feedback, timestamp):
        self.entries['trial_number'].append(trial_number)
        self.entries['target_position'].append(target_position)
        self.entries['cursor_position'].append(cursor_position)
        self.entries['target_size'].append(target_size)
        self.entries['feedback'].append(feedback)
        self.entries['timestamp'].append(timestamp)

    def save(self, dir, trial_number, result):
        filename = f'trial_log_{trial_number}_{result}.pkl'
        with open(dir + "/" + filename, 'wb') as f:
            pickle.dump(self, f)
    
    def __add__(self, obj):
        log = Log()
        log.entries['trial_number'].extend(self.entries['trial_number'])
        log.entries['trial_number'].extend(obj.entries['trial_number'])
        log.entries['target_position'].extend(self.entries['target_position'])
        log.entries['target_position'].extend(obj.entries['target_position'])
        log.entries['cursor_position'].extend(self.entries['cursor_position'])
        log.entries['cursor_position'].extend(obj.entries['cursor_position'])
        log.entries['target_size'].extend(self.entries['target_size'])
        log.entries['target_size'].extend(obj.entries['target_size'])
        log.entries['feedback'].extend(self.entries['feedback'])
        log.entries['feedback'].extend(obj.entries['feedback'])
        log.entries['timestamp'].extend(self.entries['timestamp'])
        log.entries['timestamp'].extend(obj.entries['timestamp'])
        return log

class InformationBlock:
    def __init__(self, 
                 screen,
                 config: CurricularFittsConfig,
                 font_size: int = 50,
                 header_font_size: int = 30,
                 countdown_seconds: int = 60):
        self.screen = screen
        self.config = config
        self.rect = pygame.Rect(0, 0, config.block_width, config.block_height)
        self.font = pygame.font.Font(None, font_size)
        self.header_font = pygame.font.Font(None, header_font_size)
        self.update_countdown_seconds(countdown_seconds)
        self.update_trial_number(0)
        self.update_metadata("")
        self.color = config.color_cursor  # Default text color

    def update_trial_number(self, trial_number):
        self.trial_number = trial_number

    def update_metadata(self, metadata):
        self.metadata = metadata
    
    def update_countdown_seconds(self, countdown_seconds):
        self.countdown_seconds = countdown_seconds
        self.start_time = time.time()

    def get_remaining_time(self):
        elapsed_time = time.time() - self.start_time
        remaining_time = self.countdown_seconds - elapsed_time
        return max(remaining_time, 0)

    def draw(self):
        pygame.draw.rect(self.screen, self.config.color_block_fill, self.rect)
        pygame.draw.rect(self.screen, self.config.color_block_border, self.rect, 2)  # Draw border

        headers = ["Trial:", "Time:", "Metadata:"]
        values = [f"{self.trial_number}", f"{self.get_remaining_time():.1f}s", self.metadata]

        header_surfaces = [self.header_font.render(header, True, self.color) for header in headers]
        value_surfaces = [self.font.render(value, True, self.color) for value in values]

        column_width = self.rect.width // 3

        for i, header_surface in enumerate(header_surfaces):
            header_x = self.rect.x + (i * column_width) + (column_width - header_surface.get_width()) // 2
            value_x = self.rect.x + (i * column_width) + (column_width - value_surfaces[i].get_width()) // 2

            header_y = self.rect.y + 5
            value_y = header_y + header_surface.get_height() + 5

            self.screen.blit(header_surface, (header_x, header_y))
            self.screen.blit(value_surfaces[i], (value_x, value_y))


class Cursor:
    def __init__(self, 
                 screen,
                 config: CurricularFittsConfig):
        self.screen = screen
        self.config = config
        self.radius = config.cursor_radius
        self.color = config.color_cursor  # Cursor color
        self.position = [0, 0]  # Initial position

    def update(self, update_direction):
        if update_direction is not None:
            self.position[0] += update_direction[0] * self.config.cursor_speed_multiplier * self.config.controller_map[0]
            self.position[1] += update_direction[1] * self.config.cursor_speed_multiplier * self.config.controller_map[1]

            # Ensure cursor stays in the bounds of the screen
            self.position[0] = max(0, self.position[0])
            self.position[0] = min(self.config.width - self.radius, self.position[0])
            self.position[1] = max(0, self.position[1])
            self.position[1] = min(self.config.height - self.radius, self.position[1])

    def draw(self):
        pygame.draw.circle(self.screen, self.color, (int(self.position[0]), int(self.position[1])), self.radius)


class CurricularFitts(Environment):
    """
    """

    def __init__(self,
                  controller: Controller,
                  config: CurricularFittsConfig,
                  target_generator: BaseTargetGenerator = None,
                  environment_ow: list[OutputWriter] = [],
                  save_file: str | None = None):
        
        super().__init__(controller, 
                         fps=config.fps,
                         log_dictionary=None,
                         save_file=save_file)
        
        self.config = config

        self.target_generator = target_generator or ConstantTargetGenerator(
            radius  = self.config.default_target_radius,
            timeout = self.config.default_timeout 
        )

        self.environment_ow = environment_ow

        self.predictions = None
        self.timestamp = None

        self.game_feedback = None
        self.model_feedback = None

    def game_setup(self):
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("LibEMG -> Curricular Fitts Law")

        self.cursor = Cursor(self.screen, self.config)
        self.target = Target(self.screen, self.config, radius=self.config.default_target_radius, speed=self.config.default_speed)
        self.log = Log()

        self.trial_number = 1
        self._start_new_trial(initial=True)

    def _start_new_trial(self,
                         initial: bool = False,
                         result:  bool = True):
        """
        Get new target parameters (size, timeout, speed) from the generator, and use this to setup the next target.
        """
        # if we've finished a trial, save the log, increment the trial counter
        if not initial:
            self.log.save(self.save_file, self.trial_number, result)
            self.log = Log()
            self.trial_number += 1
        
        self.cursor_timer = time.time()

        # get the radius, speed, and timeout from the generator
        radius, speed, timeout = self.target_generator.generate(result)

        self.timeout = timeout
        self.target.radius = radius
        self.target.speed = speed
        self.target.randomize_location()

        # update the information block
        self.info_block = InformationBlock(
            self.screen,
            self.config,
            countdown_seconds = timeout
        )
        self.info_block.update_trial_number(self.trial_number)

        self.trial_distance = np.linalg.norm([x - y for x,y in zip(self.cursor.position, self.target.position)])
    
    def _run_loop(self):
        self.input()
        self.update()
        self.draw()
        if self.trial_number > self.config.num_trials:
            self.done = True
            return

    def input(self):
        self.pygame_inputs()
        self.controller_inputs()
        self.check_collisions()
    
    def pygame_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

    def controller_inputs(self):
        # get the controller message

        # get the feedback ready for this message
        data = self.controller.get_data(self.config.controller_fields)
        if data is not None:
            self.predictions = data[0] 
            self.direction = [i*j for i,j in zip(self.predictions, self.config.controller_map)]
            self.timestamp = data[1]
            # game feedback is in the game space, i.e, down is positive y, right is positive x.
            self.game_feedback = self.config.feedback_handle(self.cursor.position, self.direction, self.target.position, self.target.radius, self.trial_distance)
            # model feedback is in the classifier space, so we need to transform it BACK via multiplying by controller_map again
            self.model_feedback = [i*j for i,j in zip(self.game_feedback, self.config.controller_map)]
            self.info = {'timestamp': self.timestamp,
                         'environment_feedback': self.model_feedback,
                         'game_feedback': self.game_feedback,
                         'trial': self.trial_number,
                         'prediction': self.predictions,
                         'direction': self.direction}
            self.log.record(self.trial_number, self.target.position, self.cursor.position, self.target.radius, self.game_feedback, self.timestamp)

            if self.environment_ow is not None:
                
                self.environment_ow[0].write(self.info)
            # make self._info,
            # save last timestamp, last controller output, etc.

    
    def check_collisions(self):
        target_rect = pygame.Rect(self.target.position[0] - self.target.radius,
                                self.target.position[1] - self.target.radius,
                                self.target.radius * 2, self.target.radius * 2)
        
        # TODO: Make a countdown for this to acquire the target
        if target_rect.collidepoint(self.cursor.position):
            self.target.contact = 1
            if time.time() - self.cursor_timer > self.config.target_countdown:
                self.info_block.update_metadata("Hit!")
                self._start_new_trial(initial=False, result=1)
        else:
            self.cursor_timer = time.time()
            self.target.contact = 0

    def update(self):
        self.target.update()
        self.cursor.update(self.predictions)
        
        if self.info_block.get_remaining_time() <= 0:
            self.info_block.update_metadata("Timeout!")
            self._start_new_trial(initial=0, result=0)

    def draw(self):
        self.screen.fill(self.config.color_bg)
        self.info_block.draw()
        self.target.draw()
        self.cursor.draw()
        pygame.display.flip()

    def save_results(self):
        self.target_generator.save_yields()