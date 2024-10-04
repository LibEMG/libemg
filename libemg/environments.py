from abc import ABC, abstractmethod
import socket
from multiprocessing import Process
from collections import deque
from typing import overload
import re
import time
import math

import numpy as np
import pygame


class Controller(ABC, Process):
    def __init__(self):
        super().__init__(daemon=True)

    @overload
    def get_data(self, info: list[str]) -> tuple | None:
        ...

    @overload
    def get_data(self, info: str) -> list[float] | None:
        ...

    def get_data(self, info: list[str] | str) -> tuple | list[float] | None:
        # Take in which types of info you need (predictions, PC), call the methods, then return
        # This method was needed because we're making this a process, so it's possible that separate calls to
        # parse_predictions and parse_proportional_control would operate on different packets
        if isinstance(info, str):
            # Cast to list
            info = [info]

        action = self._get_action()
        if action is None:
            # Action didn't occur
            return None
        
        info_function_map = {
            'predictions': self._parse_predictions,
            'pc': self._parse_proportional_control,
            'timestamp': self._parse_timestamp
        }

        data = []
        for info_type in info:
            try:
                parse_function = info_function_map[info_type]
                result = parse_function(action)           
            except KeyError as e:
                raise ValueError(f"Unexpected value for info type. Accepted parameters are: {list(info_function_map.keys())}. Got: {info_type}.") from e

            data.append(result)

        data = tuple(data)  # convert to tuple so unpacking can be used if desired
        if len(data) == 1:
            data = data[0]
        return data

    @abstractmethod
    def _parse_predictions(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    @abstractmethod
    def _parse_proportional_control(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    @abstractmethod
    def _parse_timestamp(self, action: str) -> float:
        # Grab latest timestamp
        ...
    
    @abstractmethod
    def _get_action(self) -> str | None:
        # Freeze single action so all data is parsed from that
        ...


class SocketController(Controller):
    def __init__(self, ip: str = '127.0.0.1', port: int = 12346) -> None:
        super().__init__()
        self.ip = ip
        self.port = port
        self.data = deque(maxlen=1) # only want to read a single message at a time

    @abstractmethod
    def _parse_predictions(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        # Will be specific to controller
        ...

    @abstractmethod
    def _parse_proportional_control(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        # Will be specific to controller
        ...

    @abstractmethod
    def _parse_timestamp(self, action: str) -> float:
        # Grab latest timestamp
        ...

    def _get_action(self):
        if len(self.data) > 0:
            # Grab latest prediction and remove from queue so it isn't repeated
            return self.data.pop()
        return None

    def run(self) -> None:
        # Create UDP port for reading predictions
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

        while True:
            bytes, _ = self.sock.recvfrom(1024)
            message = str(bytes.decode('utf-8'))
            if message:
                # Data received
                self.data.append(message)
    

class ClassifierController(SocketController):
    def __init__(self, output_format: str, ip: str = '127.0.0.1', port: int = 12346) -> None:
        super().__init__(ip, port)
        self.output_format = output_format
        self.error_message = f"Unexpected value for output_format. Accepted values are 'predictions' or 'probabilities'. Got: {output_format}."
        if output_format not in ['predictions', 'probabilities']:
            raise ValueError(self.error_message)
        
    def _parse_predictions(self, action: str) -> list[float]:
        if self.output_format == 'predictions':
            return [float(action.split(' ')[0])]
        elif self.output_format == 'probabilities':
            probabilities = self.parse_probabilities(action)
            return [float(np.argmax(probabilities))]

        raise ValueError(self.error_message)

    def _parse_timestamp(self, action: str) -> float:
        if self.output_format == 'predictions':
            raise ValueError("Output format is set to 'predictions', so timestamp cannot be parsed because timestamp is not sent when output_format='predictions'.")
        return float(action.split(' ')[-1])

    def _parse_proportional_control(self, action: str) -> list[float]:
        if self.output_format == 'predictions':
            return [float(action.split(' ')[1])]
        elif self.output_format == 'probabilities':
            return [float(action.split(' ')[-2])]

        raise ValueError(self.error_message)

    def parse_probabilities(self, action: str) -> list[float]:
        if self.output_format == 'predictions':
            raise ValueError("Output format is set to 'predictions', so probabilities cannot be parsed. Set output_format='probabilities' if this functionality is needed.")

        return [float(prob) for prob in action.split(' ')[:-2]]


class RegressorController(SocketController):
    def _parse_predictions(self, action: str) -> list[float]:
        outputs = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", action)
        outputs = list(map(float, outputs))
        return outputs[:-1]

    def _parse_timestamp(self, action: str) -> float:
        outputs = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", action)
        outputs = list(map(float, outputs))
        return outputs[-1]

    def _parse_proportional_control(self, action: str) -> list[float]:
        predictions = self._parse_predictions(action)
        return [1. for _ in predictions]    # proportional control is built into prediction, so return 1 for each DOF

        
# Not sure if controllers should go in here or have their own file...
# Environment base class that takes in controller and has a run method (likely some sort of map parameter to determine which class corresponds to which control action)

# Fitts should have the option for rotational.

class Environment(ABC):
    def __init__(self, controller: Controller):
        # Not totally sure that this class is needed
        # Only thing I think they have in common is if they all use pygame and have some common methods there...
        self.controller = controller

    @abstractmethod
    def run(self):
        # If there's a use case, we could a block = True flag so we'd put this in a different thread if False
        ...

    @abstractmethod
    def log(self):
        ...



# Should probably move environments to a submodule where each one has their own file...
class FittsLawTest:
    def __init__(self, controller: Controller, num_circles: int = 30, num_trials: int = 15, savefile: str = "out.pkl",
                  logging: bool = True, width: int = 1250, height: int = 750):
        pygame.init()
        self.controller = controller
        self.font = pygame.font.SysFont('helvetica', 40)
        self.screen = pygame.display.set_mode([width, height])
        self.clock = pygame.time.Clock()
        
        # logging information
        self.log_dictionary = {
            'time_stamp':        [],
            'trial_number':      [],
            'goal_circle' :      [],
            'global_clock' :     [],
            'cursor_position':   [],
            'class_label':       [],
            'current_direction': []
        }

        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.YELLOW = (255,255,0)
        self.BLUE   = (0,102,204)
        self.small_rad = 40
        self.big_rad   = 275
        self.pos_factor1 = self.big_rad/2
        self.pos_factor2 = (self.big_rad * math.sqrt(3))//2

        self.done = False
        self.VEL = 25
        self.dwell_time = 3
        self.num_of_circles = num_circles 
        self.max_trial = num_trials
        self.width = width
        self.height = height
        # self.fps = 1/(config.window_increment / 200)
        self.fps = 60
        self.savefile = savefile
        self.logging = logging
        self.trial = 0

        # interface objects
        self.circles = []
        self.cursor = pygame.Rect(self.width//2 - 7, self.height//2 - 7, 14, 14)
        self.goal_circle = -1
        self.get_new_goal_circle()
        self.current_direction = [0,0]

        # timer for checking socket
        self.window_checkpoint = time.time()

        # Socket for reading EMG
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        self.sock.bind(('127.0.0.1', 12346))
        self.timeout_timer = None
        self.timeout = 30   # (seconds)
        self.trial_duration = 0

    def draw(self):
        self.screen.fill(self.BLACK)
        self.draw_circles()
        self.draw_cursor()
        self.draw_timer()
    
    def draw_circles(self):
        if not len(self.circles):
            self.angle = 0
            self.angle_increment = 360 // self.num_of_circles
            while self.angle < 360:
                self.circles.append(pygame.Rect((self.width//2 - self.small_rad) + math.cos(math.radians(self.angle)) * self.big_rad, (self.height//2 - self.small_rad) + math.sin(math.radians(self.angle)) * self.big_rad, self.small_rad * 2, self.small_rad * 2))
                self.angle += self.angle_increment

        for circle in self.circles:
            pygame.draw.circle(self.screen, self.RED, (circle.x + self.small_rad, circle.y + self.small_rad), self.small_rad, 2)
        
        goal_circle = self.circles[self.goal_circle]
        pygame.draw.circle(self.screen, self.RED, (goal_circle.x + self.small_rad, goal_circle.y + self.small_rad), self.small_rad)
            
    def draw_cursor(self):
        pygame.draw.circle(self.screen, self.YELLOW, (self.cursor.left + 7, self.cursor.top + 7), 7)

    def draw_timer(self):
        if hasattr(self, 'dwell_timer'):
            if self.dwell_timer is not None:
                toc = time.perf_counter()
                duration = round((toc-self.dwell_timer),2)
                time_str = str(duration)
                draw_text = self.font.render(time_str, 1, self.BLUE)
                self.screen.blit(draw_text, (10, 10))

    def update_game(self):
        self.draw()
        self.run_game_process()
        self.move()
    
    def run_game_process(self):
        self.check_collisions()
        self.check_events()

    def check_collisions(self):
        circle = self.circles[self.goal_circle]
        if math.sqrt((circle.centerx - self.cursor.centerx)**2 + (circle.centery - self.cursor.centery)**2) < (circle[2]/2 + self.cursor[2]/2):
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.goal_circle))
            self.Event_Flag = True
        else:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.num_of_circles))
            self.Event_Flag = False

    def check_events(self):
        # closing window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                return
            
        data = self.controller.get_data(['predictions', 'pc'])
        self.window_checkpoint = time.time()
        
        self.current_direction = [0,0]
        if data is not None:
            # Move cursor
            # TODO: Fix this for classification
            predictions, pc = data
            if len(predictions) == 1 and len(pc) == 1:
                # Output class
                prediction = predictions[0]
                predictions = [0, 0]
                predictions[prediction] = 1.    # this won't work because there will be more classes than DOFs... somehow need to map classes to DOFs. probably need a map parameter that the user can pass in or something?

                pc = [pc, pc]

            self.current_direction[0] += self.VEL * float(predictions[0]) * pc[0]
            self.current_direction[1] -= self.VEL * float(predictions[1]) * pc[1]

            if self.logging:
                self.log(str(predictions), time.time())
            
        

        ## CHECKING FOR COLLISION BETWEEN CURSOR AND RECTANGLES
        if event.type >= pygame.USEREVENT and event.type < pygame.USEREVENT + self.num_of_circles:
            if self.dwell_timer is None:
                self.dwell_timer = time.perf_counter()
            else:
                toc = time.perf_counter()
                self.duration = round((toc - self.dwell_timer), 2)
            if self.duration >= self.dwell_time:
                self.get_new_goal_circle()
                self.dwell_timer = None
                if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                    self.trial += 1
                else:
                    if self.logging:
                        self.save_log()
                    self.done = True
        elif event.type == pygame.USEREVENT + self.num_of_circles:
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
            self.get_new_goal_circle()
            self.timeout_timer = None
            if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                self.trial += 1
            else:
                if self.logging:
                    self.save_log()
                self.done = True

    def move(self):
        # Making sure its within the bounds of the screen
        if self.cursor.left + self.current_direction[0] > 0 and self.cursor.left + self.current_direction[0] < self.width:
            self.cursor.left += self.current_direction[0]
        if self.cursor.top + self.current_direction[1] > 0 and self.cursor.top + self.current_direction[1] < self.height:
            self.cursor.top += self.current_direction[1]
    
    def get_new_goal_circle(self):
        if self.goal_circle == -1:
            self.goal_circle = 0
            self.next_circle_in = self.num_of_circles//2
            self.circle_jump = 0
        else:
            self.goal_circle =  (self.goal_circle + self.next_circle_in )% self.num_of_circles
            if self.circle_jump == 0:
                self.next_circle_in = self.num_of_circles//2 + 1
                self.circle_jump = 1
            else:
                self.next_circle_in = self.num_of_circles // 2
                self.circle_jump = 0
        self.timeout_timer = None
        self.trial_duration = 0


    def log(self, label, timestamp):
        circle = self.circles[self.goal_circle]
        self.log_dictionary['time_stamp'].append(timestamp)
        self.log_dictionary['trial_number'].append(self.trial)
        self.log_dictionary['goal_circle'].append((circle.centerx, circle.centery, circle[2]))
        self.log_dictionary['global_clock'].append(time.perf_counter())
        self.log_dictionary['cursor_position'].append((self.cursor.centerx, self.cursor.centery, self.cursor[2]))
        self.log_dictionary['class_label'].append(label) 
        self.log_dictionary['current_direction'].append(self.current_direction)

    def save_log(self):
        # Adding timestamp
        with open(self.savefile, 'wb') as f:
            pickle.dump(self.log_dictionary, f)

    def run(self):
        while not self.done:
            # updated frequently for graphics & gameplay
            self.update_game()
            pygame.display.update()
            self.clock.tick(self.fps)
            pygame.display.set_caption(str(self.clock.get_fps()))
        pygame.quit()
