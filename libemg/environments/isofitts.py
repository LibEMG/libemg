import math
import time

import pygame
import numpy as np

from libemg.environments.controllers import Controller
from libemg.environments._base import Environment

class IsoFitts(Environment):
    def __init__(self, controller: Controller, prediction_map = None, num_circles: int = 30, num_trials: int = 15, dwell_time: float = 3.0, timeout: float = 30.0, 
                 velocity: float = 25.0, save_file: str | None = None, width: int = 1250, height: int = 750, fps: int = 60):
        # logging information
        log_dictionary = {
            'time_stamp':        [],
            'trial_number':      [],
            'goal_circle' :      [],
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
        self.small_rad = 40
        self.big_rad   = 275
        self.pos_factor1 = self.big_rad/2
        self.pos_factor2 = (self.big_rad * math.sqrt(3))//2

        self.VEL = velocity
        self.dwell_time = dwell_time
        self.num_of_circles = num_circles 
        self.max_trial = num_trials
        self.width = width
        self.height = height
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
        self.timeout_timer = None
        self.timeout = timeout   # (seconds)
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
        
        self.current_direction = [0., 0.]
        if data is not None:
            # Move cursor
            predictions, pc = data
            if len(predictions) == 1 and len(pc) == 1:
                # Output is a class/action, not a set of DOFs
                prediction = predictions[0]
                pc = pc[0]
                direction = self.prediction_map[prediction]

                if direction == 'N':
                    predictions = [0, -1]   # -ve b/c pygame origin pixel is at top left of screen
                elif direction == 'E':
                    predictions = [1, 0]
                elif direction == 'S':
                    predictions = [0, 1]    # +ve b/c pygame origin pixel is at top left of screen
                elif direction == 'W':
                    predictions = [-1, 0]
                elif direction == 'NM':
                    predictions = [0, 0]
                else:
                    raise ValueError(f"Expected prediction map to have keys 'N', 'E', 'S', 'W', and 'NM', but found key: {direction}.")
                
                pc = [pc, pc]

            self.current_direction[0] += self.VEL * float(predictions[0]) * pc[0]
            self.current_direction[1] += self.VEL * float(predictions[1]) * pc[1]

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

    def _run_helper(self):
        # updated frequently for graphics & gameplay
        self.update_game()
        pygame.display.set_caption(str(self.clock.get_fps()))
