from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from typing import overload
from queue import Empty
import socket
import re

import numpy as np
import pygame


class Controller(ABC, Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.info_function_map = {
            'predictions': self._parse_predictions,
            'pc': self._parse_proportional_control
        }
        # TODO: Maybe add a flag for continuous vs. not continuous... not sure if that's needed though

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
        # Add note to tell user that velocity control must be turned on when using proportional control for classifier
        if isinstance(info, str):
            # Cast to list
            info = [info]

        action = self._get_action()
        if action is None:
            # Action didn't occur
            return None
        

        data = []
        for info_type in info:
            try:
                parse_function = self.info_function_map[info_type]
                result = parse_function(action)           
            except KeyError as e:
                raise ValueError(f"Unexpected value for info type. Accepted parameters are: {list(self.info_function_map.keys())}. Got: {info_type}.") from e

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
    def _get_action(self) -> str | None:
        # Freeze single action so all data is parsed from that
        ...


class SocketController(Controller):
    def __init__(self, ip: str = '127.0.0.1', port: int = 12346) -> None:
        super().__init__()
        self.info_function_map['timestamp'] = self._parse_timestamp
        self.ip = ip
        self.port = port
        self.queue = Queue(maxsize=1)   # only want to read a single message at a time

    @abstractmethod
    def _parse_predictions(self, action: str) -> list[float]:
        ...

    @abstractmethod
    def _parse_proportional_control(self, action: str) -> list[float]:
        ...

    @abstractmethod
    def _parse_timestamp(self, action: str) -> float:
        ...

    def _get_action(self):
        try:
            action = self.queue.get(block=False)
        except Empty:
            action = None
        return action

    def run(self) -> None:
        # Create UDP port for reading predictions
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

        while True:
            bytes, _ = self.sock.recvfrom(1024)
            message = str(bytes.decode('utf-8'))
            if message:
                # Data received
                if self.queue.full():
                    self.queue.get(block=True)  # setting block=False may throw an Empty error
                self.queue.put(message, block=False)    # don't want to wait to add most recent message
    

class ClassifierController(SocketController):
    def __init__(self, output_format: str, num_classes: int, ip: str = '127.0.0.1', port: int = 12346) -> None:
        super().__init__(ip, port)
        self.info_function_map['probabilities'] = self._parse_probabilities  # add option for classifier to parse probabilities
        self.output_format = output_format
        self.num_classes = num_classes  # could remove this parameter if we always sent a velocity value (e.g., set it to -1 if velocity control is not enabled)
        self.error_message = f"Unexpected value for output_format. Accepted values are 'predictions' or 'probabilities'. Got: {output_format}."
        if output_format not in ['predictions', 'probabilities']:
            raise ValueError(self.error_message)
        
    def _parse_predictions(self, action: str) -> list[float]:
        if self.output_format == 'predictions':
            return [float(action.split(' ')[0])]
        elif self.output_format == 'probabilities':
            probabilities = self._parse_probabilities(action)
            return [float(np.argmax(probabilities))]

        raise ValueError(self.error_message)

    def _parse_timestamp(self, action: str) -> float:
        if self.output_format == 'predictions':
            raise ValueError("Output format is set to 'predictions', so timestamp cannot be parsed because timestamp is not sent when output_format='predictions'.")
        return float(action.split(' ')[-1])

    def _parse_proportional_control(self, action: str) -> list[float]:
        components = action.split(' ')
        if self.output_format == 'predictions':
            try:
                return [float(components[1])]
            except IndexError as e:
                raise IndexError('Attempted to parse proportional control, but no velocity value was found. Please enable velocity control in the EMGClassifier.') from e
        elif self.output_format == 'probabilities':
            # Assume that user has enabled velocity control and take the value before the timestamp
            if len(components) < (self.num_classes + 2):
                raise ValueError('Did not find velocity value in message. Please enable velocity control in the EMGClassifier.')
            return [float(components[-2])]

        raise ValueError(self.error_message)

    def _parse_probabilities(self, action: str) -> list[float]:
        if self.output_format == 'predictions':
            raise ValueError("Output format is set to 'predictions', so probabilities cannot be parsed. Set output_format='probabilities' if this functionality is needed.")

        return [float(prob) for prob in action.split(' ')[:self.num_classes]]


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


class KeyboardController(Controller):
    def __init__(self) -> None:
        # No run method b/c using pygame events in another thread doesn't work (pygame.init is required but isn't thread-safe)
        super().__init__()
        self.queue = Queue(maxsize=1)   # only want to read a single message at a time
        self.keys = [
            pygame.K_LEFT,
            pygame.K_RIGHT,
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_1,
            pygame.K_2,
            pygame.K_3,
            pygame.K_4
        ]

    def _parse_predictions(self, action: str) -> list[float]:
        return [float(action)]

    def _parse_proportional_control(self, action: str) -> list[float]:
        predictions = self._parse_predictions(action)
        return [1. for _ in predictions]    # proportional control is built into prediction, so return 1 for each key pressed

    def _get_action(self):
        keys = pygame.key.get_pressed()
        keys_pressed = [key for key in self.keys if keys[key]]
        if len(keys_pressed) == 0:
            # No data received
            keys_pressed = [-1]

        key_pressed = keys_pressed[0]   # take the first value, maybe change later to support combined keys
        return str(key_pressed)
