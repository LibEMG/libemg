from abc import ABC, abstractmethod
import socket
from multiprocessing import Process
from collections import deque
from typing import overload
import re

import numpy as np


class Controller(ABC, Process):
    def __init__(self):
        super().__init__(daemon=True)

    @overload
    def get_data(self, info: list[str]) -> tuple:
        ...

    @overload
    def get_data(self, info: str) -> list[float]:
        ...

    def get_data(self, info: list[str] | str) -> tuple | list[float] | None:
        # Take in which types of info you need (predictions, PC), call the methods, then return
        # This method was needed because we're making this a process, so it's possible that separate calls to
        # parse_predictions and parse_proportional_control would operate on different packets
        if isinstance(info, str):
            # Cast to list
            info = [info]

        action = self.get_action()
        if action is None:
            # Action didn't occur
            return None
        
        info_function_map = {
            'predictions': self.parse_predictions,
            'pc': self.parse_proportional_control,
            'timestamp': self.parse_timestamp
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
    def parse_predictions(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    @abstractmethod
    def parse_proportional_control(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    @abstractmethod
    def parse_timestamp(self, action: str) -> float:
        # Grab latest timestamp
        ...
    
    @abstractmethod
    def get_action(self) -> str | None:
        # Freeze single action so all data is parsed from that
        ...


class SocketController(Controller):
    def __init__(self, ip: str = '127.0.0.1', port: int = 12346) -> None:
        super().__init__()
        self.ip = ip
        self.port = port
        self.data = deque(maxlen=1) # only want to read a single message at a time

    @abstractmethod
    def parse_predictions(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        # Will be specific to controller
        ...

    @abstractmethod
    def parse_proportional_control(self, action: str) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        # Will be specific to controller
        ...

    @abstractmethod
    def parse_timestamp(self, action: str) -> float:
        # Grab latest timestamp
        ...

    def get_action(self):
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
        
    def parse_predictions(self, action: str) -> list[float]:
        if self.output_format == 'predictions':
            return [float(action.split(' ')[0])]
        elif self.output_format == 'probabilities':
            probabilities = self.parse_probabilities(action)
            return [float(np.argmax(probabilities))]

        raise ValueError(self.error_message)

    def parse_timestamp(self, action: str) -> float:
        if self.output_format == 'predictions':
            raise ValueError("Output format is set to 'predictions', so timestamp cannot be parsed because timestamp is not sent when output_format='predictions'.")
        return float(action.split(' ')[-1])

    def parse_proportional_control(self, action: str) -> list[float]:
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
    def parse_predictions(self, action: str) -> list[float]:
        outputs = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", action)
        outputs = list(map(float, outputs))
        return outputs[:-1]

    def parse_timestamp(self, action: str) -> float:
        outputs = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", action)
        outputs = list(map(float, outputs))
        return outputs[-1]

    def parse_proportional_control(self, action: str) -> list[float]:
        predictions = self.parse_predictions(action)
        return [1. for _ in predictions]    # proportional control is built into prediction, so return 1 for each DOF

        
# Not sure if controllers should go in here or have their own file...
# Environment base class that takes in controller and has a run method (likely some sort of map parameter to determine which class corresponds to which control action)

# Fitts should have the option for rotational.