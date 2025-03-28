from abc import ABC, abstractmethod
from typing import overload
import socket
import re
import time

import numpy as np
import pygame


class Controller(ABC):
    def __init__(self):
        """Abstract controller interface for controlling environments. Runs as a Process in a separate thread and collects control signals continuously. Call start() to start collecting control signals."""
        self.info_function_map = {
            'predictions': self._parse_predictions,
            'pc': self._parse_proportional_control,
            'timestamp': self._parse_timestamp
        }
        # TODO: Maybe add a flag for continuous vs. not continuous... not sure if that's needed though

    @overload
    def get_data(self, info: list[str]) -> tuple | None:
        ...

    @overload
    def get_data(self, info: str) -> list[float] | None:
        ...

    def get_data(self, info: list[str] | str) -> tuple | list[float] | None:
        """Get data from current action. This method should be used to access data to ensure that all parsing happens on the same action. Velocity control must be enabled when using proportional control.

        Parameters
        ----------
        info: list[str] or str
            Type of data requested. Must be a string in info_function_map.
        """
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
        """Parse the latest prediction from a message.

        Parameters
        ----------
        action: str
            Message to parse.

        Returns
        ----------
        list[float]
            List of predictions.
        """
        ...

    @abstractmethod
    def _parse_proportional_control(self, action: str) -> list[float]:
        """Parse the latest proportional control info from a message.

        Parameters
        ----------
        action: str
            Message to parse.

        Returns
        ----------
        list[float]
            List of proportional control values.
        """
        ...

    
    @abstractmethod
    def _parse_timestamp(self, action: str) -> float:
        """Parse the latest timestamp from a message.

        Parameters
        ----------
        action : str
            Message to parse.

        Returns
        -------
        float
            Timestamp.
        """
        ...

    @abstractmethod
    def _get_action(self) -> str | None:
        """Grab the latest action.

        Returns
        ----------
        str or None
            Latest action or None if no action has occurred.
        """
        ...


class SocketController(Controller):
    def __init__(self, ip: str = '127.0.0.1', port: int = 12346) -> None:
        """Controller interface for controlling environments using a UDP socket. 
        Runs as a Process in a separate thread and collects control signals continuously. Call start() to start collecting control signals.

        Parameters
        ----------
        ip: str
            IP address for UDP socket used to read messages.
        port: int
            Port for UDP socket used to read messages.
        """
        super().__init__()
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.setblocking(False)

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
            data, _ = self.sock.recvfrom(1024)
            action = str(data.decode('utf-8'))
        except BlockingIOError:
            action = None
        return action
    

class ClassifierController(SocketController):
    def __init__(self, output_format: str, num_classes: int, ip: str = '127.0.0.1', port: int = 12346) -> None:
        """Controller interface for controlling environments using a classifier.
        Runs as a Process in a separate thread and collects control signals continuously. Call start() to start collecting control signals.

        Parameters
        ----------
        output_format: str
            Output format of classifier. Accepted values are 'probabliities' and 'predictions.
        num_classes: int
            Number of classes in classification problem.
        ip: str
            IP address for UDP socket used to read messages.
        port: int
            Port for UDP socket used to read messages.
        """
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
    """Controller interface for controlling environments using a regressor.
        Runs as a Process in a separate thread and collects control signals continuously. Call start() to start collecting control signals.

        Parameters
        ----------
        ip: str
            IP address for UDP socket used to read messages.
        port: int
            Port for UDP socket used to read messages.
        """
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
        """Controller interface for controlling environments using a keyboard. start() method is not required because this doesn't run in a separate thread."""
        # No run method b/c using pygame events in another thread doesn't work (pygame.init is required but isn't thread-safe)
        super().__init__()
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

    def _parse_timestamp(self, action: str) -> float:
        return time.time()

    def _get_action(self):
        keys = pygame.key.get_pressed()
        keys_pressed = [key for key in self.keys if keys[key]]
        if len(keys_pressed) == 0:
            # No data received
            keys_pressed = [-1]

        key_pressed = keys_pressed[0]   # take the first value, maybe change later to support combined keys
        return str(key_pressed)
