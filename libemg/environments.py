from abc import ABC, abstractmethod
import socket
from multiprocessing import Process

class Controller(ABC, Process):
    def __init__(self, ip: str = '127.0.0.1', port: int = 12346) -> None:
        # This won't be relevant for keyboard controller... maybe this should be UDPController or something. Might mean we don't need a Controller class
        # and can replace the abstract class with typing.protocol?
        self.ip = ip
        self.port = port

    @abstractmethod
    def get_predictions(self) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    @abstractmethod
    def get_proportional_control(self) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    def run(self) -> None:
        # Create UDP port for reading predictions
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        # Grab predictions (+ append?) so other method can use
        # Get PC (+ append?) so other method can use
        # Since it'll be different for each one, maybe just collect messages and then methods figure out how to handle them?
        ...


# Not sure if controllers should go in here or have their own file...
# Environment base class that takes in controller and has a run method

# Fitts should have the option for rotational.