from abc import ABC, abstractmethod

class Memory(ABC):

    @abstractmethod
    def append(self, data):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def save(self):
        ...
    
    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def __add__(self, other):
        ...