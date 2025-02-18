from abc import ABC, abstractmethod
import socket
from libemg.shared_memory_manager import SharedMemoryManager
import numpy as np

class OutputWriter(ABC):
    @abstractmethod
    def write(self, info: dict) -> None:
        """
        Write the output information.
        
        Parameters
        ----------
        info : dict
            A dictionary containing output information such as timestamp,
            prediction, probability, velocity, etc.
        """
        pass
class ConsoleOutputWriter(OutputWriter):
    def write(self, info: dict) -> None:
        print(info)

class FileOutputWriter(OutputWriter):
    def __init__(self, file_path: str, file_name: str):
        self.file_path = file_path
        self.file_name = file_name
        self.handle = open(self.file_path + self.file_name, "a", newline="")

    def write(self, info: dict) -> None:
        # Format the info as a line.
        line = f"{info.get('timestamp', '')} {info.get('prediction', '')} {info.get('probability', '')} {info.get('velocity', '')}\n"
        self.handle.write(line)
        self.handle.flush()

class SocketOutputWriter(OutputWriter):
    def __init__(self, ip: str = '127.0.0.1', port: int = 12346, protocol: str = "UDP"):
        self.ip = ip
        self.port = port
        self.protocol = protocol.upper()
        self.sock = None
        self._create_socket()

    def _create_socket(self):
        if self.protocol == "UDP":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif self.protocol == "TCP":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, self.port))
        else:
            raise ValueError("Protocol must be UDP or TCP.")

    def write(self, info: dict) -> None:
        message = info.get("message", "")
        if self.sock is None:
            self._create_socket()
        if self.protocol == "UDP":
            self.sock.sendto(message.encode('utf-8'), (self.ip, self.port))
        else:
            self.sock.sendall(message.encode('utf-8'))

    def __getstate__(self):
        # Remove the socket from the state so it's not pickled.
        state = self.__dict__.copy()
        if "sock" in state:
            del state["sock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the socket in the child process.
        self.sock = None
        self._create_socket()

class SharedMemoryOutputWriter(OutputWriter):
    def __init__(self, tag: str, shape, dtype, lock, mod_fn=None):
        """
        Parameters:
            tag (str): 
                The shared memory variable tag.
            shape (tuple): 
                The shape of the shared memory variable.
            dtype: 
                The data type of the shared memory variable.
            lock (Lock): 
                A multiprocessing lock for synchronization.
            mod_fn (callable, optional): 
                A function that takes (current_data, message) and returns new data.
                If not provided, defaults to a function that simply returns the message.
        """
        self.tag = tag
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
        self.mod_fn = mod_fn if mod_fn is not None else self.default_mod_fn
        # Create a new shared memory manager and create the variable.
        self.smm = SharedMemoryManager()
        self.smm.create_variable(tag, shape, dtype, lock)

    def write(self, info: dict) -> None:
        if self.smm is None:
            raise RuntimeError("SharedMemoryOutputWriter not attached to a manager.")
        # Use the provided mod_fn to modify the shared memory variable.
        self.smm.modify_variable(self.tag, lambda data: self.mod_fn(data, info))
    
    def default_mod_fn(self, data, info):
        input_size = self.smm_manager.variables[self.tag]["shape"][0]
        data[:] = np.vstack((info[self.tag], data))[:input_size, :]
        return data

    def __getstate__(self):
        self._smm_item = self.smm.get_shared_memory_items()
        state = self.__dict__.copy()
        # Remove the non-serializable shared memory manager.
        if "smm_manager" in state:
            del state["smm_manager"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct the shared memory manager using the stored _smm_item.
        if self._smm_item is not None:
            new_mgr = SharedMemoryManager()
            tag, shape, dtype, lock = self._smm_item
            new_mgr.create_variable(tag, shape, dtype, lock)
            self.smm_manager = new_mgr
        else:
            self.smm_manager = None
    
