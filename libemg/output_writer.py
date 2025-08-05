from abc import ABC, abstractmethod
import socket
from libemg.shared_memory_manager import SharedMemoryManager
import numpy as np
import types
from multiprocessing import Lock
import struct
import libemg
from enum import IntEnum

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
    def __init__(self, tag):
        self.tag = tag

    def write(self, info: dict) -> None:
        print(str(info['timestamp'])," ", info[self.tag])

class FileOutputWriter(OutputWriter):
    def __init__(self, tag, file_path: str, file_name: str):
        self.file_path = file_path
        self.file_name = file_name
        self.handle = open(self.file_path + self.file_name, "a", newline="")

    def write(self, info: dict) -> None:
        # Format the info as a line.
        line = f"{info.get('timestamp', '')} {info.get('prediction', '')} {info.get('probability', '')} {info.get('velocity', '')}\n"
        self.handle.write(line)
        self.handle.flush()

class SocketOutputWriter(OutputWriter):
    def __init__(self, tag, ip: str = '127.0.0.1', port: int = 12346, protocol: str = "UDP"):
        self.ip = ip
        self.port = port
        self.protocol = protocol.upper()
        self.sock = None
        self.tag = tag
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
        message = str(info[self.tag]) + " " + str(info['timestamp'])
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
            state['sock'].close()
            del state["sock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the socket in the child process.
        self.sock = None
        self._create_socket()


class SIMOutputWriter(SocketOutputWriter):
    # Enum for Degrees of Actuation
    class DOAs(IntEnum):
        TH_ROT = 0
        WFE = 6
        WUR = 7
        WPS = 8

    def __init__(self, ip: str = '127.0.0.1', port: int = 12346, protocol: str = "UDP", constant_dof_speed = .10):
        super(SIMOutputWriter, self).__init__(tag=None, ip = ip, port = port, protocol = protocol)
        self.constant_dof_speed = constant_dof_speed
        self.DOA_dict = {doa: 0 for doa in self.DOAs}
        self.udp_counter = 0

    def update_dictionary(self,d,k):
        # ToDo make matching between DOFs and Actions in k modular

        # k is model output, if it is a int, it's classification, ToDo combine with regression by one-hot-encoding
        if np.issubdtype(k.dtype, np.integer):
            if k == 0:
                d[self.DOAs.TH_ROT] = np.min([d[self.DOAs.TH_ROT] + self.constant_dof_speed, 1])
            elif k == 1:
                d[self.DOAs.TH_ROT] = np.max([d[self.DOAs.TH_ROT] - self.constant_dof_speed, -1])
            if k == 3:
                d[self.DOAs.WFE] = np.max([d[self.DOAs.WFE] - self.constant_dof_speed, -1])
            elif k == 4:
                d[self.DOAs.WFE] = np.min([d[self.DOAs.WFE] + self.constant_dof_speed, 1])
            
        
        # if k is np.float => regression
        elif np.issubdtype(k.dtype, np.floating):
            d[self.DOAs.TH_ROT] = np.clip(a=d[self.DOAs.TH_ROT] + self.constant_dof_speed * k[0], a_min=-1, a_max=1)
            d[self.DOAs.WFE] = np.clip(a=d[self.DOAs.WFE] + self.constant_dof_speed * k[1], a_min=-1, a_max=1)
            d[self.DOAs.WPS] = np.clip(a=d[self.DOAs.WPS] + self.constant_dof_speed * k[2], a_min=-1, a_max=1)
        
        return d
    

    def write(self, info):
        
        self.DOA_dict = self.update_dictionary(self.DOA_dict, info["model_output"])
        
        for doa in self.DOA_dict:
            message = self.compose_udp_message(time=info["timestamp"], num_count=1, val_type=float, data_type=bytes([1, doa]), data=struct.pack("d", self.DOA_dict[doa]))

            if self.sock is None:
                self._create_socket()

            self.sock.sendto(message, (self.ip, self.port))

    def get_byte_for_type(self, py_type):
        floating_types = {float}
        signed_types = {int}
        unsigned_types = {bytes}

        if py_type not in floating_types | signed_types | unsigned_types:
            raise Exception(f"Unsupported data type: {py_type}")

        type_info = 0b00000000
        if py_type in floating_types:
            type_info |= 0b00100000
        elif py_type in signed_types:
            type_info |= 0b00010000

        type_info += struct.calcsize('d' if py_type is float else 'i')
        return type_info
    
    def compose_udp_message(self, time, num_count, val_type, data_type, data):
        message = bytearray()
        message.append(num_count)
        message.append(self.get_byte_for_type(val_type))
        message.extend(data_type)
        message.extend(struct.pack("d", time))
        message.extend(data)
        message.extend(struct.pack("H", self.udp_counter))

        self.udp_counter = (self.udp_counter + 1) % 65536
        return message

class SharedMemoryOutputWriter(OutputWriter):
    def __init__(self, tag: str, shape, dtype, lock, mod_fn=None, mod_fn_count=None):
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
        self.mod_fn = types.MethodType(mod_fn, self) if mod_fn is not None else self.default_mod_fn
        self.mod_fn_count = types.MethodType(mod_fn_count, self) if mod_fn_count is not None else self.default_mod_fn_count
        # Create a new shared memory manager and create the variable.
        self.smm = SharedMemoryManager()
        self.smm.create_variable(tag, shape, dtype, lock)
        self.smm.create_variable(tag+"_count", (1,1), np.int32, Lock())

    def write(self, info: dict) -> None:
        if self.smm is None:
            raise RuntimeError("SharedMemoryOutputWriter not attached to a manager.")
        # Use the provided mod_fn to modify the shared memory variable.
        self.smm.modify_variable(self.tag, lambda data: self.mod_fn(data, info))
        self.smm.modify_variable(self.tag + "_count", lambda data: self.mod_fn_count(data, info))
    
    def reset(self) -> None:
        if self.smm is None:
            raise RuntimeError("SharedMemoryOutputWriter not attached to a manager.")
        self.smm.modify_variable(self.tag, lambda data: np.zeros(self.shape, dtype=self.dtype))
        self.smm.modify_variable(self.tag + "_count", lambda data: 0)

    def default_mod_fn(self, data, info):
        input_size = self.smm.variables[self.tag]["shape"][0]
        data[:] = np.vstack((info[self.tag], data))[:input_size, :]
        return data
    
    def default_mod_fn_count(self, data, info):
        data[:] = data[:] + info[self.tag].shape[0]
        return data

    def __getstate__(self):
        self._smm_item = self.smm.get_shared_memory_items()
        state = self.__dict__.copy()
        # Remove the non-serializable shared memory manager.
        if "smm" in state:
            del state["smm"]
            state['mod_fn'] = self.mod_fn.__func__
            state["mod_fn_count"] = self.mod_fn_count.__func__
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct the shared memory manager using the stored _smm_item.
        if self._smm_item is not None:
            new_mgr = SharedMemoryManager()
            for i in self._smm_item:
                tag, shape, dtype, lock = i
                new_mgr.create_variable(tag, shape, dtype, lock)
            self.smm = new_mgr
        else:
            self.smm = None
        
        self.mod_fn = types.MethodType(self.__dict__['mod_fn'], self) if self.__dict__['mod_fn'] is not None else self.default_mod_fn
        self.mod_fn_count = types.MethodType(self.__dict__['mod_fn_count'], self) if self.__dict__['mod_fn_count'] is not None else self.default_mod_fn_count
    
