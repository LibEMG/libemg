import numpy as np
from multiprocessing.shared_memory import SharedMemory

class SharedMemoryManager:
    def __init__(self):
        self.variables = {}

    def create_variable(self, tag, shape, type, lock):
        if tag in self.variables.keys():
            print(f"Already have access to this variable: {tag}")
            return True
        
        # if tag exists already
        if self.find_variable(tag, shape, type, lock):
            print(f'{tag} already exists in shared memory, found variable.')
            return True
        
        try:
            sm = SharedMemory(tag, create=False)
            sm.unlink()
        except:
            pass
        
        try:
            type_size = type().itemsize
        except TypeError:
            # Passed in non-callable dtype
            type_size = type.itemsize

        smh = SharedMemory(tag, create=True, size=int(type_size * np.prod(shape)))
        data = np.ndarray((shape),dtype=type,buffer=smh.buf)
        data.fill(0)
        self.variables[tag] = {}
        self.variables[tag]["data"] = data
        self.variables[tag]["shape"] = shape
        self.variables[tag]["type"]  = type
        self.variables[tag]["smh"]  = smh
        self.variables[tag]["lock"] = lock
        return True

    def find_variable(self, tag, shape, type, lock):
        try:
            type_size = type().itemsize
        except TypeError:
            # Passed in non-callable dtype
            type_size = type.itemsize
        try:
            smh = SharedMemory(tag, size=int(type_size * np.prod(shape)))
            # create a new numpy array that uses the shared memory
            data = np.ndarray((shape), dtype=type, buffer=smh.buf)
            self.variables[tag] = {}
            self.variables[tag]["data"]   = data
            self.variables[tag]["shape"]  = shape
            self.variables[tag]["type"]   = type
            self.variables[tag]["smh"]    = smh
            self.variables[tag]["lock"]   = lock
            return True
        except FileNotFoundError:
            return False
        
    def get_variable(self, tag):
        assert tag in self.variables.keys()
        with self.variables[tag]["lock"]:
            return self.variables[tag]["data"].copy()

    def modify_variable(self, tag, fn):
        assert tag in self.variables.keys()
        with self.variables[tag]["lock"]:
            self.variables[tag]["data"][:] = fn(self.variables[tag]["data"])

    def cleanup(self, parent = True):
        for k in self.variables.keys():
            self.variables[k]["smh"].close()
            if parent:
                self.variables[k]["smh"].unlink()
        self.variables = {}
    
    def get_variable_list(self):
        result = []
        for k in self.variables.keys():
            result.append([k, self.variables[k]["shape"], self.variables[k]["type"]])
        return result