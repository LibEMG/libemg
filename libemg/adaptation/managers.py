from multiprocessing import Process, Lock, Event
import libemg
from abc import abstractmethod
from typing import Any, Tuple
import pickle
import numpy as np
import time

class MemoryManager(Process):
    """
    This object is part of the adaptation suite provided by LibEMG. The memory manager is responsible for managing the data (both the inputs and pseudolabels) that arise from
    a user-in-the-loop setting and storing them in pickled memory classes. These memory classes may represent segments of data that have been collected (e.g., a trial of Fitts' Law).
    The memory manager contains a signal to trigger the adaptation manager to load these slices of memory, which is the other half of the adaptation suite provided by LibEMG.

    The memory slices are composed by monitoring shared memory output writers from the environment (which provide a timestamp and pseudo-label), and monitoring shared memory output writers from
    the model (which provide an identical timestamp and model inputs such as EMG features). 

    Parameters
    ----------
    memory: libemg.adaptation.memory.Memory
        A custom object that defines the memory class. It should have a .append(), .save(), .load(), .reset(), and __add__ operator overload.
    smi: list
        a list containing information to construct shared memory managers that receive all data necessary to compile memories. Consult libemg.adaptation._base.get_<type_of_adaptation>_adaptation_items() for more information.
    ow: list[libemg.output_writer.OutputWriter]
        A list of LibEMG output writers that are used by this class to write information out to other processes (i.e., the adpatation manager). Consult libemg.adaptation._base.get_<type_of_adaptation>_adaptation_items() for more information.
    save_dir: str
        The location that memory slices will be saved. If adaptation is actually running, this should be same as the load_dir argument of the libemg.adaptation.managers.AdaptationManager.
    """
    def __init__(self,
                 memory: libemg.adaptation.memory.Memory,
                 smi: list,
                 ow: list[libemg.output_writer.OutputWriter],
                 save_dir: str):
        Process.__init__(self, daemon=True)

        self.signal = Event()
        
        self.memory = memory
        self.smi = smi
        self.ow = ow
        self.save_dir = save_dir

        self.environment_feedback_count = 0
        self.trial_counter = 0

    def run(self):
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        for smi in self.smi:
            self.smm.create_variable(*smi)
        
        self.memory.reset()
        while True:
            if self.signal.is_set():
                self.memory.save(self.save_dir + "memory_"+str(self.trial_counter) + ".pkl")
                break

            # check if there is data to process, and if so process it
            self.process_data()
            

    def process_data(self):
        # if there has been no new environment feedback, just continue
        environment_feedback_count = self.smm.get_variable("environment_feedback_count")
        if environment_feedback_count == self.environmentfeedback_counter:
            return
        num_to_grab = self.environment_feedback_count - environment_feedback_count 
        data = self.smm.get_variable("environment_feedback")
        data = data[:num_to_grab,:]
        # for every row in data:
        for i in range(num_to_grab):
            if data.trial_counter != self.trial_counter:
                # Save the memory
                self.memory.save(self.save_dir + "memory_"+str(self.trial_counter) + ".pkl")
                self.trial_counter += 1
                # tell the adaptation manager a new slice is ready
                self.ow[0].write({"timestamp": data.timestamp, "trial_counter": self.trial_counter})
                # Start a fresh memory
                self.memory.reset()
            # Append the data to the memory object
            self.append_to_memory(data)

        # update the environment_feedback_count
        self.environmentfeedback_counter = environment_feedback_count 

    def save_memory(self, loc: str):
        with open(loc, 'wb') as f:
            pickle.dump(self.memory, f)

    def run_helper(self, block=True):
        """
        Helper function to run the process. This is used to avoid blocking the main thread.
        """
        if block:
            self.start()
            self.join()
        else:
            self.run()

    @abstractmethod
    def setup_memorymanager() -> None:
        pass

    @abstractmethod 
    def get_data() -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def append_to_memory() -> None:
        pass

class AdaptationManager(Process):
    """
    This object is part of the adaptation suite provided by LibEMG. The adaptation manager is responsible for aggregating the slices of data that are packaged by the memory manager, and running 
    the adaptation of the model given this data.

    Parameters
    ----------
    model: Any
        A custom object that defines the model. It should have a .adapt, .save(), .load() and .predict() methods.
    smi: list
        a list containing information to construct shared memory managers that receive messages from the memory manager to update the aggregated data for adptation. Consult libemg.adaptation._base.get_<type_of_adaptation>_adaptation_items() for more information.
    ow: list[libemg.output_writer.OutputWriter]
        A list of LibEMG output writers that are used by this class to write information out to other processes (i.e., the OnlineStreamer to load the updated model). Consult libemg.adaptation._base.get_<type_of_adaptation>_adaptation_items() for more information.
    initial_memory_loc: str
        The location of the initial memory slice. A typical use for this would be to load data from screen guided training prior to gather user-in-the-loop data. Not entirely necessary, but very beneficial for stability.
    load_dir: str
        The location that the memory slices will be loaded. This should be the save_dir argument of the libemg.adaptation.managers.MemoryManager.
    save_dir: str
        The location that the adapted models will be saved. This should be the file_path argument of the libemg.emg_predictor.OnlineStreamer (the live model for the user-in-the-loop setting.)
    """
    def __init__(self,
                  model,
                  smi: list,
                  ow: list[libemg.output_writer.OutputWriter],
                  initial_memory_loc: str|None,
                  load_dir: str,
                  save_dir: str):
        Process.__init__(self, daemon=True)

        self.signal = Event()

        self.model = model
        self.smi = smi
        self.ow = ow
        self.initial_memory_loc = initial_memory_loc
        self.load_dir = load_dir
        self.save_dir = save_dir


        self.memory = self.load_memory(initial_memory_loc)

        self.memory_count = 0
        self.adaptation_count = 0

    def run_helper(self, block=True):
        """
        Helper function to run the process. This is used to avoid blocking the main thread.
        """
        if block:
            self.start()
            self.join()
        else:
            self.run()

    def run(self):
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        for smi in self.smi:
            self.smm.create_variable(*smi)
        while True:

            if self.signal.is_set():
                self.save_model(self.save_dir + "model_final.pkl")
                break
            
            memory_count = self.smm.get_variable("memory_flag")
            if  memory_count != self.memory_count:
                self.memory_count = memory_count
                # Load memory
                new_memory = self.load_memory(self.load_dir + "memory_" + str(self.memory_count) + ".pkl")
                self.memory = self.memory + new_memory

                # Save the model
                self.model.save(self.save_dir)
                # Load the next trial's memory slice
                with open(self.load_dir + "_" + str(self.memory_count) + ".pkl", "rb") as f:
                    new_memory = pickle.load(f)
                self.memory = self.memory + new_memory
            
            # Adapt the model
            self.model.adapt(self.memory)
            self.adaptation_count += 1
            self.model.save(self.save_dir + "model_" + str(self.model_count) + ".pkl")
            self.ow[0].write({"timestamp": time.time(),
                              "adaptation_count":self.adaptation_count})
    
    def load_memory(self, loc: str):
        with open(loc, 'rb') as f:
            return pickle.load(f)

    def save_model(self, loc: str):
        with open(loc, 'wb') as f:
            pickle.dump(self.model, f)