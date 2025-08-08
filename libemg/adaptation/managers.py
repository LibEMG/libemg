from multiprocessing import Process, Lock, Event
import libemg
from abc import abstractmethod
from typing import Any, Tuple
import pickle
import numpy as np
import time

class udp_feedback_config:
    UDP_CAT_FEEDBACK = 6
    UDP_SUBCAT_PSEUDO = 0
    UDP_SUBCAT_REWARD = 1

    def feedback_scaler(x):
        sign = np.sign(x)
        abs_x = np.abs(x)
        # y = 0.1*(0.1 + 0.9 / (1 + np.exp(-10 * (0.5 * abs_x - 0.5))))
        y = (0.1 + 0.9 / (1 + np.exp(-10 * (0.5 * abs_x - 0.5))))
        return sign * y

class MemoryManager(Process):
    """
    This object is part of the adaptation suite provided by LibEMG. The memory manager is responsible for managing the data (both the inputs and pseudolabels) that arise from
    a user-in-the-loop setting and storing them in pickled memory classes. These memory classes may represent segments of data that have been collected (e.g., a trial of Fitts' Law).
    The memory manager contains a signal to trigger the adaptation manager to load these slices of memory, which is the other half of the adaptation suite provided by LibEMG.

    The memory slices are composed by monitoring shared memory output writers from the environment (which provide a timestamp and pseudo-label), and monitoring shared memory output writers from
    the model (which provide an identical timestamp and model inputs such as EMG features). 

    Parameters
    ----------
    memory: Any
        A custom object that defines the memory class. It should have a .append(), .save(), .load(), .reset(), and __add__ operator overload.
    smi: list
        a list containing information to construct shared memory managers that receive all data necessary to compile memories. Consult libemg.adaptation._base.get_<type_of_adaptation>_adaptation_items() for more information.
    ow: list[libemg.output_writer.OutputWriter]
        A list of LibEMG output writers that are used by this class to write information out to other processes (i.e., the adpatation manager). Consult libemg.adaptation._base.get_<type_of_adaptation>_adaptation_items() for more information.
    save_dir: str
        The location that memory slices will be saved. If adaptation is actually running, this should be same as the load_dir argument of the libemg.adaptation.managers.AdaptationManager.
    """
    def __init__(self,
                 memory: Any,
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
        self.trial_counter = 1
        ensure_directory(self.save_dir)

    def run(self):
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        for smi in self.smi:
            if type(smi) is list:
                self.smm.create_variable(*smi)
        self.ow[0].write(0) # start at 0
        
        self.memory.reset()
        while True:
            if self.signal.is_set():
                self.memory.save(self.save_dir + "memory_"+str(self.trial_counter) + ".pkl")
                break

            # check if there is data to process, and if so process it
            self.process_data()
            

    def process_data(self):
        if 'environment_feedback_count' in self.smm.variables:
        # if there has been no new environment feedback, just continue
            environment_feedback_count = self.smm.get_variable("environment_feedback_count")[0,0]
            if environment_feedback_count == self.environment_feedback_count:
                return
            num_to_grab = environment_feedback_count - self.environment_feedback_count 
            feedback_data = self.smm.get_variable("environment_feedback")
            feedback_data = feedback_data[:num_to_grab,:]
            
            # FE: put it here
            self.environment_feedback_count = environment_feedback_count
        
        # elif PAVE feedback is used 
        elif any([type(i) == libemg.environments.controllers.SIM_UDP_Receiver for i in self.smi]):
            receiver = self.smi[np.where([type(i) == libemg.environments.controllers.SIM_UDP_Receiver for i in self.smi])[0][0]]
            udpobject = receiver._get_action()
            if udpobject is None or udpobject.category != udp_feedback_config.UDP_CAT_FEEDBACK:
                return
            
            # if correct data type for pseudolabel do the difference (info['timestamp'], info['trial'], info['environment_feedback']
            if(udpobject.sub_category == udp_feedback_config.UDP_SUBCAT_PSEUDO):              
                # get values
                message = np.array(udpobject.process_data())
                diffs = message[1::2] - message[::2] # calc should from actual
                print(diffs)
                diffs = udp_feedback_config.feedback_scaler(diffs)
                trial = udpobject.counter // 1000 # substitute later and maybe send from Unity
                feedback_data = np.hstack([np.array([[udpobject.timestamp, trial]]), [diffs]])

            # if correct data type for rewards do the reward processing
            elif(udpobject.sub_category == udp_feedback_config.UDP_SUBCAT_REWARD):
                message = udpobject.process_data()
                print(message)

        
        input_data = self.smm.get_variable('model_input')
        
        
        # for every row in data:
        for i in range(feedback_data.shape[0]):
            feedback_row = feedback_data[i,:]
            trial = feedback_row[1]
            if trial != self.trial_counter:
                # Save the memory
                self.memory.save(self.save_dir + "memory_"+str(int(self.trial_counter)) + ".pkl")
                self.ow[0].write(self.trial_counter)
                self.trial_counter = trial
                # tell the adaptation manager a new slice is ready
                
                # Start a fresh memory
                self.memory.reset()
            # Append the data to the memory object
            row_timestamp = feedback_row[0]
            # find timestamp in classifier_input
            timestamp_id = np.where(np.round(input_data[:,0], 6) == np.round(row_timestamp, 6))[0]
            input_row  = input_data[timestamp_id,1:] # start at 2nd column to remove timestamp
            self.append_to_memory(feedback_row[2:], input_row, trial-1)

    def save_memory(self, loc: str):
        with open(loc, 'wb') as f:
            pickle.dump(self.memory, f)

    def run_helper(self, block=True):
        """
        Helper function to run the process. This is used to avoid blocking the main thread.
        """
        if block:
            self.run()
        else:
            self.start()

    @abstractmethod
    def setup_memorymanager() -> None:
        pass

    @abstractmethod 
    def get_data() -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def append_to_memory(self, feedback, input, trial) -> None:
        self.memory.append(feedback, input, trial)



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
                  save_dir: str,
                  stop_condition: callable = lambda x: True,
                  notify: bool = True):
        
        Process.__init__(self, daemon=True)

        self.signal = Event()

        self.model = model
        self.smi = smi
        self.ow = ow
        self.initial_memory_loc = initial_memory_loc
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.stop_condition = stop_condition
        self.notify = notify

        self.memory = self.load_memory(initial_memory_loc)

        self.memory_count = 0
        self.adaptation_count = 0

        # ensure save and load directories exist
        ensure_directory(self.save_dir)
        ensure_directory(self.load_dir)

    def run_helper(self, block=True):
        """
        Helper function to run the process. This is used to avoid blocking the main thread.
        NOTE: if you're training on the GPU, you need to run this in the main loop. This is a CUDA thing and 
        not something that can be worked around without sacrificing multiprocessing latency elsewhere (streaming).
        """
        if block:
            self.run()
        else:
            self.start()

    def run(self):
        start_time = time.time()
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        for smi in self.smi:
            self.smm.create_variable(*smi)

        while not self.stop_condition(self.memory_count):

            if self.signal.is_set():
                break
            
            memory_count = self.smm.get_variable("memory_flag")[0,0]
            if  memory_count > self.memory_count:
                num_memories_to_load = memory_count - self.memory_count
                for m in range(num_memories_to_load):
                    # load the next memory
                    self.memory_count += 1 
                    # Load memory
                    new_memory = self.load_memory(self.load_dir + "memory_" + str(self.memory_count) + ".pkl")
                    # print(f"{self.memory_count} : {new_memory.processed_data[0].shape}")
                    self.memory = self.memory + new_memory
            
            # Adapt the model
            loss_list = self.model.adapt(self.memory)
            with open(self.save_dir + "losses.txt", 'a') as f:
                f.write(str(time.time() - start_time) + "\t" + str(loss_list) + "\n")
            self.adaptation_count += 1
            self.model.save(self.save_dir + "mdl" + str(self.adaptation_count) + ".pkl")
            if self.notify:
                self.ow[0].write(self.adaptation_count)
        
        self.save_model(self.save_dir + "model_final.pkl")
        
    
    def load_memory(self, loc: str):
        with open(loc, 'rb') as f:
            return pickle.load(f)

    def save_model(self, loc: str):
        with open(loc, 'wb') as f:
            pickle.dump(self.model, f)

def ensure_directory(directory: str) -> None:
    """
    Ensure that a directory exists. If it does not exist, create it.

    Parameters
    ----------
    directory : str
        The directory to ensure exists.
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)