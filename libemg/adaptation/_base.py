import numpy as np
from multiprocessing import Lock
from libemg.output_writer import SharedMemoryOutputWriter, SocketOutputWriter

def get_edil_adaptation_objects(num_features, num_outputs):
    """
    Get the shared memory items necessary for adaptation. This is configured for environment-dependent incremental learning.

    This config prepares the model_inputs and model_outputs to be available via shared memory, and broadcast model_outputs over a UDP port.

    Parameters
    ----------
    num_features : int
        Number of features for input in the model.
    num_outputs : int
        Number of outputs of the model (and returned by the environment).

    Returns
    -------
    smi
        Shared memory items necessary for adaptation. Pass this to the OnlineStreamer (classifier or regressor)
    """

    # Every sharedmemoryoutputwriter should have a mod_fn that describes how to modify its data.

    def mod_fn_input(self, data, info):
        new_slice = np.hstack((info['timestamp'],info['model_input'][-1,:]))
        input_size = self.smm.variables['model_input']["shape"][0]
        data[:] = np.vstack((new_slice, data))[:input_size, :]
        return data
    
    def mod_fn_output(self, data, info):
        new_slice = np.hstack((info['timestamp'],info['model_output']))
        input_size = self.smm.variables['model_output']["shape"][0]
        data[:] = np.vstack((new_slice, data))[:input_size, :]
        return data
    
    def mod_fn_env(self, data, info):
        new_slice = np.hstack((info['timestamp'], info['trial'], info['environment_feedback']))
        input_size = self.smm.variables['environment_feedback']["shape"][0]
        data[:] = np.vstack((new_slice, data))[:input_size, :]
        return data

    def mod_fn_flags(self, data, number):
        data[:] = number
        return data


    
    adapt_flag_smow = SharedMemoryOutputWriter('adapt_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags)
    memory_flag_smow = SharedMemoryOutputWriter('memory_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags)
    active_flag_smow = SharedMemoryOutputWriter('active_flag', (1,1), np.int8, Lock(), mod_fn=mod_fn_flags)
    environment_flag_smow = SharedMemoryOutputWriter('environment_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags)

    model_output_smow = SharedMemoryOutputWriter('model_output', (100, 1+num_outputs), np.float32, Lock(), mod_fn=mod_fn_output)
    model_input_smow = SharedMemoryOutputWriter('model_input', (100, 1+num_features), np.float32, Lock(), mod_fn=mod_fn_input)
    environment_feedback_smow = SharedMemoryOutputWriter('environment_feedback', (100, 1+1+num_outputs), np.float32, Lock(), mod_fn=mod_fn_env)
    
    model_output_sow = SocketOutputWriter("model_output")

    """
    The model receives messages over shared memory via the adapt tag (to notify when a new model is ready to be loaded).
    The model receives messages over the active tag (to notify when a model should halt running).
    The model writes inputs out via the model_output and model_input tags via shared memory.
        model_output contains a timestamp and DoF outputs
        model_input contains a timestamp and model inputs (i.e., features in most cases).
    The model also writes out the model_output to a socket (UDP) for the environment to read -- this can be changed in the future to use sharedmemory, but the 
    RegressionController and ClassifierController are not set up to read from shared memory yet.
    """
    model_smi = [
        adapt_flag_smow.smm.get_variable_list()[0], # notify the onlinestreamer to load a new model by this int
        active_flag_smow.smm.get_variable_list()[0], # notify the online streamer to pause running with this flag
    ]
    model_ow = [
        model_output_smow, # <- timestamp, DOF1, DOF2 ->
        model_input_smow, # <- timestamp, ---INPUTS--- ->
        model_output_sow,
    ]

    """
    The environment receives messages over the UDP port from the model to let the user take action during the game loop. This is handled pretty manually within the 
    environment, but could be made more elegant in the future. The environment_smi is then empty, but provided to keep the interface consistent and extensible.

    For adaptation, the environment writes out messages to shared memory via the environment_feedback tag. This contains a timestamp, trial number, and environment feedback (i.e., reward or pseudolabels).
    The environment also has a 
    """
    environment_smi = []
    environment_ow = [
        environment_feedback_smow, # <- timestamp, TRIAL, ENVIRONMENT FEEDBACK ->
        environment_flag_smow, # indicate when the environment is alive
    ]
    
    """
    The adaptation manager receives messages from the memory manager to indicate a new slice of memory is ready (default upon a trial being complete).
    The adaptation manager also receives messages from the environment to indicate when the environment is alive (i.e., when it is useful to continue adaptation).

    The adaptation manager writes to the model when a new model is ready to be loaded (i.e., when the model should be updated).
    """
    adaptation_manager_smi = [
        memory_flag_smow.smm.get_variable_list()[0],
        environment_flag_smow.smm.get_variable_list()[0]
    ]
    adaptation_manager_ow = [
        adapt_flag_smow
    ]

    """
    The memory manager receives messages from the model containing the inputs (e.g., EMG features), that will be used for adaptation later.
    The memory manager receives messages from the environment in response to its actions to provide feedback to ends up being a pseudo-label.
    The memory manager receives messages from the environment to indicate when the environment is alive (i.e., when it is useful to continue adaptation).

    The memory manager outputs a message to the adaptation_manager which specifies the number of memory slices that have been saved thus far (written to .pkl files).
    """
    memory_manager_smi = [
        model_input_smow.smm.get_variable_list()[0],
        environment_feedback_smow.smm.get_variable_list()[0],
        environment_flag_smow.smm.get_variable_list()[0],
    ]
    memory_manager_ow = [
        memory_flag_smow
    ]

    return model_smi, model_ow, environment_smi, environment_ow, adaptation_manager_smi, adaptation_manager_ow, memory_manager_smi, memory_manager_ow

def get_pdil_adaptation_items():
    ...

def get_uil_adaptation_items():
    ...

def get_dodr_adaptation_items():
    ...