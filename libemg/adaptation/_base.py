import numpy as np
from multiprocessing import Lock
from libemg.output_writer import SharedMemoryOutputWriter, SocketOutputWriter, SIMOutputWriter
from libemg.environments.controllers import SIM_UDP_Receiver

def mod_fn_input(self, data, info):
    new_slice = np.hstack((info['timestamp'],info['model_input_raw'][-1,:]))
    input_size = self.smm.variables['model_input']["shape"][0]
    data[:] = np.vstack((new_slice, data))[:input_size, :]
    return data

def mod_fn_output(self, data, info):
    new_slice = np.hstack((info['timestamp'],info['model_output']))
    input_size = self.smm.variables['model_output']["shape"][0]
    data[:] = np.vstack((new_slice, data))[:input_size, :]
    return data

# def mod_fn_env(self, data, info):
#     new_slice = np.hstack((info['timestamp'], info['trial'], info['environment_feedback']))
#     input_size = self.smm.variables['environment_feedback']["shape"][0]
#     data[:] = np.vstack((new_slice, data))[:input_size, :]
#     return data

def mod_fn_flags(self, data, number):
    data[:] = number
    return data

def mod_fn_flags_count(self, data, number):
    data[:] = data[:] + 1
    return data


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

    # there are two places to grab the model input (pre scaler and post scaler)
    # for the setup of my experiments, I've worked with the pre-scaler (non-normalized) model inputs.
    def mod_fn_input(self, data, info):
        new_slice = np.hstack((info['timestamp'],info['model_input_raw'][-1,:]))
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
    
    def mod_fn_flags_count(self, data, number):
        data[:] = data[:] + 1
        return data


    
    adapt_flag_smow = SharedMemoryOutputWriter('adapt_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags, mod_fn_count=mod_fn_flags_count)
    memory_flag_smow = SharedMemoryOutputWriter('memory_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags, mod_fn_count=mod_fn_flags_count)
    active_flag_smow = SharedMemoryOutputWriter('active_flag', (1,1), np.int8, Lock(), mod_fn=mod_fn_flags, mod_fn_count=mod_fn_flags_count)
    environment_flag_smow = SharedMemoryOutputWriter('environment_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags, mod_fn_count=mod_fn_flags_count)

    model_output_smow = SharedMemoryOutputWriter('model_output', (100, 1+num_outputs), np.float64, Lock(), mod_fn=mod_fn_output, mod_fn_count=mod_fn_flags_count)
    model_input_smow = SharedMemoryOutputWriter('model_input', (100, 1+num_features), np.float64, Lock(), mod_fn=mod_fn_input, mod_fn_count=mod_fn_flags_count)
    environment_feedback_smow = SharedMemoryOutputWriter('environment_feedback', (100, 1+1+num_outputs), np.float64, Lock(), mod_fn=mod_fn_env, mod_fn_count=mod_fn_flags_count)
    
    model_output_sow = SocketOutputWriter("model_output")

    # initial values for flags
    adapt_flag_smow.write(0)
    memory_flag_smow.write(0)
    active_flag_smow.write(1)
    environment_flag_smow.write(1)

    model_output_smow.reset()
    model_input_smow.reset()
    environment_feedback_smow.reset()

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
        adapt_flag_smow.smm.get_shared_memory_items()[0], # notify the onlinestreamer to load a new model by this int
        active_flag_smow.smm.get_shared_memory_items()[0], # notify the online streamer to pause running with this flag
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
        memory_flag_smow.smm.get_shared_memory_items()[0],
        environment_flag_smow.smm.get_shared_memory_items()[0]
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
        model_input_smow.smm.get_shared_memory_items()[0],
        environment_feedback_smow.smm.get_shared_memory_items()[0], # the feedback itself
        environment_feedback_smow.smm.get_shared_memory_items()[1], # the number of times the feedback has been written to (useful for only querying the new stuff to be appended to memory)
        environment_flag_smow.smm.get_shared_memory_items()[0],
    ]
    memory_manager_ow = [
        memory_flag_smow
    ]

    return model_smi, model_ow, environment_smi, environment_ow, adaptation_manager_smi, adaptation_manager_ow, memory_manager_smi, memory_manager_ow


def get_edil_adaptation_objects_crossplatform(num_features, num_outputs):
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

    # there are two places to grab the model input (pre scaler and post scaler)
    # for the setup of my experiments, I've worked with the pre-scaler (non-normalized) model inputs.



    
    adapt_flag_smow = SharedMemoryOutputWriter('adapt_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags, mod_fn_count=mod_fn_flags_count)
    memory_flag_smow = SharedMemoryOutputWriter('memory_flag', (1,1), np.int32, Lock(), mod_fn=mod_fn_flags, mod_fn_count=mod_fn_flags_count)
    active_flag_smow = SharedMemoryOutputWriter('active_flag', (1,1), np.int8, Lock(), mod_fn=mod_fn_flags, mod_fn_count=mod_fn_flags_count)

    model_output_smow = SharedMemoryOutputWriter('model_output', (100, 1+num_outputs), np.float64, Lock(), mod_fn=mod_fn_output, mod_fn_count=mod_fn_flags_count)
    model_input_smow = SharedMemoryOutputWriter('model_input', (100, 1+num_features), np.float64, Lock(), mod_fn=mod_fn_input, mod_fn_count=mod_fn_flags_count)

    model_output_sow = SIMOutputWriter(ip='127.0.0.1', port=11000)

    environment_controller = SIM_UDP_Receiver(ip='127.0.0.1', port=11001) # default from PAVE

    # initial values for flags
    adapt_flag_smow.write(0)
    memory_flag_smow.write(0)
    active_flag_smow.write(1)
    # environment_flag_smow.write(1)

    model_output_smow.reset()
    model_input_smow.reset()
    # environment_feedback_smow.reset()

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
        adapt_flag_smow.smm.get_shared_memory_items()[0], # notify the onlinestreamer to load a new model by this int
        active_flag_smow.smm.get_shared_memory_items()[0], # notify the online streamer to pause running with this flag
    ]
    model_ow = [
        model_output_smow, # <- timestamp, DOF1, DOF2 ->
        model_input_smow, # <- timestamp, ---INPUTS--- ->
        model_output_sow,
    ]

    """
    The environment should handle the write out of feedback data (i.e., if using Unity for the environment, it should have a UDP sender that writes the data out)
    """
    # No need for environment_smi or environment_smows.
    
    """
    The adaptation manager receives messages from the memory manager to indicate a new slice of memory is ready (default upon a trial being complete).
    The adaptation manager also receives messages from the environment to indicate when the environment is alive (i.e., when it is useful to continue adaptation).

    The adaptation manager writes to the model when a new model is ready to be loaded (i.e., when the model should be updated).
    """
    adaptation_manager_smi = [
        memory_flag_smow.smm.get_shared_memory_items()[0],
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
        model_input_smow.smm.get_shared_memory_items()[0],
        # environment_feedback_smow.smm.get_shared_memory_items()[0], # the feedback itself
        # environment_feedback_smow.smm.get_shared_memory_items()[1], # the number of times the feedback has been written to (useful for only querying the new stuff to be appended to memory)
        # environment_flag_smow.smm.get_shared_memory_items()[0],
        environment_controller
    ]
    memory_manager_ow = [
        memory_flag_smow
    ]

    return model_smi, model_ow, adaptation_manager_smi, adaptation_manager_ow, memory_manager_smi, memory_manager_ow


def get_pdil_adaptation_items():
    ...

def get_uil_adaptation_items():
    ...

def get_dodr_adaptation_items():
    ...

def produce_tciil_feedback(current_location: list[int, int],
                           current_direction: list[int, int],
                           target_location: list[int, int],
                           target_size: int,
                           trial_distance: int):
    """
    Example function handle to produce a tolerant context informed incremental learning pseudo-label to be used as feedback from a 2-DoF environment.
    This is used in the default implementation of libemg.environment.curricular_environment.CurricularFittsEnvironment. This uses the procedure described
    in "Context-Informed Incremental Learning Improves Throughput and Reduces Drift in Regression-Based Myoelectric Control", Morrell et al 2025.

    Parameters
    ----------
    current location : list
        A 2DoF location where the cursor is located.
    target location : list
        A 2DoF location where the target is located.
    target size : int
        The size of the target.
    trial_distance : int
        The distance between the cursor and target at the start of the trial.
    """

    optimal_direction = get_optimal_direction(current_location, target_location)

    PC = distance_to_proportional_control(current_location, target_location, current_direction, target_size, trial_distance)
    quadrant_check = check_quadrants(current_location, current_direction, target_location)

    # silence bad directions
    pseudo_label = [val if outcome else 0 for val, outcome in zip(optimal_direction, quadrant_check)]
    # scale to correct value
    pseudo_label_scale = np.linalg.norm(pseudo_label)
    pseudo_label = [val * PC / pseudo_label_scale for val in pseudo_label]
    # pseudo_label = [0 if np.isnan(i) else i for i in pseudo_label] # remove NaNs (completely wrong quadrant is set to 0,0)
    
    return pseudo_label

def get_optimal_direction(current_location: list[int, int],
                          target_location: list[int, int]) -> list[int, int]:
    return [i - j for i, j in zip(target_location, current_location)]

def distance_to_proportional_control(current_location, target_location, current_direction, target_size, trial_distance) -> float:
    distance = np.linalg.norm(np.array(current_location) - np.array(target_location))
    in_target = distance < target_size
    if in_target:
        # step_in_dir = [ x + 0.01*y for x,y in zip(current_location, current_direction)]
        # if np.linalg.norm(np.array(step_in_dir) - np.array(target_location)) < distance:
        #     PC = 0.05 # if we're still approaching the center of the target, output speed is 0.1
        # else:
        PC = 0 # if we're in the circle, but not approaching the center, output speed is 0
    else:
        # trial distance makes sense as the normalizer for most shooting tasks, but sometimes with random distance targets, the new target can be very close
        # in which case, the user wouldn't hit the max proportinal control value for that trial. 
        # it probably makes more sense to just normalize by a consant value
        #PC = min(1.41, np.sqrt((distance - target_size)/trial_distance))
        calculated_percentile = (distance - target_size) / 300
        calculated_pc = 0.1+0.9/(1+np.exp(-10*(calculated_percentile-0.5)))
        PC = min(1., calculated_pc) 
        # the gameplay region in CurricularFittsLaw is about 1000 pixels, so any distance greater than half the playable 
        # area should evoke max speed.
    return PC

def check_quadrants(current_location, current_direction, target_location) -> list[bool, bool]:
    margins = [abs(i - j) for i, j in zip(target_location, current_location)]
    del_margins = [abs(i - (0.01 * k + j)) for i, j, k in zip(target_location, current_location, current_direction)]
    outcome = []
    for i, j in zip(margins, del_margins):
        if i > j :
            # better after the step
            outcome.append(True)
        else:
            # worse after the step
            outcome.append(False)
    return outcome
