from multiprocessing import Process

import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

from libemg.shared_memory_manager import SharedMemoryManager


class MindroveStreamer(Process):
    def __init__(self, shared_memory_items):
        super().__init__(daemon=True)
        self.shared_memory_items = shared_memory_items

    def run(self):
        def write_emg(emg):
            def add_to_buffer(buffer):
                new_buffer = np.vstack((emg, buffer))  # put new data on top
                new_buffer = new_buffer[:buffer.shape[0], :]    # ensure buffer stays the same size
                return new_buffer
            smm.modify_variable('emg', add_to_buffer)
            smm.modify_variable('emg_count', lambda x: x + 1)

        # Initialize shared memory in this process
        smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            smm.create_variable(*item)

        params = MindRoveInputParams()
        board_id = BoardIds.MINDROVE_WIFI_BOARD
        board_shim = BoardShim(board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        num_samples = 1 # number of samples to grab and add to buffer at a time
        emg_channels = board_shim.get_emg_channels(board_id)    # we can get ppg channels from a similar method

        while True:
            data = board_shim.get_board_data(num_samples=num_samples)   # grabs data from ringbuffer AND DELETES IT
            if data is None or data.shape[1] == 0:
                continue

            # data is of shape: (num_params, num_samples)
            emg = data[emg_channels].T  # we expect data as (num_samples, num_channels)
            write_emg(emg)

        # TODO: Call cleanup somehow
        # if board_shim.is_prepared():
        #     board_shim.release_session()

    

