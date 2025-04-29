from multiprocessing import Process, Event

import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

from libemg.shared_memory_manager import SharedMemoryManager


class MindroveStreamer(Process):
    def __init__(self, shared_memory_items):
        super().__init__(daemon=True)
        self.shared_memory_items = shared_memory_items
        params = MindRoveInputParams()  # not sure if we can get the IP here to avoid having to connect to the Wifi network...
        self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        self.board_shim = BoardShim(self.board_id, params)
        self.smm = SharedMemoryManager()
        self._stop_event = Event()

    def run(self):
        def write_emg(emg):
            def add_to_buffer(buffer):
                new_buffer = np.vstack((emg, buffer))  # put new data on top
                new_buffer = new_buffer[:buffer.shape[0], :]    # ensure buffer stays the same size
                return new_buffer
            self.smm.modify_variable('emg', add_to_buffer)
            self.smm.modify_variable('emg_count', lambda x: x + 1)

        # Initialize shared memory in this process
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        self.board_shim.prepare_session()
        self.board_shim.start_stream()
        num_samples = 1 # number of samples to grab and add to buffer at a time
        emg_channel_mask = self.board_shim.get_emg_channels(self.board_id)    # we can get ppg channels from a similar method

        try:
            while not self._stop_event.is_set():
                data = self.board_shim.get_board_data(num_samples=num_samples)   # grabs data from ringbuffer AND DELETES IT
                if data is None or data.shape[1] == 0:
                    continue

                # data is of shape: (num_params, num_samples)
                emg = data[emg_channel_mask].T  # we expect data as (num_samples, num_channels)
                write_emg(emg)
        finally:
            # This will only be called in the case of an exception or interrupt, but won't be called when parent process dies (b/c daemon=True)
            self._cleanup()

    def stop(self):
        self._stop_event.set()
        self.join()

    def _cleanup(self):
        if self.board_shim.is_prepared():
            self.board_shim.release_session()
        self.smm.cleanup()

