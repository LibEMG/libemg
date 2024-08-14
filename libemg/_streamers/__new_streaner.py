from libemg.shared_memory_manager import SharedMemoryManager
from multiprocessing import Process


"""
This class will inherit from the Process class. The goal is to read data and pass it to the Shared Memory object.
"""
class NewStreamer(Process):
    def __init__(self, shared_memory_items:   list = []):
        Process.__init__(self, daemon=True)
        self.shared_memory_items = shared_memory_items
        # TODO: Pass in whatever parameters you will need here.

    """
    This function is required for the streamer to work. In this function you should have a while loop
    that continuously listens for new data from the device and update the shared memory object.
    """
    def run(self):
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        #TODO: Fille out the rest (see any of the other streamers in the _streamers folder for examples)        
    
