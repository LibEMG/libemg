import libemg
import time

if __name__ == "__main__":
    ls, smi = libemg.streamers.leap_streamer()

    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in smi:
        while not (smm.find_variable(*item)):
            time.sleep(0.5)


    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)

    odh.log_to_file()

    while True:
        time.sleep(1)