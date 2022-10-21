import time 
import socket
import numpy as np
from multiprocessing import Process

class MockEMGStreamer:
    """
    The point of this class is to simulate real-time EMG data. 
    """
    def __init__(self, file, num_channels=8, sampling_rate=10, port=12345, ip="127.0.0.1"):
          self.num_channels = num_channels
          self.sampling_rate = sampling_rate
          self.port = port 
          self.ip = ip
          self.data = np.loadtxt(file, delimiter=",")
          self.process = Process(target=self._stream_thread, daemon=True)
    
    def stream(self):
        self.process.start()

    def _stream_thread(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        index = 0
        while True and index < len(self.data):
            sock.sendto(bytes(str(list(self.data[index][:self.num_channels])), "utf-8"), (self.ip, self.port))
            index += 1
            time.sleep(1/self.sampling_rate)


