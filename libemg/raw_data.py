import multiprocessing

class RawData:
    def __init__(self):
        self.emg_lock = multiprocessing.Lock()
        self.imu_lock = multiprocessing.Lock()
        self.emg_data = []
        self.imu_data = []

    def get_imu(self):
        return self.imu_data

    def get_emg(self):
        return list(self.emg_data)

    def add_emg(self,data):
        with self.emg_lock:
            self.emg_data.append(data)

    def add_imu(self,data):
        with self.imu_lock:
            self.imu_data.append(data)
    
    def reset_emg(self):
        with self.emg_lock:
            self.emg_data = []

    def reset_imu(self):
        with self.imu_lock:
            self.imu_data = []
    
    def adjust_increment(self, window, increment):
        with self.emg_lock:
            self.emg_data = self.emg_data[-window:]
            self.emg_data = self.emg_data[increment:window]
