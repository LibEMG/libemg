import multiprocessing

class RawData:
    def __init__(self):
        self.emg_lock = multiprocessing.Lock()
        self.imu_lock = multiprocessing.Lock()
        self.other_lock = multiprocessing.Lock()
        self.emg_data = []
        self.imu_data = []
        self.other_modalities = {}

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
    
    def reset_imu(self):
        with self.imu_lock:
            self.imu_data = []
    
    def reset_emg(self):
        with self.emg_lock:
            self.emg_data = []
    
    def adjust_increment(self, window, increment):
        with self.emg_lock:
            self.emg_data = self.emg_data[-window:]
            self.emg_data = self.emg_data[increment:window]

    def instantialize_other(self, other):
        self.other_modalities[other] = []

    def get_others(self):
        return self.other_modalities
    
    def add_other(self, other, data):
        assert self.check_other(other) == True
        self.check_other(other)
        with self.other_lock:
            self.other_modalities[other].append(data)
    
    def reset_others(self):
        for k in self.other_modalities.keys():
            self.other_modalities[k] = []

    def check_other(self, other):
        return other in self.other_modalities

