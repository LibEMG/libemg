import multiprocessing

class RawData:
    def __init__(self):
        self.emg_lock = multiprocessing.Lock()
        self.imu_lock = multiprocessing.Lock()
        # self.other_lock = multiprocessing.Lock()
        self.emg_data = []
        self.imu_data = []
        # self.other_modalities = {}

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

    # def instantialize_others(self, others):
    #     for o in others:
    #         self.other_modalities[o] = []

    # def get_other(self, other):
    #     self.check_other(other)
    #     return self.other_modalities['other']
    
    # def add_other(self, other, data):
    #     self.check_other(other)
    #     with self.other_lock:
    #         self.other_modalities[other].append(data)
    
    # def reset_other(self, other):
    #     self.check_other()
    #     with self.other_lock:
    #         self.other_modalities[other] = []

    # def check_other(self, other):
    #     assert other in self.other_modalities
