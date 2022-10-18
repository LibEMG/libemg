class RawData:
    # TODO: Make thread safe
    def __init__(self):
        self.emg_data = []
        self.imu_data = []

    def get_imu(self):
        return self.imu_data

    def get_emg(self):
        return list(self.emg_data)

    def add_emg(self,data):
        self.emg_data.append(data)

    def add_imu(self,data):
        self.imu_data.append(data)
    
    def reset_emg(self):
        self.emg_data = []
    
    def adjust_increment(self, window, increment):
        self.emg_data = self.emg_data[-window:]
        self.emg_data = self.emg_data[increment:window]
