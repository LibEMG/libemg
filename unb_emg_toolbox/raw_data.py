class RawData:
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