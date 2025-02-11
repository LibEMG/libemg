import socket
import struct
import numpy as np
import pickle
from libemg.shared_memory_manager import SharedMemoryManager
from multiprocessing import Process, Event, Lock

class DelsysEMGStreamer(Process):
    """
    Delsys Trigno wireless EMG system.

    Requires the Trigno Control Utility to be running.

    ----------
    shared_memory_items : list
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype, Lock()].
    emg : bool
        Enable EMG streaming
    imu : bool
        Enable IMU streaming
    recv_ip : str
        The ip address the device is connected to (likely 'localhost').
    cmd_port : int
        Port of TCU command messages.
    data_port : int
        Port of TCU data access.
    total_channels : int
        Total number of channels supported by the device.
    timeout : float
        Number of seconds before socket returns a timeout exception

    Attributes
    ----------
    BYTES_PER_CHANNEL : int
        Number of bytes per sample per channel. EMG and accelerometer data
    CMD_TERM : str
        Command string termination.

    Notes
    -----
    Implementation details can be found in the Delsys SDK reference:
    http://www.delsys.com/integration/sdk/
    """

    BYTES_PER_CHANNEL = 4
    CMD_TERM = '\r\n\r\n'
    EMG_PORT = 50043
    IMU_PORT = 50044

    def __init__(self, 
                 shared_memory_items:   list = [],
                 emg=True,
                 imu=False,
                 recv_ip:               str = 'localhost', 
                 cmd_port:              str = 50040, 
                 data_port:             str = 50043, 
                 aux_port:              str = 50044,
                 channel_list:        str = list(range(8)), 
                 timeout:               int = 10):
        """
        Note: data_port 50043 refers to the current port that EMG data is being streamed to. For older devices, the EMG data_port may be 50041 (e.g., the Delsys Trigno)
        """
        Process.__init__(self, daemon=True)

        self.connected = False
        self.signal = Event()
        self.shared_memory_items = shared_memory_items

        self.emg = emg
        self.imu = imu

        self.emg_handlers = []
        self.imu_handlers = []

        self.host = recv_ip
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.aux_port = aux_port
        self.channel_list = channel_list
        self.timeout = timeout

        self._min_recv_size = 16 * self.BYTES_PER_CHANNEL

    def connect(self):
        # create command socket and consume the servers initial response
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self._comm_socket.recv(1024)

        # create the data socket
        self._data_socket = socket.create_connection(
            (self.host, self.data_port), self.timeout)
        self._data_socket.setblocking(1)

        # create the aux data socket
        self._aux_socket = socket.create_connection(
            (self.host, self.aux_port), self.timeout)
        self._aux_socket.setblocking(1)

        self._send_cmd('START')

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)
    
    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    def run(self):
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        def write_emg(emg):
            # update the samples in "emg"
            self.smm.modify_variable("emg", lambda x: np.vstack((np.flip(emg,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("emg_count", lambda x: x + emg.shape[0])
        self.add_emg_handler(write_emg)

        def write_imu(imu):
            # update the samples in "imu"
            self.smm.modify_variable("imu", lambda x: np.vstack((np.flip(imu,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("imu_count", lambda x: x + imu.shape[0])
            # sock.sendto(data_arr, (self.ip, self.port))
        self.add_imu_handler(write_imu)

        self.connect()

        while True:
            try:
                if self.emg:
                    packet = self._data_socket.recv(self._min_recv_size)
                    data = np.asarray(struct.unpack('<'+'f'*16, packet))
                    data = data[self.channel_list]
                    if len(data.shape)==1:
                        data = data[None, :]
                    for e in self.emg_handlers:
                        e(data)
                if self.imu:
                    packet = self._aux_socket.recv(self._min_recv_size)
                    data = np.asarray(struct.unpack('<'+'f'*16, packet))
                    assert np.any(data!=0), "IMU not currently working"
                    data = data
                    if len(data.shape)==1:
                        data = data[None, :]
                    for i in self.imu_handlers:
                        i(data)
                A = 1

            except Exception as e:
                print("Error Occurred: " + str(e))
                continue

            if self.signal.is_set():
                self.cleanup()
                break
        print("LibEMG -> DelsysStreamer (process ended).")
        
    def cleanup(self):
        self._send_cmd('STOP')
        print("LibEMG -> DelsysStreamer (streaming stopped).")
        self._comm_socket.close()
        print("LibEMG -> DelsysStreamer (comm socket closed).")
        

    
    def __del__(self):
        try:
            self._comm_socket.close()
        except:
            pass

    def _send_cmd(self, command):
        self._comm_socket.send(self._cmd(command))
        resp = self._comm_socket.recv(128)
        self._validate(resp)

    @staticmethod
    def _cmd(command):
        return bytes("{}{}".format(command, DelsysEMGStreamer.CMD_TERM),
                     encoding='ascii')

    @staticmethod
    def _validate(response):
        s = str(response)
        if 'OK' not in s:
            print("warning: TrignoDaq command failed: {}".format(s))

    
