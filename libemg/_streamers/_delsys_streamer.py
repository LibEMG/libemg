import socket
import struct
import numpy
import pickle


"""
Some credit for the Trigno conenction goes to 
github: https://github.com/axopy
name: Kenneth Lyons
"""

class DelsysEMGStreamer:
    """
    Delsys Trigno wireless EMG system.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    host : str
        IP address the TCU server is running on.
    cmd_port : int
        Port of TCU command messages.
    data_port : int
        Port of TCU data access.
    rate : int
        Sampling rate of the data source.
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

    def __init__(self, stream_ip='localhost', stream_port='12345', recv_ip='localhost', cmd_port=50040, data_port=50043, total_channels=list(range(8)), timeout=10):
        """
        Note: data_port 50043 refers to the current port that EMG data is being streamed to. For older devices, the EMG data_port may be 50041 (e.g., the Delsys Trigno)"""
        self.host = recv_ip
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.total_channels = total_channels
        self.timeout = timeout
        self.stream_ip = stream_ip
        self.stream_port = stream_port

        self._min_recv_size = 16 * self.BYTES_PER_CHANNEL

        self._initialize()
        self.start()
    def _initialize(self):

        # create command socket and consume the servers initial response
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self._comm_socket.recv(1024)

        # create the data socket
        self._data_socket = socket.create_connection(
            (self.host, self.data_port), self.timeout)
        self._data_socket.setblocking(1)

    def start(self):
        """
        Tell the device to begin streaming data.

        You should call ``read()`` soon after this, though the device typically
        takes about two seconds to send back the first batch of data.
        """
        self._send_cmd('START')

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Parameters
        ----------
        num_samples : int
            Number of samples to read per channel.

        Returns
        -------
        data : ndarray, shape=(total_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        while True:
            packet = self._data_socket.recv(self._min_recv_size)
            data = numpy.asarray(struct.unpack('<'+'f'*16, packet))
            data = data[self.total_channels]
            data_arr = pickle.dumps(data)
            self.sock.sendto(data_arr, (self.stream_ip, self.stream_port))

    def stop(self):
        """Tell the device to stop streaming data."""
        self._send_cmd('STOP')

    def reset(self):
        """Restart the connection to the Trigno Control Utility server."""
        self._initialize()

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

    def start_stream(self):
        self.read()