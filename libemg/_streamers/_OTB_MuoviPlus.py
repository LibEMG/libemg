import socket
import pickle
import numpy as np
import signal
import atexit

from multiprocessing import Event, Process
from libemg.shared_memory_manager import SharedMemoryManager
from crc.crc import Crc8, CrcCalculator

"""
OT Bioelettronica
MuoviPlus device: 
    Up to 64 EMG channels 
    + 4 IMU channels 
    + Buffer 
    + Ramp
Copyright (c) 2024 Simone Posella
Check OTBioelettronica website for protocol configuration and document
"""


class OTBMuoviPlus:
    def __init__(self, stream_ip='0.0.0.0', stream_port=54321,
                 conv_factor=0.000286):

        self.CONVERSION_FACTOR = conv_factor
        self.ip_address = stream_ip
        self.port = stream_port
        self.connection = None
        self.sq_socket = None
        self.sample_from_channels = None
        self.number_of_channels = None
        self.sample_frequency = None
        self.bytes_in_sample = None
        self.start_command = None

    def initialize(self):
        self.start_command = self.create_bin_command(1)[0]
        self.sample_from_channels = [0 for i in range(self.number_of_channels)]
        self.sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sq_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    def start(self):
        self.connection = self.connect_to_sq(self.sq_socket, self.ip_address, self.port, self.start_command)

    def stop(self):
        self.disconnect_from_sq(self.connection)

    # Create the binary command which is sent to Muovi+
    # Decide here how to set the device
    # Typical configuration are:
    # EMG: 2000 Hz - 2 byte per sample
    # EEG:  500 Hz - 3 byte per sample
    def create_bin_command(self, start=1):

        # Refer to the communication protocol for details about these variables:
        EMG = 1  # 1 = EMG, 0 = EEG
        Mode = 0  # 0 = 32Ch Monop, 1 = 16Ch Monop, 2 = 32Ch ImpCk, 3 = 32Ch Test

        # Number of acquired channel depending on the acquisition mode
        NumChanVsMode = np.array([70, 70, 70, 70])

        self.number_of_channels = None
        self.sample_frequency = None
        self.bytes_in_sample = None

        if EMG == 1:
            self.bytes_in_sample = 2
        else:
            self.bytes_in_sample = 3

        # Create the command to send to Muovi
        command = EMG * 8 + Mode * 2 + start
        self.number_of_channels = NumChanVsMode[Mode]
        if EMG == 0:
            self.sample_frequency = 500  # Sampling frequency = 500 for EEG
        else:
            self.sample_frequency = 2000  # Sampling frequency = 2000 for EMG

        if EMG == 1:
            self.bytes_in_sample = 2
        else:
            self.bytes_in_sample = 3

        if (
                not self.number_of_channels or
                not self.sample_frequency or
                not self.bytes_in_sample):
            raise Exception(
                "Could not set number_of_channels "
                "and/or and/or bytes_in_sample")

        command_in_bytes = self.integer_to_bytes(command)
        return (command_in_bytes,
                self.number_of_channels,
                self.sample_frequency,
                self.bytes_in_sample)

    # Convert integer to bytes
    def integer_to_bytes(self, command):
        return int(command).to_bytes(1, byteorder="big")

    # Convert byte-array value to an integer value and apply two's complement
    def convert_bytes_to_int(self, bytes_value, bytes_in_sample):
        value = None
        if bytes_in_sample == 2:
            # Combine 2 bytes to a 16 bit integer value
            value = \
                bytes_value[0] * 256 + \
                bytes_value[1]
            # See if the value is negative and make the two's complement
            if value >= 32768:
                value -= 65536
        elif bytes_in_sample == 3:
            # Combine 3 bytes to a 24 bit integer value
            value = \
                bytes_value[0] * 65536 + \
                bytes_value[1] * 256 + \
                bytes_value[2]
            # See if the value is negative and make the two's complement
            if value >= 8388608:
                value -= 16777216
        else:
            raise Exception(
                "Unknown bytes_in_sample value. Got: {}, "
                "but expecting 2 or 3".format(bytes_in_sample))
        return value

    # Convert channels from bytes to integers
    def bytes_to_integers(self,
                          sample_from_channels_as_bytes,
                          number_of_channels,
                          bytes_in_sample,
                          output_milli_volts):
        channel_values = []
        # Separate channels from byte-string. One channel has
        # "bytes_in_sample" many bytes in it.
        for channel_index in range(number_of_channels):
            channel_start = channel_index * bytes_in_sample
            channel_end = (channel_index + 1) * bytes_in_sample
            channel = sample_from_channels_as_bytes[channel_start:channel_end]

            # Convert channel's byte value to integer
            value = self.convert_bytes_to_int(channel, bytes_in_sample)

            # Convert bio measurement channels to milli volts if needed
            # The 4 last channels (Auxiliary and Accessory-channels)
            # are not to be converted to milli volts
            if output_milli_volts and channel_index < (number_of_channels - 6):
                value *= self.CONVERSION_FACTOR
            channel_values.append(value)
        return channel_values

    #     Read raw byte stream from data logger. Read one sample from each
    #     channel. Each channel has 'bytes_in_sample' many bytes in it.
    def read_raw_bytes(self, connection, number_of_all_channels, bytes_in_sample):
        buffer_size = number_of_all_channels * bytes_in_sample
        new_bytes = connection.recv(buffer_size)
        return new_bytes

    def read(self):
        sample_from_channels_as_bytes = self.read_raw_bytes(
            self.connection,
            self.number_of_channels,
            self.bytes_in_sample)

        sample_from_channels = self.bytes_to_integers(
            sample_from_channels_as_bytes,
            self.number_of_channels,
            self.bytes_in_sample,
            output_milli_volts=False)

        return sample_from_channels

    # Connect to Muovi TCP socket and send start command
    def connect_to_sq(self,
                      sq_socket,
                      ip_address,
                      port,
                      start_command):
        sq_socket.bind((ip_address, port))
        sq_socket.listen(1)
        print('Waiting for connection...')
        conn, addr = sq_socket.accept()
        print('Connection from address: {0}'.format((addr)))
        conn.send(start_command)
        return conn

    # Disconnect from Sessantaquattro+ by sending a stop command
    def disconnect_from_sq(self, conn):
        if conn is not None:
            (stop_command,
             _,
             __,
             ___) = self.create_bin_command(start=0)
            conn.send(stop_command)
            conn.shutdown(2)
            conn.close()
        else:
            raise Exception(
                "Can't disconnect because the"
                "connection is not established")


class OTBMuoviPlusStreamer:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def start_stream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        device = OTBMuoviPlus()
        device.initialize()
        device.start()

        #Last channels excluded from visualization
        additional_channel = 6 #IMU (4 channels) + RAMP + BUFF
        n_byte = 2
        while True:
            try:
                samples = device.read()
                newsamples = samples[:(len(samples) - additional_channel * n_byte)]
                # print(samples)
                data = pickle.dumps(list(newsamples))
                sock.sendto(data, (self.ip, self.port))
            except Exception as e:
                print(e)
                print("Worker Stopped.")
                device.stop()
                quit()


class PacketParser:
    @staticmethod
    def parse_raw(n, packet):
        """
        Parse raw binary packet into signed 16-bit EMG values.
        Assumes each packet consists of 64 samples per channel and 9 bytes of overhead.
        """
        try:
            int_data = np.frombuffer(packet, dtype='>i2').astype(np.int32)
            int_data[int_data > 32767] -= 65536

            if int_data.shape[0] < n * (292 // 2):
                return None

            parsed_packets = []
            for i in range(n):
                base = 2 * i * (64 + 9)
                ch1_start = base
                ch1_end = base + 64
                ch2_start = base + 64 + 9
                ch2_end = ch2_start + 64

                ch1 = int_data[ch1_start:ch1_end]
                ch2 = int_data[ch2_start:ch2_end]

                if len(ch1) == 64 and len(ch2) == 64:
                    parsed_packets.append(np.concatenate((ch1, ch2)))

            return parsed_packets

        except Exception as e:
            print(f"[PacketParser Error] {e}")
            return None
        

class OTBMuoviPlusEMGStreamer(Process):
    """
    Handles EMG streaming from the OTB MuoviPlus device using a TCP socket.
    Parses, filters, and stores data into shared memory for real-time access.
    """

    # Define the start and stop signals for the device
    START_SIGNAL = [0b00000101, 0b01011011, 0b01001011]
    STOP_TCP = [0b00000000]

    def __init__(self, ip: str, port: int, shared_memory_items: list, emg_channels=128):
        super().__init__()
        self.ip = ip
        self.port = port
        self.shared_memory_items = shared_memory_items
        self.emg_channels = emg_channels

        self.client = None
        self.stop_event = Event()
        self.smm = SharedMemoryManager()
        
        # Graceful exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._handle_exit_signal)


    def run(self):
        """Main loop that runs in its own process to stream EMG data."""

        self._setup_shared_memory()
        
        try:
            # Connect to the device and start streaming
            self._connect()
            self._send_packet(self.START_SIGNAL)

            while not self.stop_event.is_set():
                # Receive raw data packets from the device
                raw = self.client.recv(292 * 8)
                if not raw:
                    break

                # Parse the raw data into EMG packets
                emg_packets = PacketParser.parse_raw(n=8, packet=raw)

                # Process the received EMG packets
                if emg_packets and len(emg_packets) == 8:
                    emg_packets = [self._filter_channels(packet) for packet in emg_packets]
                    self._write_emg_data(emg_packets)

        except Exception as e:
            print(f"[OTBStreamer Error] {e}")
        finally:
            self.cleanup()
    
    
    def stop(self):
        """Stop the streaming process."""
        self.stop_event.set()
        self._send_packet(self.STOP_TCP)
        self.join()
    

    def _connect(self):
        """Connect to the OTB MuoviPlus device."""
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(5)
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client.connect((socket.gethostbyname(self.ip), self.port))
        print(f"[OTBStreamer] Connected to {self.ip}:{self.port}")


    def _send_packet(self, sig_bits):
        """Send a packet to the OTB MuoviPlus device."""
        if self.client:
            packet = bytearray(sig_bits)
            crc_calc = CrcCalculator(Crc8.MAXIM_DOW)
            packet.append(crc_calc.calculate_checksum(packet))
            self.client.send(packet)

    
    def _disconnect(self):
        """Disconnect from the OTB MuoviPlus device."""
        if self.client:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.client.close()
            self.client = None
            print("[OTBStreamer] Disconnected")

    
    def _filter_channels(self, packet):
        """
        Filters out the unused 64 channels if only 64 channels are configured.
        """
        if self.emg_channels == 128 or len(packet) != 128:
            return packet
        
        first_half, second_half = packet[:64], packet[64:]

        if np.all(first_half == 0):
            return second_half
        elif np.all(second_half == 0):
            return first_half
        else:
            return first_half + second_half

    
    def _write_emg_data(self, packets):
        """
        Write EMG data to shared memory, FIFO-style.
        """
        emg_array = np.array(packets)
        num_samples = emg_array.shape[0]

        def update_buffer(buffer):
            return np.vstack((emg_array, buffer[:-num_samples]))

        self.smm.modify_variable('emg', update_buffer)
        self.smm.modify_variable('emg_count', lambda x: x + num_samples)


    def _setup_shared_memory(self):
        """Initialize shared memory variables."""
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)


    def cleanup(self):
        self._disconnect()
        self.smm.cleanup()


    def _handle_exit_signal(self, signum, frame):
        print(f"[OTBStreamer] Received exit signal {signum}, cleaning up.")
        self.cleanup()
