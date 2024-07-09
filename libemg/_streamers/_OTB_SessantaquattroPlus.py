import socket
import pickle

"""
OT Bioelettronica
Sessantaquattro+ device: 
Up to 64 EMG channels (32, 32+32, 64) 
    + 4 IMU channels 
    + 2 AUX channels 
    + Buffer 
    + Ramp
Copyright (c) 2024 Simone Posella

check OTBioelettronica website for protocol configuration and document
"""


class OTBSessantaquattroPlus:
    def __init__(self, stream_ip='0.0.0.0', stream_port=45454,
                 conv_factor=0.000286):

        self.CONVERSION_FACTOR = conv_factor
        self.ip_address = stream_ip
        self.port = stream_port
        self.connection = None
        self.sq_socket = None
        self.sample_from_channels = None
        self.start_command = self.create_bin_command(1)[0]

    def initialize(self):
        self.sample_from_channels = [0 for i in range(self.number_of_channels - 8)]
        self.start_command = self.create_bin_command(1)[0]
        self.sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sq_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    def start(self):
        self.connection = self.connect_to_sq(self.sq_socket, self.ip_address, self.port, self.start_command)

    def stop(self):
        self.disconnect_from_sq(self.connection)

    # Create the binary command which is sent to Sessantaquattro+
    # Decide here how to set the device
    # Typical configuration are:
    # EMG: 2000 Hz - 2 byte per sample
    # EEG:  500 Hz - 3 byte per sample
    def create_bin_command(self, start=1):
        rec = 0
        trig = 0
        ext = 0
        hpf = 1
        hres = 0
        mode = 0
        nch = 3
        fsamp = 2
        getset = 0

        command = 0
        command = command + start
        command = command + rec * 2
        command = command + trig * 4
        command = command + ext * 16
        command = command + hpf * 64
        command = command + hres * 128
        command = command + mode * 256
        command = command + nch * 2048
        command = command + fsamp * 8192
        command = command + getset * 32768

        self.number_of_channels = None
        self.sample_frequency = None
        self.bytes_in_sample = None

        if nch == 0:
            if mode == 1:
                self.number_of_channels = 12
            else:
                self.number_of_channels = 16
        elif nch == 1:
            if mode == 1:
                self.number_of_channels = 16
            else:
                self.number_of_channels = 24
        elif nch == 2:
            if mode == 1:
                self.number_of_channels = 24
            else:
                self.number_of_channels = 40
        elif nch == 3:
            if mode == 1:
                self.number_of_channels = 40
            else:
                self.number_of_channels = 72
        else:
            raise Exception('Wrong value for nch. Got: {0}', nch)

        if fsamp == 0:
            if mode == 3:
                self.sample_frequency = 2000
            else:
                self.sample_frequency = 500
        elif fsamp == 1:
            if mode == 3:
                self.sample_frequency = 4000
            else:
                self.sample_frequency = 1000
        elif fsamp == 2:
            if mode == 3:
                self.sample_frequency = 8000
            else:
                self.sample_frequency = 2000
        elif fsamp == 3:
            if mode == 3:
                self.sample_frequency = 16000
            else:
                self.sample_frequency = 4000
        else:
            raise Exception('wrong value for fsamp. Got: {fsamp}', fsamp)

        if hres == 1:
            self.bytes_in_sample = 3
        else:
            self.bytes_in_sample = 2

        if (
                not self.number_of_channels or
                not self.sample_frequency or
                not self.bytes_in_sample):
            raise Exception(
                "Could not set number_of_channels "
                "and/or and/or bytes_in_sample")

        return (self.integer_to_bytes(command),
                self.number_of_channels,
                self.sample_frequency,
                self.bytes_in_sample)

    # Convert integer to bytes
    def integer_to_bytes(self, command):
        return int(command).to_bytes(2, byteorder="big")

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
            if output_milli_volts and channel_index < (number_of_channels - 8):
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

        return sample_from_channels_as_bytes
        #return sample_from_channels

    # Connect to Sessantaquattro+'s TCP socket and send start command
    def connect_to_sq(self,
                      sq_socket,
                      ip_address,
                      port,
                      start_command):
        self.sq_socket.bind((ip_address, port))
        self.sq_socket.listen(1)
        print('Waiting for connection...')
        conn, addr = self.sq_socket.accept()
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


class OTBSessantaquattroPlusStreamer:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def start_stream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        device = OTBSessantaquattroPlus()
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
