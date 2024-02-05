import socket
import pickle
import struct

"""
OT Bioelettronica
Syncstation device: 
Up to 8 probes with many combinations.
Connectable probes are:
    - Muovi (32 EMG ch. + IMU + buffer + ramp)
    - Muovi+ (64 EMG ch. + IMU + buffer + ramp)
    - Sessantaquatro (64 EMG ch. + buffer + ramp)
    - Sessantaquatro+ (64 EMG ch. + IMU + buffer + ramp)
    - Due+ (2 EMG + IMU + buffer + ramp)
    
    + 3 AUX channels
    + 1 Load cell channel
There is ONE RESTRICTION RULE due to the packet coming from the Syncstation
In order to be compatible with LibEMG the data must be read from the socket and forwarded to LibEMG
For this reason, at the moment, is hard to organize data from the Syncstation
convert them separately depending on the probe configuration. 
For this reason, all the probes must be set in the same configuration (EMG or EEG)

Copyright (c) 2024 Simone Posella
Check OTBioelettronica website for protocol configuration and document
"""

class OTBSyncstation:
    def __init__(self, stream_ip='192.168.76.1', stream_port=54320,
                 conv_factor=0.000286):

        self.CONVERSION_FACTOR = conv_factor
        self.ip_address = stream_ip
        self.port = stream_port
        self.connection = None
        self.sq_socket = None
        self.sample_from_channels = None
        self.number_of_channels = None
        self.sample_frequency = None
        self.bytes_in_sample = 2
        self.total_number_of_bytes = 0
        self.start_command_length = 0
        self.start_command = self.create_bin_command(1)[0]
        self.DeviceEN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.EMG = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.Mode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.SizeComm = sum(self.DeviceEN)

        NumChan = [38, 38, 38, 38, 70, 70, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

        Error = any(self.DeviceEN[i] > 1 for i in range(16))
        if Error:
            print("Error, set DeviceEN values equal to 0 or 1")
            exit()

        Error = any(self.EMG[i] > 1 for i in range(16))
        if Error:
            print("Error, set EMG values equal to 0 or 1")
            exit()

        Error = any(self.Mode[i] > 3 for i in range(16))
        if Error:
            print("Error, set Mode values between to 0 and 3")
            exit()

    # Function to calculate CRC8
    def CRC8(self, Vector, Len):
        crc = 0
        j = 0

        while Len > 0:
            Extract = Vector[j]
            for i in range(8, 0, -1):
                Sum = crc % 2 ^ Extract % 2
                crc //= 2

                if Sum > 0:
                    str_crc = []
                    a = format(crc, '08b')
                    b = format(140, '08b')
                    for k in range(8):
                        str_crc.append(int(a[k] != b[k]))

                    crc = int(''.join(map(str, str_crc)), 2)

                Extract //= 2

            Len -= 1
            j += 1

        return crc

    def initialize(self):
        self.sample_from_channels = [0 for i in range(self.number_of_channels)]
        self.start_command = self.create_bin_command(1)[0]
        self.sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sq_socket.settimeout(10)

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
        self.DeviceEN = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.EMG = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.Mode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.NumChan = [38, 38, 38, 38, 70, 70, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

        Error = any(self.DeviceEN[i] > 1 for i in range(16))
        if Error:
            print("Error, set DeviceEN values equal to 0 or 1")
            exit()

        Error = any(self.EMG[i] > 1 for i in range(16))
        if Error:
            print("Error, set EMG values equal to 0 or 1")
            exit()

        Error = any(self.Mode[i] > 3 for i in range(16))
        if Error:
            print("Error, set Mode values between to 0 and 3")
            exit()

        self.SizeComm = sum(self.DeviceEN)

        NumEMGChanMuovi = 0
        NumAUXChanMuovi = 0
        NumEMGChanSessn = 0
        NumAUXChanSessn = 0
        NumEMGChanDuePl = 0
        NumAUXChanDuePl = 0
        muoviEMGChan = []
        muoviAUXChan = []
        sessnEMGChan = []
        sessnAUXChan = []
        duePlEMGChan = []
        duePlAUXChan = []

        sampFreq = 2000
        TotNumChan = 0
        TotNumByte = 0
        ConfStrLen = 1
        ConfString = [0] * 18

        ConfString[0] = self.SizeComm * 2 + start

        for i in range(16):
            if self.DeviceEN[i] == 1:
                ConfString[ConfStrLen] = (i * 16) + self.EMG[i] * 8 + self.Mode[i] * 2 + 1

                if i < 5:
                    muoviEMGChan.extend(list(range(TotNumChan + 1, TotNumChan + 33)))
                    muoviAUXChan.extend(list(range(TotNumChan + 33, TotNumChan + 39)))
                    NumEMGChanMuovi += 32
                    NumAUXChanMuovi += 6
                elif i > 6:
                    duePlEMGChan.extend(list(range(TotNumChan + 1, TotNumChan + 3)))
                    duePlAUXChan.extend(list(range(TotNumChan + 3, TotNumChan + 9)))
                    NumEMGChanDuePl += 2
                    NumAUXChanDuePl += 6
                else:
                    sessnEMGChan.extend(list(range(TotNumChan + 1, TotNumChan + 65)))
                    sessnAUXChan.extend(list(range(TotNumChan + 65, TotNumChan + 71)))
                    NumEMGChanSessn += 64
                    NumAUXChanSessn += 6

                TotNumChan += self.NumChan[i]

                if self.EMG[i] == 1:
                    TotNumByte += self.NumChan[i] * 2
                else:
                    TotNumByte += self.NumChan[i] * 3

                if self.EMG[i] == 1:
                    sampFreq = 2000

                ConfStrLen += 1

        SyncStatChan = list(range(TotNumChan, TotNumChan + 7))
        TotNumChan += 6
        TotNumByte += 12

        ConfString[ConfStrLen] = 0  # Placeholder for CRC8 calculation

        ConfString[ConfStrLen] = self.CRC8(ConfString, ConfStrLen)
        ConfStrLen += 1
        StartCommand = ConfString[0:ConfStrLen]

        self.start_command_length = ConfStrLen
        self.total_number_of_bytes = TotNumByte
        self.number_of_channels = TotNumChan
        self.sample_frequency = sampFreq
        self.bytes_in_sample = 2

        if (
                not self.number_of_channels or
                not self.sample_frequency or
                not self.bytes_in_sample):
            raise Exception(
                "Could not set number_of_channels "
                "and/or and/or bytes_in_sample")

        return (StartCommand,
                self.number_of_channels,
                self.sample_frequency,
                self.bytes_in_sample)

    # Convert integer to bytes
    def integer_to_bytes(self, command, length):
        return int(command).to_bytes(length, byteorder="big")

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
            if output_milli_volts and channel_index < (number_of_channels): #subtract 6 to remove last 6 channels (number_of_channels-6)
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
        # return sample_from_channels

    # Connect to Sessantaquattro+'s TCP socket and send start command
    def connect_to_sq(self,
                      sq_socket,
                      ip_address,
                      port,
                      start_command):

        conn = sq_socket.connect((ip_address, port))
        print('Connection to Syncstation: {0}'.format((ip_address)))
        sendcommand = struct.pack('B' * self.start_command_length, *start_command)
        sq_socket.sendall(sendcommand)
        print("Inviato Start Command ", sendcommand)
        return conn

    # Disconnect from Sessantaquattro+ by sending a stop command
    def disconnect_from_sq(self, conn):
        if conn is not None:
            ConfString = [0] * 18

            # Send the stop command to syncstation
            for i in range(18):
                ConfString[i] = 0
            ConfString[1] = self.CRC8(ConfString, 1)
            StopCommand = ConfString[0:1]
            packed_data = struct.pack('B' * len(StopCommand), *StopCommand)
            print("Mandato comando di Stop")
            self.sq_socket.sendall(packed_data)
            # Close the TCP socket
            self.sq_socket.close()
        else:
            raise Exception(
                "Can't disconnect because the"
                "connection is not established")


class OTBSyncstationStreamer:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def start_stream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        device = OTBSyncstation()
        device.initialize()
        device.start()
        blockData = device.number_of_channels * device.bytes_in_sample
        print("blockdata: ", blockData)

        while True:
            try:

                samples = device.sq_socket.recv(blockData)
                sample_from_channels = device.bytes_to_integers(
                    samples,
                    device.number_of_channels,
                    device.bytes_in_sample,
                    output_milli_volts=True)
                data = pickle.dumps(list(sample_from_channels))
                sock.sendto(data, (self.ip, self.port))
            except Exception as e:
                print(e)
                print("Worker Stopped.")
                device.stop()
                quit()
