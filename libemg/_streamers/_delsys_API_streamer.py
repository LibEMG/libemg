from libemg.shared_memory_manager import SharedMemoryManager
from multiprocessing import Process, Event, Lock

import numpy as np

class DataKernel():
    def __init__(self, trigno_base):
        self.TrigBase = trigno_base
        self.packetCount = 0
        self.sampleCount = 0
        self.allcollectiondata = [[]]
        self.channel1time = []

    def processData(self, data_queue):
        """Processes the data from the DelsysAPI and place it in the data_queue argument"""
        outArr = self.GetData()
        if outArr is not None:
            for i in range(len(outArr)):
                self.allcollectiondata[i].extend(outArr[i][0].tolist())
            try:
                for i in range(len(outArr[0])):
                    if np.asarray(outArr[0]).ndim == 1:
                        data_queue.append(list(np.asarray(outArr, dtype='object')[0]))
                    else:
                        data_queue.append(list(np.asarray(outArr, dtype='object')[:, i]))
                try:
                    self.packetCount += len(outArr[0])
                    self.sampleCount += len(outArr[0][0])
                except:
                    pass
            except IndexError:
                pass

    def processYTData(self, data_queue):
        """Processes the data from the DelsysAPI and place it in the data_queue argument"""
        outArr = self.GetYTData()
        if outArr is not None:
            for i in range(len(outArr)):
                self.allcollectiondata[i].extend(outArr[i][0].tolist())
            try:
                yt_outArr = []
                for i in range(len(outArr)):
                    chan_yt = outArr[i]
                    chan_ydata = np.asarray([k.Item2 for k in chan_yt[0]], dtype='object')
                    yt_outArr.append(chan_ydata)

                data_queue.append(list(yt_outArr))

                try:
                    self.packetCount += len(outArr[0])
                    self.sampleCount += len(outArr[0][0])
                except:
                    pass
            except IndexError:
                pass

    def GetData(self):
        """ Check if data ready from DelsysAPI via Aero CheckDataQueue() - Return True if data is ready
            Get data (PollData)
            Organize output channels by their GUID keys

            Return array of all channel data
        """

        dataReady = self.TrigBase.CheckDataQueue()                      # Check if DelsysAPI real-time data queue is ready to retrieve
        if dataReady:
            DataOut = self.TrigBase.PollData()                          # Dictionary<Guid, List<double>> (key = Guid (Unique channel ID), value = List(Y) (Y = sample value)
            outArr = [[] for i in range(len(DataOut.Keys))]             # Set output array size to the amount of channels being outputted from the DelsysAPI

            channel_guid_keys = list(DataOut.Keys)                      # Generate a list of all channel GUIDs in the dictionary
            for j in range(len(DataOut.Keys)):                          # loop all channels
                chan_data = DataOut[channel_guid_keys[j]]               # Index a single channels data from the dictionary based on unique channel GUID (key)
                outArr[j].append(np.asarray(chan_data, dtype='object')) # Create a NumPy array of the channel data and add to the output array

            return outArr
        else:
            return None

    def GetYTData(self):
        """ YT Data stream only available when passing 'True' to Aero Start() command i.e. TrigBase.Start(True)
            Check if data ready from DelsysAPI via Aero CheckYTDataQueue() - Return True if data is ready
            Get data (PollYTData)
            Organize output channels by their GUID keys

            Return array of all channel data
        """

        dataReady = self.TrigBase.CheckYTDataQueue()                        # Check if DelsysAPI real-time data queue is ready to retrieve
        if dataReady:
            DataOut = self.TrigBase.PollYTData()                            # Dictionary<Guid, List<(double, double)>> (key = Guid (Unique channel ID), value = List<(T, Y)> (T = time stamp in seconds Y = sample value)
            outArr = [[] for i in range(len(DataOut.Keys))]                 # Set output array size to the amount of channels being outputted from the DelsysAPI

            channel_guid_keys = list(DataOut.Keys)                          # Generate a list of all channel GUIDs in the dictionary
            for j in range(len(DataOut.Keys)):                              # loop all channels
                chan_yt_data = DataOut[channel_guid_keys[j]]                # Index a single channels data from the dictionary based on unique channel GUID (key)
                outArr[j].append(np.asarray(chan_yt_data, dtype='object'))  # Create a NumPy array of the channel data and add to the output array

            return outArr
        else:
            return None
        

class DelsysAPIStreamer(Process):
    def __init__(self, key, license, dll_folder = 'resources/', shared_memory_items:   list = [],
                       emg:                   bool = True):
        Process.__init__(self, daemon=True)

        self.dll_folder = dll_folder
        self.connected = False
        self.signal = Event()
        self.shared_memory_items = shared_memory_items
        self.key = key 
        self.license = license
        
        self.emg = emg
        
        self.emg_handlers = []
    
    def connect(self, key, license):
        try:
            self.trigbase.ValidateBase(key, license)
        except Exception as e:
            if "product not licensed." in str(e):
                print("Error: Key/License Not Validated\nClose the program and paste your key/license into TrignoBase.py file\nContact support@delsys.com if you have not received your APi key/license")
            elif "no RF subsystem found" in str(e):
                print("Error: Trigno system not found\nPlease make sure your base station or lite dongle is plugged in via USB\nVisit our website to request a quote or contact support@delsys.com")
            else:
                print(str(e))
            print(Exception)
        
        self.sensors = self.scan()
        self.select_sensors()

    def scan(self):
        try:
            f = self.trigbase.ScanSensors().Result
        except Exception as e:
            print(str(e))

        all_scanned_sensors = self.trigbase.GetScannedSensorsFound()
        print("Sensors Found:\n")
        for sensor in all_scanned_sensors:
            print("(" + str(sensor.PairNumber) + ") " +
                sensor.FriendlyName + "\n" +
                sensor.Configuration.ModeString + "\n")
        return all_scanned_sensors

    def select_sensors(self):
        sensor_count = len(self.sensors)
        for i in range(sensor_count):
            self.trigbase.SelectSensor(i)

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def run(self):
        try:
            from collections import deque
            from pythonnet import load
            load("coreclr")
            import clr
        except RuntimeError as e:
            raise RuntimeError('.NET runtime not found, so DelsysAPI Streamer cannot run. Please ensure that a .NET runtime >8.0 is installed. Exiting run() method.') from e

        clr.AddReference(self.dll_folder + "DelsysAPI")
        clr.AddReference("System.Collections")
        from Aero import AeroPy
        # Set up shared memory 
        self.trigbase = AeroPy()
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        def write_emg(emg):
            # update the samples in "emg"
            self.smm.modify_variable("emg", lambda x: np.vstack((np.flip(emg,0), x))[:x.shape[0],:])
            # update the number of samples retrieved
            self.smm.modify_variable("emg_count", lambda x: x + emg.shape[0])
        self.add_emg_handler(write_emg)

        self.connect(self.key, self.license)

        if self.trigbase.GetPipelineState() == 'Connected':
            self.trigbase.Configure(False, False)
            channelcount = 0
            channelobjects = []
            datahandler = DataKernel(self.trigbase)
            emg_idxs = []

            for i in range(len(self.sensors)):

                selectedSensor = self.trigbase.GetSensorObject(i)
                print("(" + str(selectedSensor.PairNumber) + ") " + str(selectedSensor.FriendlyName))

                if len(selectedSensor.TrignoChannels) > 0:
                    print("--Channels")

                    for channel in range(len(selectedSensor.TrignoChannels)):
                        sample_rate = round(selectedSensor.TrignoChannels[channel].SampleRate, 3)
                        print("----" + selectedSensor.TrignoChannels[channel].Name + " (" + str(sample_rate) + " Hz)")
                        channelcount += 1
                        channelobjects.append(channel)
                        datahandler.allcollectiondata.append([])

                        emg_idxs.append("EMG" in selectedSensor.TrignoChannels[channel].Name)

            self.trigbase.Start(False)
            
            while True:
                try:
                    outArr = datahandler.GetData()
                    # convert to one single np array
                    if outArr is not None:
                        emg_data = []
                        for i in range(len(outArr)):
                            if emg_idxs[i]:
                                emg_data.append(outArr[i][0])
                        emg_data = np.array(emg_data).T
                        if emg_data.shape[1] == 1:
                            emg_data = emg_data[:,None]
                        for e in self.emg_handlers:
                            e(emg_data)
                except Exception as e:
                    print("LibEMG -> DelsysAPIStreamer: Error ocurred " + str(e))
                    
                if self.signal.is_set():
                    self.cleanup()
                    break
            print("LibEMG -> DelsysStreamer (process ended).")

    def cleanup(self):
        pass

    def __del__(self):
        pass
