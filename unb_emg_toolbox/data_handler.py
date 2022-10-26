import numpy as np
import os
import re
import socket
import csv
import ast
import matplotlib.pyplot as plt
from glob import glob
from itertools import compress
from datetime import datetime
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from unb_emg_toolbox.raw_data import RawData
from unb_emg_toolbox.utils import get_windows, _get_mode_windows

class DataHandler:
    def __init__(self):
        self.data = []
        pass

    def get_data(self):
        pass

class OfflineDataHandler(DataHandler):
    """OfflineDataHandler class - responsible for collecting all offline data in a directory.

    The purpose of this class is to facilitate the process of accumulating offline training
    and testing data. This class is extensible to a variety of file and folder structures. 
    """
    def __init__(self):
        super().__init__()
    
    #TODO: Evan - document 
    def get_data(self, dataset_folder="", dictionary={},  delimiter=","):
        self._get_data_helper(delimiter, dataset_folder, dictionary)
    
    def parse_windows(self, window_size, window_increment):
        """Parses windows based on the acquired data from the get_data function.

        Parameters
        ----------
        window_size: int
            The number of samples in a window. 
        window_increment: int
            The number of samples that advances before next window.
        """
        return self._parse_windows_helper(window_size, window_increment)

    #TODO: Evan - document 
    def isolate_data(self, key, values):
        assert key in self.extra_attributes
        assert type(values) == list 
        return self._isolate_data_helper(key,values)

    def _parse_windows_helper(self, window_size, window_increment):
        metadata_ = {}
        for i, file in enumerate(self.data):
            # emg data windowing
            windows = get_windows(file,window_size,window_increment)
            if "windows_" in locals():
                windows_ = np.concatenate((windows_, windows))
            else:
                windows_ = windows
            # metadata windowing
            for k in self.extra_attributes:
                if type(getattr(self,k)[i]) != np.ndarray:
                    file_metadata = np.ones((windows.shape[0])) * getattr(self, k)[i]
                else:
                    file_metadata = _get_mode_windows(getattr(self,k)[i], window_size, window_increment)
                if k not in metadata_.keys():
                    metadata_[k] = file_metadata
                else:
                    metadata_[k] = np.concatenate((metadata_[k], file_metadata))

            
        return windows_, metadata_

    def _get_data_helper(self, delimiter, folder_location, filename_dic):
        data = []
        # you can insert custom member variables that will be collected from the filename using the dictionary
        # this gives at least a tiny bit of flexibility around what is recorded aside from the data
        dictionary_keys = filename_dic.keys()
        keys = [k for k in dictionary_keys if not (k.endswith("_regex") or k.endswith("_column"))]
        for k in keys:
            if not hasattr(self, k):
                setattr(self, k, [])
        self.extra_attributes = keys

        if not os.path.isdir(folder_location):
            print("Invalid dataset directory: " + folder_location)
        
        # get all files in directory
        files = [y for x in os.walk(folder_location) for y in glob(os.path.join(x[0], '*.csv'))]
        for f in files:
            file_data = np.genfromtxt(f,delimiter=delimiter)
            # collect the data from the file
            if "data_column" in dictionary_keys:
                self.data.append(file_data[:, filename_dic["data_column"]])
            else:
                self.data.append(file_data)
            # also collect the metadata from the filename
            for k in keys:
                if k + "_regex" in dictionary_keys:
                    k_val = re.findall(filename_dic[k+"_regex"],f)[0]
                    k_id  = filename_dic[k].index(k_val)
                    setattr(self, k, getattr(self,k)+[k_id])
                elif k + "_column" in dictionary_keys:
                    setattr(self, k, getattr(self,k)+[file_data[:,filename_dic[k+"_column"]]])

    def _isolate_data_helper(self, key, values):
        new_odh = OfflineDataHandler()
        setattr(new_odh, "extra_attributes", self.extra_attributes)
        key_attr = getattr(self, key)
        
        # if these end up being ndarrays, it means that the metadata was IN the csv file.
        
        if type(key_attr[0]) == np.ndarray:
            # for every file (list element)
            data = []
            for f in range(len(key_attr)):
                keep_mask = list([i in values for i in key_attr[f]])
                data.append(self.data[f][keep_mask,:])
            setattr(new_odh, "data", data)

            for k in self.extra_attributes:
                key_value = getattr(self, k)
                if type(key_value[0]) != np.ndarray:
                    # if the other metadata was not in the csv file (i.e. subject label in filename but classes in csv), then just keep it
                    setattr(new_odh, k, key_value)
                else:
                    # the other metadata that is in the csv file should be sliced the same way as the ndarray
                    key = []
                    for f in range(len(key_attr)):
                        keep_mask = list([i in values for i in key_attr[f]])
                        
                        key.append(key_value[f][keep_mask,:])
                    setattr(new_odh, k, key)
        else:
            keep_mask = list([i in values for i in key_attr])
            setattr(new_odh, "data", list(compress(self.data, keep_mask)))
            for k in self.extra_attributes:
                setattr(new_odh, k,list(compress(getattr(self, k), keep_mask)))
        return new_odh

    def visualize():
        pass

class OnlineDataHandler(DataHandler):
    """OnlineDataHandler class - responsible for collecting data streamed in through TCP socket.

    This class is extensible to any device as long as the data is being streamed over TCP.
    Note, you should change either file, std_out or emg_arr to True for anything meaningful
    to happen.

    Parameters
    ----------
    port: int (optional), default = 12345
        The TCP port to listen for events on. 
    ip: string (optional), default = '127.0.0.1'
        The TCP ip to listen for events on.
    file_path: string (optional), default = "raw_emg.csv"
        The path of the file to write the raw EMG to. This only gets written to if the file parameter is set to true.
    file: bool (optional): default = False
        If True, all data acquired over the TCP port will be written to a file specified by the file_path parameter.
    std_out: bool (optional): default = False
        If True, all data acquired over the TCP port will be written to std_out.
    emg_arr: bool (optional): default = True
        If True, all data acquired over the TCP port will be written to an array object that can be accessed.
    """
    def __init__(self, port=12345, ip='127.0.0.1', file_path="raw_emg.csv", file=False, std_out=False, emg_arr=False):
        self.port = port 
        self.ip = ip
        self.options = {'file': file, 'file_path': file_path, 'std_out': std_out, 'emg_arr': emg_arr}
        
        # Deal with threading:
        BaseManager.register('RawData', RawData)
        manager = BaseManager()
        manager.start()
        self.raw_data = manager.RawData()
        self.listener = Process(target=self._listen_for_data_thread, args=[self.raw_data], daemon=True,)
    
    def get_data(self):
        """Starts listening in a seperate process for data streamed over TCP. 

        The options (file, std_out, and emg_arr) will determine what happens with this data.
        """
        self.listener.start()
        
    def visualize(self, num_samples=500):
        """Visualize the incoming raw EMG in a plot.

        Parameters
        ----------
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        """
        plt.style.use('ggplot')
        plt.title("Raw Data")
        while True:
            data = np.array(self.raw_data.get_emg())
            if len(data) > num_samples:
                data = data[-num_samples:]
            if len(data) > 0:
                plt.clf()
                plt.title("Raw Data")
                for i in range(0,len(data[0])):
                    x = list(range(0,len(data)))
                    plt.plot(x, data[:,i], label="CH"+str(i))
                plt.legend(loc = 'lower right')
            plt.pause(0.1)
    
    def _listen_for_data_thread(self, raw_data):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.bind((self.ip, self.port))
        if self.options['file']:
            open(self.options['file_path'], "w").close()
        while True:
            data, _ = sock.recvfrom(1024)
            data = data.decode("utf-8")
            if data:
                timestamp = datetime.now()
                if self.options['std_out']:
                    print(data + " " + str(timestamp))  
                if self.options['file']:
                    with open(self.options['file_path'], 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(ast.literal_eval(data))
                if self.options['emg_arr']:
                    raw_data.add_emg(ast.literal_eval(data))