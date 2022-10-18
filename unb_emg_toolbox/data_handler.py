import numpy as np
import os
import re
import socket
import csv
import ast
from glob import glob
from itertools import compress
from datetime import datetime
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from unb_emg_toolbox.raw_data import RawData
from unb_emg_toolbox.utils import get_windows

class DataHandler:
    def __init__(self):
        self.data = []
        pass

    def get_data(self):
        pass

class OfflineDataHandler(DataHandler):
    '''
    OfflineDataHandler class - responsible for collecting all offline data in a directory.
    '''
    # TODO: Add option for testing and training folders
    def __init__(self):
        super().__init__()
    
    def get_data(self, dataset_folder="", dictionary={},  delimiter=","):
        self._get_data_helper(delimiter, dataset_folder, dictionary)
    
    def parse_windows(self, window_size, window_increment):
        return self._parse_windows_helper(window_size, window_increment)

    def isolate_data(self, key, values):
        assert key in self.extra_attributes
        assert type(values) == list 
        return self._isolate_data_helper(key,values)

    def _parse_windows_helper(self, window_size, window_increment):
        metadata_ = {}
        for i, file in enumerate(self.data):
            windows = get_windows(file,window_size,window_increment)
            for k in self.extra_attributes:
                if k not in metadata_.keys():
                    metadata_[k] = np.ones((windows.shape[0])) * getattr(self, k)[i]
                else:
                    metadata_[k] = np.concatenate((metadata_[k], np.ones((windows.shape[0])) * getattr(self, k)[i]))
            if "windows_" in locals():
                windows_ = np.concatenate((windows_, windows))
            else:
                windows_ = windows
            
        return windows_, metadata_

    def _get_data_helper(self, delimiter, folder_location, filename_dic):
        data = []
        # you can insert custom member variables that will be collected from the filename using the dictionary
        # this gives at least a tiny bit of flexibility around what is recorded aside from the data
        dictionary_keys = filename_dic.keys()
        keys = [k for k in dictionary_keys if not k.endswith("_regex")]
        for k in keys:
            if not hasattr(self, k):
                setattr(self, k, [])
        self.extra_attributes = keys

        if not os.path.isdir(folder_location):
            print("Invalid dataset directory: " + folder_location)
        
        # get all files in directory
        files = [y for x in os.walk(folder_location) for y in glob(os.path.join(x[0], '*.csv'))]
        for f in files:
            # collect the data from the file
            self.data.append(np.genfromtxt(f,delimiter=delimiter))
            # also collect the metadata from the filename
            for k in keys:
                k_val = re.findall(filename_dic[k+"_regex"],f)[0]
                k_id  = filename_dic[k].index(k_val)
                setattr(self, k, getattr(self,k)+[k_id])

    def _isolate_data_helper(self, key, values):
        new_odh = OfflineDataHandler()
        setattr(new_odh, "extra_attributes", self.extra_attributes)
        key_attr = getattr(self, key)
        keep_mask = list([i in values for i in key_attr])
        setattr(new_odh, "data", list(compress(self.data, keep_mask)))
        for k in self.extra_attributes:
            setattr(new_odh, k,list(compress(getattr(self, k), keep_mask)))
        return new_odh


class OnlineDataHandler(DataHandler):
    '''
    OnlineDataHandler class - responsible for collecting data streamed in through TCP socket.
    '''
    def __init__(self, port=12345, ip='127.0.0.1', file_path="raw_emg.csv", file=False, std_out=False, emg_arr=False):
        self.port = port 
        self.ip = ip
        self.options = {'file': file, 'file_path': file_path, 'std_out': std_out, 'emg_arr': emg_arr}
        
        # Deal with threading:
        BaseManager.register('RawData', RawData)
        manager = BaseManager()
        manager.start()
        self.raw_data = manager.RawData()
        self.listener = Process(target=self.listen_for_data_thread, args=[self.raw_data], daemon=True,)
    
    def get_data(self):
        self.listener.start()
    
    def listen_for_data_thread(self, raw_data):
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