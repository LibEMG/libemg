import numpy as np
import os
import re
import socket
import csv
import ast
from datetime import datetime
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from unb_emg_toolbox.raw_data import RawData
from unb_emg_toolbox.utils import get_windows

class DataHandler:
    def __init__(self, num_classes):
        self.training_windows = []
        self.testing_windows = []
        self.training_labels = []
        self.testing_labels = []
        self.train_data = []
        self.test_data = []
        self.num_classes = num_classes

    def get_data(self):
        pass

class OfflineDataHandler(DataHandler):
    '''
    OfflineDataHandler class - responsible for collecting all offline data in a directory.
    '''
    # TODO: Add option for testing and training folders
    def __init__(self, num_classes, train_folder, train_dic={}, test_folder=None, test_dic=None):
        super().__init__(num_classes)
        self.train_folder_loc = train_folder
        self.test_folder_loc = test_folder
        self.train_filename_dic = train_dic
        self.test_filename_dic = test_dic
    
    def get_data(self, delimiter=","):
        self.train_data = self._get_data_helper(delimiter, self.train_folder_loc, self.train_filename_dic)
        if self.test_folder_loc:
            self.test_data = self._get_data_helper(delimiter, self.test_folder_loc, self.test_filename_dic)
    
    def parse_windows(self, window_size, window_increment):
        self._parse_training_windows_helper(window_size, window_increment)
        self._parse_test_windows_helper(window_size, window_increment)

    #TODO: I think this might get messed if the folders aren't in sequential order? 
    def _parse_test_windows_helper(self, window_size, window_increment):
        for i, rep in enumerate(self.test_data):
            windows = get_windows(rep,window_size,window_increment)
            if len(self.testing_windows) > 0:
                self.testing_windows = np.concatenate((self.testing_windows, windows))
            else:
                self.testing_windows = windows
            self.testing_labels = self.testing_labels + [(i % self.num_classes) for _ in range(len(windows))]

    def _parse_training_windows_helper(self, window_size, window_increment):
        for i, rep in enumerate(self.train_data):
            windows = get_windows(rep,window_size,window_increment)
            if len(self.training_windows) > 0:
                self.training_windows = np.concatenate((self.training_windows, windows))
            else:
                self.training_windows = windows
            self.training_labels = self.training_labels + [(i % self.num_classes) for _ in range(len(windows))]

    def _get_data_helper(self, delimiter, folder_location, filename_dic):
        data = []
        # you can insert custom member variables that will be collected from the filename using the dictionary
        # this gives at least a tiny bit of flexibility around what is recorded aside from the data
        dictionary_keys = filename_dic.keys()
        keys = [k for k in dictionary_keys if not k.endswith("_regex")]
        for k in keys:
            if not hasattr(self, k):
                setattr(self, k, [])

        if not os.path.isdir(folder_location):
            print("Invalid dataset directory: " + folder_location)
        
        # get all files in directory
        files = os.listdir(folder_location)
        for f in files:
        
            # collect the data from the file
            data.append(np.genfromtxt(folder_location + "/" + f,delimiter=delimiter))
            # also collect the metadata from the filename
            for k in keys:
                k_val = re.findall(filename_dic[k+"_regex"],f)[0]
                k_id  = filename_dic[k].index(k_val)
                setattr(self, k, getattr(self,k)+[k_id])
        return data

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