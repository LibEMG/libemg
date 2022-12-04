import numpy as np
import os
import re
import socket
import csv
import pickle
import time
import math
import wfdb
import copy
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from pathlib import Path
from glob import glob
from itertools import compress
from datetime import datetime
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from libemg.raw_data import RawData
from libemg.utils import get_windows, _get_mode_windows

class DataHandler:
    def __init__(self):
        self.data = []
        pass

    def _get_repeating_values(self, data):
        repeats = 0
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                repeats += 1
        return repeats


    def _get_num_channels(self, data):
        return len(data[0])

    def _get_sampling_rate(self, data, time):
        return int(math.ceil(len(data)/time))

    def _get_resolution(self, data):
        return int(math.ceil(math.log2(len(np.unique(data)))))
    
    def _get_max_value(self, data):
        return np.max(data)
    
    def _get_min_value(self, data):
        return np.min(data)


class OfflineDataHandler(DataHandler):
    """OfflineDataHandler class - responsible for collecting all offline data in a directory.

    The purpose of this class is to facilitate the process of accumulating offline training
    and testing data. This class is extensible to a variety of file and folder structures. 
    """
    def __init__(self):
        super().__init__()
    

    def get_data(self, folder_location="", filename_dic={}, delimiter=",", mrdf_key='p_signal'):
        """Method to collect data from a folder into the OfflineDataHandler object. Metadata can be collected either from the filename
        specifying <tag>_regex keys in the filename_dic, or from within the .csv or .txt files specifying <tag>_columns in the filename_dic.

        Parameters
        ----------
        folder_location: str
            Location of the dataset relative to current file path
        filename_dic: dict
            dictionary containing the values of the metadata and the regex or columns associated with that metadata.
        delimiter: int
            How the columns of the files are separated in the .txt or .csv files.
        """
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
        files.extend([y for x in os.walk(folder_location) for y in glob(os.path.join(x[0], '*.txt'))])
        files.extend([y for x in os.walk(folder_location) for y in glob(os.path.join(x[0], '*.hea'))])

        # Convert all file paths to unix style
        files = [Path(f).as_posix() for f in files]

        # check files meet all regex
        regex_keys = [filename_dic[k] for k in dictionary_keys if k.endswith("_regex")]
        self._check_file_regex(files, regex_keys)

        for f in files:
            if '.hea' in f:
                # The key is the emg key that is in the mrdf file
                file_data = (wfdb.rdrecord(f.replace('.hea',''))).__getattribute__(mrdf_key)
            else:
                file_data = np.genfromtxt(f,delimiter=delimiter)
            # collect the data from the file
            if "data_column" in dictionary_keys:
                self.data.append(file_data[:, filename_dic["data_column"]])
            else:
                self.data.append(file_data)
            # also collect the metadata from the filename
            for k in keys:
                #TODO: There is an issue here - talk to evan
                if k + "_regex" in dictionary_keys:
                    k_val = re.findall(filename_dic[k+"_regex"],f)[0]
                    k_id  = filename_dic[k].index(k_val)
                    metadata_column = k_id * np.ones((file_data.shape[0],1), dtype=int)
                    setattr(self, k, getattr(self,k)+[metadata_column])
                elif k + "_column" in dictionary_keys:
                    column = file_data[:,filename_dic[k+"_column"]]
                    if type(filename_dic[k]) == list:
                        k_id = np.array([filename_dic[k].index(i) for i in column])
                        metadata_column = np.expand_dims(k_id, axis=1)
                    else:
                        # if a tuple is passed in (range of values)
                        # we can put a check here later
                        metadata_column = np.expand_dims(column,1)
                    setattr(self, k, getattr(self,k)+[metadata_column])
    
    def active_threshold(self, num_std=1.5, nm_label=0, class_attribute="class"):
        """Replaces class labels with the no motion label when amplitude is outside of the no motion range. This is
        an in-place operation on the offline data handler.

        Parameters
        ----------
        num_std: float
            The number of standard deviations that must be exceeded to be considered an active class.
        nm_label: int
            The class label associated with the no motion class.
        class_attribute: string
            The key used in the dictionary for classes when the dataset was constructed. This should exist in 
            the extra_attributes of the offline data handler.
        """
        # get the mean and std of channel amplitudes for nm class
        assert class_attribute in self.extra_attributes
        for data, labels in zip(self.data, getattr(self, class_attribute)):
            nm_ids = (labels == nm_label).reshape(-1)
            new_mavs = np.mean(np.abs(data[nm_ids,:]),axis=1)# mean across channels
            nm_mavs = np.concatenate((nm_mavs, new_mavs)) if "nm_mavs" in locals() else new_mavs
        nm_mav_mean = np.mean(nm_mavs)
        nm_mav_std  = np.std(nm_mavs)
        # act on the data, replacing all class labels that don't exceed the threshold
        class_copy = getattr(self, class_attribute)
        num_relabel = 0
        total_samples = 0
        for i, (data, labels) in enumerate(zip(self.data, getattr(self, class_attribute))):
            nm_labels = labels == nm_label
            mavs = np.mean(np.abs(data),axis=1)
            relabel_ids = mavs < nm_mav_mean + num_std * nm_mav_std
            num_relabel += sum(relabel_ids) - sum(nm_labels) # only count changed from active -> nm
            total_samples += relabel_ids.shape[0] - sum(nm_labels) # only count changes from active -> nm
            class_copy[i][relabel_ids] = nm_label
        setattr(self, class_attribute, class_copy)
        print(f"{num_relabel} of {total_samples} active class samples were relabelled to no motion.")
    
    def parse_windows(self, window_size, window_increment):
        """Parses windows based on the acquired data from the get_data function.

        Parameters
        ----------
        window_size: int
            The number of samples in a window. 
        window_increment: int
            The number of samples that advances before next window.
        
        Returns
        ----------
        windows_: array_like
            A np.ndarray of size windows x channels x samples
        metadata_: array_like
            A dictionary containing np.ndarrays for each metadata tag of the dataset. Each window will
            have an associated value for each metadata. Therefore, the dimensions of the metadata should be Wx1 for each field.
        """
        return self._parse_windows_helper(window_size, window_increment)

    def _parse_windows_helper(self, window_size, window_increment):
        """Function that actually performs windowing on the OfflineDataHandler after error checking has been performed.

        Parameters
        ----------
        window_size: int
            The number of samples in a window. 
        window_increment: int
            The number of samples that advances before next window.
        
        Returns
        ----------
        windows_: array_like
            A np.ndarray of size windows x channels x samples
        metadata_: array_like
            A dictionary containing np.ndarrays for each metadata tag of the dataset. Each window will
            have an associated value for each metadata. Therefore, the dimensions of the metadata should be Wx1 for each field.
        """
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
    
    def isolate_channels(self, channels):
        """Entry point for isolating a certain range of channels. 

        Parameters
        ----------
        channels: array_like
            A list of values that you want to isolate. (e.g. [0,1,2]). Indexing starts at 0.
            
        Returns
        ----------
        OfflineDataHandler
            returns a new offline data handler with only the data that satisfies the requested slice.
        """
        # Validate channel list
        for c in channels:
            if c < 0 or c >= len(self.data[0][0]):
                print("Invalid channel list - index: " + str(c))
                return 
        new_odh = copy.deepcopy(self)
        # TODO: Optimize this
        for i in range(0, len(new_odh.data)):
            new_odh.data[i] = new_odh.data[i][:,channels]
        return new_odh
    
    def isolate_data(self, key, values):
        """Entry point for isolating a single key of data within the offline data handler. First, error checking is performed within this method, then
        if it passes, the isolate_data_helper is called to make a new OfflineDataHandler that contains only that data.

        Parameters
        ----------
        key: str
            The metadata key that will be used to filter (i.e., "subject", "rep", "class", "set", whatever you'd like).
        values: array_like
            A list of values that you want to isolate. (e.g. [0, 1,2,3]). Indexing starts at 0.
            
        Returns
        ----------
        OfflineDataHandler
            returns a new offline data handler with only the data that satisfies the requested slice.
        """
        assert key in self.extra_attributes
        assert type(values) == list 
        return self._isolate_data_helper(key,values)

    def _isolate_data_helper(self, key, values):
        """Function that actually performs the isolation of OfflineDataHandler.data according to the elements of metadata[key] being in the values list. 

        Parameters
        ----------
        key: str
            The metadata key that will be used to filter (i.e., "subject", "rep", "class", "set", whatever you'd like)
        values: list
            
        Returns
        ----------
        OfflineDataHandler
            returns a new offline data handler with only the data that satisfies the requested slice.
        """
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
    
    def _check_file_regex(self, files, regex_keys):
        """Function that verifies that the list of files in the dataset folder agree with the metadata regex in the dictionary. It is assumed that
        if the filename does not match the regex there is either a mistake is creating the regex or those files are not intended to be loaded. The
        number of files that were excluded are printed to the console, and the excluded files are removed from the files variable (list passed by
        reference so any changes in the function scope will persist outside the function scope)

        Parameters
        ----------
        files: array_like
            A list containing the path (str) of all the files found in the dataset folder that end in .csv or .txt
        regex_keys: array_like
            A list containing the dictionary keys passed during the dataset loading process that indicate metadata to be extracted
            from the path.
            
        Returns
        ----------
        None
        """
        num_files = len(files)
        removed_files = []
        for f in files:
            violations = 0
            for k in regex_keys:
                # regex failed to return a value
                if len(re.findall(k,f)) == 0:
                    violations += 1
            if violations:
                removed_files.append(f)
        [files.remove(rf) for rf in removed_files]
        print(f"{len(removed_files)} of {num_files} files violated regex and were excluded")
    
    def visualize():
        pass

class OnlineDataHandler(DataHandler):
    """OnlineDataHandler class - responsible for collecting data streamed in through UDP socket.

    This class is extensible to any device as long as the data is being streamed over UDP.
    Note, you should change either file, std_out or emg_arr to True for anything meaningful
    to happen.

    Parameters
    ----------
    port: int (optional), default = 12345
        The UDP port to listen for events on. 
    ip: string (optional), default = '127.0.0.1'
        The UDP ip to listen for events on.
    file_path: string (optional), default = "raw_emg.csv"
        The path of the file to write the raw EMG to. This only gets written to if the file parameter is set to true.
    file: bool (optional): default = False
        If True, all data acquired over the UDP port will be written to a file specified by the file_path parameter.
    std_out: bool (optional): default = False
        If True, all data acquired over the UDP port will be written to std_out.
    emg_arr: bool (optional): default = True
        If True, all data acquired over the UDP port will be written to an array object that can be accessed.
    """
    def __init__(self, port=12345, ip='127.0.0.1', file_path="raw_emg.csv", file=False, std_out=False, emg_arr=False):
        self.port = port 
        self.ip = ip
        self.options = {'file': file, 'file_path': file_path, 'std_out': std_out, 'emg_arr': emg_arr}
        
        if not file and not std_out and not emg_arr:
            raise Exception("Set either file, std_out, or emg_arr parameters or this class will have no functionality.")

        # Deal with threading:
        BaseManager.register('RawData', RawData)
        manager = BaseManager()
        manager.start()
        self.raw_data = manager.RawData()
        self.listener = Process(target=self._listen_for_data_thread, args=[self.raw_data], daemon=True,)
    
    def start_listening(self):
        """Starts listening in a seperate process for data streamed over UDP. 

        The options (file, std_out, and emg_arr) will determine what happens with this data.
        """
        self.listener.start()

    def stop_listening(self):
        """Terminates the process listening for data.
        """
        self.listener.terminate()

    def analyze_hardware(self, analyze_time=10):
        """Analyzes several metrics from the hardware:
        (1) sampling rate
        (2) resolution
        (3) min val
        (4) max val
        (5) time between samples
        (6) reapeating values
        (7) number of channels

        Parameters
        ----------
        analyze_time: int (optional), default=10 (seconds)
            The time in seconds that you want to analyze the device for. 
        """
        if not self._check_streaming():
            return

        btwn_sample_time = []
        self.raw_data.reset_emg()
        st = time.time()
        current_size = 0
        print("Starting analysis " + "(" + str(analyze_time) + "s)... We suggest that you elicit varying contractions and intensities to get an accurate analysis.")
        while(time.time() - st < analyze_time):
            if len(self.raw_data.get_emg()) > current_size:
                current_size += 1
                btwn_sample_time.append(time.time())
        emg_data = self.raw_data.get_emg().copy()
        print("Sampling Rate: " + str(self._get_sampling_rate(emg_data,analyze_time)))
        print("Num Channels: " + str(self._get_num_channels(emg_data)))
        print("Max Value: " + str(self._get_max_value(emg_data)))
        print("Min Value: " + str(self._get_min_value(emg_data)))
        print("Resolution: " + str(self._get_resolution(emg_data)) + " bits")
        for i in range(1,len(btwn_sample_time)):
            btwn_sample_time[i-1] = btwn_sample_time[i] - btwn_sample_time[i-1]
        btwn_sample_time.pop()
        print("Time Between Samples - Mean: " + str(np.mean(btwn_sample_time)) + "s STD: " + str(np.std(btwn_sample_time)) + "s")
        print("Repeating Values: " + str(self._get_repeating_values(emg_data)))
        
        self.stop_listening()
        print("Analysis sucessfully complete. ODH process has stopped.")

    def visualize(self, num_samples=500, y_axes=None):
        """Visualize the incoming raw EMG in a plot (all channels together).

        Parameters
        ----------
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        y_axes: array_like (optional)
            A list of two elements consisting of the y-axes.
        """
        pyplot.style.use('ggplot')
        if not self._check_streaming():
            return
        num_channels = len(self.raw_data.get_emg()[0])
        emg_plots = []
        figure, ax = pyplot.subplots()
        figure.suptitle('Raw Data', fontsize=16)
        for i in range(0,num_channels):
            emg_plots.append(ax.plot([],[],label="CH"+str(i+1)))
        figure.legend()
        
        def update(frame):
            data = np.array(self.raw_data.get_emg())
            if len(data) > num_samples:
                data = data[-num_samples:]
            if len(data) > 0:
                x_data = list(range(0,len(data)))
                for i in range(0,num_channels):
                    y_data = data[:,i]
                    emg_plots[i][0].set_data(x_data, y_data)
                figure.gca().relim()
                figure.gca().autoscale_view()
                if not y_axes is None:
                    figure.gca().set_ylim(y_axes)
            return emg_plots,

        animation = FuncAnimation(figure, update, interval=100)
        pyplot.show()

    def visualize_channels(self, channels, num_samples=500, y_axes=None):
        """Visualize individual channels (each channel in its own plot).

        Parameters
        ----------
        channels: list
            A list of channels to graph indexing starts at 0.
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        y_axes: array_like (optional)
            A list of two elements consisting of the y-axes.
        """
        pyplot.style.use('ggplot')
        emg_plots = []
        figure, axs = pyplot.subplots(len(channels), 1)
        figure.suptitle('Raw Data', fontsize=16)
        for i in range(0,len(channels)):
            axs[i].set_ylabel("Channel " + str(channels[i]))
            emg_plots.append(axs[i].plot([],[]))

        def update(frame):
            data = np.array(self.raw_data.get_emg())
            if len(data) > num_samples:
                data = data[-num_samples:]
            if len(data) > 0:
                x_data = list(range(0,len(data)))
                for i in range(0,len(channels)):
                    y_data = data[:,i]
                    emg_plots[i][0].set_data(x_data, y_data)
                
                    axs[i].relim()
                    axs[i].autoscale_view()
                    if not y_axes is None:
                        axs[i].set_ylim(y_axes)
            return emg_plots,

        animation = FuncAnimation(figure, update, interval=100)
        pyplot.show()


    def _listen_for_data_thread(self, raw_data):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.bind((self.ip, self.port))
        if self.options['file']:
            open(self.options['file_path'], "w").close()
        while True:
            data = sock.recv(4096)
            if data:
                data = pickle.loads(data)
                timestamp = datetime.now()
                if self.options['std_out']:
                    print(str(data) + " " + str(timestamp))  
                if self.options['file']:
                    with open(self.options['file_path'], 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(data)
                if self.options['emg_arr']:
                    raw_data.add_emg(data)

    def _check_streaming(self, timeout=10):
        wt = time.time()
        while(True):
            if len(self.raw_data.get_emg()) > 0: 
                return True
            if time.time() - wt > timeout:
                print("Not reading any data.... Check hardware connection.")
                return False