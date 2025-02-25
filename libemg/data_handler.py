from abc import ABC, abstractmethod
from typing import Callable, Sequence
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import re
import time
import math
import wfdb
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Axes
from scipy.ndimage import zoom
from scipy.signal import decimate
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from pathlib import Path
from glob import glob
from multiprocessing import Process
from multiprocessing import Process, Event
from libemg.feature_extractor import FeatureExtractor
from libemg.shared_memory_manager import SharedMemoryManager
from scipy.signal import welch
from libemg.utils import get_windows, _get_fn_windows, _get_mode_windows, make_regex

class RegexFilter:
    """
    Filters files based on filenames that match the associated regex pattern and grabs metadata based on the regex pattern.

    Parameters
    ----------
    left_bound: str
        The left bound of the regex.
    right_bound: str
        The right bound of the regex.
    values: list
        The values between the two regexes.
    description: str
        Description of filter - used to name the metadata field. Pass in an empty string to filter files without storing the values as metadata.
    """
    def __init__(self, left_bound: str, right_bound: str, values: Sequence[str], description: str):
        if values is None:
            raise ValueError('Expected a list of values for RegexFilter, but got None. Using regex wildcard is not supported with the RegexFilter.')
        self.pattern = make_regex(left_bound, right_bound, values)
        self.values = values
        self.description = description

    def get_matching_files(self, files: Sequence[str]):
        """Filter out files that don't match the regex pattern and return the matching files.

        Parameters
        ----------
        files: list
            List of potential files that need to be filtered.

        Returns
        ----------
        matching_files: list
            List of files that match regex pattern.
        """
        matching_files = [file for file in files if len(re.findall(self.pattern, file)) != 0]
        return matching_files

    def get_metadata(self, filename: str):
        """Get metadata from the filename.

        Parameters
        ----------
        filename: str
            Name of file.

        Returns
        ----------
        metadata_idx: int
            Index of value (relative to list of values passed in).
        """
        # this is how it should work to be the same as the ODH, but we can maybe discuss redoing this so it saves the actual value instead of the indices. might be confusing to pass values to get data but indices to isolate it. also not sure if it needs to be arrays
        val = re.findall(self.pattern, filename)[0]
        idx = self.values.index(val)
        return idx


class MetadataFetcher(ABC):
    """
    Describes a type of metadata and implements a method to fetch it.

    Parameters
    ----------
    description: str
        Description of metadata.
    """
    def __init__(self, description: str):
        self.description = description

    @abstractmethod
    def __call__(self, filename: str, file_data: npt.NDArray, all_files: Sequence[str]) -> npt.NDArray:
        """Fetch metadata. Must return a (N x M) numpy.ndarray, where N is the number of samples in the EMG data and M is the number of columns in the metadata.
        If a single value array is returned (0D or 1D), it will be cast to a N x 1 array where all values are the original value.

        Parameters
        ----------
        filename: str
            Name of data file.
        file_data: np.ndarray
            Data within file.
        all_files: list
            List of filenames containing all files within data directory.

        Returns
        ----------
        metadata: np.ndarray
            Array containing the metadata corresponding to the provided file.
        """
        raise NotImplementedError("Must implement __call__ method.")


class FilePackager(MetadataFetcher):
    """Package data file with another file that contains relevant metadata (e.g., a labels file). Cycles through all files
    that match the RegexFilter and packages a data file with a metadata file based on a packaging function.

    Parameters
    ----------
    regex_filter: RegexFilter
        Used to find the type of metadata files. The description of this RegexFilter is used to assign the name of the field for this metadata in the OfflineDataHandler.
    package_function: callable or Sequence[RegexFilter]
        Function handle used to determine if two files should be packaged together (i.e., found the metadata file that goes with the data file).
        Takes in the filename of a metadata file and the filename of the data file. Should return True if the files should be packaged together and False if not.
        Alternatively, a list of RegexFilters can be passed in and a function will be created that packages files only if the regex metadata of the data filename
        and metadata filename match.
    align_method: str or callable, default='zoom'
        Method for aligning the samples of the metadata file and data file. Pass in 'zoom' for the metadata file to be zoomed using spline interpolation to the size of the data file or 
        pass in a callable that takes in the metadata and the EMG data and returns the aligned metadata.
    load: callable, str, or None, default=None
        Determines how metadata file is loaded. If a custom loading function, should take in the filename and return an array. If a string, 
        it is assumed to be the MRDF key of a .hea file. If None is passed, the metadata is loaded based on the file extension (only .csv and .txt are supported).
    column_mask: list or None, default=None
        List of integers corresponding to the indices of the columns that should be extracted from the raw file data. If None is passed, all columns are extracted.
    """
    def __init__(self, regex_filter: RegexFilter, package_function: Callable[[str, str], bool] | Sequence[RegexFilter], 
                 align_method: str | Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = 'zoom', load: Callable[[str], npt.NDArray] | str | None = None, column_mask: Sequence[int] | None = None):
        super().__init__(regex_filter.description)
        self.regex_filter = regex_filter
        self.package_filters = None

        if isinstance(package_function, Sequence):
            # Create function to ensure metadata matches
            self.package_filters = copy.deepcopy(package_function)
            package_function = self._match_regex_patterns
        self.package_function = package_function
        self.align_method = align_method
        self.load = load
        self.column_mask = column_mask

    def _match_regex_patterns(self, metadata_file: str, data_file: str):
        assert self.package_filters is not None, 'Attempting to match package filters, but None found.'
        for filter in self.package_filters:
            if len(filter.get_matching_files([metadata_file])) == 0:
                # Doesn't match filters
                return False
            matching_metadata = filter.get_metadata(metadata_file) == filter.get_metadata(data_file)
            if not matching_metadata:
                return False
        return True
        

    def __call__(self, filename: str, file_data: npt.NDArray, all_files: Sequence[str]):
        potential_files = self.regex_filter.get_matching_files(all_files)
        packaged_files = [Path(potential_file) for potential_file in potential_files if self.package_function(potential_file, filename)]
        if len(packaged_files) != 1:
            # I think it's easier to enforce a single file per FilePackager, but we could build in functionality to allow multiple files then just vstack all the data if there's a use case for that.
            raise ValueError(f"Found {len(packaged_files)} files to be packaged with {filename} when trying to package {self.regex_filter.description} file (1 file should be found). Please check filter and package functions.")
        packaged_file = packaged_files[0]
        suffix = packaged_file.suffix
        packaged_file = packaged_file.as_posix()

        if callable(self.load):
            # Passed in a custom loading function
            packaged_file_data = self.load(packaged_file)
        elif isinstance(self.load, str):
            # Passed in a MRDF key
            assert suffix == '.hea', f"Provided string for load parameter, but packaged file doesn't have extension .hea. Please pass in a custom load function and/or ensure the correct file is packaged."
            packaged_file_data = (wfdb.rdrecord(packaged_file.replace('.hea', ''))).__getattribute__(self.load)
        elif suffix == '.txt':
            packaged_file_data = np.loadtxt(packaged_file, delimiter=',')
        elif suffix == '.csv':
            packaged_file_data = pd.read_csv(packaged_file)
            packaged_file_data = packaged_file_data.to_numpy()
        else:
            raise ValueError("Unsupported filetype when loading packaged files - expected filetypes are .csv and .txt. Pass in a callable loading function to load files of other types.")

        # Align with EMG data
        if self.align_method == 'zoom':
            zoom_rate = file_data.shape[0] / packaged_file_data.shape[0]
            zoom_factor = (zoom_rate, 1)    # only align the 0th axis (samples)
            packaged_file_data = zoom(packaged_file_data, zoom=zoom_factor)
        elif callable(self.align_method):
            packaged_file_data = self.align_method(packaged_file_data, file_data)
        else:
            raise ValueError('Unexpected value for align_method. Please pass in a callable or a supported string (e.g., zoom).')

        if self.column_mask is not None:
            # Only grab data at specified columns
            packaged_file_data = packaged_file_data[:, self.column_mask]

        if packaged_file_data.ndim == 1:
            # Ensure 2D array
            packaged_file_data = np.expand_dims(packaged_file_data, axis=1)

        return packaged_file_data


class ColumnFetcher(MetadataFetcher):
    """
    Fetch metadata from columns within data file.

    Parameters
    ----------
    description: str
        Description of metadata.
    column_mask: list or int
        Integers corresponding to indices of columns that should be fetched.
    values: list or None, default=None
        List of potential values within metadata column. If a list is passed in, the metadata will be stored as the location (index) of the value within the provided list. If None, the value within the columns will be stored.
    """
    def __init__(self, description: str, column_mask: Sequence[int] | int, values: Sequence | None = None):
        super().__init__(description)
        self.column_mask = column_mask
        self.values = values

    def __call__(self, filename: str, file_data: npt.NDArray, all_files: Sequence[str]):
        metadata = file_data[:, self.column_mask]
        if isinstance(self.values, list):
            # Convert to indices of provided values
            metadata = np.array([self.values.index(i) for i in metadata])

        return metadata


class DataHandler:
    def __init__(self):
        self.data = []
        pass

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
    """OfflineDataHandler class - responsible for collecting all offline data.

    The purpose of this class is to facilitate the process of accumulating offline training
    and testing data. This class is extensible to a wide range of file and folder structures. 
    """
    def __init__(self):
        super().__init__()
    
    def __add__(self, other):
        # Concatenate two OfflineDataHandlers together
        if not isinstance(other, OfflineDataHandler):
            raise ValueError("Incorrect type used when concatenating OfflineDataHandlers.")
        self_attributes = self.__dict__.keys()
        other_attributes = other.__dict__.keys()
        if not self_attributes == other_attributes:
            # Objects don't have the same attributes
            raise ValueError("Objects being concatenated must have the same attributes.")
        
        new_odh = OfflineDataHandler()
        for self_attribute, other_attribute in zip(self_attributes, other_attributes):
            # Concatenate attributes together
            new_value = []
            new_value.extend(getattr(self, self_attribute))
            new_value.extend(getattr(other, other_attribute))
            if self_attribute == 'extra_attributes':
                # Remove duplicates
                new_value = list(np.unique(new_value))
            # Set attributes of new handler
            setattr(new_odh, self_attribute, new_value)
        return new_odh
        
    def get_data(self, folder_location: str, regex_filters: Sequence[RegexFilter], metadata_fetchers: Sequence[MetadataFetcher] | None = None, delimiter: str = ',',
                 mrdf_key: str = 'p_signal', skiprows: int = 0, data_column: Sequence[int] | None = None, downsampling_factor: int | None = None):
        """Method to collect data from a folder into the OfflineDataHandler object. The relevant data files can be selected based on passing in 
        RegexFilters, which will filter out non-matching files and grab metadata from the filename based on their provided description. Data can be labelled with other
        sources of metadata via passed in MetadataFetchers, which will associate metadata with each data file.


        Parameters
        ----------
        folder_location: str
            Location of the dataset relative to the current file path.
        regex_filters: list
            List of RegexFilters used to filter data files to the desired set of files. Metadata for each RegexFilter
            will be pulled from the filename and stored as a field.
        metadata_fetchers: list or None, default=None
            List of MetadataFetchers used to associate metadata with each data file (e.g., FilePackager). If the provided MetadataFetchers do not suit your needs,
            you may inherit from the MetadataFetcher class to create your own. If None is passed, no extra metadata is fetched (other than from filenames via regex).
        delimiter: str, default=','
            Specifies how columns are separated in .txt or .csv data files.
        mrdf_key: str, default='p_signal'
            Key in mrdf file associated with EMG data.
        skiprows: int, default=0
            The number of rows to skip in the file (e.g., .csv or .txt) starting from the top row.
        data_column: list or None, default=None
            List of indices representing columns of data in data file. If a list is passed in, only the data at these columns will be stored as EMG data.
        downsampling_factor: int or None, default=None
            Factor to downsample by. Signal is first filtered and then downsampled. See scipy.signal.decimate for more details (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html#scipy-signal-decimate).

        Raises
        ------
        ValueError:
            Raises ValueError if folder_location is not a valid directory.
        """
        def append_to_attribute(name, value):
            if name == '':
                # Don't want this data saved to data handler, so skip it
                return
            if not hasattr(self, name):
                setattr(self, name, [])
                self.extra_attributes.append(name)
            current_value = getattr(self, name)
            setattr(self, name, current_value + [value])

        if not os.path.isdir(folder_location):
            raise ValueError(f"Folder location {folder_location} is not a directory.")

        if metadata_fetchers is None:
            metadata_fetchers = []
        self.extra_attributes = []
        # Fetch data files
        all_files = []
        for pattern in ['*.csv', '*.txt', '*.hea']:
            all_files.extend([y for x in os.walk(folder_location) for y in glob(os.path.join(x[0], pattern))])
        all_files = [Path(f).as_posix() for f in all_files]
        data_files = copy.deepcopy(all_files)
        for regex_filter in regex_filters:
            data_files = regex_filter.get_matching_files(data_files)
        print(f"{len(data_files)} data files fetched out of {len(all_files)} files.")

        # Read data from files
        for file in data_files:
            if '.hea' in file:
                # The key is the emg key that is in the mrdf file
                file_data = (wfdb.rdrecord(file.replace('.hea',''))).__getattribute__(mrdf_key)
            else:
                file_data = np.genfromtxt(file,delimiter=delimiter, skip_header=skiprows)
                if len(file_data.shape) == 1:
                    # some devices may have one channel -> make sure it interprets it as a 2d array
                    file_data = np.expand_dims(file_data, 1)
            
            if downsampling_factor is not None:
                file_data = decimate(file_data, downsampling_factor, axis=0)

            if data_column is not None:
                # collect the data from the file
                self.data.append(file_data[:, data_column])
            else:
                self.data.append(file_data)

            # Fetch metadata from filename
            for regex_filter in regex_filters:
                metadata_idx = regex_filter.get_metadata(file)
                metadata = metadata_idx * np.ones((file_data.shape[0], 1), dtype=int)
                append_to_attribute(regex_filter.description, metadata)

            # Fetch remaining metadata
            for metadata_fetcher in metadata_fetchers:
                metadata = metadata_fetcher(file, file_data, all_files)
                if metadata.ndim == 0 or metadata.shape[0] == 1:
                    # Cast to array with the same # of samples as EMG data
                    metadata = np.full((file_data.shape[0], 1), fill_value=metadata)
                if metadata.ndim == 1:
                    # Ensure that output is always 2D array
                    metadata = np.expand_dims(metadata, axis=1)
                append_to_attribute(metadata_fetcher.description, metadata)
            
    def active_threshold(self, nm_windows, active_windows, active_labels, num_std=3, nm_label=0, silent=True):
        """Returns an update label list of the active labels for a ramp contraction.

        Parameters
        ----------
        nm_windows: list
            The no motion windows that are used to establish the threshold. 
        active_windows: list
            The active windows that should be thresholded. 
        active_labels: list
            The active window labels that need to be updated.
        num_std: int (default=3)
            The number of standard deviations away from the no motion class that are relabeled.
        nm_label: int
            The class label associated with the no motion class.
        silent: bool (default=True)
            If False, it will print out the number of active windows that were relabeled.
        """
        num_relabeled = 0
        fe = FeatureExtractor()

        # Get mean and STD of no motion
        nm_mavs = fe.extract_features(['MAV'], nm_windows)['MAV']
        nm_mean = np.mean(nm_mavs, axis=1)
        nm_mav_mean = np.mean(nm_mean)
        nm_mav_std = np.std(nm_mean)

        a_mavs = fe.extract_features(['MAV'], active_windows)['MAV']
        for i in range(0,len(a_mavs)):
            if np.mean(a_mavs[i]) < nm_mav_mean + num_std * nm_mav_std:
                active_labels[i] = nm_label
                num_relabeled += 1
        if not silent:
            print(f"{num_relabeled} of {len(active_labels)} active class windows were relabelled to no motion.")
        return active_labels
    
    def parse_windows(self, window_size, window_increment, metadata_operations=None):
        """Parses windows based on the acquired data from the get_data function.

        Parameters
        ----------
        window_size: int
            The number of samples in a window. 
        window_increment: int
            The number of samples that advances before next window.
        metadata_operations: dict or None (optional),default=None
            Specifies which operations should be performed on metadata attributes when performing windowing. By default,
            all metadata is stored as its mode in a window. To change this behaviour, specify the metadata attribute as the key and
            the operation as the value in the dictionary. The operation (value) should either be an accepted string (mean, median, last_sample) or
            a function handle that takes in an ndarray of size (window_size, ) and returns a single value to represent the metadata for that window. Passing in a string
            will map from that string to the specified operation. The windowing of only the attributes specified in this dictionary will be modified - all other
            attributes will default to the mode. If None, all attributes default to the mode. Defaults to None.
        
        Returns
        ----------
        list
            A np.ndarray of size windows x channels x samples.
        list
            A dictionary containing np.ndarrays for each metadata tag of the dataset. Each window will
            have an associated value for each metadata. Therefore, the dimensions of the metadata should be Wx1 for each field.
        """
        return self._parse_windows_helper(window_size, window_increment, metadata_operations)

    def _parse_windows_helper(self, window_size, window_increment, metadata_operations):
        common_metadata_operations = {
            'mean': np.mean,
            'median': np.median,
            'last_sample': lambda x: x[-1]
        }
        window_data = []
        metadata = {k: [] for k in self.extra_attributes}
        for i, file in enumerate(self.data):
            # emg data windowing
            window_data.append(get_windows(file,window_size,window_increment))
    
            for k in self.extra_attributes:
                if type(getattr(self,k)[i]) != np.ndarray:
                    file_metadata = np.ones((window_data[-1].shape[0])) * getattr(self, k)[i]
                else:
                    if metadata_operations is not None:
                        if k in metadata_operations.keys():
                            # do the specified operation
                            operation = metadata_operations[k]
                            
                            if isinstance(operation, str):
                                try:
                                    operation = common_metadata_operations[operation]
                                except KeyError as e:
                                    raise KeyError(f"Unexpected metadata operation string. Please pass in a function or an accepted string {tuple(common_metadata_operations.keys())}. Got: {operation}.")
                            file_metadata = _get_fn_windows(getattr(self,k)[i], window_size, window_increment, operation)
                        else:
                            file_metadata = _get_mode_windows(getattr(self,k)[i], window_size, window_increment)
                    else:
                        file_metadata = _get_mode_windows(getattr(self,k)[i], window_size, window_increment)
                
                metadata[k].append(file_metadata)
            
        return np.vstack(window_data), {k: np.concatenate(metadata[k], axis=0) for k in metadata.keys()}

    
    def isolate_channels(self, channels):
        """Entry point for isolating a certain range of channels. 

        Parameters
        ----------
        channels: list
            A list of values (i.e., channels) that you want to isolate. (e.g., [0,1,2]). Indexing starts at 0.
            
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
    
    def isolate_data(self, key, values, fast=True):
        """Entry point for isolating a single key of data within the offline data handler. First, error checking is performed within this method, then
        if it passes, the isolate_data_helper is called to make a new OfflineDataHandler that contains only that data.

        Parameters
        ----------
        key: str
            The metadata key that will be used to filter (e.g., "subject", "rep", "class", "set", whatever you'd like).
        values: list
            A list of values that you want to isolate. (e.g. [0,1,2,3]). Indexing starts at 0.
        fast: Boolean (default=False)
            If true, it iterates over the median value for each EMG element. This should be used when parsing on things like reps, subjects, classes, etc.
            
        Returns
        ----------
        OfflineDataHandler
            returns a new offline data handler with only the data that satisfies the requested slice.
        """
        assert key in self.extra_attributes
        assert type(values) == list 
        return self._isolate_data_helper(key,values,fast)

    def _isolate_data_helper(self, key, values,fast):
        new_odh = OfflineDataHandler()
        setattr(new_odh, "extra_attributes", self.extra_attributes)
        key_attr = getattr(self, key)
        for e in self.extra_attributes:
            setattr(new_odh, e, [])
                
        for f in range(len(key_attr)):
            if fast:
                if key_attr[f][0][0] in values:
                    keep_mask = [True] * len(key_attr[f])
                else:
                    keep_mask = [False] * len(key_attr[f])
            else:
                keep_mask = list([i in values for i in key_attr[f]])
            
            if self.data[f][keep_mask,:].shape[0]> 0:
                new_odh.data.append(self.data[f][keep_mask,:])
                for e in self.extra_attributes:
                    updated_arr = getattr(new_odh, e)
                    updated_arr.append(getattr(self, e)[f][keep_mask])
                    setattr(new_odh, e, updated_arr)

        return new_odh
    
    def visualize():
        pass


class OnlineDataHandler(DataHandler):
    """OnlineDataHandler class - responsible for collecting data streamed over shared memory.

    This class is extensible to any device as long as the data is being streamed over shared memory.
    By default this will start writing to an array of EMG data stored in memory.

    Parameters
    ----------
    shared_memory_items: Object
        The shared memory object returned from the streamer.
    channel_mask: list or None (optional), default=None
        Mask of active channels to use online. Allows certain channels to be ignored when streaming in real-time. If None, all channels are used.
        Defaults to None.
    """
    def __init__(self, shared_memory_items, channel_mask = None):
        self.shared_memory_items = shared_memory_items
        self.prepare_smm()
        self.log_signal = Event()
        self.visualize_signal = Event()        
        self.fi = None
        self.channel_mask = channel_mask
    
    def prepare_smm(self):
        self.modalities = []
        self.smm = SharedMemoryManager()
        for i in self.shared_memory_items:
            counter = 0
            while not self.smm.find_variable(*i):
                counter += 1
                time.sleep(0.5)
                if counter > 5:
                    print(f"Not finding key {i[0]} in shared memory...waiting.")
            if "_count" in i[0]:
                continue
            self.modalities.append(i[0])

    def stop_all(self):
        """Terminates the processes spawned by the ODH.
        """
        self.stop_log()
        self.stop_visualize()

    def stop_log(self):
        self.log_signal.set()
        time.sleep(0.5)
        self.log_signal.clear()

    def stop_visualize(self):
        self.visualize_signal.set()
        time.sleep(0.5)
        self.visualize_signal.clear()

    def install_filter(self, fi):
        """Install a filter to be used on the online stream of data.
        
        Parameters
        ----------
        fi: libemg.filter object
            The filter object that you'd like to run on the online data.
        """
        self.fi = fi

    def install_channel_mask(self, mask):
        """Install a channel mask to isolate certain channels for online streaming.

        Parameters
        ----------
        mask: list or None (optional), default=None
            Mask of active channels to use online. Allows certain channels to be ignored when streaming in real-time. If None, all channels are used.
            Defaults to None.
        """
        self.channel_mask = mask


    def analyze_hardware(self, analyze_time=10):
        """Analyzes several metrics from the hardware:
        (1) sampling rate
        (2) resolution
        (3) min val
        (4) max val
        (5) number of channels

        Parameters
        ----------
        analyze_time: int (optional), default=10 (seconds)
            The time in seconds that you want to analyze the device for. 
        """
        if not self._check_streaming():
            return

        self.reset()
        st = time.time()
        print("Starting analysis " + "(" + str(analyze_time) + "s)... We suggest that you elicit varying contractions and intensities to get an accurate analysis.")
        counters = {}
        data = {}
        for mod in self.modalities:
            counters[mod]=0
            data[mod]=[]
        while(time.time() - st < analyze_time):
            vals, count = self.get_data()
            for mod in self.modalities:
                num_new_samples = count[mod][0][0]-counters[mod]
                if num_new_samples > 0:
                    data[mod] = [vals[mod][:num_new_samples,:]] + data[mod]
                    counters[mod] += num_new_samples

        for key in data.keys():
            print('--------- ' + str(key) + ' ---------')
            t_data = np.vstack(data[key])
            print("Sampling Rate: " + str(self._get_sampling_rate(t_data,analyze_time)))
            print("Num Channels: " + str(self._get_num_channels(t_data)))
            print("Max Value: " + str(self._get_max_value(t_data)))
            print("Min Value: " + str(self._get_min_value(t_data)))
            print("Resolution: " + str(self._get_resolution(t_data)) + " bits")
        
        print("Analysis sucessfully complete. ODH process has stopped.")

    def visualize(self, num_samples=500, block=True):
        """Visualize the incoming raw EMG in a plot (all channels together).

        Parameters
        ----------
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        block: Boolean (optional), default=False
            Blocks the main thread if True.
        """
        if block:
            self._visualize(num_samples)
        else:
            p = Process(target=self._visualize, kwargs={"num_samples":num_samples}, daemon=True)
            p.start()

    def _visualize(self, num_samples):
        self.prepare_smm()

        pyplot.style.use('ggplot')
        plots = []
        fig, ax = pyplot.subplots(len(self.modalities), 1,squeeze=False)
        def on_close(event):
            self.visualize_signal.set()
        fig.canvas.mpl_connect('close_event', on_close)
        fig.suptitle('Raw Data', fontsize=16)
        for i,mod in enumerate(self.modalities):
            num_channels = self.smm.get_variable(mod).shape[1]
            for j in range(0,num_channels):
                plots.append(ax[i][0].plot([],[],label=mod+"_CH"+str(j+1)))
        
        fig.legend()
        
        def update(frame):
            data, _ = self.get_data(N=0,filter=True)
            line = 0
            for i, mod in enumerate(self.modalities):
                for j in range(data[mod].shape[1]):
                    data[mod][:,j] = data[mod][:,j] - np.mean(data[mod][:,j])
                inter_channel_amount = 1.5 * np.max(data[mod])
                if len(data[mod]) > num_samples:
                    data[mod] = data[mod][:num_samples,:]
                if len(data[mod]) > 0:
                    x_data = list(range(0,data[mod].shape[0]))
                    num_channels = data[mod].shape[1]
                    for j in range(0,num_channels):
                        y_data = data[mod][:,j]
                        plots[line][0].set_data(x_data, y_data +inter_channel_amount*j)
                        line += 1
            for i in range(len(self.modalities)):
                ax[i][0].relim()
                ax[i][0].autoscale_view()
                ax[i][0].set_title(self.modalities[i])
            return plots,
    
        while True:
            animation = FuncAnimation(fig, update, interval=100, repeat=False)
            pyplot.show()
            if self.visualize_signal.is_set():
                print("ODH->visualize ended.")
                break

    def visualize_channels(self, channels, num_samples=500, y_axes=None):
        """Visualize individual channels (each channel in its own plot).

        Parameters
        ----------
        channels: list
            A list of channels to graph indexing starts at 0.
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        y_axes: list (optional)
            A list of two elements consisting of the y-axes.
        """
        self.prepare_smm()
        pyplot.style.use('ggplot')
        while not self._check_streaming():
            pass
        emg_plots = []
        fig, ax = pyplot.subplots()
        fig.suptitle('Raw Data', fontsize=16)
        for i in range(0,len(channels)):
            emg_plots.append(ax.plot([],[],label="CH"+str(channels[i])))

        def update(frame):
            data, _ = self.get_data()
            data = data['emg']
            data = data[:,channels]
            inter_channel_amount = 1.5 * np.max(data)
            if len(data) > num_samples:
                data = data[:num_samples,:]
            if len(data) > 0:
                x_data = list(range(0,data.shape[0]))
            
                for i in range(data.shape[1]):
                    y_data = data[:,i]
                    emg_plots[i][0].set_data(x_data, y_data +inter_channel_amount*i)
                fig.gca().relim()
                fig.gca().autoscale_view()
            return emg_plots,

        animation = FuncAnimation(fig, update, interval=100)
        pyplot.show()
    
    def visualize_heatmap(self, num_samples = 500, feature_list = None, remap_function = None, cmap = None):
        """Visualize heatmap representation of EMG signals. This is commonly used to represent HD-EMG signals.

        Parameters
        ----------
        num_samples: int (optional), default=500
            The number of samples to average over (i.e., window size) when showing heatmap.
        feature_list: list or None (optional), default=None
            List of feature representations to extract, where each feature will be shown in a different subplot. 
            Compatible with all features in libemg.feature_extractor.get_feature_list() that return a single value per channel (e.g., MAV, RMS). 
            If a feature type that returns multiple values is passed, an error will be thrown. If None, defaults to MAV.
        remap_function: callable or None (optional), default=None
            Function pointer that remaps raw data to a format that can be represented by an image (such as np.reshape). Takes in an array and should return
            an array. If None, no remapping is done.
        cmap: colormap or None (optional), default=None
            matplotlib colormap used to plot heatmap.
        """
        # Create figure
        pyplot.style.use('ggplot')
        if not self._check_streaming():
            # Not reading any data
            return
        
        if feature_list is None:
            # Default to MAV
            feature_list = ['MAV']

        if cmap is None:
            cmap = cm.viridis   # colourmap to determine heatmap style
        
        def extract_data():
            data, _ = self.get_data()
            data = data['emg']
            if len(data) > num_samples:
                # Only look at the most recent num_samples samples (essentially extracting a single window)
                data = data[:num_samples]
            # Extract features along each channel
            windows = data[np.newaxis].transpose(0, 2, 1)   # add axis and tranpose to convert to (windows x channels x samples)
            fe = FeatureExtractor()
            feature_set_dict = fe.extract_features(feature_list, windows, array=False)
            assert isinstance(feature_set_dict, dict), f"Expected dictionary of features. Got: {type(feature_set_dict)}."
            if remap_function is not None:
                # Remap raw data to image format
                for key in feature_set_dict:
                    feature_set_dict[key] = remap_function(feature_set_dict[key]).squeeze() # squeeze to remove extra axis added for windows
            return feature_set_dict

        # Analyze data stream to determine min/max values for normalizing
        analyze_time = 5
        print(f"Analyzing data stream for {analyze_time} seconds to determine min/max values for each feature value. Please rest, then perform a contraction at max intensity.")
        start_time = time.time()
        normalization_values = {}
        while (time.time() - start_time) < analyze_time:
            features = extract_data()
            for feature, feature_data in features.items():
                if feature not in normalization_values.keys():
                    normalization_values[feature] = (feature_data.min(), feature_data.max())
                else:
                    old_min, old_max = normalization_values[feature]
                    current_min = min(old_min, feature_data.min())
                    current_max = max(old_max, feature_data.max())
                    normalization_values[feature] = (current_min, current_max)

        
        # Format figure
        sample_data = extract_data()    # access sample data to determine heatmap size
        fig, axs = plt.subplots(len(sample_data.keys()), 1)
        if isinstance(axs, Axes):
            axs = np.array([axs])
        fig.suptitle(f'HD-EMG Heatmap')
        plots = []
        for (feature_key, feature_data), ax in zip(sample_data.items(), axs):
            ax.set_title(f'{feature_key}')
            ax.set_xlabel('Electrode Row')
            ax.set_ylabel('Electrode Column')
            ax.grid(visible=False)  # disable grid
            ax.set_xticks(range(feature_data.shape[1]))
            ax.set_yticks(range(feature_data.shape[0]))
            im = ax.imshow(np.zeros(shape=feature_data.shape), cmap=cmap, animated=True)
            plt.colorbar(im)
            plots.append(im)
        plt.tight_layout()
            

        def update(frame):
            # Update function to produce live animation
            data = extract_data()
                
            if len(data) > 0:
                # Loop through feature plots
                for feature, plot in zip(data.items(), plots):
                    feature_key, feature_data = feature
                    feature_min, feature_max = normalization_values[feature_key]
                    # Normalize to properly display colours
                    normalized_data = (feature_data - feature_min) / (feature_max - feature_min)
                    # Convert to coloured map
                    heatmap_data = cmap(normalized_data)
                    plot.set_data(heatmap_data) # update plot
            return plots, 
        
        animation = FuncAnimation(fig, update, interval=100)
        pyplot.show()

    def visualize_feature_space(self, feature_dic, window_size, window_increment, sampling_rate, hold_samples=20, projection="PCA", classes=None, class_labels=None, normalize=True):
        """Visualize a live pca plot. This is reliant on previously collected training data.

        Parameters
        ----------
        feature_dic: dict
            A dictionary consisting of the different features acquired through screen guided training. This is the output from the 
            extract_features method.
        window_size: int
            The number of samples in a window. 
        window_increment: int
            The number of samples that advances before next window.
        sampling_rate: int
            The sampling rate of the device. This impacts the refresh rate of the plot. 
        hold_samples: int (optional), default=20
            The number of live samples that are shown on the plot.
        projection: string (optional), default=PCA
            The projection method. Currently, the only available option, is PCA.
        classes: list
            A list of classes that is associated with each feature index.
        class_labels: list
            A list of class labels for the legend. 
        normalize: boolean
            Whether the user wants to scale features to zero mean and unit standard deviation before projection (recommended).
        """
        from sklearn.decomposition import PCA
        pyplot.style.use('ggplot')
        feature_list = feature_dic.keys()
        fe = FeatureExtractor()

        if projection == "PCA":
            for i, k in enumerate(feature_dic.keys()):
                feature_matrix = feature_dic[k] if i == 0 else np.hstack((feature_matrix, feature_dic[k]))

            if normalize:
                feature_means = np.mean(feature_matrix, axis=0)
                feature_stds  = np.std(feature_matrix, axis=0)
                feature_matrix = (feature_matrix - feature_means) / feature_stds

            fig, ax = plt.subplots()
            pca = PCA(n_components=feature_matrix.shape[1]) 

            if classes is not None:
                class_list = np.unique(classes)
    
            train_data = pca.fit_transform(feature_matrix)
            if classes is not None:
                for c in class_list:
                    class_ids = classes == c
                    c_label = "tr "+str(int(c)),
                    if class_labels is not None:
                        c_label = class_labels[c]
                    ax.plot(train_data[class_ids,0], train_data[class_ids,1], marker='.', alpha=0.75, label=c_label, linestyle="None")
            else:
                ax.plot(train_data[:,0], train_data[:,1], marker=".", label="tr", linestyle="None")
            
            graph = ax.plot(0, 0, marker='+', color='black', alpha=0.75, label="new_data", linestyle="None")

            fig.legend()
            self.reset()

            pc1 = [] 
            pc2 = []      

            def update(frame):
                data, counts = self.get_data(N=window_size)
                if counts['emg'] > window_size:
                    data = data['emg'][::-1]
                    window = get_windows(data, window_size, window_size)
                    features = fe.extract_features(feature_list, window)
                    for i, k in enumerate(features.keys()):
                        formatted_data = features[k] if i == 0 else np.hstack((formatted_data, features[k]))
                    
                    if normalize:
                        formatted_data = (formatted_data-feature_means)/feature_stds

                    data = pca.transform(formatted_data)
                    pc1.append(data[0,0])
                    pc2.append(data[0,1])

                    pc1_data = pc1[-hold_samples:]
                    pc2_data = pc2[-hold_samples:]
                    graph[0].set_data(pc1_data, pc2_data)

                    ax.relim()
                    ax.autoscale_view()

            animation = FuncAnimation(fig, update, interval=(1000/sampling_rate * window_increment))
            plt.show()

    def get_data(self, N=0, filter=True):
        """Grab the data in the shared memory buffer across all modalities.
 
        Parameters
        ----------
        N : int
            Number of samples to grab from the shared memory items. If zero, grabs all data.
        filter: bool
            Apply the installed filters to the data prior to returning or not.
 
        Returns
        ----------
        val: dict
            A dictionary with keys corresponding to the modalities. Each key will have a np.ndarray of data returned.
        count: dict
            A dictionary with keys corresponding to the modalities. Each key will have an int corresponding to the number
            of samples received since the streamer began (or the last reset call).
        """
        val   = {}
        count = {}
        for mod in self.modalities:
            data = self.smm.get_variable(mod)
            if filter:
                if self.fi is not None:
                    if mod == "emg": # TODO: enable filter for each modality
                        data = self.fi.filter(data)
            if N != 0:
                val[mod]   = data[:N,:]
            else:
                val[mod]   = data[:,:]
            if self.channel_mask is not None:
                val[mod] = val[mod][:, self.channel_mask]
            count[mod] = self.smm.get_variable(mod+"_count")
        return val,count

    def reset(self, modality=None):
        """Reset the data within the shared memory buffer.
 
        Parameters
        ----------
        modality: str
            The modality that should be reset. If None, all modalities are reset.
        """
        if modality == None:
            modality = self.modalities
        else:
            modality = [modality]
        for mod in modality:
            self.smm.modify_variable(mod, lambda x: np.zeros_like(x))
            self.smm.modify_variable(mod+"_count", lambda x: np.zeros_like(x))

    def log_to_file(self, block=False, file_path='', timestamps=True):
        """Logs the raw data being read to a file.

        Parameters
        ----------
        block: bool (optional), default=False 
            If true, the main thread will be blocked. 
        file_path: int (optional), default=''
            The prefix to the file path that will be logged for each modality.
        timestamps: bool (optional), default=True
            If true, this will log the timestamps with each recording.
        """
        print("ODH->log_to_file begin.")
        self.file_path = file_path
        self.timestamps = timestamps
        if block:
            self._log_to_file()
            print("ODH->log_to_file ended.")
        else:
            p = Process(target=self._log_to_file, daemon=True)
            p.start()

    def _log_to_file(self):

        files = {}
        # start shared memory manager to access sensor
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.find_variable(*item)
        # initialize sample count for all modalities
        last_count = {}
        for m in self.modalities:
            last_count[m] = 0
        while True:
            timestamp = time.time()
            vals, counts = self.get_data(N=0, filter=False)
            for m in vals.keys():
                new_count       = counts[m][0,0]
                num_new_samples = new_count - last_count[m]
                new_samples     = vals[m][:num_new_samples,:]
                last_count[m] = new_count
                if num_new_samples:
                    if not m in files.keys():
                        files[m] = open(self.file_path + m + '.csv', "a", newline='')
                    if self.timestamps:
                        np.savetxt(files[m], np.hstack((np.ones((new_samples.shape[0],1))*timestamp, new_samples)))
                        # check to see if they're in the right order, or if they need to be reversed again!
                    else:
                        np.savetxt(files[m], new_samples)
            if self.log_signal.is_set():
                print("ODH->log_to_file ended.")
                break

    def _check_streaming(self, timeout=15):
        wt = time.time()
        emg_count = self.smm.get_variable("emg_count")
        while(True):
            emg_count2 = self.smm.get_variable("emg_count")
            if emg_count != emg_count2: 
                return True
            if time.time() - wt > timeout:
                print("Not reading any data.... Check hardware connection.")
                return False
            
    def start_listening(self):
        print("LibEMG>v1.0 no longer requires online_data_handler.start_listening().\nThis is deprecated.")
        pass