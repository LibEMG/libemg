import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from unb_emg_toolbox.data_handler import OfflineDataHandler, OnlineDataHandler

class Filter:
    """ A class that will perform filtering on: (1) OfflineDataHandler, (2) OnlineDataHandler, 
    and (3) RawData in numpy.ndarrays.

    Parameters
    ----------
    sampling_frequency: int
        The sampling frequency of the device used. This must be known for the 
        digital filters to do what is intended.
    """
    def __init__(self, sampling_frequency):
        self.sampling_frequency = sampling_frequency

    def install_filters(self, filter_dictionary={}):
        '''Install a particular filter.

        Installing filters is required prior to filtering being performed. Filters are created using the scipy.signal package.
        The necessary parameters for these fitlers are included in a dictionary. When multiple filters are intending to be used
        at a time, install them sequentially by calling this function for each filter. If there is an intended order for the filters
        then install them in the intended processing order.

        Parameters
        ----------
        filter_dictionary: dictionary
            A dictionary containing the necessary parameters for defining a single filter.
        
        Examples
        -------
        >>> # create a notch filter for removing power line interference
        >>> filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
        >>> # create a filter for removing high frequency noise and low frequency motion artefacts
        >>> filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
        >>> # create a filter for low frequency motion artefacts
        >>> filter_dictionary={ "name": "highpass", "cutoff": 20, "order":2}
        '''
        installed_filter = {"name":filter_dictionary["name"]}
        if filter_dictionary["name"] == "notch":
            assert filter_dictionary["cutoff"]/(self.sampling_frequency/2) < 1 # cutoff given too high for nyquist rate
            installed_filter["b"], installed_filter["a"] = scipy.signal.iirnotch(w0=filter_dictionary["cutoff"],
                                                                                 Q=filter_dictionary["cutoff"]/filter_dictionary["bandwidth"],
                                                                                 fs=self.sampling_frequency)
        elif filter_dictionary["name"] in ["lowpass","highpass","bandpass","bandstop"]:
            # normalize cutoff by nyquist rate (sampling frequency/2)
            if type(filter_dictionary["cutoff"])==list:
                cutoff = [i/(self.sampling_frequency/2) for i in filter_dictionary["cutoff"]]
                assert sum([i > 1 for i in cutoff]) == 0 # cutoff given too high for nyquist rate
            else:
                cutoff = filter_dictionary["cutoff"]/(self.sampling_frequency/2)
                assert cutoff < 1 # cutoff given too high for nyquist rate
            
            installed_filter["b"], installed_filter["a"] = scipy.signal.butter(N=filter_dictionary["order"],
                                                                               Wn=cutoff,
                                                                               btype=filter_dictionary["name"])
        elif filter_dictionary["name"] == "standardize":
            assert "data" in list(filter_dictionary.keys())
            installed_filter["mean"], installed_filter["std"] = self._get_standardization_params(filter_dictionary["data"])

        if hasattr(self, "filters"):
            setattr(self, "filters", getattr(self, "filters")+[(installed_filter)])
        else:
            setattr(self, "filters", [installed_filter])


    def filter(self, data):
        ''' Run installed filters on data.

        Parameters
        ----------
        data: array_like    
            The data that will be passed through the filters.

        Returns
        ------- 
        array_like
            Returns the filtered data.
        '''
        if not hasattr(self, "filters"):
            print("No filters have been installed")
            return data

        if type(data) == OfflineDataHandler:
            # make sure we have data to filter
            self._filter_offline_data_handler(data)
        elif type(data) == OnlineDataHandler:
            return self._filter_online_data_handler(data)
        elif type(data) == np.ndarray:
            return self._filter_np_ndarray(data)
        # should we also accomodate the RawData class?
        else:
            print("An unsupported data type was passed into the function")

    def _filter_offline_data_handler(self, data):
        ''' Helper function that runs the installed filters across the various files contained in the offline data handler.
        The data contained in the OfflineDataHandler is updated to reflect the filtered data.

        Paramaters
        ----------
        data: OfflineDataHandler    
            The data that will be passed through the filters.
        '''
        assert hasattr(data,"data")
        for f in range(len(data.data)):
            data.data[f] = self._run_filter(data.data[f])


    def _filter_online_data_handler(self, data):
        ''' Helper function that runs the installed filters on recently captured data in the online data handler.

        Paramaters
        ----------
        data: OnlineDataHandler    
            The data that will be passed through the filters.

        Returns
        ------- 
        np.ndarray
            Data that has been filtered.
        '''
        # this just gets the data in the tcpip port. it isnt the full window
        # I advise using the np_ndarray with a full window (see demo/filter_example.py -> online_filtering_demo())
        emg_data = np.array(data.raw_data.get_emg())
        assert emg_data.shape[0] > 5
        return self._run_filter(emg_data)

    def _filter_np_ndarray(self, data):
        ''' Helper function that runs the installed filters on an np.ndarray.

        Paramaters
        ----------
        data: np.ndarray    
            The data that will be passed through the filters.

        Returns
        ------- 
        np.ndarray
            Data that has been filtered.
        '''
        return self._run_filter(data)

    def _run_filter(self, matrix):
        ''' Helper function that actually runs the installed filters on an np.ndarray. This is where the actual processing happens.

        Paramaters
        ----------
        matrix: np.ndarray    
            The data that will be passed through the filters.

        Returns
        ------- 
        matrix: np.ndarray
            Data that has been filtered.
        '''
        for fl in range(len(self.filters)):
            if self.filters[fl]["name"] == "standardize":
                matrix = (matrix - self.filters[fl]["mean"]) / self.filters[fl]["std"]
            elif self.filters[fl]["name"] in ["lowpass","highpass","bandpass","bandstop","notch"]:
                matrix = scipy.signal.filtfilt(self.filters[fl]["b"],
                                            self.filters[fl]["a"],
                                            matrix,
                                            axis=0)
        return matrix

    def _get_standardization_params(self, odh):
        ''' Helper function that computes the mean and standard deviation of data contained in an OfflineDataHandler.

        Paramaters
        ----------
        odh: OfflineDataHandler   
            The data that parameters will be computed from.

        Returns
        ------- 
        mean: np.ndarray
            channel-wise means.
        std:  np.ndarray
            channel-wise standard deviations.
        '''
        data = np.concatenate(odh.data)
        filter_mean = np.mean(data,axis=0)
        filter_std  = np.std(data, axis=0)
        assert (filter_std != 0).any()
        return filter_mean, filter_std

    def visualize_filters(self):
        '''Visualizes the bode plot of the installed filters.
        '''
        fig, ax = plt.subplots(len(self.filters), 2, figsize=(10, 5*len(self.filters)))
        for fl in range(len(self.filters)):
            if self.filters[fl]["name"] == "standardize":
                continue# no visualization of standardize filter
            filter_name = self.filters[fl]["name"]
            freq, h = scipy.signal.freqz(self.filters[fl]["b"],
                                         self.filters[fl]["a"],
                                         fs=self.sampling_frequency)
            # Plot
            ax[fl,0].plot(freq, 20*np.log10(abs(h)), color='blue')
            ax[fl,0].set_title(f"F#{fl}: { filter_name} Magnitude Response")
            ax[fl,0].set_ylabel("Amplitude (dB)", color='blue')
            ax[fl,0].set_xlim([0, 100])
            ax[fl,0].set_ylim([-25, 10])
            ax[fl,0].grid(True)
            ax[fl,1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
            ax[fl,1].set_title(f"F#{fl}: {filter_name} Phase Response")
            ax[fl,1].set_ylabel("Angle (degrees)", color='green')
            ax[fl,1].set_xlabel("Frequency (Hz)")
            ax[fl,1].set_xlim([0, 100])
            ax[fl,1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
            ax[fl,1].set_ylim([-90, 90])
            ax[fl,1].grid(True)
        plt.show()
    
    def visualize():
        pass
            