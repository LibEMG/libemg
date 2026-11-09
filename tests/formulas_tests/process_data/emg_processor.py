import numpy as np
from libemg.filtering import Filter
from libemg.utils import get_windows, _get_mode_windows
from libemg.feature_extractor import FeatureExtractor

class EMGProcessor:
    def filtering(self, signal):
        # (1) Filter the EMG data 
        filter = Filter(2000) # fs = 2000 Hz
        #filter_dictionary = {"name": "notch","cutoff":60,"bandwidth":3} # 60 Hz
        filter.install_filters({"name": "notch","cutoff":60,"bandwidth":3})
        #filter_dictionary = {"name":"highpass","cutoff":20,"order":4} # 20 Hz
        filter.install_filters({"name":"highpass","cutoff":20,"order":4})
        #filter_dictionary = {"name":"lowpass","cutoff":450,"order":4} # 450 Hz
        filter.install_filters({"name":"lowpass","cutoff":450,"order":4})
        filtered_emg = filter.filter(signal)

        return filtered_emg

    def get_windows(self, signal, labels):
        # (2) Get Windows 
        window_size = 320 
        window_inc = 192 
        windows = get_windows(signal, window_size, window_inc)
        window_labels = _get_mode_windows(labels, window_size, window_inc)

        return windows, window_labels

    def features(self, windows):
        # (3) Feature extractor Demo 
        fe = FeatureExtractor() 
        feats = fe.extract_features(['MAV', 'ZC', 'SSC', 'WL'], windows)
        
        # combine features → matrix
        combined = []
        
        for value in feats.values():
            value = np.array(value)

            if value.ndim == 1:
                combined.append(value.reshape(-1, 1))
            else:
                combined.append(value)

        return np.hstack(combined)

    def processor(self, signal, labels):
        filtered = self.filtering(signal)

        windows, win_labels = self.get_windows(filtered, labels)

        feats = self.features(windows)

        return feats, win_labels

    def get_rep_indices(self, window_labels):
        reps = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: []
        }
        i = 0
        while (i < len(window_labels)):
            curr = window_labels[i]
            start = i
            while (i< len(window_labels) and window_labels[i] == curr):
                i += 1
            end = i
            reps[int(curr)].append((start,end))
        return reps