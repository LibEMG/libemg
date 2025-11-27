import numpy as np
import torch.nn.functional as F
import torch 
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows
import pyautogui
import time 
import statistics
import json 

class DiscreteControl:
    """
    The temporary discrete control class for interfacing the cross-user Myo model made available at: <insert git repo here>.
    The model currently supports 5 gestures: Close, Flexion, Extension, Open, Pinch. 
    These gestures can be mapped to keyboard keys for controlling applications.

    Parameters
    ----------
    odh: OnlineDataHandler
        The online data handler object for streaming EMG data.
    window_size: int
        The window size (in samples) to use for splitting up each template.
    increment: int
        The increment size (in samples) for the sliding window. 
    model: torch.nn.Module
        The trained PyTorch model for gesture classification.
    buffer: int, optional
        The size of the prediction buffer to use for mode filtering. Default is 1.
    key_mapping: dict, optional
        A dictionary mapping gesture names to keyboard keys. Default maps 'Close' to 'c', 'Flexion' to 'f', 'Extension' to 'e', 'Open' to 'o', and 'Pinch' to 'p'.
    debug: bool, optional
        If True, enables debug mode with additional print statements. Default is True.
    """
    def __init__(self, odh, window_size, increment, save_folder='data/', config='config.json', buffer=5, key_mapping={'Close':'c', 'Flexion':'f', 'Extension':'e', 'Open':'o', 'Pinch':'p'}, debug=True):
        self.odh = odh
        self.window_size = window_size
        self.increment = increment
        self.buffer_size = buffer
        self.model = None 
        self.template_size = None
        self.min_template_size = None 
        self.key_mapping = key_mapping
        self.debug = debug
        self.save_folder = save_folder
        self.config = config 
        self.last_model_path = None 
        self.load_config()

    def load_config(self):
        data = json.load(open(self.config, 'r'))
        self.min_template_size = data['min_template_size']
        self.template_size = data['template_size']
        model_path = data['model_path']
        if model_path != self.last_model_path:
            device = torch.device('cpu')
            self.model = torch.load(model_path, map_location=device)
            self.model.to(device)
            self.model.eval()
            self.last_model_path = model_path
    
    def run(self):
        """
        Main loop for gesture detection.
        Runs a sliding window over incoming EMG data and makes predictions based on the trained model.
        """
        gesture_mapping = ['Nothing', 'Close', 'Flexion', 'Extension', 'Open', 'Pinch']
        expected_count = self.min_template_size
        buffer = []
        time_since_load = time.time()
        nm_data = []

        while True:
            # Reload config every 10 seconds
            if time.time() - time_since_load > 10:
                self.load_config()
                time_since_load = time.time()
                
            # Get and process EMG data
            _, counts = self.odh.get_data(self.window_size)
            if counts['emg'][0][0] >= expected_count:
                data, counts = self.odh.get_data(min([self.template_size, counts['emg'][0][0]]))
                emg = data['emg'][::-1]
                feats = self._get_features([emg], self.window_size, self.increment, None, None)
                pred, _ = self._predict(feats[0])
                buffer.append(pred)
                mode_pred = statistics.mode(buffer[-self.buffer_size:])
                if mode_pred != 0: 
                    if self.debug:
                        print(str(time.time()) + ' ' + gesture_mapping[mode_pred])
                    self._key_press(mode_pred, gesture_mapping)
                    
                    # Save datga 
                    np.save(self.save_folder + str(mode_pred) + '_' + str(int(time.time())) + '.npy', emg)
                    if len(nm_data) > 0:
                        np.save(self.save_folder + str(0) + '_' + str(int(time.time())) + '.npy', nm_data[0])
                    
                    nm_data = []
                    self.odh.reset()
                    expected_count = self.min_template_size
                    buffer = []
                else:
                    nm_data.append(emg)
                    expected_count += 10

    def _key_press(self, pred, mapping):
        if mapping[pred] in self.key_mapping:
            pyautogui.press(self.key_mapping[mapping[pred]])
    
    def _predict(self, gest, device='cpu'):
        g_tensor = torch.tensor(np.expand_dims(np.array(gest, dtype=np.float32), axis=0), dtype=torch.float32).to(device)
        with torch.no_grad(): 
            output = self.model.forward_once(g_tensor)
            pred = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1).max().item()
        return pred, prob
    
    def _get_features(self, data, window_size, window_inc, feats, feat_dic):
        fe = FeatureExtractor()
        data = np.array([get_windows(d, window_size, window_inc) for d in data], dtype='object')
        if feats is None:
            return data 
        if feat_dic is not None:
            feats = np.array([fe.extract_features(feats, d, array=True, feature_dic=feat_dic) for d in data], dtype='object')
        else:
            feats = np.array([fe.extract_features(feats, np.array(d, dtype='float'), array=True) for d in data], dtype='object')
        feats = np.nan_to_num(feats, copy=True, nan=0, posinf=0, neginf=0)
        return feats