import numpy as np
import torch.nn.functional as F
import torch 
from playsound import playsound
import matplotlib.pyplot as plt
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows
import pyautogui
import time 
import statistics

class DiscreteControl:
    """
    A class for detecting gestures using the Teager-Kaiser energy operator on EMG signals.
    """
    
    def __init__(self, odh, window_size, increment, threshold=100, buffer=20, subject=None, model=None):
        self.odh = odh
        self.window_size = window_size
        self.increment = increment
        self.threshold = threshold
        self.buffer_size = buffer
        self.subject = subject
        self.model = model 
    
    def run(self):
        """
        Main loop for gesture detection.
        Continuously monitors EMG data and detects gestures based on energy thresholds.
        """
        gesture_mapping = ['Nothing', 'Close', 'Flexion', 'Extension', 'Open', 'Pinch']
        expected_count = 250

        while True:
            buffer = []

            # Get and process EMG data
            data, counts = self.odh.get_data(self.window_size)
            if counts['emg'][0][0] >= expected_count:
                data, counts = self.odh.get_data(250)
                emg = data['emg'][::-1]
                feats = self.get_features([emg], 10, 5, None, None)
                pred, _ = self.predict(feats[0])
                buffer.append(pred)
                mode_pred = statistics.mode(buffer[-20:])
                if mode_pred != 0: 
                    print(str(time.time()) + ' ' + gesture_mapping[mode_pred])
                    self.key_press(mode_pred, gesture_mapping)
                    self.odh.reset()
                    expected_count = 250
                    buffer = []
                else:
                    expected_count += 10

    def key_press(self, pred, mapping):
        if mapping[pred] == 'Close':
            pyautogui.press('c')
        elif mapping[pred] == 'Flexion':
            pyautogui.press('f')
        elif mapping[pred] == 'Extension':
            pyautogui.press('e')
        elif mapping[pred] == 'Open':
            pyautogui.press('o')
        elif mapping[pred] == 'Pinch':
            pyautogui.press('p')
        playsound('Other/click.wav')
    
    def predict(self, gest, device='cpu'):
        g_tensor = torch.tensor([gest], dtype=torch.float32).to(device)
        with torch.no_grad(): 
            output = self.model.forward_once(g_tensor)
            pred = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1).max().item()
        return pred, prob
    
    def get_features(self, data, window_size, window_inc, feats, feat_dic):
        fe = FeatureExtractor()
        data = np.array([get_windows(d, window_size, window_inc) for d in data], dtype='object')
        if feats is None:
            return data 
        if feat_dic is not None:
            feats = np.array([fe.extract_features(feats, d, array=True, feature_dic=feat_dic) for d in data], dtype='object')
        else:
            feats = np.array([fe.extract_features(feats, np.array(d, dtype='float'), array=True) for d in data], dtype='object')
        feats = np.nan_to_num(feats, copy=True, nan=0, posinf=0, neginf=0)
        # expected shape: (NFiles,) -> (Time, channel)
        return feats