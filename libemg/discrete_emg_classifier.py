from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
import pyautogui
import numpy as np
import torch
from playsound import playsound
from libemg.utils import get_windows
 
class OnlineEMGDiscreteClassifier:
    """OnlineEMGClassifier.
 
    Given a DiscreteEMGClassifier and additional information, this class will stream class predictions over UDP in real-time.
 
    Parameters
    ----------
    offline_classifier: EMGClassifier
        An EMGClassifier object.
    port: int (optional), default = 12346
        The port used for streaming predictions over UDP.
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    """
    def __init__(self, model, online_data_handler, port=12346, ip='127.0.0.1'):
        self.port = port
        self.ip = ip
        self.model = model
        self.raw_data = online_data_handler.raw_data
        self.process = Process(target=self._run_helper, daemon=True,)
        self.feats = []
        
    def run(self, block=True):
        """Runs the classifier - continuously streams predictions over UDP.
 
        Parameters
        ----------
        block: bool (optional), default = True
            If True, the run function blocks the main thread. Otherwise it runs in a
            seperate process.
        """
        if block:
            self._run_helper()
        else:
            self.process.start()
 
    def stop_running(self):
        """Kills the process streaming classification decisions.
        """
        self.process.terminate()
 
    def _run_helper(self):
        classes = ['NOTHING', 'CLOSE', 'FLEXION', 'EXTENSION', 'OPEN', 'PINCH']
        keys = [None, 'C', 'F', 'E', 'O', 'P']
        print("Running Classifier")
        fe = FeatureExtractor()
        self.raw_data.reset_emg()
        while True:
            # Lets base everything off of EMG
            if len(self.raw_data.get_emg()) >= 260:
                emg_data = np.array([self.raw_data.get_emg()[-260:]])
                features = self.get_features(fe, emg_data, 10, 5, ['WENG'], {'WENG_fs': 200})
                pred = self.model['mlp'].predict(self.model['fe'].forward_once(torch.tensor(features, dtype=torch.float32)))[0]
                if pred != 0:
                    self.raw_data.adjust_increment(260, 260)
                    pyautogui.press(keys[pred])
                    playsound('Other/connect.mp3')
                self.raw_data.adjust_increment(260, 10)
 
    def _get_data_helper(self):
        data = np.array(self.raw_data.get_emg())
        if self.filters is not None:
            try:
                data = self.filters.filter(data)
            except:
                pass
        return data
    
    
    def get_features(self, fe, data, window_size, window_inc, feats, feat_dic={}):
        data = np.array([get_windows(d, window_size, window_inc) for d in data])
        feats = np.array([fe.extract_features(feats, d, array=True, feature_dic=feat_dic) for d in data])
        feats = np.nan_to_num(feats, copy=True, nan=0, posinf=0, neginf=0)
        return feats