from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
import numpy as np
import time
import pyautogui
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
    def __init__(self, offline_classifiers, online_data_handler, keys, port=12346, ip='127.0.0.1'):
        self.port = port
        self.ip = ip
        self.keys = keys
        self.classifiers = offline_classifiers
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
        from playsound import playsound
        print("Running Classifier")
        fe = FeatureExtractor()
        self.raw_data.reset_emg()
        from collections import deque
        queues = [deque(maxlen=5) for _ in range(0, len(self.classifiers))]
        prob_queues = [deque(maxlen=5) for _ in range(0, len(self.classifiers))]
        while True:
            # Lets base everything off of EMG
            if len(self.raw_data.get_emg()) >= 260:
                emg_data = np.array([self.raw_data.get_emg()[-260:]])
                features = self.get_features(fe, emg_data, 5, 5, ['RMS'])
                for i, c in enumerate(self.classifiers):
                    preds, probs = c.predict(features)
                    queues[i].append(preds[0])
                    prob_queues[i].append(max(probs[0]))
                    if len(list(queues[i])) == 5 and sum(list(queues[i])) == 0 and np.mean(list(prob_queues[i])) >= self.raw_data.get_rejection_thresholds()[i]:
                        self.raw_data.adjust_increment(260, 260)
                        queues = [deque(maxlen=5) for _ in range(0, len(self.classifiers))]
                        prob_queues = [deque(maxlen=5) for _ in range(0, len(self.classifiers))]
                        if self.keys[i] is not None:
                            pyautogui.press(self.keys[i])
                        playsound('Other/connect.mp3')
                    self.raw_data.adjust_increment(260, 5)
 
    def _get_data_helper(self):
        data = np.array(self.raw_data.get_emg())
        if self.filters is not None:
            try:
                data = self.filters.filter(data)
            except:
                pass
        return data
    
    
    def get_features(self, fe, data, window_size, window_inc, feats):
        data = np.array([get_windows(d, window_size, window_inc) for d in data])
        feats = np.array([fe.extract_features(feats, d, array=True) for d in data])
        feats = np.nan_to_num(feats, copy=True, nan=0, posinf=0, neginf=0)
        return feats
    