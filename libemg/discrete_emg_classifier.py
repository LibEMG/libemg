from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
import numpy as np
import time
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
    def __init__(self, offline_classifier, online_data_handler, port=12346, ip='127.0.0.1'):
        self.port = port
        self.ip = ip
        self.classifier = offline_classifier
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
        print("Running Classifier")
        gesture_mapping = ['No Gesture', 'Fist', 'Wave In', 'Wave Out', 'Open', 'Double Tap']
        fe = FeatureExtractor()
        self.raw_data.reset_emg()
        while True:
            # Lets base everything off of EMG 
            if len(self.raw_data.get_emg()) >= 40:
                emg_data = np.array([self.raw_data.get_emg()[-40:]])
                window = get_windows(emg_data[-40:][:], 40, 40)
                features = fe.extract_features(['WENG'], window, feature_dic={'WENG_fs': 200}, array=True)
                preds = self.classifier.predict([features])
                self.raw_data.adjust_increment(40, 20)
                print(gesture_mapping[preds[0]])

            # if len(self.raw_data.get_emg()) >= 40: # Defaulting to 300 samples for the myo armband
            #     emg_data = np.array([self.raw_data.get_emg()[-200:]])

            #     self.raw_data.adjust_increment(200, 10)
            #     emg_feats = self.get_features(fe, emg_data, 5, 5, ['RMS'])
                
            #     preds = self.classifier.predict(emg_feats, [-1])
            #     probs = self.classifier.predict(emg_feats, [-1], prob=True) #, imu=imu_feats, ppg=ppg_feats)
            #     if probs >= 0.99:
            #         print(gesture_mapping[preds[0]])
            #     else:
            #         print('Rejected!')

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
    