#
# Author: Bárbara Marinho Muniz
# Date: 2026-10-05
#

import pandas as pd
import numpy as np
from .read import read
import os
from .paths import get_session_files
from .emg_processor import EMGProcessor 
from scipy.stats import mode

emg = EMGProcessor()

class Development():
    """
    def get_labels(self,reps, n_samples):
        labels = np.full(n_samples, -1)

        for cls in range(9):
            for start, end in reps[cls]:
                labels[start:end] = cls

        return labels
    """

    def combine_reps(self,rep1,rep2,offset):
        new_rep = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: []
        }
        for key in rep1:
            for start,end in rep1[key]:
                new_rep[key].append((start,end))
            for start,end in rep2[key]:
                new_rep[key].append((start+offset,end+offset))
        return new_rep

    def analyze(self,sets):
        if sets is None:
            return
        
        (set1, labels1), (set2, labels2), (set3, labels3) = read(sets)

        feats1, win_labels1 = emg.processor(set1, labels1)
        feats2, win_labels2 = emg.processor(set2, labels2)
        feats3, win_labels3 = emg.processor(set3, labels3)
        
        reps1 = emg.get_rep_indices(win_labels1) # reps indices for set1 given session x
        reps2 = emg.get_rep_indices(win_labels2) # reps indices for set2 given session x
        reps3 = emg.get_rep_indices(win_labels3) # reps indices for set3 given session x
        
        return ([feats1,feats2], [feats3], [reps1,reps2], [reps3], 5)

    def main(self):
        folder = r"tests\data\1\Session 02"
        if os.path.isdir(folder):
            sets = get_session_files(folder)
            features_train, features_test, reps_train, reps_test, classes = self.analyze(sets)
            return (features_train, features_test, reps_train, reps_test, classes)
        else:
            print("Invalid folder")
