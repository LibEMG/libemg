from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler
import numpy as np
import pandas as pd
import h5py

class EMG2POSE(Dataset):
    def __init__(self, dataset_folder="Meta/emg2pose_data/"):
        self.mapping = {'FingerPinches1': 'AllFingerPinchesThumbSwipeThumbRotate', 'Object1': 'CoffeePanicPete', 'Counting1': 'CountingUpDownFaceSideAway', 'Counting2': 'CountingUpDownFingerWigglingSpreading', 'DoorknobFingerGraspFistGrab': 'DoorknobFingerGraspFistGrab', 'Throwing': 'FastPongFronthandBackhandThrowing', 'Abduction': 'FingerAbductionSeries', 'FingerFreeform': 'FingerFreeform', 'FingerPinches2': 'FingerPinchesSingleFingerPinchesMultiple', 'HandHandInteractions': 'FingerTouchPalmClapmrburns', 'Wiggling1': 'FingerWigglingSpreading', 'Punch': 'GraspPunchCloseFar', 'Gesture1': 'HandClawGraspFlicks', 'StaticHands': 'HandDeskSeparateClaspedChest', 'FingerPinches3': 'HandOverHandAllFingerPinchesThumbSwipeThumbRotate', 'Wiggling2': 'HandOverHandCountingUpDownFingerWigglingSpreading', 'Unconstrained': 'unconstrained', 'Gesture2': 'HookEmHornsOKScissors', 'FingerPinches4': 'IndexPinchesMiddlePinchesThumbswipes', 'Pointing': 'IndividualFingerPointingSnap', 'Freestyle1': 'OneHandedFreeStyle', 'Object2': 'PlayBlocksChess', 'Draw': 'PokeDrawPinchRotateclosefar', 'Poke': 'PokePinchCloseFar', 'Gesture3': 'ShakaVulcanPeace', 'ThumbsSwipes': 'ThumbsSwipesWholeHand', 'ThumbRotations': 'ThumbsUpDownThumbRotationsCWCCWP', 'Freestyle2': 'TwoHandedFreeStyle', 'WristFlex': 'WristFlexionAbduction'}

        Dataset.__init__(self, 
                        2000, 
                        32, 
                        'Ctrl Labs Armband', 
                        193, 
                        self.mapping, 
                        'TODO',
                        "A large dataset from ctrl-labs (Meta) for joint angle estimation. Note that not all subjects have all stages.",
                        "https://openreview.net/forum?id=b5n3lKRLzk")
        self.dataset_folder = dataset_folder

    def check_files(self):
        if not self.check_exists(self.dataset_folder):
            print("Please download the dataset from: https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar")
            return False
        
        if not self.check_exists(self.dataset_folder + 'metadata.csv'):
            print("Could not find metadata file... Please make sure this is downloaded and in the folder.")
            return False
        
        return True

    # This split works for stage generalization - takes the average across stages, though 
    def prepare_data(self, split = False, subjects = None, test_stages = None):
        """Prepares the EMG2POSE dataset.
        
        Parameters
        ----------
        subjects: list
            A list of subject indexes. 
        test_stages: list (default=['Wiggling2', 'Gesture3', 'Gesture2', 'Counting2', 'FingerFreeform', 'Counting1'])
            A list of stages to use for training. See self.mapping for options. If a user doens't have that testing stage then it is ignored.
        """
        if test_stages:
            for t in test_stages:
                assert t in self.mapping.keys()
        else:
            test_stages = ['Wiggling2', 'Gesture3', 'Gesture2', 'Counting2', 'FingerFreeform', 'Counting1']

        train_stages = []
        for k in self.mapping.keys():
            if k not in test_stages:
                train_stages.append(k)

        # (1) Make sure everything is downloaded 
        self.check_files()

        # (2) Load metadata file 
        df = pd.read_csv(self.dataset_folder + 'metadata.csv')
        subject_ids = []
        for user in np.unique(df['user']):
            if np.unique(df[df['user'] == user]['stage']).shape[0] == 29:
                subject_ids.append(user)
        subject_ids = np.array(subject_ids)

        if subjects:
            subject_ids = subject_ids[subjects]
        
        odh_tr = OfflineDataHandler()
        odh_tr.subjects = []
        odh_tr.labels = []
        odh_tr.stages = []
        odh_tr.reps = []
        odh_tr.extra_attributes = ['subjects', 'labels', 'stages', 'reps']

        odh_te = OfflineDataHandler()
        odh_te.subjects = []
        odh_te.labels = []
        odh_te.stages = []
        odh_te.reps = []
        odh_te.extra_attributes = ['subjects', 'labels', 'stages', 'reps']

        # (3) Iterate through subjects and grab all of the relevant files 
        for s_i, s in enumerate(subject_ids):
            sub_mask = df['user'] == s
            gestures = [self.mapping[v] for v in np.hstack([train_stages, test_stages])]
            reps = [0] * len(gestures)
            gesture_mask = df['stage'].isin(gestures)

            # Get all files for that subject 
            files = df['filename'][(sub_mask) & (gesture_mask)]
            files = [f.replace('left', '') for f in files]
            files = [f.replace('right', '') for f in files]
            for f in np.unique(files):
                # Check that files exists otherwise skip
                if not (self.check_exists(self.dataset_folder + '/' + f + 'left.hdf5') and self.check_exists(self.dataset_folder + '/' + f + 'right.hdf5')):
                    continue

                left = h5py.File(self.dataset_folder + '/' + f + 'left.hdf5', "r")
                right = h5py.File(self.dataset_folder + '/' + f + 'right.hdf5', "r")
                gest = df[df['filename'] == f + 'right']['stage'].item()
                gesture_name = list(self.mapping.keys())[list(self.mapping.values()).index(gest)]

                emg_left = left['emg2pose']['timeseries']['emg']
                emg_right = right['emg2pose']['timeseries']['emg']
                min_idx = min([len(emg_left), len(emg_right)])

                ja_left = left['emg2pose']['timeseries']['joint_angles']
                ja_right = right['emg2pose']['timeseries']['joint_angles']

                if gesture_name in train_stages:
                    odh_tr.data.append(np.hstack([emg_left[0:min_idx], emg_right[0:min_idx]]))
                    odh_tr.labels.append(np.hstack([ja_left[0:min_idx], ja_right[0:min_idx]]))
                    odh_tr.stages.append(np.ones((len(odh_tr.data[-1]), 1)) * gestures.index(gest))
                    odh_tr.subjects.append(np.ones((len(odh_tr.data[-1]), 1)) * s_i)
                    odh_tr.reps.append(np.ones((len(odh_tr.data[-1]), 1)) * reps[gestures.index(gest)])
                else:
                    odh_te.data.append(np.hstack([emg_left[0:min_idx], emg_right[0:min_idx]]))
                    odh_te.labels.append(np.hstack([ja_left[0:min_idx], ja_right[0:min_idx]]))
                    odh_te.stages.append(np.ones((len(odh_te.data[-1]), 1)) * gestures.index(gest))
                    odh_te.subjects.append(np.ones((len(odh_te.data[-1]), 1)) * s_i)
                    odh_te.reps.append(np.ones((len(odh_te.data[-1]), 1)) * reps[gestures.index(gest)])
                reps[gestures.index(gest)] += 1
                
        odh_all = odh_tr + odh_te
        data = odh_all
        if split:
            data = {'All': odh_all, 'Train': odh_tr, 'Test': odh_te}
        return data 
        
    

