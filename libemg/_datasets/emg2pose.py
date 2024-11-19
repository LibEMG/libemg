from libemg._datasets.dataset import Dataset
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
                        "A large dataset from ctrl-labs (Meta) for joint angle estimation.",
                        "https://openreview.net/forum?id=b5n3lKRLzk")
        self.dataset_folder = dataset_folder

    def prepare_data(self, split = False, subjects = None, train_stages = None, test_stages = None):
        """Prepares the EMG2POSE dataset.
        
        Parameters
        ----------
        subjects: list
            A list of subject indexes. 
        train_stages: list (default=first 20)
            A list of stages to use for training. See self.mapping for options. 
        test: list (default=last 9)
            A list of stages to use for testing. See self.mapping for options. 
        """
        # Make sure all train and test stages are valid 
        if train_stages:
            for ts in train_stages:
                assert ts in list(self.mapping.keys())
        else:
            train_stages = list(self.mapping.keys())[0:20]
        if test_stages:
            for ts in test_stages:
                assert ts in list(self.mapping.keys())
        else:
           test_stages = list(self.mapping.keys())[20:29]

        # (1) Make sure everything is downloaded 
        if not self.check_exists(self.dataset_folder):
            print("Please download the dataset from: https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar")
            return 
        
        if not self.check_exists(self.dataset_folder + 'metadata.csv'):
            print("Could not find metadata file... Please make sure this is downloaded and in the folder.")

        # (2) Load metadata file 
        df = pd.read_csv(self.dataset_folder + 'metadata.csv')
        subject_ids = np.unique(df['user']) # Unique subject IDs 
        if subjects:
            subject_ids = subject_ids[subjects]
                
        # (3) Iterate through subjects and grab all of the relevant files 
        for s_i, s in enumerate(subject_ids):
            sub_mask = df['user'] == s
            gestures = [self.mapping[v] for v in np.hstack([train_stages, test_stages])]
            gesture_mask = df['stage'].isin(gestures)

            tr_data = []
            te_data = []
            tr_labels = []
            te_labels = []
            # Get all files for that subject 
            files = df['filename'][(sub_mask) & (gesture_mask)]
            for f in np.array(files):
                if 'left' in f:
                    continue # Just for now TODO: Fix this !!!!!!!
                h5_data = h5py.File(self.dataset_folder + '/' + f + '.hdf5', "r")
                gesture_name = df[df['filename'] == f]['stage'].item()
                print(gesture_name)
                print(train_stages)
                if gesture_name in train_stages:
                    tr_data.append(h5_data['emg2pose']['timeseries']['emg'])
                    tr_labels.append(h5_data['emg2pose']['timeseries']['joint_angles'])
                else:
                    te_data.append(h5_data['emg2pose']['timeseries']['emg'])
                    te_labels.append(h5_data['emg2pose']['timeseries']['joint_angles'])
            
            print(len(tr_data))
            print(len(te_data))
