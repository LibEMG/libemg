from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler
import numpy as np
from libemg.utils import *
import os
import warnings
from typing import Any, Dict, Iterable
import h5py
import numpy as np
from scipy.io import loadmat


# FIXED GLOBAL GESTURE MAP
GESTURE_MAP = {         # Matches EPN-612 Class IDs
    "relax": 0,
    "fist": 1,
    "wave in": 2,
    "wave out": 3,
    "open": 4,
    "pinch": 5,
    "up": 6,
    "down": 7,
    "left": 8,
    "right": 9,
    "forward": 10,
    "backward": 11,
}

# MATLAB-derived fixed gesture order (rep 1..180)
GESTURE_ORDER_180 = (
    ["relax"] * 15 +
    ["wave in"] * 15 +
    ["wave out"] * 15 +
    ["fist"] * 15 +
    ["open"] * 15 +
    ["pinch"] * 15 +
    ["up"] * 15 +
    ["down"] * 15 +
    ["left"] * 15 +
    ["right"] * 15 +
    ["forward"] * 15 +
    ["backward"] * 15
)
assert len(GESTURE_ORDER_180) == 180

# Assigning integer labels to devices
DEVICE_MAP = {
    "myo": 0,
    "gForce": 1,
}


# ======== UTILS ========
def to_scalar(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.item()
    return x

def write_h5_scalar(group: h5py.Group, name, value):
    name = str(name)
    value = to_scalar(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        group.create_dataset(name, data=value)
    else:
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=np.array(str(value), dtype=dt))


# ======== METADATA EXTRACTION ========
def extract_metadata(userData) -> Dict[str, Any]:
    meta = {}

    def extract_struct(section):
        out = {}
        for field in section._fieldnames:
            out[field] = to_scalar(getattr(section, field))
        return out

    meta["userInfo"] = extract_struct(userData.userInfo)
    meta["extraInfo"] = extract_struct(userData.extraInfo)
    meta["deviceInfo"] = extract_struct(userData.deviceInfo)
    meta["userGroup"] = to_scalar(userData.userGroup)
    meta["gestureNameMap"] = {str(v): k for k, v in GESTURE_MAP.items()}

    return meta


# ======== CORE PROCESSING ========
def process_user(
    mat_path: str,
    out_path: str,
    subject_id: int,
    is_training_group: bool):
    userData = loadmat(mat_path, squeeze_me=True, 
                       struct_as_record=False)["userData"]

    reps_written = 0

    with h5py.File(out_path, "w") as h5:
        # ---- META ----
        meta_grp = h5.create_group("meta")
        meta = extract_metadata(userData)

        for section, values in meta.items():
            if isinstance(values, dict):
                sec_grp = meta_grp.create_group(section)
                for k, v in values.items():
                    write_h5_scalar(sec_grp, k, v)
            else:
                write_h5_scalar(meta_grp, section, values)

        # ---- REPS ----
        reps_grp = h5.create_group("reps")

        def process_block(block, rep_offset: int, max_reps: int):
            nonlocal reps_written
            for i in range(max_reps):
                rep_id = rep_offset + i
                entry = block[i]

                if not hasattr(entry, "emg"):
                    warnings.warn(
                        f"Missing EMG (subject={subject_id}, rep={rep_id})"
                    )
                    continue

                gesture = GESTURE_ORDER_180[i]
                classe = GESTURE_MAP[gesture]

                emg = np.asarray(entry.emg, dtype=np.float32)
                point_begins = np.asarray(entry.pointGestureBegins, dtype=np.int64)

                rep_grp = reps_grp.create_group(f"rep_{rep_id:03d}")
                rep_grp.create_dataset("emg", data=emg)
                rep_grp.create_dataset("gesture", data=classe)
                rep_grp.create_dataset("subject", data=subject_id)
                rep_grp.create_dataset("rep", data=rep_id)
                rep_grp.create_dataset("point_begins", data=point_begins)

                reps_written += 1

        # training block: reps 0..179
        process_block(userData.training, rep_offset=0, max_reps=180)

        # testing block only for training users: reps 180..359
        if is_training_group and hasattr(userData, "testing"):
            process_block(userData.testing, rep_offset=180, max_reps=180)

    print(f"Finished user subject={subject_id} | "
            f"reps extracted={reps_written} | "
            f"output={out_path}")


# ======== DATASET WALKER ========
def process_dataset(root_in: str, root_out: str):
    for split in ["training", "testing"]:
        in_split = os.path.join(root_in, split)
        out_split = os.path.join(root_out, split)
        os.makedirs(out_split, exist_ok=True)

        user_dirs = sorted(d for d in os.listdir(in_split) if d.startswith("user_"))

        print(f"\n=== Processing split: {split} ===")

        for idx, user_dir in enumerate(user_dirs):
            subject_id = idx
            mat_path = os.path.join(in_split, user_dir, "userData.mat")
            out_path = os.path.join(out_split, f"{user_dir}.h5")

            print(f"Starting {user_dir} -> subject={subject_id}")

            process_user(mat_path=mat_path, out_path=out_path,
                        subject_id=subject_id, 
                        is_training_group=(split == "training"))

# ======== MAIN DATASET CLASS ========
class EMGEPN100(Dataset):
    def __init__(self, dataset_folder: str='DATASET_85'):
        Dataset.__init__(self, 
                         sampling={'myo': 200, 'gForce': 500}, 
                         num_channels={'myo': 8, 'gForce': 8}, 
                         recording_device=['myo', 'gForce'], 
                         num_subjects=85, 
                         gestures= GESTURE_MAP,      # Matches EPN-612 static classes IDs
                         num_reps="30 Reps x 12 Gestures x 43 Users (Train group), 15 Reps x 12 Gestures x 42 Users (Test group) --> Cross User Split",
                         description="Multi-hardware EMG dataset for 12 different hand gesture categories using the myo armband and the G-force armband.", 
                         citation="https://doi.org/10.3390/s22249613")
        self.resolution_bit = {'myo': 8, 'gForce': 12}
        self.dataset_folder = dataset_folder
        self.url = "https://laboratorio-ia.epn.edu.ec/es/recursos/dataset/emg-imu-epn-100"

    def _get_odh(self, processed_root, subjects, 
                 segment, relabel_seg, channel_last):

        splits = {"training", "testing"}
        odhs = []

        for split in splits:
            split_dir = os.path.join(processed_root, split)
            user_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".h5"))

            odh = OfflineDataHandler()
            odh.subjects = []
            odh.classes = []
            odh.reps = []
            odh.devices = []
            odh.sampling_rates = []
            odh.extra_attributes = ['subjects', 'classes', 'reps',
                                    'devices', 'sampling_rates']

            for user_file in user_files:
                path = os.path.join(split_dir, user_file)

                with h5py.File(path, "r") as f:
                    subject = int(f["reps"]["rep_000"]["subject"][()])
                    subject += 43 if split == "testing" else 0                      # 43 training group subjects and 42 testing
                    if subjects is not None:
                        if subject not in subjects:
                            continue

                    reps = f["reps"]
                    device_str = f["meta/deviceInfo/DeviceType"][()].decode("utf-8")
                    device = DEVICE_MAP[device_str]
                    fs = float(f["meta/deviceInfo/emgSamplingRate"][()])

                    for rep_name in reps:
                        rep_grp = reps[rep_name]

                        gst = int(rep_grp["gesture"][()])
                        rep_id = int(rep_grp["rep"][()])

                        _emg = rep_grp["emg"][:].astype(np.float32, copy=False)      # [T, CH]
                        if not channel_last:
                            _emg = np.transpose(_emg, (1, 0))                        # [CH, T]

                        if segment and gst != 0:
                            point_begins =  rep_grp["point_begins"][()]
                            emg = _emg[point_begins:]
                        else:
                            emg = _emg

                        # ---- Preparing ODH ----
                        odh.data.append(emg)
                        odh.classes.append(np.ones((len(emg), 1)) * gst)
                        odh.subjects.append(np.ones((len(emg), 1)) * subject)
                        odh.reps.append(np.ones((len(emg), 1)) * rep_id)
                        odh.devices.append(np.ones((len(emg), 1)) * device)
                        odh.sampling_rates.append(np.ones((len(emg), 1)) * fs)

                        if segment and gst != 0 and relabel_seg is not None:
                            assert type(relabel_seg) is int
                            gst = relabel_seg

                            emg = _emg[:point_begins]

                            odh.data.append(emg)
                            odh.classes.append(np.ones((len(emg), 1)) * gst)
                            odh.subjects.append(np.ones((len(emg), 1)) * subject)
                            odh.reps.append(np.ones((len(emg), 1)) * rep_id)
                            odh.devices.append(np.ones((len(emg), 1)) * device)
                            odh.sampling_rates.append(np.ones((len(emg), 1)) * fs)

            odhs.append(odh)

        return odhs


    def prepare_data(self,
                    split: bool = False,
                    segment: bool = True,
                    relabel_seg: int | None = None,
                    channel_last: bool = True,
                    subjects: Iterable[int] | None = None) -> OfflineDataHandler:
        """Return processed EPN100 dataset as LibEMG ODH.

        Parameters
        ----------
        split: bool or None (optional), default=False 
            Whether to return seperate training and testing ODHs.
        window_ms: float or None (optional), default=None 
            Windows size in ms (for feature extraction). There are two different sensors used in this dataset with different sampling rates.
        stride_ms: float or None (optional), default=None 
            Window stride (increment) size in ms (for feature extraction). There are two different sensors used in this dataset with different sampling rates.
        segment: bool, default=True 
            Whether crop the segment before 'pointGestureBeging' index in the dataset.
        relabel_seg: int or None (optional), default=0 
            If not False, this arg will be used as the relabeling value.
        channel_last: bool, default=True,
            Shape will be (, T, CH) if True otherwise (, CH, T)
        subjects: Iterable[int] or None (optional), default=None
            Subjects to be included in the processed dataset.

        Returns
        ----------
        Dic or OfflineDataHandler
            A dictionary of 'All', 'Train' and 'Test' ODHs of processed data or a single OfflineDataHandler if split is False.
        """
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)) and \
            (not self.check_exists( self.dataset_folder + "PROCESSED")):
            raise FileNotFoundError("Please download the EPN100+ dataset from: {} "
                                    "and place 'testing' and 'training' folders inside: "
                                    "'{}' folder.".format(self.url, self.dataset_folder)) 
        
        if (not self.check_exists( self.dataset_folder + "PROCESSED")):
            process_dataset(self.dataset_folder, self.dataset_folder + "PROCESSED")

        odh_tr, odh_te = self._get_odh(self.dataset_folder + "PROCESSED", 
                                    subjects, segment, relabel_seg, channel_last)
        
        return {'All': odh_tr + odh_te, 'Train': odh_tr, 'Test': odh_te} \
                if split else odh_tr + odh_te
    
    def get_device_ID(self, device_name: str):
        """
        Get device label ID by name
        
        Parameters
        ----------
        device_name: str
            Name of the requested device.
            
        Returns
        ----------
        int
            Device's ID
        """

        return DEVICE_MAP[device_name]




        