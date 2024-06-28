import time
import socket
import pickle
import numpy as np
from multiprocessing import Process, Event, Lock
from libemg._streamers._myo_streamer import MyoStreamer
from libemg._streamers._delsys_streamer import DelsysEMGStreamer
import platform
if platform.system() != 'Linux':
    from libemg._streamers._oymotion_windows_streamer import Gforce
else: 
    from libemg._streamers._oymotion_streamer import OyMotionStreamer
from libemg._streamers._emager_streamer import EmagerStreamer
from libemg._streamers._sifi_bridge_streamer import SiFiBridgeStreamer
from libemg._streamers._leap_streamer import LeapStreamer


def leap_streamer(shared_memory_items=None,
                  arm_basis = True,
                  arm_width = False,
                  hand_direction = False,
                  elbow = False,
                  grab_angle = False,
                  grab_strength = False,
                  palm_normal = True,
                  palm_position = True,
                  palm_velocity = True,
                  palm_width = False,
                  pinch_distance = False,
                  pinch_strength = False,
                  handedness = True,
                  hand_r = False,
                  hand_s = False,
                  sphere_center = True,
                  sphere_radius = True,
                  wrist = True,
                  finger_bases = True,
                  btip_position = False,
                  carp_position = False,
                  dip_position = False,
                  finger_direction = True,
                  finger_extended = False,
                  finger_length = False,
                  mcp_position = False,
                  pip_position = False,
                  stabilized_tip_position=False,
                  tip_position=True,
                  tip_velocity=False,
                  tool=False,
                  touch_distance = True,
                  touch_zone = True,
                  finger_width=False):
    if shared_memory_items is None:
        shared_memory_items = []
        # leap is 115 FPS -> 115 Hz normally.
        if arm_basis:
            shared_memory_items.append(["arm_basis",       (230,11), np.double])
            shared_memory_items.append(["arm_basis_count", (1,1),    np.int32])
        if arm_width:
            shared_memory_items.append(["arm_width",       (230,3), np.double])
            shared_memory_items.append(["arm_width_count", (1,1),    np.int32])
        if hand_direction:
            shared_memory_items.append(["hand_direction",       (230,5), np.double])
            shared_memory_items.append(["hand_direction_count", (1,1),    np.int32])
        if elbow:
            shared_memory_items.append(["elbow",       (230,5), np.double])
            shared_memory_items.append(["elbow_count", (1,1),    np.int32])
        if grab_angle:
            shared_memory_items.append(["grab_angle",       (230,3), np.double])
            shared_memory_items.append(["grab_angle_count", (1,1),    np.int32])
        if grab_strength:
            shared_memory_items.append(["grab_strength",       (230,3), np.double])
            shared_memory_items.append(["grab_strength_count", (1,1),    np.int32])
        if palm_normal:
            shared_memory_items.append(["palm_normal",       (230,5), np.double])
            shared_memory_items.append(["palm_normal_count", (1,1),    np.int32])
        if palm_position:
            shared_memory_items.append(["palm_position",       (230,5), np.double])
            shared_memory_items.append(["palm_position_count", (1,1),    np.int32])
        if palm_velocity:
            shared_memory_items.append(["palm_velocity",       (230,5), np.double])
            shared_memory_items.append(["palm_velocity_count", (1,1),    np.int32])
        if palm_width:
            shared_memory_items.append(["palm_width",       (230,3), np.double])
            shared_memory_items.append(["palm_width_count", (1,1),    np.int32])    
        if pinch_distance:
            shared_memory_items.append(["pinch_distance",       (230,3), np.double])
            shared_memory_items.append(["pinch_distance_count", (1,1),    np.int32]) 
        if pinch_strength:
            shared_memory_items.append(["pinch_strength",       (230,3), np.double])
            shared_memory_items.append(["pinch_strength_count", (1,1),    np.int32]) 
        if handedness:
            shared_memory_items.append(["handedness",       (230,3), np.double])
            shared_memory_items.append(["handedness_count", (1,1),    np.int32]) 
        if hand_r:
            shared_memory_items.append(["hand_r",       (230,5), np.double])
            shared_memory_items.append(["hand_r_count", (1,1),    np.int32]) 
        if hand_s:
            shared_memory_items.append(["hand_s",       (230,3), np.double])
            shared_memory_items.append(["hand_s_count", (1,1),    np.int32])
        if sphere_center:
            shared_memory_items.append(["sphere_center",       (230,5), np.double])
            shared_memory_items.append(["sphere_center_count", (1,1),    np.int32])
        if sphere_radius:
            shared_memory_items.append(["sphere_radius",       (230,3), np.double])
            shared_memory_items.append(["sphere_radius_count", (1,1),    np.int32])
        if wrist:
            shared_memory_items.append(["wrist",       (230,5), np.double])
            shared_memory_items.append(["wrist_count", (1,1),    np.int32])
        if finger_bases:
            shared_memory_items.append(["finger_bases",       (230,38), np.double])
            shared_memory_items.append(["finger_bases_count", (1,1),    np.int32])
        if btip_position:
            shared_memory_items.append(["btip_position",       (230,5), np.double])
            shared_memory_items.append(["btip_position_count", (1,1),    np.int32])
        if carp_position:
            shared_memory_items.append(["carp_position",       (230,5), np.double])
            shared_memory_items.append(["carp_position_count", (1,1),    np.int32])
        if dip_position:
            shared_memory_items.append(["dip_position",       (230,5), np.double])
            shared_memory_items.append(["dip_position_count", (1,1),    np.int32])
        if finger_direction:
            shared_memory_items.append(["finger_direction",       (230,5), np.double])
            shared_memory_items.append(["finger_direction_count", (1,1),    np.int32])
        if finger_extended:
            shared_memory_items.append(["finger_extended",       (230,3), np.double])
            shared_memory_items.append(["finger_extended_count", (1,1),    np.int32])
        if finger_length:
            shared_memory_items.append(["finger_length",       (230,3), np.double])
            shared_memory_items.append(["finger_length_count", (1,1),    np.int32])
        if mcp_position:
            shared_memory_items.append(["mcp_position",       (230,5), np.double])
            shared_memory_items.append(["mcp_position_count", (1,1),    np.int32])
        if pip_position:
            shared_memory_items.append(["pip_position",       (230,5), np.double])
            shared_memory_items.append(["pip_position_count", (1,1),    np.int32])
        if stabilized_tip_position:
            shared_memory_items.append(["stabilized_tip_position",       (230,5), np.double])
            shared_memory_items.append(["stabilized_tip_position_count", (1,1),    np.int32])
        if tip_position:
            shared_memory_items.append(["tip_position",       (230,5), np.double])
            shared_memory_items.append(["tip_position_count", (1,1),    np.int32])
        if tip_velocity:
            shared_memory_items.append(["tip_velocity",       (230,5), np.double])
            shared_memory_items.append(["tip_velocity_count", (1,1),    np.int32])
        if tool:
            shared_memory_items.append(["tool",       (230,3), np.double])
            shared_memory_items.append(["tool_count", (1,1),    np.int32])
        if touch_distance:
            shared_memory_items.append(["touch_distance",       (230,3), np.double])
            shared_memory_items.append(["touch_distance_count", (1,1),    np.int32])
        if touch_zone:
            shared_memory_items.append(["touch_zone",       (230,3), np.double])
            shared_memory_items.append(["touch_zone_count", (1,1),    np.int32])
        if finger_width:
            shared_memory_items.append(['finger_width', (230,3), np.double])
            shared_memory_items.append(['finger_width_count', (1,1), np.int32])

    for item in shared_memory_items:
        item.append(Lock())
    
    ls = LeapStreamer(shared_memory_items)
    ls.start()
    return ls, shared_memory_items


def sifibridge_streamer(version="1_1",
                 shared_memory_items = None,
                 ecg=False,
                 emg=True, 
                 eda=False,
                 imu=False,
                 ppg=False,
                 notch_on=True, notch_freq=60,
                 emg_fir_on = True,
                 emg_fir=[20,450],
                 eda_cfg = True,
                 fc_lp = 0, # low pass eda
                 fc_hp = 5, # high pass eda
                 freq = 250,# eda sampling frequency
                 streaming=False):
    """The UDP streamer for the sifi armband. 
    This function connects to the sifi bridge and streams its data over UDP. This is used
    for the SiFi biopoint and bioarmband.
    Note that the IMU is acc_x, acc_y, acc_z, quat_w, quat_x, quat_y, quat_z.
    Parameters
    ----------
    version: string (option), default = '1.2'
        The version for the sifi streamer.
    shared_memory_items, default = []
        The key, size, datatype, and multiprocessing Lock for all data to be shared between processes.
    Examples
    ---------
    >>> sifibridge_streamer()
    """

    if shared_memory_items is None:
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (3000,8), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
        if imu:
            shared_memory_items.append(["imu",       (100,10), np.double])
            shared_memory_items.append(["imu_count", (1,1),    np.int32])
        if ecg:
            shared_memory_items.append(["ecg",       (100,10), np.double])
            shared_memory_items.append(["ecg_count", (1,1),    np.int32])
        if eda:
            shared_memory_items.append(["eda",       (100,10), np.double])
            shared_memory_items.append(["eda_count", (1,1),    np.int32])
        if ppg:
            shared_memory_items.append(["ppg",       (100,10), np.double])
            shared_memory_items.append(["ppg_count", (1,1),    np.int32])

    for item in shared_memory_items:
        item.append(Lock())
    sb = SiFiBridgeStreamer(version=version,
                            shared_memory_items=shared_memory_items,
                            notch_on=notch_on,
                            ecg=ecg,
                            emg=emg,
                            eda=eda,
                            imu=imu,
                            ppg=ppg,
                            notch_freq=notch_freq,
                            emgfir_on=emg_fir_on,
                            emg_fir = emg_fir,
                            eda_cfg = eda_cfg,
                            fc_lp = fc_lp, # low pass eda
                            fc_hp = fc_hp, # high pass eda
                            freq = freq,# eda sampling frequency
                            streaming=streaming)
    sb.start()
    return sb, shared_memory_items


def mock_emg_stream(file_path, num_channels, sampling_rate=100, port=12345, ip="127.0.0.1"):
    """Streams EMG from a test file over UDP.

    This function can be used to simulate raw EMG being streamed over a UDP port. The main purpose 
    of this function would be to explore real-time interactions without the need for a physical 
    device. Note: This will start up a seperate process to stream data over. Additionally, 
    this uses the time module and as such the sampling rate may not be perfect and there may 
    be some latency.

    Parameters
    ----------
    file_path: string
        The path of the csv file where the EMG data is located. 
    num_channels: int
        The number of channels to stream. This should be <= to 
        the number of columns in the CSV.
    sampling_rate: int (optional), default=100
        The desired sampling rate in Hz.
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string (option), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    
    Examples
    ----------
    >>> mock_emg_stream("stream_data.csv", num_channels=8, sampling_rate=100)
    """
    Process(target=_stream_thread, args=(file_path, num_channels, sampling_rate, port, ip), daemon=True).start()

def _stream_thread(file_path, num_channels, sampling_rate, port, ip):
    data = np.loadtxt(file_path, delimiter=",")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    index = 0
    while index < len(data):
        val = time.time() + (1000/sampling_rate)/1000
        while time.time() < val:
            pass
        data_arr = pickle.dumps(list(data[index][:num_channels]))
        sock.sendto(data_arr, (ip, port))
        index += 1

def myo_streamer(
    shared_memory_items = None,
    emg = True, 
    imu = False,
    filtered=True):
    """The UDP streamer for the myo armband. 

    This function connects to the Myo and streams its data over UDP. It leverages the PyoMyo 
    library. Note: this version requires the blue dongle to be plugged in.

    Parameters
    ----------
    filtered: bool (optional), default=True
        If True, the data is the filtered data. Otherwise it is the raw unfiltered data.

    Examples
    ---------
    >>> myo_streamer()
    """
    if shared_memory_items is None:
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (1000,8), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
        if imu:
            shared_memory_items.append(["imu",       (250,10), np.double])
            shared_memory_items.append(["imu_count", (1,1),    np.int32])

    for item in shared_memory_items:
        item.append(Lock())
    myo = MyoStreamer(filtered, emg, imu, shared_memory_items)
    myo.start()
    return myo, shared_memory_items

def delsys_streamer(shared_memory_items = None,
                    emg = True, 
                    imu = False,
                    delsys_ip='localhost',
                    cmd_port=50040, 
                    emg_port=50043, 
                    channel_list=list(range(8)),
                    timeout=10):
    """The streamer for the Delsys device (Avanti/Trigno). 

    This function connects to the Delsys and streams its data over UDP. Note that you must have the Delsys Control Utility
    installed for this to work.

    Parameters
    ----------
    shared_memory_items : list
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype].
    cmd_port: int (optional), default=50040.
        The port that commands are sent to the Delsys system (ie., the start command and the stop command.)
    delsys_port: int (optional), default=50043. 
        The port that the Delsys is streaming over. Note this value reflects the EMG data port for the Delsys Avanti system. For the Trigno system (legacy), the port is 50041.
    delsys_ip: string (optional), default='localhost'
        The ip that the Delsys is streaming over.
    channel_list: list, default=[0,1,2,3,4,5,6,7].
        The channels (i.e., electrodes) that are being used in the experiment. The Delsys will send 16 channels over the delsys_ip, but we only take the active channels to be streamed over the stream_ip/stream_port.
    timeout : int
        Timeout for commands sent to Delsys.
    Examples
    ---------
    >>> delsys_streamer()
    """
    if shared_memory_items is None:
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (3000,len(channel_list)), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
        if imu:
            shared_memory_items.append(["imu",       (500,3), np.double])
            shared_memory_items.append(["imu_count", (1,1),    np.int32])
    for item in shared_memory_items:
        item.append(Lock())
    
    delsys = DelsysEMGStreamer(shared_memory_items=shared_memory_items,
                                emg=emg,
                                imu=imu,
                                recv_ip=delsys_ip,
                                cmd_port=cmd_port,
                                data_port=emg_port,
                                channel_list=channel_list,
                                timeout=timeout)
    delsys.start()
    return delsys, shared_memory_items


def oymotion_streamer(shared_memory_items=None, sampling_rate=1000, emg=True,imu=False):
    """The UDP streamer for the oymotion armband. 

    This function connects to the oymotion and streams its data over UDP. It leverages the gforceprofile 
    library. Note: this version requires the dongle to be plugged in. Note, you should run this with sudo
    and using sudo -E python to preserve your environment.

    Parameters
    ----------
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    platform: string ('windows', 'linux', or 'mac'), default='windows'
        The platform being used. Linux uses a different streamer than mac and windows. 
    sampling_rate: int (optional), default=1000 (options: 1000 or 500)
        The sampling rate wanted from the device. Note that 1000 Hz is 8 bit resolution and 500 Hz is 12 bit resolution

    Examples
    ---------
    >>> oymotion_streamer()
    """
    
    if sampling_rate == 1000:
        res = 8
    elif sampling_rate == 500:
        res = 12
    else:
        raise Exception("Invalid sampling frequency provided.")

    if shared_memory_items == None:
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (sampling_rate*2,8), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
        if imu:
            shared_memory_items.append(["imu",       (100,10), np.double])
            shared_memory_items.append(["imu_count", (1,1),    np.int32])
    for item in shared_memory_items:
        item.append(Lock())

    operating_system = platform.system().lower()

    # I'm only addressing this atm.
    if operating_system == "windows" or operating_system == 'mac':
        oym = Gforce(sampling_rate, res, emg, imu, shared_memory_items)
        oym.start()
    else:
        # This has not been updated to the new memory manager methods.
        # oym = OyMotionStreamer(ip, port, sampRate=sampling, resolution=res)
        # oym.start_stream()
        raise Exception("Oymotion Streamer is not implemented for Linux.")
    return oym, shared_memory_items



def emager_streamer(shared_memory_items = None):
    """The streamer for the emager armband. 

    This function connects to the emager cuff and streams its data over a serial port and access it via shared memory.

    Parameters
    ----------
    shared_memory_items: list, default=12345
        The shared memory items used to access specific tags in shared memory (e.g., emg and emg_count). If None is passed,
        defaults values are created and used to create the streamer.

    Examples
    ---------
    >>> emager_streamer()
    """
    if shared_memory_items is None:
        # Create defaults
        shared_memory_items = []
        shared_memory_items.append(['emg', (2000, 64), np.double])  # buffer size doesn't have a huge effect - pretty much as long as it's bigger than window size
        shared_memory_items.append(['emg_count', (1, 1), np.int32])

    for item in shared_memory_items:
        item.append(Lock())
    ema = EmagerStreamer(shared_memory_items)
    ema.start()
    return ema, shared_memory_items

