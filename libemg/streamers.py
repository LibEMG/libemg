import time
import socket
import pickle
import platform
import numpy as np

from multiprocessing import Process, Event, Lock
from libemg._streamers._myo_streamer import MyoStreamer
from libemg._streamers._delsys_streamer import DelsysEMGStreamer
from libemg._streamers._delsys_API_streamer import DelsysAPIStreamer
if platform.system() != 'Linux':
    from libemg._streamers._oymotion_windows_streamer import Gforce
else: 
    from libemg._streamers._oymotion_streamer import OyMotionStreamer
from libemg._streamers._emager_streamer import EmagerStreamer
from libemg._streamers._sifi_bridge_streamer import SiFiBridgeStreamer
from libemg._streamers._leap_streamer import LeapStreamer
from libemg._streamers._mindrove import MindroveStreamer

def sifi_biopoint_streamer(
    name = "BioPoint_v1_3",
    shared_memory_items = None,
    ecg = False,
    emg = True, 
    eda = False,
    imu = False,
    ppg = False,
    filtering = True, 
    emg_notch_freq = 60,
    emg_bandpass = (20,450),
    eda_bandpass = (0,5),
    eda_freq = 0,
    streaming=False,
    mac= None
):
    """
    The streamer for the SiFi BioPoint. 
    
    This function connects to SiFi Bridge and streams its data to the SharedMemory.
    
    **Note**: The IMU keys are:
    
    - Acceleration: ax, ay, az
    - Quaternions: qw, qx, qy, qz
    
    Parameters
    ----------
    
    device: string, default = BioPoint_v1_3
        The name or MAC of the device.
    shared_memory_items, default = []
        The key, size, datatype, and multiprocessing Lock for all data to be shared between processes.
    ecg, default = False
        Enable electrocardiography recording from the main sensor unit.
    emg, default = True
        Enable electromyography recording.
    eda, default = False
        Enable electrodermal recording.
    imu, default = False
        Enable inertial measurement unit recording
    ppg, default = False
        The flag to enable photoplethysmography recording
    filtering, default = True
        Enable on-device filtering, including bandpass filters and notch filters.
    emg_notch_freq, default = 60
        EMG notch filter frequency, useful for eliminating Mains power interference. Can be {None, 50, 60} Hz.
    emg_bandpass, default = (20, 450)
        The low and high cutoff frequency of the EMG bandpass filter.
    eda_bandpass, default = (0, 5)
        The low and high cutoff frequency of the EDA bandpass filter.
    eda_freq, default = 0
        The excitation signal frequency for EDA/BIOZ. Setting an AC value may inject a lot of noise into the EMG sensor.
    streaming, default = False
        Whether to package the modalities together within packets for lower latency, only supported for BioPoint v1.3 and up.
    mac, default = None:  
        Optional MAC address the device to connect to, useful when multiple devices are in the vicinity and you want to connect to a specific one.
          
    Returns
    ----------
    
    Object: streamer
        The sifi streamer process object.
    Object: shared memory
        The shared memory items list to be passed to the OnlineDataHandler.
    
    Examples
    ---------
    
    >>> streamer, shared_memory = sifibridge_streamer()
    """

    if shared_memory_items is None:
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (4000,1), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
        if imu:
            shared_memory_items.append(["imu",       (200,7), np.double])
            shared_memory_items.append(["imu_count", (1,1),    np.int32])
        if ecg:
            shared_memory_items.append(["ecg",       (1000,1), np.double])
            shared_memory_items.append(["ecg_count", (1,1),    np.int32])
        if eda:
            shared_memory_items.append(["eda",       (200,1), np.double])
            shared_memory_items.append(["eda_count", (1,1),    np.int32])
        if ppg:
            shared_memory_items.append(["ppg",       (200,4), np.double])
            shared_memory_items.append(["ppg_count", (1,1),    np.int32])

    for item in shared_memory_items:
        item.append(Lock())
        
    sb = SiFiBridgeStreamer(
        name,
        shared_memory_items,
        ecg,
        emg,
        eda,
        imu,
        ppg,
        filtering,
        emg_notch_freq,
        emg_bandpass,
        eda_bandpass,
        eda_freq,
        streaming,
        mac
    )
    sb.start()
    return sb, shared_memory_items


def sifi_bioarmband_streamer(
    name = "BioPoint_v1_1",
    shared_memory_items = None,
    ecg = False,
    emg = True, 
    eda = False,
    imu = False,
    ppg = False,
    filtering = True, 
    emg_notch_freq = 60,
    emg_bandpass = (20,450),
    eda_bandpass = (0,5),
    eda_freq = 0,
    streaming = False,
    mac = None
):
    """
    The streamer for the SiFi BioArmband. 
    
    This function connects to SiFi Bridge and streams its data to the SharedMemory.
    
    **Note**: The IMU keys are:
    
    - Acceleration: ax, ay, az
    - Quaternions: qw, qx, qy, qz
        
    Parameters
    ----------
    
    name: string, default = BioArmband
        The name of the Sifi Device. For example: BioArmband, BioPoint_v1_3, etc.
    shared_memory_items, default = []
        The key, size, datatype, and multiprocessing Lock for all data to be shared between processes.
    ecg, default = False
        Enable electrocardiography recording from the main sensor unit.
    emg, default = True
        Enable electromyography recording.
    eda, default = False
        Enable electrodermal recording.
    imu, default = False
        Enable inertial measurement unit recording
    ppg, default = False
        The flag to enable photoplethysmography recording
    filtering, default = True
        Enable on-device filtering, including bandpass filters and notch filters.
    emg_notch_freq, default = 60
        EMG notch filter frequency, useful for eliminating Mains power interference. Can be {None, 50, 60} Hz.
    emg_bandpass, default = (20, 450)
        The low and high cutoff frequency of the EMG bandpass filter.
    eda_bandpass, default = (0, 5)
        The low and high cutoff frequency of the EDA bandpass filter.
    eda_freq, default = 0
        The excitation signal frequency for EDA/BIOZ.  Setting an AC value may inject a lot of noise into the EMG sensor.
    streaming, default = False
        Whether to package the modalities together within packets for lower latency, only supported for BioPoint v1.3 and up.
    mac, default = None:  
        Optional MAC address the device to connect to, useful when multiple devices are in the vicinity and you want to connect to a specific one.
         
    Returns
    ----------
    
    Object: streamer
        The sifi streamer process object.
    Object: shared memory
        The shared memory items list to be passed to the OnlineDataHandler.
    
    Examples
    ---------
    
    >>> streamer, shared_memory = sifibridge_streamer()
    """

    if shared_memory_items is None:
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (3000,8), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
        if imu:
            shared_memory_items.append(["imu",       (200,7), np.double])
            shared_memory_items.append(["imu_count", (1,1),    np.int32])
        if ecg:
            shared_memory_items.append(["ecg",       (1000,1), np.double])
            shared_memory_items.append(["ecg_count", (1,1),    np.int32])
        if eda:
            shared_memory_items.append(["eda",       (200,1), np.double])
            shared_memory_items.append(["eda_count", (1,1),    np.int32])
        if ppg:
            shared_memory_items.append(["ppg",       (200,4), np.double])
            shared_memory_items.append(["ppg_count", (1,1),    np.int32])

    for item in shared_memory_items:
        item.append(Lock())

        
    sb = SiFiBridgeStreamer(
        name,
        shared_memory_items,
        ecg,
        emg,
        eda,
        imu,
        ppg,
        filtering,
        emg_notch_freq,
        emg_bandpass,
        eda_bandpass,
        eda_freq,
        streaming,
        mac
    )

    sb.start()
    return sb, shared_memory_items

def myo_streamer(
    shared_memory_items : list | None = None,
    emg                 : bool = True, 
    imu                 : bool = False,
    filtered            : bool=True):
    """The streamer for the myo armband. 

    This function connects to the Myo. It leverages the PyoMyo 
    library. Note: this version requires the blue dongle to be plugged in.

    Parameters
    ----------
    shared_memory_items : list (optional)
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype].
    emg : bool (optional)
        Specifies whether EMG data should be forwarded to shared memory.
    imu : bool (optional)
        Specifies whether IMU data should be forwarded to shared memory.
    filtered : bool (optional), default=True
        If True, the data is the filtered data. Otherwise it is the raw unfiltered data.
    Returns
    ----------
    Object: streamer
        The sifi streamer object.
    Object: shared memory
        The shared memory object.
    Examples
    ---------
    >>> streamer, shared_memory = myo_streamer()
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

def delsys_streamer(shared_memory_items : list | None = None,
                    emg                 : bool = True, 
                    imu                 : bool = False,
                    delsys_ip           : str = 'localhost',
                    cmd_port            : int = 50040, 
                    emg_port            : int = 50043, 
                    aux_port            : int = 50044,
                    channel_list        : list = list(range(8)),
                    timeout             : int = 10):
    """The streamer for the Delsys device (Avanti/Trigno). 

    This function connects to the Delsys. Note that you must have the Delsys Control Utility
    installed for this to work.

    Parameters
    ----------
    shared_memory_items : list (optional)
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype].
    delsys_ip: string (optional), default='localhost'
        The ip that the Delsys is streaming over.
    cmd_port: int (optional), default=50040.
        The port that commands are sent to the Delsys system (ie., the start command and the stop command.)
    emg_port: int (optional), default=50043. 
        The port that the Delsys is streaming over. Note this value reflects the EMG data port for the Delsys Avanti system. For the Trigno system (legacy), the port is 50041.
    aux_port: int (optional), default=50044.
        The port that the Delsys is streaming IMU data over.
    channel_list: list, default=[0,1,2,3,4,5,6,7].
        The channels (i.e., electrodes) that are being used in the experiment. The Delsys will send 16 channels over the delsys_ip, but we only take the active channels to be streamed over the stream_ip/stream_port.
    timeout : int
        Timeout for commands sent to Delsys.
    Returns
    ----------
    Object: streamer
        The delsys streamer object.
    Object: shared memory
        The shared memory object.
    Examples
    ---------
    >>> streamer, shared_memory = delsys_streamer()
    """
    if shared_memory_items is None:
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (3000,len(channel_list)), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
        if imu:
            shared_memory_items.append(["imu",       (500,6), np.double])
            shared_memory_items.append(["imu_count", (1,1),    np.int32])
    for item in shared_memory_items:
        item.append(Lock())
    
    delsys = DelsysEMGStreamer(shared_memory_items=shared_memory_items,
                                emg=emg,
                                imu=imu,
                                recv_ip=delsys_ip,
                                cmd_port=cmd_port,
                                data_port=emg_port,
                                aux_port=aux_port,
                                channel_list=channel_list,
                                timeout=timeout)
    delsys.start()
    return delsys, shared_memory_items


def delsys_api_streamer(license             : str,
                        key                 : str,
                        num_channels        : int | None = None,
                        dll_folder          : str = 'resources/',
                        shared_memory_items : list | None = None,
                        emg                 : bool = True):
    """The streamer for the Delsys devices that use their new C#.NET API. 

    This function connects to the Delsys. Note that you must have the Delsys .dll files (found here: https://github.com/delsys-inc/Example-Applications/tree/main/Python/resources), 
    C#.NET 8.0 SDK, and the delsys license + key. Additionally, for using any device that connects over USB, make sure that the usb driver is version >= 6.0.0.

    Parameters
    ----------
    license : str
        Delsys license
    key : str
        Delsys key
    num_channels: int
        The number of delsys sensors you are using. Not used if shared_memory_items is passed, otherwise is a required parameter.
    dll_folder: string : optional (default='resources/')
        The location of the DLL files installed from the Delsys Github.
    shared_memory_items : list (optional)
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype].
    emg : bool : (optional)
        Whether to collect emg data or not.
    Returns
    ----------
    Object: streamer
        The delsys streamer object.
    Object: shared memory
        The shared memory object.
    Examples
    ---------
    >>> streamer, shared_memory = delsys_api_streamer(LICENSE, KEY, num_channels=4)
    """
    if shared_memory_items is None:
        assert num_channels is not None, f"No shared memory items were passed, so num_channels must be set. Please set num_channels to the number of Delsys sensors. Got: {num_channels}."
        shared_memory_items = []
        if emg:
            shared_memory_items.append(["emg",       (5300,num_channels), np.double])
            shared_memory_items.append(["emg_count", (1,1),    np.int32])
    for item in shared_memory_items:
        item.append(Lock())
    
    delsys = DelsysAPIStreamer(key, license, dll_folder, shared_memory_items=shared_memory_items, emg=emg)
    delsys.start()
    return delsys, shared_memory_items

def oymotion_streamer(shared_memory_items : list | None = None,
                      sampling_rate       : int = 1000,
                      emg                 : bool = True,
                      imu                 : bool = False):
    """The streamer for the oymotion armband. 

    This function connects to the oymotion and streams its data. It leverages the gforceprofile 
    library. Note: this version requires the dongle to be plugged in. Note, you should run this with sudo
    and using sudo -E python to preserve your environment in Linux.

    Parameters
    ----------
    shared_memory_items : list (optional)
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype].
    sampling_rate: int (optional), default=1000 (options: 1000 or 500)
        The sampling rate wanted from the device. Note that 1000 Hz is 8 bit resolution and 500 Hz is 12 bit resolution
    emg : bool (optional),
        Detemines whether EMG data will be forwarded
    imu : bool (optional),
        Determines whether IMU data will be forwarded
    Returns
    ----------
    Object: streamer
        The oymotion streamer object
    Object: shared memory
        The shared memory object
    Examples
    ---------
    >>> streamer, shared_memory = oymotion_streamer()
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
    if operating_system == "windows" or operating_system == 'darwin':
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
    shared_memory_items : list (optional)
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype].
    Returns
    ----------
    Object: streamer
        The emager streamer object.
    Object: shared memory
        The shared memory object.
    Examples
    ---------
    >>> streamer, shared_memory = emager_streamer()
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

#TODO: Update docs
def leap_streamer(shared_memory_items : list | None =None,
                  arm_basis : bool = True,
                  arm_width : bool = False,
                  hand_direction : bool = False,
                  elbow : bool = False,
                  grab_angle : bool = False,
                  grab_strength : bool = False,
                  palm_normal : bool = True,
                  palm_position : bool = True,
                  palm_velocity : bool = True,
                  palm_width : bool = False,
                  pinch_distance : bool = False,
                  pinch_strength : bool = False,
                  handedness : bool = True,
                  hand_r : bool = False,
                  hand_s : bool = False,
                  sphere_center : bool = True,
                  sphere_radius : bool = True,
                  wrist : bool = True,
                  finger_bases : bool = True,
                  btip_position : bool = False,
                  carp_position : bool = False,
                  dip_position : bool = False,
                  finger_direction : bool = True,
                  finger_extended : bool = False,
                  finger_length : bool = False,
                  mcp_position : bool = False,
                  pip_position : bool = False,
                  stabilized_tip_position : bool = False,
                  tip_position : bool = True,
                  tip_velocity : bool = False,
                  tool : bool = False,
                  touch_distance : bool = True,
                  touch_zone : bool = True,
                  finger_width : bool = False):
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


def mindrove_streamer(shared_memory_items = None):
    """The streamer for the Mindrove EMG armband.

    This function starts a process that listens for EMG data using Mindrove's Python SDK.
    Please ensure that you are connected to the Mindrove armband's Wifi network before starting this 

    Parameters
    ----------
    shared_memory_items : list (optional)
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype]
        Tags "emg" and "emg_count" must be provided. If set to None, shared memory items are automatically created
        based on the device properties.
    """
    if shared_memory_items is None:
        shared_memory_items = [
            ['emg', (1000, 8), np.double],
            ['emg_count', (1, 1), np.int32]
        ]

    for item in shared_memory_items:
        if len(item) == 3:
            item.append(Lock())

    m = MindroveStreamer(shared_memory_items)
    m.start()
    return m, shared_memory_items

