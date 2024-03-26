import time
import socket
import pickle
import numpy as np
from multiprocessing import Process, Event, Lock
from libemg._streamers._myo_streamer import MyoStreamer
from libemg._streamers._delsys_streamer import DelsysEMGStreamer
import platform
if platform.system() != 'Linux':
    from libemg._streamers._oymotion_windows_streamer import Gforce, oym_start_stream
else: 
    from libemg._streamers._oymotion_streamer import OyMotionStreamer
from libemg._streamers._emager_streamer import EmagerStreamer
from libemg._streamers._sifi_bridge_streamer import SiFiBridgeStreamer

def sifibridge_streamer(version="1.2",
                 shared_memory_items = [["emg",       (7500,8), np.double],
                                        ["emg_count", (1,1),    np.int32]], #TODO: Make default include everything
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

def delsys_streamer(stream_ip='localhost', stream_port=12345, delsys_ip='localhost',cmd_port=50040, emg_port=50043, channel_list=list(range(8))):
    """The UDP streamer for the Delsys device (Avanti/Trigno). 

    This function connects to the Delsys and streams its data over UDP. Note that you must have the Delsys Control Utility
    installed for this to work.

    Parameters
    ----------
    stream_port: int (optional), default=12345
        The desired port to stream over. 
    stream_ip: string (option), default = 'localhost'
        The ip used for streaming predictions over UDP.
    cmd_port: int (optional), default=50040.
        The port that commands are sent to the Delsys system (ie., the start command and the stop command.)
    delsys_port: int (optional), default=50043. 
        The port that the Delsys is streaming over. Note this value reflects the EMG data port for the Delsys Avanti system. For the Trigno system (legacy), the port is 50041.
    delsys_ip: string (optional), default='localhost'
        The ip that the Delsys is streaming over.
    channel_list: list, default=[0,1,2,3,4,5,6,7].
        The channels (i.e., electrodes) that are being used in the experiment. The Delsys will send 16 channels over the delsys_ip, but we only take the active channels to be streamed over the stream_ip/stream_port.

    Examples
    ---------
    >>> delsys_streamer()
    """
    delsys = DelsysEMGStreamer(stream_ip = stream_ip,
                            stream_port = stream_port,
                            recv_ip=delsys_ip,
                            cmd_port=cmd_port,
                            data_port=emg_port,
                            total_channels=channel_list,
                            timeout=10)
    p = Process(target=delsys.start_stream, daemon=True)
    p.start()
    return p


def oymotion_streamer(ip='127.0.0.1', port=12345, platform='windows', sampling_rate=1000):
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
    platform.lower()
    sampling = 1000
    res = 8
    if sampling_rate == 500:
        res = 12
        sampling = 500

    if platform == "windows" or platform == 'mac':
        oym = Gforce(ip,port)
        p = Process(target=oym_start_stream, args=(oym,sampling,), daemon=True)
        p.start()
        return p
    else:
        oym = OyMotionStreamer(ip, port, sampRate=sampling, resolution=res)
        oym.start_stream()
    return 0



def emager_streamer(ip='127.0.0.1', port=12345):
    """The UDP streamer for the emager armband. 

    This function connects to the emager cuff and streams its data over UDP.

    Parameters
    ----------
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string (option), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.

    Examples
    ---------
    >>> emager_streamer()
    """
    ema = EmagerStreamer(ip, port)
    p = Process(target=ema.start_stream, daemon=True)
    p.start()
    return p
