import time
import socket
import pickle
import platform
import numpy as np
from multiprocessing import Process
from libemg._streamers._sifi_streamer import SiFiLabServer
from libemg._streamers._myo_streamer import MyoStreamer
from libemg._streamers._delsys_streamer import DelsysEMGStreamer
if platform.system() != 'Linux':
    from libemg._streamers._oymotion_windows_streamer import Gforce, oym_start_stream
else: 
    from libemg._streamers._oymotion_streamer import OyMotionStreamer
from libemg._streamers._emager_streamer import EmagerStreamer
from libemg._streamers._sifi_bridge_streamer import SiFiBridgeStreamer
from libemg._streamers._OTB_SessantaquattroPlus import OTBSessantaquattroPlusStreamer
from libemg._streamers._OTB_MuoviPlus import OTBMuoviPlusStreamer
from libemg._streamers._OTB_Muovi import OTBMuoviStreamer
from libemg._streamers._OTB_Syncstation import OTBSyncstationStreamer

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
    ip: string optional, default = '127.0.0.1'
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

def myo_streamer(filtered=True, ip='127.0.0.1', port=12345, imu=False):
    """The UDP streamer for the myo armband. 

    This function connects to the Myo and streams its data over UDP. It leverages the PyoMyo 
    library. Note: this version requires the blue dongle to be plugged in.

    Parameters
    ----------
    filtered: bool (optional), default=True
        If True, the data is the filtered data. Otherwise it is the raw unfiltered data.
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    imu: bool (optional), default=False
        If True, the IMU data will also stream. Columns: 0,1,2,3 (quaternion), 4,5,6 (accelerometer), 7,8,9 (gyroscope). 
    Examples
    ---------
    >>> myo_streamer()
    """
    myo = MyoStreamer(filtered, ip, port, imu)
    p = Process(target=myo.start_stream, daemon=True)
    p.start()
    return p

def sifibridge_streamer(ip='127.0.0.1', port=12345, version="1.2",
                 ecg=False,
                 emg=True, 
                 eda=False,
                 imu=False,
                 ppg=False,
                 notch_on=True, notch_freq=60,
                 emg_fir_on = True,
                 emg_fir=[20,450],
                 other=False):
    """The UDP streamer for the sifi armband. 
    This function connects to the sifi bridge and streams its data over UDP. This is used
    for the SiFi biopoint and bioarmband.
    Note that the IMU is acc_x, acc_y, acc_z, quat_w, quat_x, quat_y, quat_z.
    Parameters
    ----------
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    version: string optional, default = '1.2'
        The version for the sifi streamer.
    Examples
    ---------
    >>> sifibridge_streamer()
    """
    sb = SiFiBridgeStreamer(ip, port,notch_on=notch_on,
                            ecg=ecg,
                            emg=emg,
                            eda=eda,
                            imu=imu,
                            ppg=ppg,
                            notch_freq=notch_freq,
                            emgfir_on=emg_fir_on,
                            emg_fir = emg_fir,
                            other=other)
    p = Process(target=sb.start_stream, daemon=True)
    p.start()
    return p

def sifi_streamer(stream_port=12345, stream_ip='127.0.0.1', sifi_port=5000, sifi_ip='127.0.0.1'):
    """The UDP streamer for the sifi cuff. 

    This function connects to the Sifi cuff and streams its data over UDP. Note that you must have the Sifi UI
    installed for this to work.

    Parameters
    ----------
    stream_port: int (optional), default=12345
        The desired port to stream over. 
    stream_ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    sifi_port: int (optional), default=5000
        The port that the SIFI cuff is streaming over.
    sifi_ip: string (optional), default='127.0.0.1'
        The ip that the SIFI cuff is streaming over.

    Examples
    ---------
    >>> sifi_streamer()
    """
    sifi = SiFiLabServer(stream_port=stream_port, stream_ip=stream_ip, sifi_port=sifi_port, sifi_ip=sifi_ip)
    p = Process(target=sifi.start_stream, daemon=True)
    p.start()
    return p

def delsys_streamer(stream_ip='localhost', stream_port=12345, delsys_ip='localhost',cmd_port=50040, emg_port=50043, channel_list=list(range(8))):
    """The UDP streamer for the Delsys device (Avanti/Trigno). 

    This function connects to the Delsys and streams its data over UDP. Note that you must have the Delsys Control Utility
    installed for this to work.

    Parameters
    ----------
    stream_port: int (optional), default=12345
        The desired port to stream over. 
    stream_ip: string optional, default = 'localhost'
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
    ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.

    Examples
    ---------
    >>> emager_streamer()
    """
    ema = EmagerStreamer(ip, port)
    p = Process(target=ema.start_stream, daemon=True)
    p.start()
    return p

def otb_syncstation_streamer(ip='localhost', port=12345):
    """OT Bioelettronica sync station streamer.

    Parameters
    ----------
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.

    Examples
    ---------
    >>> otb_syncstation_streamer()
    """
    syncstation = OTBSyncstationStreamer(ip, port)
    p = Process(target=syncstation.start_stream, daemon=True)
    p.start()
    return p


def otb_muovi_plus_streamer(ip='localhost', port=12345):
    """OT Bioelettronica muovi+ streamer.

    Parameters
    ----------
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.

    Examples
    ---------
    >>> otb_muovi_plus_streamer()
    """
    muoviplus = OTBMuoviPlusStreamer(ip, port)
    p = Process(target=muoviplus.start_stream, daemon=True)
    p.start()
    return p

def otb_muovi_streamer(ip='localhost', port=12345):
    """OT Bioelettronica muovi streamer.

    Parameters
    ----------
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.

    Examples
    ---------
    >>> otb_muovi_streamer()
    """
    muovi = OTBMuoviStreamer(ip, port)
    p = Process(target=muovi.start_stream, daemon=True)
    p.start()
    return p


def otb_sessantaquattro_plus_streamer(ip='localhost', port=12345):
    """OT Bioelettronica sessantaquatro streamer.

    Parameters
    ----------
    port: int (optional), default=12345
        The desired port to stream over. 
    ip: string optional, default = '127.0.0.1'
        The ip used for streaming predictions over UDP.

    Examples
    ---------
    >>> otb_sessantaquattro_plus_streamer()
    """
    sessantaquattro_plus = OTBSessantaquattroPlusStreamer(ip, port)
    p = Process(target=sessantaquattro_plus.start_stream, daemon=True)
    p.start()
    return p