import numpy as np
import time
import socket
from multiprocessing import Process

def get_windows(data, window_size, window_increment):
    """Extracts windows from a given set of data.

    Parameters
    ----------
    data: array_like
        An NxM stream of data with N samples and M channels
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.

    Returns
    ----------
    array_like
        The set of windows extracted from the data as a NxCxL where N is the number of windows, C is the number of channels 
        and L is the length of each window. 

    Examples
    ---------
    >>> data = np.loadtxt('data.csv', delimiter=',')
    >>> windows = get_windows(data, 100, 50)
    """
    num_windows = int((data.shape[0]-window_size)/window_increment) + 1
    windows = []
    st_id=0
    ed_id=st_id+window_size
    for w in range(num_windows):
        windows.append(data[st_id:ed_id,:].transpose())
        st_id += window_increment
        ed_id += window_increment
    return np.array(windows)

def _get_mode_windows(data, window_size, window_increment):
    windows = get_windows(data, window_size, window_increment)
    # we want to get the mode along the final dimension
    mode_of_windows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=windows.astype(np.int64))
    
    return mode_of_windows.squeeze()

def make_regex(left_bound, right_bound, values=[]):
    """Regex creation helper for the data handler.

    The Data Handler relies on regexes to parse the file/folder structures and extract data. 
    This function makes the creation of regexes easier.

    Parameters
    ----------
    left_bound: string
        The left bound of the regex.
    right_bound: string
        The right bound of the regex.
    values: array_like
        The values between the two regexes.

    Returns
    ----------
    string
        The created regex.
    
    Examples
    ---------
    >>> make_regex(left_bound = "_C_", right_bound="_EMG.csv", values = [0,1,2,3,4,5])
    """
    left_bound_str = "(?<="+ left_bound +")"
    mid_str = "["
    for i in values:
        mid_str += i + "|"
    mid_str = mid_str[:-1]
    mid_str += "]+"
    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str

def mock_emg_stream(file_path, num_channels, sampling_rate=100, port=12345, ip="127.0.0.1"):
    """Streams EMG from a test file over TCP.

    This function can be used to simulate raw EMG being streamed over a TCP port. The main purpose 
    of this function would be to explore real-time interactions without the need for a physical 
    device. Note: This will start up a seperate process to stream data over. Additionally, 
    this uses the time module and as such the sampling rate may not be perfect.

    Parameters
    ----------
    file_path: string
        The path of the csv file where the EMG data is located. 
    num_channels: int
        The number of channels to stream. This should be <= to 
        the number of columns in the CSV.
    sampling_rate: int (optional), default=100
        The desired sampling rate.
    port: int (optional), defaul=12345
        The desired port to stream over. 
    ip: string (option), default = '127.0.0.1'
        The ip used for streaming predictions over TCP.
    
    Examples
    ----------
    >>> mock_emg_stream("stream_data.csv", num_channels=8, sampling_rate=100)
    """
    Process(target=_stream_thread, args=(file_path, num_channels, sampling_rate, port, ip), daemon=True).start()

def _stream_thread(file_path, num_channels, sampling_rate, port, ip):
    data = np.loadtxt(file_path, delimiter=",")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    index = 0
    while True and index < len(data):
        _ = time.perf_counter() + (1000/sampling_rate)/1000
        while time.perf_counter() < _:
            pass
        sock.sendto(bytes(str(list(data[index][:num_channels])), "utf-8"), (ip, port))
        index += 1
