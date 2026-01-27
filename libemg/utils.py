import os

import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


def get_windows(data, window_size, window_increment, channel_last=False):
    """Extracts windows from a given set of data.

    Parameters
    ----------
    data: list
        An NxM stream of data with N samples and M channels
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    channel_last: bool, default=False
        Output will be NxLxC if True.

    Returns
    ----------
    list
        The set of windows extracted from the data as a NxCxL where N is the number of windows, C is the number of channels 
        and L is the length of each window. Output will be NxLxC if channel_last is True.

    Examples
    ---------
    >>> data = np.loadtxt('data.csv', delimiter=',')
    >>> windows = get_windows(data, 100, 50)
    """
    data = np.array(data)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=-1)

    T = data.shape[0]
    starts = np.arange(0, T - window_size + 1, window_increment)
    idx = starts[:, None] + np.arange(window_size)[None, :]

    windows = data[idx]
    if not channel_last:
        windows = np.transpose(windows, (0, 2, 1))

    return windows

def _get_mode_windows(data, window_size, window_increment):
    windows = get_windows(data, window_size, window_increment)
    # we want to get the mode along the final dimension
    mode_of_windows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=windows.astype(np.int64))
    
    return mode_of_windows.squeeze()

def _get_fn_windows(data, window_size, window_increment, fn):
    windows = get_windows(data, window_size, window_increment)
    # we want to apply the function along the final dimension
    
    if type(fn) is list:
        fn_of_windows = windows
        for i in fn:
            fn_of_windows = np.apply_along_axis(lambda x: i(x), axis=2, arr=fn_of_windows)
    else:
        fn_of_windows = np.apply_along_axis(lambda x: fn(x), axis=2, arr=windows)
    return fn_of_windows.squeeze()

def make_regex(left_bound, right_bound, values = None):
    """Regex creation helper for the data handler.

    The OfflineDataHandler relies on regexes to parse the file/folder structures and extract data. 
    This function makes the creation of regexes easier.

    Parameters
    ----------
    left_bound: string
        The left bound of the regex.
    right_bound: string
        The right bound of the regex.
    values: list or None (optional), default = None
        The values between the two regexes. If None, will try to find the values using a wildcard. Defaults to None.

    Returns
    ----------
    string
        The created regex.
    
    Examples
    ---------
    >>> make_regex(left_bound = "_C_", right_bound="_EMG.csv", values = [0,1,2,3,4,5])
    """
    left_bound_str = "(?<="+ left_bound +")"

    if values is None:
        # Apply wildcard
        mid_str = '(.*?)'
    else:
        mid_str = "(?:"
        for i in values:
            mid_str += i + "|"
        mid_str = mid_str[:-1]
        mid_str += ")"

    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str
