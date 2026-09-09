import gc
import os

import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


def _release_interactive_plot(show):
    """Run something that opens an interactive plot, and dispose of it here.

    An interactive matplotlib window is a pile of Tk widgets -- around twenty
    ``tkinter.Variable`` and ``tkinter.PhotoImage`` objects per figure, most of
    them the toolbar. Closing the window does not finalize them: they become
    ordinary garbage, freed whenever some thread next triggers a collection. If
    that thread is not the one running the Tk main loop, every single ``__del__``
    marshals its Tcl call to the main loop, waits a full second for a loop that
    has already exited, then gives up with ``RuntimeError: main thread is not in
    main loop``. Twenty objects is twenty seconds of that thread stopped dead --
    long enough for a shared-memory logger to miss an entire recording, since
    what the device produces meanwhile is overwritten before it can be read.

    Giving the plot a call frame of its own is what makes the cleanup possible.
    The figure and every closure over it die with that frame, so by the time
    ``show`` has returned there is nothing left holding the window together and
    the collection below finalizes all of it here, on the thread that owns the
    Tk main loop, where the calls are direct and immediate.

    Parameters
    ----------
    show: callable
        Builds the figure, shows it, and closes it (``matplotlib.pyplot.close``)
        before returning. It must not hand the figure back or store it on
        anything that outlives the call, or the window survives the collection
        and is left for a worker thread to trip over.

    Returns
    ----------
    result: object
        Whatever ``show`` returned.
    """
    try:
        return show()
    finally:
        gc.collect()


def get_windows(data, window_size, window_increment):
    """Extracts windows from a given set of data.

    Parameters
    ----------
    data: list
        An NxM stream of data with N samples and M channels
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.

    Returns
    ----------
    list
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
    for _ in range(num_windows):
        if data.ndim == 1:
            windows.append([data[st_id:ed_id].transpose()]) # One Channel EMG
        else:
            windows.append(data[st_id:ed_id,:].transpose())
        st_id += window_increment
        ed_id += window_increment
    return np.array(windows)

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
