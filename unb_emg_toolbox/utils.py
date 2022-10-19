import numpy as np

def get_windows(data, window_size, window_increment):
    """Extracts windows from a given set of data.

    Parameters:
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

def get_mode_windows(data, window_size, window_increment):
    """Calculate the mode of values seen across a set of windows.

    Rather than capturing an entire window, this function gets the mode of values seen across samples in the window.
    This is useful for metadata.

    Parameters:
    ----------
    data: array_like
        An NxM stream of data with N samples and M channels.
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.

    Returns
    ----------
    array_like
        The mode calculated across a sequence of windows.
    """
    windows = get_windows(data, window_size, window_increment)
    # we want to get the mode along the final dimension
    mode_of_windows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=windows.astype(np.int64))
    
    return mode_of_windows.squeeze()

def make_regex(left_bound, right_bound, values=[]):
    """Regex creation helper for the data handler.

    The Data Handler relies on regexes to parse the file/folder structures and extract data. 
    This function makes the creation of regexes easier.

    Parameters:
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
    """
    left_bound_str = "(?<="+ left_bound +")"
    mid_str = "["
    for i in values:
        mid_str += i + "|"
    mid_str = mid_str[:-1]
    mid_str += "]+"
    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str