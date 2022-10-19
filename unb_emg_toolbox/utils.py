from asyncio import WindowsProactorEventLoopPolicy
import numpy as np

def get_windows(data, window_size, window_increment):
        '''
        data is a NxM stream of data with N samples and M channels (numpy array)
        window_size is number of samples in window
        window_increment is number of samples that advances before the next window
        '''
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
    '''
    data is a NxM stream of data with N samples and M channels (numpy array)
    window_size is a number of samples in window
    window_increment is number of samples that advances before next window
    
    Rather than capturing an entire window, this function gets the mode of values seen across samples in the window.
    This is useful for metadata
    '''
    windows = get_windows(data, window_size, window_increment)
    # we want to get the mode along the final dimension
    mode_of_windows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=windows.astype(np.int64))
    
    return mode_of_windows.squeeze()

def make_regex(left_bound, right_bound, values=[]):
    left_bound_str = "(?<="+ left_bound +")"
    mid_str = "["
    for i in values:
        mid_str += i + "|"
    mid_str = mid_str[:-1]
    mid_str += "]+"
    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str