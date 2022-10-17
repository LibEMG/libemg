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

def create_folder_dictionary(left_bound, right_bound, class_values, rep_values):
    classes_regex = _make_regex(left_bound="C_", right_bound="_E", values=class_values)
    reps_regex = _make_regex(left_bound="R_", right_bound="_C", values=rep_values)
    dic = {
        "classes": class_values,
        "reps":    rep_values,
        "classes_regex": classes_regex, #r"(?<=_)[^#]+(?=.txt)",# the # here means only a number is accepted... need to find a more wildcard solution
        "reps_regex": reps_regex # "(?<=e_)[]+(?=_)"
    }
    return dic

def _make_regex(left_bound, right_bound, values=[]):
    regex = ""
    left_bound_str = "(?<="+ left_bound +")"
    mid_str = "["
    for i in values:
        mid_str += i + "|"
    mid_str = mid_str[:-1]
    mid_str += "]+"
    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str