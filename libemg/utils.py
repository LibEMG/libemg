import os

import numpy as np
from PIL import Image, UnidentifiedImageError


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

def make_regex(left_bound, right_bound, values=[]):
    """Regex creation helper for the data handler.

    The OfflineDataHandler relies on regexes to parse the file/folder structures and extract data. 
    This function makes the creation of regexes easier.

    Parameters
    ----------
    left_bound: string
        The left bound of the regex.
    right_bound: string
        The right bound of the regex.
    values: list
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
    mid_str = "(?:"
    for i in values:
        mid_str += i + "|"
    mid_str = mid_str[:-1]
    mid_str += ")"
    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str

def make_gif(frames, output_filepath = 'libemg.gif', duration = 100):
    """Save a .gif video file from a list of images.


    Parameters
    ----------
    frames: list
        List of frames, where each element is a PIL.Image object.
    output_filepath: string (optional), default='libemg.gif'
        Filepath of output file.
    duration: int (optional), default=100
        Duration of each frame in milliseconds.
    
    """
    frames[0].save(
        output_filepath,
        save_all=True,
        append_images=frames[1:],   # append remaining frames
        format='GIF',
        duration=duration,
        loop=0  # infinite loop
    )

def make_gif_from_directory(directory_path, output_filepath = 'libemg.gif', match_filename_function = None, 
                            delete_images = False, duration = 100):
    """Save a .gif video file from image files in a specified directory. Accepts all images that be read using
    PIL.Image.open().


    Parameters
    ----------
    directory_path: string
        Path to directory that contains images.
    output_filepath: string (optional), default='libemg.gif'
        Filepath of output file.
    match_filename_function: Callable or None (optional), default=None
        Match function that determines which images in directory to use to create .gif. The match function should only expect a filename
        as a parameter and return True if the image should be used to create the .gif, otherwise it should return False. 
        If None, reads in all images in the directory.
    delete_images: bool (optional), default=False
        True if images used to create .gif should be deleted, otherwise False.
    duration: int (optional), default=100
        Duration of each frame in milliseconds.
    """
    if match_filename_function is None:
        # Combine all images in directory
        match_filename_function = lambda x: True
    frames = []
    filenames = os.listdir(directory_path)
    matching_filenames = [] # images used to create .gif

    for filename in filenames:
        absolute_path = os.path.join(directory_path, filename)
        if match_filename_function(filename):
            # File matches the user pattern and is an accepted image format
            try:
                image = Image.open(absolute_path)
                frames.append(image)
                matching_filenames.append(absolute_path)
            except UnidentifiedImageError:
                # Reading non-image file
                print(f'Skipping {absolute_path} because it is not an image file.')
    
    # Make .gif from frames
    make_gif(frames, output_filepath, duration=duration)

    if delete_images:
        # Delete all images used to create .gif
        for filename in matching_filenames:
            os.remove(filename)
