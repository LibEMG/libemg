import os

import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


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
    """Save a .gif video file from image files in a specified directory. Accepts all image types that can be read using
    PIL.Image.open(). Appends images in alphabetical order.


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
    filenames.sort()    # sort alphabetically
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
            
def _convert_plot_to_image(fig):
    """Convert a matplotlib Figure to a PIL.Image object.

    Parameters
    ----------
    fig: matplotlib.pyplot.Figure
        Figure that should be saved as an image.
    
    Returns
    ----------
    PIL.Image
        Matplotlib Figure as a PIL.Image.
    """
    canvas = FigureCanvasAgg(fig)
    
    # Get RGBA buffer
    canvas.draw()
    rgba_buffer = canvas.buffer_rgba()

    # Convert the buffer to a PIL Image
    return Image.frombytes('RGBA', canvas.get_width_height(), rgba_buffer)

def _add_image_label_axes(fig):
    """Add axes to a matplotlib Figure for displaying images in the top, right, bottom, and left of the Figure. 
    
    Parameters
    ----------
    fig: matplotlib.pyplot.Figure
        Figure to add image axes to.
    
    Returns
    ----------
    np.ndarray
        Array of matplotlib axes objects. The location in the array corresponds to the location of the axis in the figure.
    """
    # Make 3 x 3 grid
    grid_shape = (3, 3)
    gs = fig.add_gridspec(grid_shape[0], grid_shape[1], width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

    # Create subplots using the gridspec
    axs = np.empty(shape=grid_shape, dtype=object)
    for row_idx in range(grid_shape[0]):
        for col_idx in range(grid_shape[1]):
            ax = plt.subplot(gs[row_idx, col_idx])
            if (row_idx, col_idx) != (1, 1):
                # Disable axis for images, not for main plot
                ax.axis('off')
            axs[row_idx, col_idx] = ax
    
    return axs


def make_regression_training_gif(coordinates, output_filepath = 'libemg.gif', duration = 100, title = '', xlabel = '', ylabel = '', axis_images = None, save_coordinates = False,
                                 third_dof_display = 'size', show_direction = False, show_countdown = False, verbose = False):
    """Save a .gif file of an icon moving around a 2D plane. Can be used for regression training.
    
    Parameters
    ----------
    coordinates: numpy.ndarray
        N x M matrix, where N is the number of frames and M is the number of DOFs. Order is x-axis, y-axis, and third DOF (either rotation or target radius).
        Each row contains the value for x position, y position, and / or third DOF depending on how many DOFs are passed in.
    output_filepath: string (optional), default='libemg.gif'
        Filepath of output file.
    duration: int (optional), default=100
        Duration of each frame in milliseconds.
    title: string (optional), default=''
        Title of plot.
    xlabel: string (optional), default=''
        Label for x-axis.
    ylabel: string (optional), default=''
        Label for y-axis.
    axis_images: dict (optional), default=None
        Dictionary mapping compass directions to images. Images will be displayed in the corresponding compass direction (i.e., 'N' correponds to the top of the image).
        Valid keys are 'NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE'. If None, no images will be displayed.
    save_coordinates: bool (optional), default=False
        True if coordinates should be saved to a .txt file for ground truth values, otherwise False.
    third_dof_display: string (optional), default='size'
        Determines how the third DOF is displayed. Valid values are 'size' (third DOF is target size), 'rotation' (third DOF is rotation in degrees).
    show_direction: bool (optional), default=False
        True if the direction of the icon should be displayed as a faded icon, otherwise False.
    show_countdown: bool (optional), default=False
        True if a countdown should be displayed below the target, otherwise False.
    """
    # Plotting functions
    def plot_circle(xy, radius, edgecolor, facecolor, alpha = 1.0):
        circle = Circle(xy, radius=radius, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
        plt.gca().add_patch(circle)
        
    def plot_dot(frame_coordinates, alpha = 1.0, colour = 'black'):
        # Parse coordinates
        x = frame_coordinates[0]
        y = frame_coordinates[1]
        # Dot properties
        size = 50
        plt.scatter(x, y, s=size, c=colour, alpha=alpha)
    
    def plot_arrow(frame_coordinates, alpha = 1.0, colour = 'black'):
        # Parse coordinates
        x_tail = frame_coordinates[0]
        y_tail = frame_coordinates[1]
        angle = frame_coordinates[2]

        # Map from [-1, 1] to degrees
        angle_degrees = np.interp(angle, [-1, 1], [0, 360])
        # Convert angle to radians
        arrow_angle_radians = np.radians(angle_degrees)
        # Arrow properties
        arrow_length = 0.1
        head_size = 0.05
        # Calculate arrow head coordinates
        x_head = x_tail + arrow_length * np.cos(arrow_angle_radians)
        y_head = y_tail + arrow_length * np.sin(arrow_angle_radians)
        plt.arrow(x_tail, y_tail, x_head - x_tail, y_head - y_tail, head_width=head_size, head_length=head_size, fc=colour, ec=colour, alpha=alpha)
    
    def plot_target(frame_coordinates, alpha = 1.0, colour = 'red'):
        # Parse coordinates
        x = frame_coordinates[0]
        y = frame_coordinates[1]
        z = frame_coordinates[2]

        min_radius = 0.05
        max_radius = 0.2
        radius = np.interp(z, [-1, 1], [min_radius, max_radius])  # map z value from [-1, 1] to actual limits of target
        
        # Plot target
        xy = (x, y)
        limit_alpha = 0.4
        plot_circle(xy, radius=radius, edgecolor='none', facecolor=colour, alpha = alpha) # plot target
        plot_circle(xy, radius=max_radius, edgecolor='black', facecolor='none', alpha=limit_alpha)   # plot max boundary
        plot_circle(xy, radius=min_radius, edgecolor='black', facecolor='black', alpha=limit_alpha)   # plot min boundary
        
    plot_icon = plot_dot if coordinates.shape[1] == 2 else plot_arrow
    if coordinates.shape[1] == 2:
        # Plot a dot if 2 DOFs were passed in
        plot_icon = plot_dot
    elif third_dof_display == 'rotation':
        # Degrees passed in, so plot arrow
        plot_icon = plot_arrow
    elif third_dof_display == 'size':
        # Plot target of varying size
        plot_icon = plot_target
    else:
        # Unexpected format
        raise ValueError("Please pass in 'rotation' or 'size' for third_dof_display.")
    
    # Calculate direction changes
    direction_changes = np.sign(np.diff(coordinates, axis=0, n=1))[:-1] * np.diff(coordinates, axis=0, n=2) / duration
    direction_change_indices = np.where(np.abs(direction_changes) > 1e-8)[0] + 1 # add 1 to align with coordinates
    direction_change_indices = np.append(direction_change_indices, coordinates.shape[0] - 1)    # append final frame position
    
    # Format figure
    fig = plt.figure(figsize=(8, 8))
    axis_limits = (-1.25, 1.25)
    if axis_images is not None:
        axs = _add_image_label_axes(fig)
        loc_axis_map = {
            'NW': axs[0, 0],
            'N': axs[0, 1],
            'NE': axs[0, 2],
            'W': axs[1, 0],
            'E': axs[1, 2],
            'SW': axs[2, 0],
            'S': axs[2, 1],
            'SE': axs[2, 2]
        }
        for loc, image in axis_images.items():
            ax = loc_axis_map[loc]
            ax.imshow(image)
        # Set main axis so icon is drawn correctly
        plt.sca(axs[1, 1])    
        ax = axs[1, 1]
    else:
        ax = plt.gca()
    fig.suptitle(title)
    fig.tight_layout()
        
    frames = []
    direction_change_idx = None
    target_alpha = 0.05
    for frame_idx, frame_coordinates in enumerate(coordinates):
        if verbose and frame_idx % 10 == 0:
            print(f'Frame {frame_idx} / {coordinates.shape[0]}')
        # Plot icon
        plot_icon(frame_coordinates)

         # Format axis
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ticks = [-1., -0.5, 0, 0.5, 1.]
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.axis('equal')
        ax.set(xlim=axis_limits, ylim=axis_limits)

        # Show boundaries
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an), 'b--', alpha=0.7)
        
        # Plot additional information
        if show_direction:
            # Show path until a change in direction
            nearest_direction_change_idx = np.where(direction_change_indices - frame_idx >= 0)[0][0]     # get nearest direction change frame that hasn't passed
            new_direction_change_idx = direction_change_indices[nearest_direction_change_idx]  
            if new_direction_change_idx != direction_change_idx:
                # Update value
                direction_change_idx = direction_change_indices[nearest_direction_change_idx]
                target_alpha = 0.05  # reset alpha
            else:
                # Add in fade
                target_alpha += 0.01
                target_alpha = min(0.4, target_alpha) # limit alpha to 0.4
            plot_icon(coordinates[direction_change_idx], alpha=target_alpha, colour='green')
        if show_countdown:
            # Show countdown below target
            try:
                matching_indices = np.where(np.all(coordinates == frame_coordinates, axis=1))[0]
                future_matching_indices = matching_indices[np.where(matching_indices >= frame_idx)[0]] # only look at indices that are in the future, not the past
                steady_state_indices = future_matching_indices[np.where(np.diff(future_matching_indices) != 1)[0]]
                if steady_state_indices.size > 0:
                    # Found end of current steady state
                    final_steady_state_idx = steady_state_indices[0]
                else:
                    # No other steady states at these coordinates, so just take the end of the current one
                    final_steady_state_idx = future_matching_indices[-1]
                time_until_movement = (final_steady_state_idx - frame_idx) * duration / 1000   # convert from frames to seconds
                if time_until_movement >= 0.25:
                    # Only show countdown if the steady state is longer than 1 second
                    plt.text(frame_coordinates[0] - 0.03, frame_coordinates[1] - 0.2, str(int(time_until_movement)), fontweight='bold', c='red')
            except IndexError:
                # Did not find steady state
                pass
            
        # Save frame
        frame = _convert_plot_to_image(fig)
        frames.append(frame)
        plt.cla()   # clear axis
            
       
    
    # Save file
    make_gif(frames, output_filepath=output_filepath, duration=duration)
    if save_coordinates:
        # Save coordinates in .txt file
        filename_no_extension = os.path.splitext(output_filepath)[0]
        labels_filepath = filename_no_extension + '.txt'
        np.savetxt(labels_filepath, coordinates, delimiter=',')
