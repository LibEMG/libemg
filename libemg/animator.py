from pathlib import Path
import os
from typing import Callable, Sequence
import warnings

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
import cv2


class Animator:
    def __init__(self, output_filepath: str = 'libemg.gif', fps: int = 24):
        """Animator object for creating .gif files from a list of images.
        
        Parameters
        ----------
        output_filepath: string (optional), default='libemg.gif'
            Path to output file.
        fps: int (optional), default=24
            Frames per second of output file.
        """
        self.output_filepath = output_filepath
        self.fps = fps
        self.duration = 1000 // fps  # milliseconds per frame
        _, video_format = os.path.splitext(output_filepath) # set format to file extension
        self.video_format = video_format
    
    def convert_time_to_frames(self, duration_seconds: float):
        """Calculate the number of frames from the desired duration.
        
        Parameters
        ----------
        duration_seconds: float
            Duration of video in seconds.
        
        Returns
        ----------
        int
            Number of frames for given duration.
        """
        return duration_seconds * self.fps
    
    def save_mp4(self, frames: Sequence[Image.Image]):
        """Save a .mp4 video file from a list of images.


        Parameters
        ----------
        frames: list
            List of frames, where each element is a PIL.Image object.
        """
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video = cv2.VideoWriter(self.output_filepath, fourcc, fps=self.fps, frameSize=frames[0].size)
        
        for frame in frames:
            img = frame.copy()
            bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video.write(bgr_img)
        video.release()
    
    def save_video(self, frames: Sequence[Image.Image]):
        """Save a video file from a list of images.


        Parameters
        ----------
        frames: list
            List of frames, where each element is a PIL.Image object.
        """
        if self.video_format == '.gif':
            # Make .gif from frames
            self.save_gif(frames)
        elif self.video_format == '.mp4':
            # Make .mp4 from frames
            self.save_mp4(frames)
        else:
            # Unrecognized format
            raise ValueError(f'Unrecognized format {format}.')
        
    def save_gif(self, frames: Sequence[Image.Image]):
        """Save a .gif video file from a list of images.


        Parameters
        ----------
        frames: list
            List of frames, where each element is a PIL.Image object.
        """
        frames[0].save(
            self.output_filepath,
            save_all=True,
            append_images=frames[1:],   # append remaining frames
            format='GIF',
            duration=self.duration,
            loop=0  # infinite loop
        )
    
    def save_video_from_directory(self, directory_path: str, match_filename_function: Callable[[str], bool] | None = None, 
                                  delete_images: bool = False):
        """Save a video file from image files in a specified directory. Accepts all image types that can be read using
        PIL.Image.open(). Appends images in alphabetical order.


        Parameters
        ----------
        directory_path: string
            Path to directory that contains images.
        match_filename_function: Callable or None (optional), default=None
            Match function that determines which images in directory to use to create video. The match function should only expect a filename
            as a parameter and return True if the image should be used to create the video, otherwise it should return False. 
            If None, reads in all images in the directory.
        delete_images: bool (optional), default=False
            True if images used to create video should be deleted, otherwise False.
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
        
        self.save_video(frames)

        if delete_images:
            # Delete all images used to create .gif
            for filename in matching_filenames:
                os.remove(filename)


class PlotAnimator(Animator):
    def __init__(self, output_filepath: str ='libemg.gif', fps: int = 24, show_direction: bool = False, show_countdown: bool = False, show_boundary: bool = False,
                 figsize: tuple[int, int] = (6, 6), dpi: int = 80, tpd: int = 2):
        """Animator object specifically for plots.
        
        Parameters
        ----------
        output_filepath: string (optional), default='libemg.gif'
            Path to output file.
        fps: int (optional), default=24
            Frames per second of output file.
        show_direction: bool (optional), default=False
            True if the direction of the icon should be displayed as a faded icon, otherwise False.
        show_countdown: bool (optional), default=False
            True if a countdown should be displayed below the target, otherwise False.
        show_boundary: bool (optional), default=False
            True if a circle of radius 1 should be displayed as boundaries, otherwise False.
        figsize: tuple (optional), default=(6, 6)
            Size of figure in inches.
        dpi: int (optional), default=80
            Dots per inch of figure.
        tpd: int (optional), default=2
            Time (in seconds) for icon to travel a distance of 1.
        """
        super().__init__(output_filepath, fps)
        self.show_direction = show_direction
        self.show_countdown = show_countdown
        self.show_boundary = show_boundary
        self.figsize = figsize
        self.dpi = dpi
        self.tpd = tpd
        self.fpd = fps * self.tpd  # number of frames to generate to travel a distance of 1
    
    
    def convert_distance_to_frames(self, coordinates1: npt.NDArray[np.float_], coordinates2: npt.NDArray[np.float_]):
        """Calculate the number of frames needed to move from coordinates1 to coordinates2.
        
        Parameters
        ----------
        coordinates1: numpy.ndarray
            1D array where each element corresponds to the value along a different DOF.
        coordinates2: numpy.ndarray
            1D array where each element corresponds to the value along a different DOF. DOFs must line up
            with coordinates1.
        
        Returns
        ----------
        int
            Number of frames to travel for given distance.
        """
        distance = np.linalg.norm(coordinates2 - coordinates1)  # calculate euclidean distance between two points
        return int(distance * self.fpd)
    
    @staticmethod
    def _normalize_to_unit_distance(x: npt.NDArray[np.float_], y: npt.NDArray[np.float_]):
        """Normalize coordinates to a unit circle distance.
        
        Parameters
        ----------
        x: numpy.ndarray
            1D array of x coordinates.
        y: numpy.ndarray
            1D array of y coordinates.
        
        Returns
        ----------
        numpy.ndarray
            N x 2 matrix, where N is the number of coordinates. Each row contains the normalized x and y coordinates.
        """
        assert x.shape == y.shape, "x and y must be the same length."
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)    # ignore divide by 0 warnings
            # Calculate angles
            angles = np.arctan(y / x)
            np.arctan2(y, x, angles, where=np.isnan(angles))    # only replace where angles are nan
            
            # Normalize by angles (max distance of 1)
            normalized_x = x * np.abs(np.cos(angles))   # absolute value so the direction of the original coordinates is not changed
            normalized_y = y * np.abs(np.sin(angles))   # absolute value so the direction of the original coordinates is not changed
            normalized_coordinates = np.concatenate((normalized_x.reshape(-1, 1), normalized_y.reshape(-1, 1)), axis=1)
        
        assert normalized_coordinates.shape == (x.shape[0], 2)
        return normalized_coordinates

    @staticmethod
    def _convert_plot_to_image(fig: Figure):
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
    
    def _format_figure(self):
        """Set Figure to desired format."""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = plt.gca()
        return fig, ax
    
    def _preprocess_coordinates(self, coordinates: npt.NDArray[np.float_]):
        """Modify coordinates before plotting (e.g., normalization).
        
        Parameters
        ----------
        coordinates: numpy.ndarray
            N x M matrix, where N is the number of frames and M is the number of DOFs. Order is x-axis, y-axis, and third DOF (either rotation or target radius).
            Each row contains the value for x position, y position, and / or third DOF depending on how many DOFs are passed in.
        """
        return coordinates
    
    def _show_boundary(self):
        """Plot boundary to axis."""
        pass    # going to be different for each implementation, so don't implement it here

    def _show_countdown(self, coordinates: npt.NDArray[np.float_], text: str):
        """Show a countdown based on the current coordinates and frame index.
        
        Parameters
        ----------
        coordinates: numpy.ndarray
            2 item array. The first element is the x-coordinate and the second element is the y-coordinate.
        text: str
            Text to show.
        """
        plt.text(coordinates[0], coordinates[1], text, fontweight='bold', c='red', ha='center', va='center')
    
    def _show_direction(self, coordinates: npt.NDArray[np.float_], alpha: float = 1.0):
        """Show the direction of the next part of the movement.
        
        Parameters
        ----------
        coordinates: numpy.ndarray
            1 x M matrix, where M is the number of DOFs. Order is x-axis, y-axis, and third DOF (either rotation or target radius).
            Each row contains the value for x position, y position, and / or third DOF depending on how many DOFs are passed in.
        alpha: float (optional), default=1.0
            Alpha value of icon.
        """
        self.plot_icon(coordinates, alpha=alpha, colour='green')

    
    def plot_icon(self, coordinates: npt.NDArray[np.float_], alpha: float = 1.0, colour: str = 'black'):
        """Plot target / icon on axis.
        
        Parameters
        ----------
        coordinates: numpy.ndarray
            1D array where each element corresponds to the value along a different DOF.
        alpha: float (optional), default=1.0
            Alpha value of icon.
        colour: string (optional), default='black'
            Colour of icon.
        """
        plt.plot(coordinates[0], coordinates[1], alpha=alpha, c=colour)
    
    
    def save_plot_video(self, coordinates: npt.NDArray[np.float_], title: str = '', xlabel: str = '', ylabel: str = '', save_coordinates: bool = False, verbose: bool = False):
        """Save a video file of an icon moving around a 2D plane.
        
        Parameters
        ----------
        coordinates: numpy.ndarray
            N x M matrix, where N is the number of frames and M is the number of DOFs. Order is x-axis, y-axis, and third DOF (either rotation or target radius).
            Each row contains the value for x position, y position, and / or third DOF depending on how many DOFs are passed in.
        title: string (optional), default=''
            Title of plot.
        xlabel: string (optional), default=''
            Label for x-axis.
        ylabel: string (optional), default=''
            Label for y-axis.
        save_coordinates: bool (optional), default=False
            True if coordinates should be saved to a .txt file for ground truth values, otherwise False.
        show_direction: bool (optional), default=False
            True if the direction of the icon should be displayed as a faded icon, otherwise False.
        show_countdown: bool (optional), default=False
            True if a countdown should be displayed below the target, otherwise False.
        show_boundary: bool (optional), default=False
            True if a circle of radius 1 should be displayed as boundaries, otherwise False.
        verbose: bool (optional), default=False
            True if progress should be printed to console, otherwise False.
        """
        if save_coordinates:
            # Save coordinates in .txt file
            labels_filepath = Path(self.output_filepath).with_suffix('.txt')
            labels_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(labels_filepath, coordinates, delimiter=',')
        
        # Format figure
        fig, ax = self._format_figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig.suptitle(title)
        fig.tight_layout()

        split_steady_state_indices = None
        steady_state_start_indices = None
        steady_state_end_indices = None
        if self.show_direction or self.show_countdown:
            # Calculate steady states
            diff = np.diff(coordinates, axis=0)
            max_diff = np.max(np.abs(diff), axis=1)
            all_steady_state_indices = np.where(max_diff < 0.01)[0] # find where the differences at each frame is less than some threshold
            try:
                # Only take starts and ends of each segment
                split_steady_state_indices = np.split(all_steady_state_indices, np.where(np.diff(all_steady_state_indices) != 1)[0] + 1)    # add 1 to align diff with original
                split_steady_state_indices.append(np.array([coordinates.shape[0] - 1])) # append last frame
                steady_state_start_indices = np.array([segment[0] for segment in split_steady_state_indices])
                steady_state_end_indices = np.array([segment[-1] for segment in split_steady_state_indices])
            except IndexError as e:
                raise IndexError('Could not find steady state frames. If these features are desired, please pass in steady state frames (i.e., consecutive frames with the same value).') from e
                
        # Adjust coordinates if desired
        coordinates = self._preprocess_coordinates(coordinates)
        
        frames = []
        current_steady_state_idx = 0
        target_alpha = 0.05
        for frame_idx, frame_coordinates in enumerate(coordinates):
            if verbose and frame_idx % 10 == 0:
                print(f'Frame {frame_idx} / {coordinates.shape[0]}')
            
            # Plot additional information
            if self.show_boundary:
                # Show boundaries
                self._show_boundary()
            
            if (self.show_direction or self.show_countdown) and split_steady_state_indices is not None and steady_state_start_indices is not None and steady_state_end_indices is not None:
                # Calculate next steady state frame
                next_steady_state_idx = min(current_steady_state_idx, len(steady_state_end_indices) - 1)    # limit max index
                if frame_idx > steady_state_end_indices[current_steady_state_idx]:
                    current_steady_state_idx += 1
                    target_alpha = 0.05 # reset alpha
                
                if self.show_direction:
                    # Show path until a change in direction
                    next_steady_state_idx = current_steady_state_idx + 1 if frame_idx > steady_state_start_indices[current_steady_state_idx] else current_steady_state_idx
                    next_steady_state_start = steady_state_start_indices[next_steady_state_idx]
                    target_alpha += 0.01    # add in fade
                    target_alpha = min(0.4, target_alpha) # limit alpha to 0.4
                    self._show_direction(coordinates[next_steady_state_start], alpha=target_alpha)
                    
                if self.show_countdown:
                    # Show countdown during steady state
                    steady_state_end = steady_state_end_indices[current_steady_state_idx]
                    time_until_movement = (steady_state_end - frame_idx) * self.duration / 1000   # convert from frames to seconds
                    if time_until_movement >= 0.25 and frame_idx in split_steady_state_indices[current_steady_state_idx]:
                        # Only show countdown if the steady state is longer than 1 second
                        self._show_countdown(frame_coordinates, str(int(time_until_movement)))
                
            # Plot icon
            self.plot_icon(frame_coordinates)
                
            # Save frame
            frame = self._convert_plot_to_image(fig)
            frames.append(frame)
            # Clear axis while retaining formatting
            for artist in ax.lines + ax.collections + ax.patches + ax.texts:
                artist.remove()
        
        # Save file
        self.save_video(frames)


class CartesianPlotAnimator(PlotAnimator):
    def __init__(self, output_filepath: str = 'libemg.gif', fps: int = 24, show_direction: bool = False, show_countdown: bool = False, show_boundary: bool = False, normalize_distance: bool = False,
                 axis_images: dict[str, Image.Image] | None = None, figsize: tuple[int, int] = (6, 6), dpi: int = 80, tpd: int = 2):
        """Animator object for creating video files from a list of coordinates on a cartesian plane.
        
        Parameters
        ----------
        output_filepath: string (optional), default='libemg.gif'
            Path to output file.
        fps: int (optional), default=24
            Frames per second of output file.
        show_direction: bool (optional), default=False
            True if the direction of the icon should be displayed as a faded icon, otherwise False.
        show_countdown: bool (optional), default=False
            True if a countdown should be displayed below the target, otherwise False.
        show_boundary: bool (optional), default=False
            True if a circle of radius 1 should be displayed as boundaries, otherwise False.
        normalize_distance: bool (optional), default=False
            True if the distance between each coordinate should be normalized to 1, otherwise False.
        axis_images: dict (optional), default=None
            Dictionary mapping compass directions to images. Images will be displayed in the corresponding compass direction (i.e., 'N' correponds to the top of the image).
            Valid keys are 'NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE'. If None, no images will be displayed.
        figsize: tuple (optional), default=(6, 6)
            Size of figure in inches.
        dpi: int (optional), default=80
            Dots per inch of figure.
        tpd: int (optional), default=2
            Time (in seconds) for icon to travel a distance of 1.
        """
        super().__init__(output_filepath, fps, show_direction, show_countdown, show_boundary, figsize, dpi, tpd)
        self.normalize_distance = normalize_distance
        self.axis_images = axis_images
    
    def _format_figure(self):
        fig, ax = super()._format_figure()
        axis_limits = (-1.25, 1.25)
        if self.axis_images is not None:
            ax.axis('off')  # hide default axis
            axs = self._add_image_label_axes(fig)
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
            for loc, image in self.axis_images.items():
                ax = loc_axis_map[loc]
                ax.imshow(image)
            # Set main axis so icon is drawn correctly
            plt.sca(axs[1, 1])    
            ax = axs[1, 1]
        
        ticks = [-1., -0.5, 0, 0.5, 1.]
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.axis('equal')
        ax.set(xlim=axis_limits, ylim=axis_limits)
        return fig, ax
    
    def _preprocess_coordinates(self, coordinates: npt.NDArray[np.float_]):
        coordinates = super()._preprocess_coordinates(coordinates)

        if self.normalize_distance:
            # Normalize to unit circle distance
            coordinates = self._normalize_to_unit_distance(coordinates[:, 0], coordinates[:, 1])
        return coordinates
    
    def _show_boundary(self):
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an), 'b--', alpha=0.7)
    
    def _show_countdown(self, coordinates: npt.NDArray[np.float_], text: str):
        x = coordinates[0]
        y = coordinates[1] - 0.2
        return super()._show_countdown((x, y), text)
    
    @staticmethod
    def _add_image_label_axes(fig: Figure):
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



class ScatterPlotAnimator(CartesianPlotAnimator):
    def __init__(self, output_filepath: str = 'libemg.gif', fps: int = 24, show_direction: bool = False, show_countdown: bool = False, show_boundary: bool = False, normalize_distance: bool = False, 
                 axis_images: dict[str, Image.Image] | None = None, plot_line: bool = False, figsize: tuple[int, int] = (6, 6), dpi: int = 80, tpd: int = 2):
        """Animator object for creating video files from a list of coordinates on a cartesian plane shown as a scatter plot.
        
        Parameters
        ----------
        output_filepath: string (optional), default='libemg.gif'
            Path to output file.
        fps: int (optional), default=24
            Frames per second of output file.
        show_direction: bool (optional), default=False
            True if the direction of the icon should be displayed as a faded icon, otherwise False.
        show_countdown: bool (optional), default=False
            True if a countdown should be displayed below the target, otherwise False.
        show_boundary: bool (optional), default=False
            True if a circle of radius 1 should be displayed as boundaries, otherwise False.
        normalize_distance: bool (optional), default=False
            True if the distance between each coordinate should be normalized to 1, otherwise False.
        axis_images: dict (optional), default=None
            Dictionary mapping compass directions to images. Images will be displayed in the corresponding compass direction (i.e., 'N' correponds to the top of the image).
            Valid keys are 'NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE'. If None, no images will be displayed.
        plot_line: bool (optional), default=False
            True if a line should be plotted between the origin and the current point, otherwise False.
        figsize: tuple (optional), default=(6, 6)
            Size of figure in inches.
        dpi: int (optional), default=80
            Dots per inch of figure.
        tpd: int (optional), default=2
            Time (in seconds) for icon to travel a distance of 1.
        """
        super().__init__(output_filepath, fps, show_direction, show_countdown, show_boundary, normalize_distance, axis_images, figsize, dpi, tpd)
        self.plot_line = plot_line

    
    def plot_icon(self, coordinates: npt.NDArray[np.float_], alpha: float = 1.0, colour: str = 'black'):
        # Parse coordinates
        x = coordinates[0]
        y = coordinates[1]
        # Dot properties
        size = 50
        plt.scatter(x, y, s=size, c=colour, alpha=alpha)
        if self.plot_line and alpha == 1.0:
            # TODO: Add a _is_target() method to detect this
            # Plot line to current point, but not to target
            plt.plot([0, x], [0, y], c=colour, linewidth=5)


class ArrowPlotAnimator(CartesianPlotAnimator):
    def plot_icon(self, coordinates: npt.NDArray[np.float_], alpha: float = 1.0, colour: str = 'black'):
        # Parse coordinates
        x_tail = coordinates[0]
        y_tail = coordinates[1]
        angle = coordinates[2]

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


class TargetPlotAnimator(CartesianPlotAnimator):
    @staticmethod
    def _plot_circle(xy: tuple[float, float], radius: float, edgecolor: str, facecolor: str, alpha: float = 1.0):
        circle = Circle(xy, radius=radius, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
        plt.gca().add_patch(circle)
    
    def plot_icon(self, coordinates: npt.NDArray[np.float_], alpha: float = 1.0, colour: str = 'black'):
        # Parse coordinates
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]

        min_radius = 0.05
        max_radius = 0.2
        radius = np.interp(z, [-1, 1], [min_radius, max_radius])  # map z value from [-1, 1] to actual limits of target
        
        # Plot target
        xy = (x, y)
        limit_alpha = 0.4
        TargetPlotAnimator._plot_circle(xy, radius=radius, edgecolor='none', facecolor=colour, alpha = alpha) # plot target
        TargetPlotAnimator._plot_circle(xy, radius=max_radius, edgecolor='black', facecolor='none', alpha=limit_alpha)   # plot max boundary
        TargetPlotAnimator._plot_circle(xy, radius=min_radius, edgecolor='black', facecolor='black', alpha=limit_alpha)   # plot min boundary


class BarPlotAnimator(PlotAnimator):
    def __init__(self, bar_labels: Sequence[str], output_filepath: str = 'libemg.gif', fps: int = 24, show_direction: bool = False, show_countdown: bool = False, show_boundary: bool = False,
                 figsize: tuple[int, int] = (6, 6), dpi: int = 80, tpd: int = 2):
        """Animator object for creating video files from a list of coordinates on a cartesian plane shown as a bar plot.
        
        Parameters
        ----------
        bar_labels: list
            List of labels for each bar.
        output_filepath: string (optional), default='libemg.gif'
            Path to output file.
        fps: int (optional), default=24
            Frames per second of output file.
        show_direction: bool (optional), default=False
            True if the direction of the icon should be displayed as a faded icon, otherwise False.
        show_countdown: bool (optional), default=False
            True if a countdown should be displayed below the target, otherwise False.
        show_boundary: bool (optional), default=False
            True if a circle of radius 1 should be displayed as boundaries, otherwise False.
        normalize_distance: bool (optional), default=False
            True if the distance between each coordinate should be normalized to 1, otherwise False.
        axis_images: dict (optional), default=None
            Dictionary mapping compass directions to images. Images will be displayed in the corresponding compass direction (i.e., 'N' correponds to the top of the image).
            Valid keys are 'NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE'. If None, no images will be displayed.
        plot_line: bool (optional), default=False
            True if a line should be plotted between the origin and the current point, otherwise False.
        figsize: tuple (optional), default=(6, 6)
            Size of figure in inches.
        dpi: int (optional), default=80
            Dots per inch of figure.
        tpd: int (optional), default=2
            Time (in seconds) for icon to travel a distance of 1.
        """
        super().__init__(output_filepath, fps, show_direction, show_countdown, show_boundary, figsize, dpi, tpd)
        self.bar_labels = bar_labels
        self.bar_width = 0.4
    
    def _format_figure(self):
        fig, ax = super()._format_figure()
        axis_limits = (-1.25, 1.25)
        ax.set(ylim=axis_limits)
        return fig, ax
    
    def _plot_border(self, coordinates: npt.NDArray[np.float_], edgecolor: str = 'black'):
        plt.bar(self.bar_labels, coordinates, color='none', edgecolor=edgecolor, linewidth=2, width=self.bar_width)
    
    def _show_direction(self, coordinates: npt.NDArray[np.float_], alpha: float = 1):
        self._plot_border(coordinates, edgecolor='green')
        
    def _show_countdown(self, coordinates: npt.NDArray[np.float_], text: str):
        adjustment = 0.05
        for label, dof_value in zip(self.bar_labels, coordinates):
            modifier = -adjustment if dof_value < 0 else adjustment
            super()._show_countdown((label, dof_value + modifier), text)
    
    def _show_boundary(self):
        self._plot_border(-1)
        self._plot_border(1)
    
    def plot_icon(self, coordinates: npt.NDArray[np.float_], alpha: float = 1, colour: str = 'black'):
        plt.bar(self.bar_labels, coordinates, alpha=alpha, color=colour, width=self.bar_width)

        axis_limits = plt.gca().get_ylim()
        for label in self.bar_labels:
            try:
                negative_label, positive_label = label.split(' / ')
                plt.text(label, axis_limits[0] + 0.1, negative_label, ha='center', va='center')
                plt.text(label, axis_limits[1] - 0.1, positive_label, ha='center', va='center')
            except ValueError:
                break


class SingleDirectionBarPlotAnimator(BarPlotAnimator):
    def _format_figure(self):
        fig, ax = super()._format_figure()
        ax.set(ylim=(0, 1.25)) # set limits only positive instead of both directions
        return fig, ax
