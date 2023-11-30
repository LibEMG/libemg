import os
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


class Animator:
    def __init__(self, output_filepath = 'libemg.gif', fps = 24):
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
    
    def convert_time_to_frames(self, duration_seconds):
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
        
    def save_gif(self, frames):
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
    
    def save_gif_from_directory(self, directory_path, match_filename_function = None, 
                            delete_images = False):
        """Save a .gif video file from image files in a specified directory. Accepts all image types that can be read using
        PIL.Image.open(). Appends images in alphabetical order.


        Parameters
        ----------
        directory_path: string
            Path to directory that contains images.
        match_filename_function: Callable or None (optional), default=None
            Match function that determines which images in directory to use to create .gif. The match function should only expect a filename
            as a parameter and return True if the image should be used to create the .gif, otherwise it should return False. 
            If None, reads in all images in the directory.
        delete_images: bool (optional), default=False
            True if images used to create .gif should be deleted, otherwise False.
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
        self.save_gif(frames)

        if delete_images:
            # Delete all images used to create .gif
            for filename in matching_filenames:
                os.remove(filename)


class PlotAnimator(Animator):
    def __init__(self, output_filepath='libemg.gif', fps=24, show_direction = False, show_countdown = False, show_boundary = False):
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
        """
        super().__init__(output_filepath, fps)
        self.show_direction = show_direction
        self.show_countdown = show_countdown
        self.show_boundary = show_boundary
        self.fpd = fps * 2  # number of frames to generate to travel a distance of 1
    
    
    def convert_distance_to_frames(self, coordinates1, coordinates2):
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
    
    def _format_figure(self):
        """Set Figure to desired format."""
        fig = plt.figure()
        ax = plt.gca()
        return fig, ax
    
    def _preprocess_coordinates(self, coordinates):
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

    def _show_countdown(self, coordinates, frame_idx):
        """Show a countdown based on the current coordinates and frame index.
        
        Parameters
        ----------
        coordinates: numpy.ndarray
            N x M matrix, where N is the number of frames and M is the number of DOFs. Order is x-axis, y-axis, and third DOF (either rotation or target radius).
            Each row contains the value for x position, y position, and / or third DOF depending on how many DOFs are passed in.
        frame_idx: int
            Current frame index.
        """
        frame_coordinates = coordinates[frame_idx]
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
            time_until_movement = (final_steady_state_idx - frame_idx) * self.duration / 1000   # convert from frames to seconds
            if time_until_movement >= 0.25:
                # Only show countdown if the steady state is longer than 1 second
                plt.text(frame_coordinates[0] - 0.03, frame_coordinates[1] - 0.2, str(int(time_until_movement)), fontweight='bold', c='red')
        except IndexError:
            # Did not find steady state
            pass
    
    def plot_icon(self, coordinates):
        """Plot target / icon on axis.
        
        Parameters
        ----------
        coordinates: numpy.ndarray
            1D array where each element corresponds to the value along a different DOF.
        """
        plt.plot(coordinates[0], coordinates[1])
    
    
    def save_plot_gif(self, coordinates, title = '', xlabel = '', ylabel = '', save_coordinates = False, verbose = False):
        """Save a .gif file of an icon moving around a 2D plane.
        
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
            filename_no_extension = os.path.splitext(self.output_filepath)[0]
            labels_filepath = filename_no_extension + '.txt'
            np.savetxt(labels_filepath, coordinates, delimiter=',')
        
        # Format figure
        fig, ax = self._format_figure()
        fig.suptitle(title)

        # Calculate direction changes
        direction_changes = np.sign(np.diff(coordinates, axis=0, n=1))[:-1] * np.diff(coordinates, axis=0, n=2) / self.duration
        direction_change_indices = np.where(np.abs(direction_changes) > 1e-8)[0] + 1 # add 1 to align with coordinates
        direction_change_indices = np.append(direction_change_indices, coordinates.shape[0] - 1)    # append final frame position
        
        # Adjust coordinates if desired
        coordinates = self._preprocess_coordinates(coordinates)
        
        frames = []
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        direction_change_idx = None
        target_alpha = 0.05
        for frame_idx, frame_coordinates in enumerate(coordinates):
            if verbose and frame_idx % 10 == 0:
                print(f'Frame {frame_idx} / {coordinates.shape[0]}')
            # Plot icon
            self.plot_icon(frame_coordinates)

            # Format axis
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            ticks = [-1., -0.5, 0, 0.5, 1.]
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.axis('equal')
            ax.set(xlim=xlim, ylim=ylim)    # maintain the same axis limits

            # Plot additional information
            if self.show_boundary:
                # Show boundaries
                self._show_boundary()
            
            if self.show_direction:
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
                self.plot_icon(coordinates[direction_change_idx], alpha=target_alpha, colour='green')
                
            if self.show_countdown:
                # Show countdown during steady state
                self._show_countdown(coordinates, frame_idx)
                
            # Save frame
            frame = self._convert_plot_to_image(fig)
            frames.append(frame)
            plt.cla()   # clear axis
        
        # Save file
        self.save_gif(frames)


class CartesianPlotAnimator(PlotAnimator):
    def __init__(self, output_filepath = 'libemg.gif', fps = 24, show_direction = False, show_countdown = False, show_boundary = False, normalize_distance = False, axis_images = None):
        """Animator object for creating .gif files from a list of coordinates on a cartesian plane.
        
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
        """
        super().__init__(output_filepath, fps, show_direction, show_countdown, show_boundary)
        self.normalize_distance = normalize_distance
        self.axis_images = axis_images
    
    def _format_figure(self):
        fig = plt.figure(figsize=(8, 8))
        axis_limits = (-1.25, 1.25)
        if self.axis_images is not None:
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
        else:
            ax = plt.gca()
        ax.set(xlim=axis_limits, ylim=axis_limits)
        fig.tight_layout()
        return fig, ax
    
    def _preprocess_coordinates(self, coordinates):
        coordinates = super()._preprocess_coordinates(coordinates)

        if self.normalize_distance:
            # Normalize to unit circle distance
            coordinates = self._normalize_to_unit_distance(coordinates[:, 0], coordinates[:, 1])
        return coordinates
    
    def _show_boundary(self):
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an), 'b--', alpha=0.7)
    
    
    @staticmethod
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

    @staticmethod
    def _normalize_to_unit_distance(x, y):
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


class ScatterPlotAnimator(CartesianPlotAnimator):
    def __init__(self, output_filepath = 'libemg.gif', fps = 24, show_direction = False, show_countdown = False, show_boundary = False, normalize_distance = False, axis_images = None, 
                 plot_line = False):
        """Animator object for creating .gif files from a list of coordinates on a cartesian plane shown as a scatter plot.
        
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
        """
        super().__init__(output_filepath, fps, show_direction, show_countdown, show_boundary, normalize_distance, axis_images)
        self.plot_line = plot_line

    
    def plot_icon(self, frame_coordinates, alpha = 1.0, colour = 'black'):
        # Parse coordinates
        x = frame_coordinates[0]
        y = frame_coordinates[1]
        # Dot properties
        size = 50
        plt.scatter(x, y, s=size, c=colour, alpha=alpha)
        if self.plot_line and alpha == 1.0:
            # Plot line to current point, but not to target
            plt.plot([0, x], [0, y], c=colour, linewidth=5)


class ArrowPlotAnimator(CartesianPlotAnimator):
    def plot_icon(self, frame_coordinates, alpha = 1.0, colour = 'black'):
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


class TargetPlotAnimator(CartesianPlotAnimator):
    @staticmethod
    def _plot_circle(xy, radius, edgecolor, facecolor, alpha = 1.0):
        circle = Circle(xy, radius=radius, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
        plt.gca().add_patch(circle)
    
    def plot_icon(self, frame_coordinates, alpha = 1.0, colour = 'black'):
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
        TargetPlotAnimator._plot_circle(xy, radius=radius, edgecolor='none', facecolor=colour, alpha = alpha) # plot target
        TargetPlotAnimator._plot_circle(xy, radius=max_radius, edgecolor='black', facecolor='none', alpha=limit_alpha)   # plot max boundary
        TargetPlotAnimator._plot_circle(xy, radius=min_radius, edgecolor='black', facecolor='black', alpha=limit_alpha)   # plot min boundary
