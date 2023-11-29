import os
from PIL import Image, UnidentifiedImageError

class Animator:
    
    def __init__(self, output_filepath = 'libemg.gif', duration = 100):
        self.output_filepath = output_filepath
        self.duration = duration
        
    def make_gif(self, frames):
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
    
    def make_gif_from_directory(self, directory_path, match_filename_function = None, 
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
        self.make_gif(frames)

        if delete_images:
            # Delete all images used to create .gif
            for filename in matching_filenames:
                os.remove(filename)
