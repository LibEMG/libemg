
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
        output_filepath: string (optional), default='libemg.gif'
            Filepath of output file.
        duration: int (optional), default=100
            Duration of each frame in milliseconds.
        
        """
        frames[0].save(
            self.output_filepath,
            save_all=True,
            append_images=frames[1:],   # append remaining frames
            format='GIF',
            duration=self.duration,
            loop=0  # infinite loop
        )