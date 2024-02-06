from PIL import Image
import numpy as np
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import cv2

class Media:
    def __init__(self):
        pass

    def from_file(self, location, fps=24):
        self.type = location.split(".")[-1]
        self.fps = fps
        if self.type == "mp4":
            self.import_video(location)
        elif self.type in ["png", "jpg", "bmp"]:
            self.import_picture(location)
        elif self.type == "gif":
            self.import_gif(location)    

    def import_picture(self, location):
        self.file_content = Image.open(location)

    def import_gif(self, location):
        self.file_content = Image.open(location)
        self.frame = 0
        self.file_content.seek(self.frame)
        self.n_frames = self.file_content.n_frames
        self.frame_times = np.linspace(0, self.n_frames/self.fps, int(self.n_frames))

    def import_video(self, location):
        # get video capture ready
        self.video_capture = cv2.VideoCapture(location)
        self.n_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_times = np.linspace(0, self.n_frames/self.fps, int(self.n_frames))
        # get first frame
        self.frame = 0
        _, cv2_image  = self.video_capture.read()
        cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGBA)
        self.file_content =  Image.fromarray(cv2_image)

    def from_numpy(self, numpy_array):
        self.file_content = Image.fromarray(numpy_array)
        self.type = "png"
    
    def reset(self):
        if self.type == "gif":
            self.frame = 0
            self.file_content.seek(self.frame)
        if self.type == "mp4":
            self.frame = 0
            self.video_capture.set(cv2.CAP_PROP_FRAME_COUNT, 0)
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

    def advance(self):
        assert hasattr(self, "file_content")
        if self.type == "gif":
            if self.frame + 1 < self.file_content.n_frames:
                self.frame += 1
                self.file_content.seek(self.frame)
            else:
                print("End of gif reached.")
        if self.type == "mp4":
            # print(self.frame)
            self.frame += 1
            ret, cv2_image  = self.video_capture.read()
            if not ret:
                print("End of video reached")
            else:
                cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGBA)
                self.file_content =  Image.fromarray(cv2_image)
    
    def advance_to(self, play_time):
        if not hasattr(self, "frame_times"):
            return
        # find the closest time
        del_times = np.abs(self.frame_times - play_time)
        closest_frame = np.argmin(del_times)
        if self.type == "gif":
            if closest_frame < self.file_content.n_frames:
                self.frame = closest_frame
                self.file_content.seek(self.frame)
        if self.type == "mp4":
            self.frame = closest_frame
            self.video_capture.set(1, self.frame)
            ret, cv2_image  = self.video_capture.read()
            if not ret:
                print("End of video reached")
            else:
                cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGBA)
                self.file_content =  Image.fromarray(cv2_image)

    def get_dpg_formatted_texture(self, width, height, grayscale=False):
        dpg_img = self.file_content.resize((width, height))
        if grayscale:
            dpg_img = dpg_img.convert("L")
        dpg_img = dpg_img.convert("RGBA")
        dpg_img = np.asfarray(dpg_img, dtype='f').ravel()
        dpg_img = np.true_divide(dpg_img, 255.0)
        return dpg_img

def set_texture(tag, texture, width, height, format=dpg.mvFormat_Float_rgba):
    with dpg.texture_registry(show=False):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value=texture)
        else:
            dpg.add_raw_texture(width=width,
                                height=height,
                                default_value=texture,
                                tag=tag,
                                format=format)

def init_matplotlib_canvas(width=720, height=480):
    plt.figure(figsize=(width/80,height/80), dpi=80)
    
def matplotlib_to_numpy():
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    return image