from PIL import Image
import numpy as np
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt

class Media:
    def __init__(self):
        pass

    def from_file(self, location, fps=24):
        self.type = location.split(".")[-1]
        self.file_content = Image.open(location)
        # fps will re-render .png or frames of gif at desired rate
        # you don't want this large even for .png
        self.fps = fps
        if self.type == "gif":
            self.frame = 0
            self.file_content.seek(self.frame)
        
    def from_numpy(self, numpy_array):
        self.file_content = Image.fromarray(numpy_array)
        self.type = "png"
    
    def reset(self):
        if self.type == "gif":
            self.frame = 0
            self.file_content.seek(self.frame)

    def advance(self):
        assert hasattr(self, "file_content")
        if self.type == "gif":
            if self.frame + 1 < self.file_content.n_frames:
                self.frame += 1
                self.file_content.seek(self.frame)
            else:
                print("End of gif reached.")
    
    def get_dpg_formatted_texture(self, width, height, grayscale=False):
        dpg_img = self.file_content.resize((width, height))
        if grayscale:
            dpg_img = dpg_img.convert("L")
        dpg_img = dpg_img.convert("RGB")
        dpg_img = np.asfarray(dpg_img, dtype='f').ravel()
        dpg_img = np.true_divide(dpg_img, 255.0)
        return dpg_img

def set_texture(tag, texture, width, height, format=dpg.mvFormat_Float_rgb):
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