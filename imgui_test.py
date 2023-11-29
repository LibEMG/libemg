import dearpygui.dearpygui as dpg
from PIL import Image
import numpy as np
import libemg
import os
from itertools import compress
import time
import csv
import copy

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
            
    
    def reset(self):
        if self.type == ".gif":
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

class LibEMGGUI:
    def __init__(self, 
                 width=1920,
                 height=1080,
                 args = None):
        self.args = args
        self.window_init(width, height)
        
    
    def window_init(self, width, height):
        dpg.create_context()
        dpg.create_viewport(title="LibEMG",
                            width=width,
                            height=height)
        dpg.setup_dearpygui()
        

        self.file_menu_init()

        dpg.show_viewport()

        # dpg.configure_app(manual_callback_management=True)
        # while dpg.is_dearpygui_running():
        #     jobs = dpg.get_callback_queue()
        #     dpg.run_callbacks(jobs)
        #     dpg.render_dearpygui_frame()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def file_menu_init(self):

        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Exit")
                
            with dpg.menu(label="Data"):
                dpg.add_menu_item(label="Collect Data", callback=self.data_collection_callback)
                dpg.add_menu_item(label="Import Data")
                dpg.add_menu_item(label="Export Data")
                dpg.add_menu_item(label="Inspect Data")

    def data_collection_callback(self):
        self.dcp = DataCollectionPanel(**self.args)
        self.dcp.spawn_configuration_window()

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

class DataCollectionPanel:
    def __init__(self,
                 odh = None,
                 num_reps=3,
                 rep_time=3,
                 media_folder='media/',
                 data_folder='data/',
                 rest_time=2,
                 auto_advance=True,
                 exclude_files=[],
                 args = None):
        
        self.odh = odh
        self.num_reps = num_reps
        self.rep_time = rep_time
        self.media_folder = media_folder
        self.data_folder  = data_folder
        self.rest_time = rest_time
        self.auto_advance=auto_advance
        self.exclude_files = exclude_files
        
    
    def spawn_configuration_window(self):
        with dpg.window(tag="__dc_configuration_window", label="Data Collection Configuration"):
            
            dpg.add_spacer(height=50)
            dpg.add_text(label="Training Menu")
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):

                dpg.add_table_column(label="")
                dpg.add_table_column(label="")
                dpg.add_table_column(label="")
                # REP ROW
                with dpg.table_row(): 
                    with dpg.group(horizontal=True):
                        dpg.add_text("Num Reps: ")
                        dpg.add_input_text(default_value=self.num_reps,
                                        tag="__num_reps",
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Time Per Rep")
                        dpg.add_input_text(default_value=self.rep_time,
                                        tag="__rep_time",
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Time Between Reps")
                        dpg.add_input_text(default_value=self.rest_time,
                                        tag="__rest_time", 
                                        width=100)
                # FOLDER ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Media Folder:")
                        dpg.add_input_text(default_value=self.media_folder, 
                                        tag="__media_folder", width=250)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Output Folder:")
                        dpg.add_input_text(default_value=self.data_folder, 
                                        tag="__output_folder",
                                        width=250)
                # CHECKBOX ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Auto-Advance")
                        dpg.add_checkbox(default_value=self.auto_advance,
                                        tag="__auto_advance")
                # BUTTON ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Start", callback=self.start_callback)
        
        dpg.set_primary_window("__dc_configuration_window", True)

    def start_callback(self):
        if self.odh and len(self.odh.raw_data.get_emg()) > 0:
            self.get_settings()
            dpg.delete_item("__dc_configuration_window")
            media_list = self.gather_media()
            self.spawn_collection_window(media_list)
            print("start")

    def get_settings(self):
        self.num_reps      = int(dpg.get_value("__num_reps"))
        self.rep_time      = float(dpg.get_value("__rep_time"))
        self.rest_time     = float(dpg.get_value("__rest_time"))
        self.media_folder  = dpg.get_value("__media_folder")
        self.output_folder = dpg.get_value("__output_folder")
        self.auto_advance  = bool(dpg.get_value("__auto_advance"))

    def gather_media(self):
        # find everything in the media folder
        files = os.listdir(self.media_folder)
        valid_files = [file.endswith((".gif",".png")) for file in files]
        files = list(compress(files, valid_files))
        collection_conf = []
        for rep in range(self.num_reps):
            for motion_class in files:
                # entry for collection of rep
                media = Media()
                media.from_file(self.media_folder + motion_class)
                collection_conf.append([media,motion_class.split('.')[0],rep,self.rep_time])
        return collection_conf

    def spawn_collection_window(self, media_list):
        # open first frame of gif
        texture = media_list[0][0].get_dpg_formatted_texture(width=720,height=480)
        set_texture("collection_visual", texture, width=720, height=480)

        with dpg.window(label="Collection Window",
                        tag="__dc_collection_window"):
            dpg.add_spacer(height=50)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=275, height=10)
                dpg.add_text(default_value="Collection Menu")
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__prompt_spacer",width=300, height=10)
                dpg.add_text(media_list[0][1], tag="__prompt")
            dpg.add_image("collection_visual")
            dpg.add_progress_bar(tag="progress", default_value=0.0,width=720)
            
        dpg.set_primary_window("__dc_collection_window", True)

        self.run_sgt(media_list)
    
    def run_sgt(self, media_list):
        i = 0
        while i < len(media_list):
            # set the prompt to be the label of the motion (from the file)
            dpg.set_value("__prompt", value=media_list[i][1])
            dpg.set_item_width("__prompt_spacer", 300)
            # reset the frame of the file to 0 (useful for gif, irrelevant for png)
            media_list[i][0].reset()
            texture = media_list[i][0].get_dpg_formatted_texture(width=720,height=480)
            set_texture("collection_visual", texture, 720, 480)

            # reset emg data
            self.odh.raw_data.reset_emg()

            # initialize motion and frame timers
            motion_timer = time.perf_counter_ns()
            frame_timer  = time.perf_counter_ns()
            # play the gif (or keep image on screen) until motion timer > self.rep_time
            while (time.perf_counter_ns() - motion_timer)/1e9 < self.rep_time:
                time_remaining = 1/media_list[i][0].fps - (time.perf_counter_ns() - frame_timer)/1e9
                time.sleep(max(0, time_remaining))
                frame_timer = time.perf_counter_ns()
                # update visual
                media_list[i][0].advance()
                
                texture = media_list[i][0].get_dpg_formatted_texture(width=720,height=480)
                set_texture("collection_visual", texture, 720, 480)
                # update progress bar
                progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*self.rep_time))
                dpg.set_value("progress", value = progress)
                # reset frame timer
                
            
            self.save_data(self.output_folder + "C_" + str(media_list[i][1]) + "_R_" + str(media_list[i][2]) + ".csv")

            i = i+1
            # pause / redo goes here!

            # do the rest
            if self.rest_time and i < len(media_list):
                dpg.set_value("__prompt", value="Up next: "+media_list[i][1])
                dpg.set_item_width("__prompt_spacer", 250)
                texture = media_list[i][0].get_dpg_formatted_texture(width=720,height=480, grayscale=True)
                set_texture("collection_visual", texture, 720, 480)

                # initialize motion and frame timers
                motion_timer = time.perf_counter_ns()
                frame_timer  = time.perf_counter_ns()
                while (time.perf_counter_ns() - motion_timer)/1e9 < self.rest_time:
                    time_remaining = 1/media_list[i][0].fps - (time.perf_counter_ns() - frame_timer)/1e9
                    time.sleep(max(0, time_remaining))
                    frame_timer = time.perf_counter_ns()
                    # update visual
                    media_list[i][0].advance()
                    
                    texture = media_list[i][0].get_dpg_formatted_texture(width=720,height=480, grayscale=True)
                    set_texture("collection_visual", texture, 720, 480)
                    # update progress bar
                    progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*self.rest_time))
                    dpg.set_value("progress", value = progress)
            # 
            
        
    def save_data(self, filename):
        data = self.odh.raw_data.get_emg()
        with open(filename, "w", newline='', encoding='utf-8') as file:
            emg_writer = csv.writer(file)
            for row in data:
                emg_writer.writerow(row)
        self.odh.raw_data.reset_emg()


if __name__ == "__main__":
    p = libemg.streamers.sifibridge_streamer(version="1_1")
    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()
    args = {
        "odh"         : odh,
        "media_folder": "media/",
        "data_folder" : "data/",
        "num_reps"    : 5,
        "rep_time"    : 3,
        "rest_time"   : 1,
        "auto_advance":True,
    }
    gui = LibEMGGUI(args = args)



    # media = Media()
    # media.from_file("images/Hand_Close.png")
    # texture = media.get_dpg_formatted_texture(width=720, height=480)
    # with dpg.texture_registry(show=True):
    #     dpg.add_raw_texture(width=720,
    #                         height=480,
    #                         default_value=texture,
    #                         tag="__dcs_image",
    #                         format=dpg.mvFormat_Float_rgb)