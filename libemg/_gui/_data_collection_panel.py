import dearpygui.dearpygui as dpg
import numpy as np
import os
from itertools import compress
import time
import csv
import json
from datetime import datetime
from ._utils import Media, set_texture, init_matplotlib_canvas, matplotlib_to_numpy

import threading
import matplotlib.pyplot as plt


class DataCollectionPanel:
    def __init__(self,
                 num_reps=3,
                 rep_time=3,
                 media_folder='media/',
                 data_folder='data/',
                 rest_time=2,
                 auto_advance=True,
                 exclude_files=[],
                 gui = None):
        
        self.num_reps = num_reps
        self.rep_time = rep_time
        self.media_folder = media_folder
        self.data_folder  = data_folder
        self.rest_time = rest_time
        self.auto_advance=auto_advance
        self.exclude_files = exclude_files
        self.gui = gui

        self.widget_tags = {"configuration":['__dc_configuration_window','__dc_num_reps','__dc_rep_time','__dc_rest_time', '__dc_media_folder',\
                                             '__dc_auto_advance'],
                            "collection":   ['__dc_collection_window', '__dc_prompt_spacer', '__dc_prompt', '__dc_progress', '__dc_redo_button'],
                            "visualization": ['__vls_visualize_window']}
        

    def cleanup_window(self, window_name):
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)      
    
    def spawn_configuration_window(self):
        self.cleanup_window("configuration")
        self.cleanup_window("collection")
        self.cleanup_window("visualization")
        with dpg.window(tag="__dc_configuration_window", label="Data Collection Configuration"):
            
            # dpg.add_spacer(height=50)
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
                                        tag="__dc_num_reps",
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Time Per Rep")
                        dpg.add_input_text(default_value=self.rep_time,
                                        tag="__dc_rep_time",
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Time Between Reps")
                        dpg.add_input_text(default_value=self.rest_time,
                                        tag="__dc_rest_time", 
                                        width=100)
                # FOLDER ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Media Folder:")
                        dpg.add_input_text(default_value=self.media_folder, 
                                        tag="__dc_media_folder", width=250)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Output Folder:")
                        dpg.add_input_text(default_value=self.data_folder, 
                                        tag="__dc_output_folder",
                                        width=250)
                # CHECKBOX ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Auto-Advance")
                        dpg.add_checkbox(default_value=self.auto_advance,
                                        tag="__dc_auto_advance")
                # BUTTON ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Start", callback=self.start_callback)
                        dpg.add_button(label="Visualize", callback=self.visualize_callback)
        
        # dpg.set_primary_window("__dc_configuration_window", True)



    def start_callback(self):
        if self.gui.online_data_handler and len(self.gui.online_data_handler.raw_data.get_emg()) > 0:
            self.get_settings()
            dpg.delete_item("__dc_configuration_window")
            self.cleanup_window("configuration")
            media_list = self.gather_media()

            self.spawn_collection_thread = threading.Thread(target=self.spawn_collection_window, args=(media_list,))
            self.spawn_collection_thread.start()
            # self.spawn_collection_window(media_list)

    def get_settings(self):
        self.num_reps      = int(dpg.get_value("__dc_num_reps"))
        self.rep_time      = float(dpg.get_value("__dc_rep_time"))
        self.rest_time     = float(dpg.get_value("__dc_rest_time"))
        self.media_folder  = dpg.get_value("__dc_media_folder")
        self.output_folder = dpg.get_value("__dc_output_folder")
        self.auto_advance  = bool(dpg.get_value("__dc_auto_advance"))

    def gather_media(self):
        # find everything in the media folder
        files = os.listdir(self.media_folder)
        valid_files = [file.endswith((".gif",".png",".mp4")) for file in files]
        files = list(compress(files, valid_files))
        self.num_motions = len(files)
        collection_conf = []
        # make the collection_details.json file
        collection_details = {}
        collection_details["num_motions"] = self.num_motions
        collection_details["num_reps"]    = self.num_reps
        collection_details["classes"] =   [f.split('.')[0] for f in files]
        collection_details["class_map"] = {index: f.split('.')[0] for index, f in enumerate(files)}
        collection_details["time"]    = datetime.now().isoformat()
        with open(self.output_folder + "collection_details.json", 'w') as f:
            json.dump(collection_details, f)

        # make the media list for SGT progression
        for rep_index in range(self.num_reps):
            for class_index, motion_class in enumerate(files):
                # entry for collection of rep
                media = Media()
                media.from_file(self.media_folder + motion_class)
                collection_conf.append([media,motion_class.split('.')[0],class_index,rep_index,self.rep_time])
        return collection_conf

    def spawn_collection_window(self, media_list):
        # open first frame of gif
        texture = media_list[0][0].get_dpg_formatted_texture(width=720,height=480)
        set_texture("__dc_collection_visual", texture, width=720, height=480)

        with dpg.window(label="Collection Window",
                        tag="__dc_collection_window",
                        width=800,
                        height=800):
            dpg.add_spacer(height=50)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=275, height=10)
                dpg.add_text(default_value="Collection Menu")
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_prompt_spacer",width=300, height=10)
                dpg.add_text(media_list[0][1], tag="__dc_prompt")
            dpg.add_image("__dc_collection_visual")
            dpg.add_progress_bar(tag="__dc_progress", default_value=0.0,width=720)
            
        # dpg.set_primary_window("__dc_collection_window", True)

        self.run_sgt(media_list)
        # clean up the window
        
        dpg.delete_item("__dc_collection_window")
        self.cleanup_window("collection")
        # open config back up
        self.spawn_configuration_window()
    
    def run_sgt(self, media_list):
        self.i = 0
        self.advance = True
        while self.i < len(media_list):

            # do the rest
            if self.rest_time and self.i < len(media_list):
                self.play_collection_visual(media_list[self.i], active=False)
                media_list[self.i][0].reset()
            
            self.play_collection_visual(media_list[self.i], active=True)
            
            self.save_data(self.output_folder + "C_" + str(media_list[self.i][2]) + "_R_" + str(media_list[self.i][3]) + ".csv")
            last_rep = media_list[self.i][3]
            self.i = self.i+1
            if self.i  == len(media_list):
                break
            current_rep = media_list[self.i][3]
            # pause / redo goes here!
            if last_rep != current_rep  or (not self.auto_advance):
                self.advance = False
                dpg.add_button(tag="__dc_redo_button", label="Redo", callback=self.redo_collection_callback, parent="__dc_collection_window")
                dpg.add_button(tag="__dc_continue_button", label="Continue", callback=self.continue_collection_callback, parent="__dc_collection_window")
                while not self.advance:
                    time.sleep(0.1)
                    dpg.configure_app(manual_callback_management=True)
                    jobs = dpg.get_callback_queue()
                    dpg.run_callbacks(jobs)
                dpg.configure_app(manual_callback_management=False)
        
    def redo_collection_callback(self):
        if self.auto_advance:
            self.i      = self.i - self.num_motions
        else:
            self.i      = self.i - 1 
        dpg.delete_item("__dc_redo_button")
        dpg.delete_item("__dc_continue_button")
        self.advance = True
    
    def continue_collection_callback(self):
        dpg.delete_item("__dc_redo_button")
        dpg.delete_item("__dc_continue_button")
        self.advance = True

    def play_collection_visual(self, media, active=True):
        if active:
            timer_duration = self.rep_time
            dpg.set_value("__dc_prompt", value=media[1])
            dpg.set_item_width("__dc_prompt_spacer", 300)
        else:
            timer_duration = self.rest_time
            dpg.set_value("__dc_prompt", value="Up next: "+media[1])
            dpg.set_item_width("__dc_prompt_spacer", 250)
        
        
        texture = media[0].get_dpg_formatted_texture(width=720,height=480, grayscale=not(active))
        set_texture("__dc_collection_visual", texture, 720, 480)
        self.gui.online_data_handler.raw_data.reset_emg()
        # initialize motion and frame timers
        motion_timer = time.perf_counter_ns()
        while (time.perf_counter_ns() - motion_timer)/1e9 < timer_duration:
            time.sleep(1/media[0].fps) # never refresh faster than media fps
            # update visual
            media[0].advance_to((time.perf_counter_ns() - motion_timer)/1e9)
            texture = media[0].get_dpg_formatted_texture(width=720,height=480, grayscale=not(active))
            set_texture("__dc_collection_visual", texture, 720, 480)
            # update progress bar
            progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*timer_duration))
            dpg.set_value("__dc_progress", value = progress)        
    
    def save_data(self, filename):
        data = self.gui.online_data_handler.raw_data.get_emg()
        with open(filename, "w", newline='', encoding='utf-8') as file:
            emg_writer = csv.writer(file)
            for row in data:
                emg_writer.writerow(row)
        self.gui.online_data_handler.raw_data.reset_emg()

    def visualize_callback(self):
        self.visualization_thread = threading.Thread(target=self._run_visualization_helper)
        self.visualization_thread.start()
        # self.visualization_process = Process(target=self._run_visualization_helper, daemon=True,)
        # self.visualization_process.start()
        # self._run_visualization_helper()
    
    def _run_visualization_helper(self):

        self.last_new_data_size = 0

        self.visualization_horizon = 5000
        self.visualization_rate    = 24
        # initialize blank data object
        self.data = np.zeros((self.visualization_horizon, np.stack(self.gui.online_data_handler.raw_data.get_emg()).shape[1]))
        self.visualizing = True
        init_matplotlib_canvas(width=800, height=600)
        self.plot_handles = plt.plot(self.data)

        plt.xlabel("Samples")
        plt.ylabel("Values")
        self.update_data()
        img = matplotlib_to_numpy()
        media = Media()
        media.from_numpy(img)
        texture = media.get_dpg_formatted_texture(width=720, height=480)
        set_texture("__vls_plot", texture, width=720, height=480)

        self.cleanup_window("visualization")
        with dpg.window(tag="__vls_visualization_window",
                        label="Live Signal Visualization Configuration",
                        width=800,
                        height=600):
            
            dpg.add_text(label="Visualization Panel")
            dpg.add_image("__vls_plot")
            dpg.add_button(label="Quit", callback=self.quit_visualization_callback)
        
        dpg.configure_app(manual_callback_management=True)
        
        while self.visualizing:
            
            self.update_data()
            img = matplotlib_to_numpy()
            media = Media()
            media.from_numpy(img)
            texture = media.get_dpg_formatted_texture(width=720, height=480)
            set_texture("__vls_plot", texture, width=720, height=480)
            # time.sleep(1/self.visualization_rate)
            jobs = dpg.get_callback_queue()
            dpg.run_callbacks(jobs)

        

        dpg.delete_item("__vls_visualization_window")
        self.cleanup_window("visualization")
        dpg.configure_app(manual_callback_management=False)
    
    def quit_visualization_callback(self):
        self.visualizing = False

    def update_data(self):
        new_data = self.gui.online_data_handler.raw_data.get_emg()
        if len(new_data):
            new_data = np.stack(new_data)
            current_new_data_size = new_data.shape[0]
            if current_new_data_size == self.last_new_data_size:
                return
            if current_new_data_size > self.last_new_data_size:
                new_data = new_data[self.last_new_data_size:,:]
            self.last_new_data_size = current_new_data_size
            self.data = np.vstack((self.data, new_data))
            # self.gui.online_data_handler.raw_data.reset_emg()
            self.data = self.data[-self.visualization_horizon:,:]
            for i, h in enumerate(self.plot_handles):
                h.set_ydata(self.data[:,i])
            plt.ylim((np.min(self.data), np.max(self.data)))