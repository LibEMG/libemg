import dearpygui.dearpygui as dpg
import numpy as np
import os
from itertools import compress
import time
import csv
import json
from datetime import datetime
from ._utils import Media, set_texture


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
        
        # dpg.set_primary_window("__dc_configuration_window", True)

    def start_callback(self):
        if self.odh and len(self.odh.raw_data.get_emg()) > 0:
            self.get_settings()
            dpg.delete_item("__dc_configuration_window")
            media_list = self.gather_media()
            self.spawn_collection_window(media_list)

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
        self.num_motions = len(files)
        collection_conf = []
        # make the collection_details.json file
        collection_details = {}
        collection_details["num_motions"] = self.num_motions
        collection_details["num_reps"]    = self.num_reps
        collection_details["classes"] =   [f.split('.')[0] for f in files]
        collection_details["class_map"] = {index: f.split('.')[0] for index, f in enumerate(files)}
        collection_details["time"]    = datetime.now().isoformat()
        with open(self.data_folder + "collection_details.json", 'w') as f:
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
        set_texture("collection_visual", texture, width=720, height=480)

        with dpg.window(label="Collection Window",
                        tag="__dc_collection_window",
                        width=800,
                        height=800):
            dpg.add_spacer(height=50)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=275, height=10)
                dpg.add_text(default_value="Collection Menu")
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__prompt_spacer",width=300, height=10)
                dpg.add_text(media_list[0][1], tag="__prompt")
            dpg.add_image("collection_visual")
            dpg.add_progress_bar(tag="progress", default_value=0.0,width=720)
            
        # dpg.set_primary_window("__dc_collection_window", True)

        self.run_sgt(media_list)
        # clean up the window
        dpg.delete_item("__dc_collection_window")
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
                dpg.add_button(tag="__redo_button", label="Redo", callback=self.redo_collection_callback, parent="__dc_collection_window")
                dpg.add_button(tag="__continue_button", label="Continue", callback=self.continue_collection_callback, parent="__dc_collection_window")
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
        dpg.delete_item("__redo_button")
        dpg.delete_item("__continue_button")
        self.advance = True
    
    def continue_collection_callback(self):
        dpg.delete_item("__redo_button")
        dpg.delete_item("__continue_button")
        self.advance = True

    def play_collection_visual(self, media, active=True):
        if active:
            timer_duration = self.rep_time
            dpg.set_value("__prompt", value=media[1])
            dpg.set_item_width("__prompt_spacer", 300)
        else:
            timer_duration = self.rest_time
            dpg.set_value("__prompt", value="Up next: "+media[1])
            dpg.set_item_width("__prompt_spacer", 250)
        
        
        texture = media[0].get_dpg_formatted_texture(width=720,height=480, grayscale=not(active))
        set_texture("collection_visual", texture, 720, 480)
        self.odh.raw_data.reset_emg()
        # initialize motion and frame timers
        motion_timer = time.perf_counter_ns()
        frame_timer  = time.perf_counter_ns()
        while (time.perf_counter_ns() - motion_timer)/1e9 < timer_duration:
            time_remaining = 1/media[0].fps - (time.perf_counter_ns() - frame_timer)/1e9
            time.sleep(max(0, time_remaining))
            frame_timer = time.perf_counter_ns()
            # update visual
            media[0].advance()
            texture = media[0].get_dpg_formatted_texture(width=720,height=480, grayscale=not(active))
            set_texture("collection_visual", texture, 720, 480)
            # update progress bar
            progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*timer_duration))
            dpg.set_value("progress", value = progress)        
    
    def save_data(self, filename):
        data = self.odh.raw_data.get_emg()
        with open(filename, "w", newline='', encoding='utf-8') as file:
            emg_writer = csv.writer(file)
            for row in data:
                emg_writer.writerow(row)
        self.odh.raw_data.reset_emg()
