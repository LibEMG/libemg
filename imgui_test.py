import dearpygui.dearpygui as dpg
from PIL import Image
import numpy as np
import libemg
import os
from itertools import compress
import time
import csv
import copy



if __name__ == "__main__":
    p = libemg.streamers.sifibridge_streamer(version="1_1")
    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()
    # odh.analyze_hardware()
    args = {
        "odh"         : odh,
        "media_folder": "media/",
        "data_folder" : "data/",
        "num_reps"    : 5,
        "rep_time"    : 3,
        "rest_time"   : 1,
        "auto_advance":True,
    }
    gui = libemg.gui.GUI(args = args)


    # Useful code snippets:
    # how media is loaded and prepared:
    # media = Media()
    # media.from_file("images/Hand_Close.png")
    # texture = media.get_dpg_formatted_texture(width=720, height=480)
    # with dpg.texture_registry(show=True):
    #     dpg.add_raw_texture(width=720,
    #                         height=480,
    #                         default_value=texture,
    #                         tag="__dcs_image",
    #                         format=dpg.mvFormat_Float_rgb)

    ## # set the prompt to be the label of the motion (from the file)
    # dpg.set_value("__prompt", value=media_list[i][1])
    # dpg.set_item_width("__prompt_spacer", 300)
    # # reset the frame of the file to 0 (useful for gif, irrelevant for png)
    # media_list[i][0].reset()
    # texture = media_list[i][0].get_dpg_formatted_texture(width=720,height=480)
    # set_texture("collection_visual", texture, 720, 480)

    # # reset emg data
    # self.odh.raw_data.reset_emg()

    # # initialize motion and frame timers
    # motion_timer = time.perf_counter_ns()
    # frame_timer  = time.perf_counter_ns()
    # # play the gif (or keep image on screen) until motion timer > self.rep_time
    # while (time.perf_counter_ns() - motion_timer)/1e9 < self.rep_time:
    #     time_remaining = 1/media_list[i][0].fps - (time.perf_counter_ns() - frame_timer)/1e9
    #     time.sleep(max(0, time_remaining))
    #     frame_timer = time.perf_counter_ns()
    #     # update visual
    #     media_list[i][0].advance()
        
    #     texture = media_list[i][0].get_dpg_formatted_texture(width=720,height=480)
    #     set_texture("collection_visual", texture, 720, 480)
    #     # update progress bar
    #     progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*self.rep_time))
    #     dpg.set_value("progress", value = progress)
    #     # reset frame timer