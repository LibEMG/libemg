import csv
import time
import os
import json
from tkinter import *
from tkinter.ttk import Progressbar
from threading import Thread
from turtle import width
from PIL import ImageTk, Image
from os import walk
import random
import numpy as np
random.seed(time.time())



class ScreenGuidedTraining:
    """The Screen Guided Training module. 

    By default, this module has two purposes: 
    (1) Launching a Screen Guided Training window. 
    (2) Downloading gesture sets from our library of gestures located at:
    https://github.com/AnonSubmissions123/LibEMGGestures
    """
    def __init__(self):
        pass 

    def download_gestures(self, gesture_ids, folder, download_imgs=True, download_gifs=False, redownload=False):
        """
        Downloads gesture images (either .png or .gif) from: 
        https://github.com/AnonSubmissions123/LibEMGGestures.
        
        This function dowloads gestures using the "curl" command. 

        Parameters
        ----------
        gesture_ids: list
            A list of indexes corresponding to the gestures you want to download. A list of indexes and their respective 
            gesture can be found at https://github.com/AnonSubmissions123/LibEMGGestures.
        folder: string
            The output folder where the downloaded gestures will be saved.
        download_gif: bool (optional), default=False
            If True, the assocaited GIF will be downloaded.
        redownload: bool (optional), default=False
            If True, all files will be re-downloaded (regardless if they are already downloaed).
        """
        git_url = "https://raw.githubusercontent.com/libemg/LibEMGGestures/main/"
        gif_folder = "GIFs/"
        img_folder = "Images/"
        json_file = "gesture_list.json"
        curl_commands = "curl --create-dirs" + " -O --output-dir " + folder + " "

        files = next(walk(folder), (None, None, []))[2]

        # Check JSON file exists
        if not json_file in files or redownload:
            os.system(curl_commands + git_url + json_file)

        json_file = json.load(open(folder + json_file))

        for id in gesture_ids:
            idx = str(id)
            img_file = json_file[idx] + ".png"
            gif_file = json_file[idx] + ".gif"
            if download_imgs and (not img_file in files or redownload):
                os.system(curl_commands + git_url + img_folder + img_file)
            if download_gifs:
                if not gif_file in files or redownload:
                    os.system(curl_commands + git_url + gif_folder + gif_file)


    def launch_training(self, data_handler, num_reps=3, rep_time=3, rep_folder=None, output_folder=None, time_between_reps=3, randomize=False, continuous=False, gifs=False, exclude_files=[], width=450, height=450):
        """Launches the Screen Guided Training UI.

        Parameters
        ----------
        num_reps: int > 0
            The number of repetitions per class. 
        rep_time: int > 0
            The amount of time for each rep.
        time_between_reps: int > 0
            The amount of time between subsequent class prompts.
        rep_folder: string 
            The folder path where the images associated with each rep are located. Each image should be <class_name>.<png,jpg>.
        output_folder: string
            The folder path where the acquired data will be written to. 
        data_handler: OnlineDataHandler
            Online data handler used for acquiring raw EMG data.
        randomize: bool, default=False
            If True the classes are presented in a random order.
        continuous: bool, default=False
            If True there is no pause between reps.
        gifs: bool, default=False
            If True looks and plays gifs (this option of for discrete training).
        exclude_files: list, default=None
            A list of files (i.e., classes) to exclude. 
        width: int, default=450
            The width of the image/gif.
        height: int, default=450
            The height of the image/gif.
        """
        _SGTUI(num_reps=num_reps, rep_time=rep_time, rep_folder=rep_folder, output_folder=output_folder, data_handler=data_handler, time_between_reps=time_between_reps, randomize=randomize, continuous=continuous, gifs=gifs, exclude_files=exclude_files, width=width, height=height)


class _SGTUI:
    def __init__(self, num_reps=3, rep_time=3, rep_folder=None, output_folder=None, data_handler=None, time_between_reps=3, randomize=False, continuous=False, gifs=False, exclude_files=[], width=450, height=450):
        self.window = Tk()

        self.meta_data_dic = {}
        
        self.exclude_files = exclude_files
        self.num_reps = IntVar(value=num_reps)
        self.rep_time = IntVar(value=rep_time)
        self.rep_folder = StringVar(value=rep_folder)
        self.output_folder = StringVar(value=output_folder)
        self.time_between_reps = IntVar(value=time_between_reps)
        self.randomize = BooleanVar(value=randomize)
        self.continuous = BooleanVar(value=continuous)
        self.gifs = BooleanVar(value=gifs)
        self.inputs = []
        self.data_handler = data_handler
        self.og_inputs = []
        self.width = width
        self.height = height
        
        # For Data Accumulation Screen:
        self.pb = None
        self.cd_label = None
        self.image_label = None
        self.class_label = None
        self.rep_label = None
        self.next_rep_button = None
        self.redo_rep_button = None
        self.start_training_button = None
        self.rep_number = 0
        self.data_collecting_thread = None
        self.error_label = None
        # For UI
        self._intialize_UI()
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.window.mainloop()

    def _on_closing(self):
        self.window.destroy()

    def _accumulate_training_images(self):
        filenames = next(walk(self.rep_folder.get()), (None, None, []))[2]
        filenames.sort()
        file_types = [".jpg",".jpeg",".png"]
        if self.gifs.get():
            file_types = [".gif"]
        for file in filenames:
            if any(sub_str in file for sub_str in file_types):
                if not file in self.exclude_files:
                    self.inputs.append(file)
        self.og_inputs = list(self.inputs)
    
    def _clear_frame(self):
        for widgets in self.window.winfo_children():
            widgets.destroy()
            
    def _intialize_UI(self):
        self._clear_frame()
        self.window.title("Startup Screen")
        self.window.geometry("700x500")
        self.window.resizable(True, True)
        Label(text="Training Module", font=("Arial", 30)).pack(pady=10)
        label_font = ("Arial bold", 12)
        
        # Create Form
        form_frame = Frame(self.window)
        Label(form_frame, text="Num Reps:", font=label_font).grid(row=0, column=0, sticky=W, padx=(0,10), pady=(10,5))
        self._create_text_input(1, 0, 1, self.num_reps, form_frame)
        Label(form_frame, text="Time Per Rep:", font=label_font).grid(row=0, column=1, sticky=W, padx=(0,0), pady=(10,5))
        self._create_text_input(1, 1, 1, self.rep_time, form_frame)
        Label(form_frame, text="Time Between Rep:", font=label_font).grid(row=0, column=2, sticky=W, padx=(0,0), pady=(10,5))
        self._create_text_input(1, 2, 1, self.time_between_reps, form_frame)
        Label(form_frame, text="Input Folder:", font=label_font).grid(row=2, column=0, columnspan=3, sticky=W, pady=(10,5))
        self._create_text_input(3, 0, 3, self.rep_folder, form_frame)
        Label(form_frame, text="Output Folder:", font=label_font).grid(row=4, column=0, columnspan=3, sticky=W, pady=(10,5))
        self._create_text_input(5, 0, 3, self.output_folder, form_frame)
        Checkbutton(form_frame, text='Randomize', font=label_font, variable=self.randomize, onvalue=True, offvalue=False).grid(row=6, column=0, pady=(20,10), padx=(10,10))
        Checkbutton(form_frame, text='Continuous', font=label_font, variable=self.continuous, onvalue=True, offvalue=False).grid(row=6, column=1, pady=(20,10), padx=(10,10))
        Checkbutton(form_frame, text='GIFs', font=label_font, variable=self.gifs, onvalue=True, offvalue=False).grid(row=6, column=2, pady=(20,10), padx=(10,10))
        self.start_training_button = Button(form_frame, text = 'Start Training', font= ("Arial", 14), command=self._create_data_recording_screen)
        self.start_training_button.grid(row=7, column=0, columnspan=3, pady=(10,5))
        Button(form_frame, text = 'Visualize', font= ("Arial", 14), command=self._visualize).grid(row=8, column=0, columnspan=3, pady=(10,5))
        self.error_label = Label(form_frame, text="Online Data Handler not reading data...", font=('Arial', 12), bg='#FFFFFF', fg='#FF0000')
        self.error_label.grid(row=9, column=0, columnspan=5, sticky=N, pady=(10,5))
        form_frame.pack()

        # Listening for data in thread
        thread = Thread(target=self._listen_for_data)
        thread.daemon = True
        thread.start()
    
    def _visualize(self):
        self.data_handler.visualize()

    def _listen_for_data(self):
        self.start_training_button['state'] = 'disabled'
        # Error Checking - Waiting for ODH to start reading data
        while True:
            if self.data_handler and len(self.data_handler.raw_data.get_emg()) > 0:
                self.error_label.destroy()
                self.start_training_button['state'] = 'normal'
                break  

    def _create_text_input(self, row, col, col_span, default_text, frame):
        text_box_font = ("Arial", 12)
        entry = Entry(frame, font=text_box_font, textvariable=default_text)
        entry.grid(row=row, column=col, columnspan=col_span, sticky=N+S+W+E, padx=(0,10))
        return entry

    def _create_data_recording_screen(self):
        self._clear_frame()
        self.window.title("Data Accumulation")
        self.window.geometry("800x750")
        self.window.resizable(True, True)
        self._accumulate_training_images()

        # Create metadata
        self.meta_data_dic['continuous'] = self.continuous.get()
        self.meta_data_dic['randomize'] = self.randomize.get()
        self.meta_data_dic['gifs'] = self.gifs.get()
        
        # Create UI Elements
        self.pb = Progressbar(self.window, orient='horizontal', length=450, mode='determinate')
        self.cd_label = Label(text="X", font=("Arial", 25))
        self.image_label = _ImageLabel(self.window)
        self.class_label = Label(text="Label", font=("Arial", 25))
        self.rep_label = Label(text="Rep X of Y", font=("Arial", 25))

        # Add Elements:
        self.rep_label.pack()
        self.class_label.pack()
        self.image_label.pack()
        self.pb.pack(ipady=8, pady=10)
        self.cd_label.pack()

        # Start Data Collection...
        self._collect_data_in_thread()

    def _collect_data_in_thread(self):
        self.data_collecting_thread = Thread(target=self._collect_data)
        self.data_collecting_thread.daemon = True
        self.data_collecting_thread.start()
    
    def _collect_data(self):
        self.rep_label["text"] = "Rep " + str(self.rep_number + 1) + " of " + str(self.num_reps.get())
        if self.rep_number < int(self.num_reps.get()):
            if self.randomize.get(): 
                random.shuffle(self.inputs)
            for file in self.inputs:
                for val in range(0,2):
                    image_file = str(self.rep_folder.get() + file)
                    cd_time = int(self.time_between_reps.get())
                    if val == 0:
                        if self.continuous.get():
                            continue
                        self.image_label.unload()
                        self.image_label.load(image_file, True, self.width, self.height, self.rep_time.get(), self.time_between_reps.get())
                    else:
                        self.image_label.unload()
                        self.image_label.load(image_file, False, self.width, self.height, self.rep_time.get(), self.time_between_reps.get())
                        cd_time = self.rep_time.get()
                    self._update_class(str(file.split(".")[0]))
                    if val != 0:
                        self.data_handler.raw_data.reset_emg()
                    self._bar_count_down(cd_time, file)
            self.rep_number += 1
            self.next_rep_button = Button(self.window, text = 'Next Rep', font = ("Arial", 12), command=self._next_rep)
            self.redo_rep_button = Button(self.window, text = 'Redo Rep', font = ("Arial", 12), command=self._redo_rep)
            self.next_rep_button.pack()
            self.redo_rep_button.pack(pady = 10)
        else:
            self._intialize_UI()  
            self.rep_number = 0
            return
    
    def _redo_rep(self):
        self.rep_number -= 1
        self._next_rep()

    def _next_rep(self):
        self.next_rep_button.destroy()
        self.redo_rep_button.destroy()
        self._collect_data_in_thread()
        
    def _bar_count_down(self, seconds, file):
        self.pb['value'] = 0
        for i in range (0, seconds):
            self.cd_label['text'] = seconds - i
            self.pb['value'] += (100/seconds)
            self.window.update_idletasks()
            time.sleep(1)
            self._write_data(self.data_handler.raw_data.get_emg(), i, file)
            self.data_handler.raw_data.reset_emg() 
    
    def _update_class(self,label):
        self.class_label['text'] = "Class: " + str(label)
        self.window.update_idletasks()
    
    def _write_data(self, data, time, file):
        flag = "a"
        if time == 0:
            flag = "w"
        if not os.path.isdir(self.output_folder.get()):
            os.makedirs(self.output_folder.get()) 

        c = self.og_inputs.index(file)
        # Write EMG Files
        emg_file = self.output_folder.get() + "R_" + str(self.rep_number) + "_C_" + str(c) + ".csv"
        self.meta_data_dic[emg_file] = {
            'rep_idx': self.rep_number,
            'class_idx': self.og_inputs.index(file),
            'class_name': self.og_inputs[c].split(".")[0],
            'file_type': self.og_inputs[c].split(".")[1]
        }
        with open(emg_file, flag, newline='', encoding='utf-8') as file:
            emg_writer = csv.writer(file)
            emg_writer.writerows(data)

        # Write Metadata file
        with open(self.output_folder.get() + "metadata.json", 'w') as f: 
            f.write(json.dumps(self.meta_data_dic))

from itertools import count, cycle
"""
Credit goes to: https://pythonprogramming.altervista.org/animate-gif-in-tkinter/
"""
class _ImageLabel(Label):
    def load(self, im_file, gray, width, height, rep_time, btwn_rep_time):
        if ".gif" in im_file:
            im = Image.open(im_file)
        else:
            im = Image.open(im_file).resize((width, height))
        frames = []
        delays = []
        try:
            for i in count(1):
                if gray:
                    frames.append(ImageTk.PhotoImage(im.copy().convert('L').resize((width, height))))
                else:
                    frames.append(ImageTk.PhotoImage(im.copy().resize((width, height))))
                delays.append(im.info['duration'])
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)
        self.delays = delays
        self.delays = [int(d / (sum(delays)/(rep_time*1000))) for d in delays]
        self.times = np.cumsum(self.delays)
        self.timer = None
        self.curr_frame = 0
        
        if len(frames) == 1:
            self.config(image=next(self.frames))
        else: 
            self.next_frame()
 
    def unload(self):
        self.config(image=None)
        self.frames = None
        self.delays = None
        self.timer = None
        self.curr_frame = 0
 
    def next_frame(self):
        if self.timer == None:
            self.timer = time.time()

        if self.frames and self.delays:
            nearest_time = (self.times - (time.time() - self.timer)*1000)
            nearest_frame = np.where((nearest_time > 0))[0][0]
            while(self.curr_frame < nearest_frame):
                self.config(image=next(self.frames))
                self.curr_frame += 1
            self.config(image=next(self.frames))
            self.after(int(self.delays[self.curr_frame] - (nearest_time[nearest_frame]-self.delays[self.curr_frame])), self.next_frame)
            self.curr_frame += 1
        else:
            self.unload()


import dearpygui.dearpygui as dpg
import os
from itertools import compress
import time
from PIL import Image
import PIL
import numpy as np
import libemg
import csv

class DataCollection:
    def __init__(self,odh):
        self.odh = odh


    def launch_training(self,
                        num_reps=3,
                        rep_time=3,
                        media_folder='media/',
                        data_folder='data/',
                        rest_time=2,
                        randomize=False,
                        continuous=False,
                        gifs=False,
                        exclude_files=[],
                        width=800,
                        height=800):
        self.num_reps = num_reps
        self.rep_time = rep_time
        self.media_folder = media_folder
        self.data_folder = data_folder
        self.rest_time = rest_time
        self.randomize = randomize
        self.continuous = continuous
        self.gifs = gifs
        self.excluded_files = exclude_files
        self.width = width
        self.height = height

        self.base_gui_setup()
        self.construct_conf()
        self.start_gui()

    def base_gui_setup(self):
        dpg.create_context()
        dpg.create_viewport(title="LibEMG - SGT",
                            width=self.width,
                            height=self.height)
        dpg.setup_dearpygui()

        
    
    def start_gui(self):
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def construct_conf(self):
        with dpg.window(label="SGT - Setup",
                        tag="__setup_window",
                        width=self.width,
                        height=self.height):
            
            dpg.add_text("Training Menu")
            with dpg.group(horizontal=True) as rep_group: 
                dpg.add_text("Num Reps: ")
                dpg.add_input_text(default_value=self.num_reps,
                                tag="__num_reps",
                                width=100)
                dpg.add_text("Time Per Rep")
                dpg.add_input_text(default_value=self.rep_time,
                                tag="__rep_time",
                                width=100)
                dpg.add_text("Time Between Reps")
                dpg.add_input_text(default_value=self.rest_time,
                                tag="__rest_time", 
                                width=100)
            
            with dpg.group(horizontal=True) as folder_group:
                dpg.add_text("Media Folder:")
                dpg.add_input_text(default_value=self.media_folder, 
                                tag="__media_folder", width=250)
                dpg.add_text("Output Folder:")
                dpg.add_input_text(default_value=self.data_folder, 
                                tag="__output_folder",
                                width=250)
            
            with dpg.group(horizontal=True) as checkbox_group:
                dpg.add_text("Randomize")
                dpg.add_checkbox(default_value=self.randomize,
                                tag="__randomize")
                dpg.add_text("Continuous")
                dpg.add_checkbox(default_value=self.continuous,
                                tag="__continuous")
                dpg.add_text("GIFs")
                dpg.add_checkbox(default_value=self.gifs,
                                tag="__gifs")
            
            with dpg.group(horizontal=True) as button_group:
                dpg.add_button(label="Start", callback=self.start_callback)
                dpg.add_button(label="Visualize", callback=self.visualize_callback)

    def visualize_callback(self):
        print("visualize clicked")

    def start_callback(self):
        # only allow start to be pressed if odh receives data
        if self.odh and len(self.odh.raw_data.get_emg()) > 0:
            self.get_settings()
            dpg.delete_item("__setup_window")
            media = self.gather_media()
            self.construct_collection(media)

    def gather_media(self):
        # find everything in the media folder
        files = os.listdir(self.media_folder)
        valid_files = [file.endswith((".gif",".png")) for file in files]
        files = list(compress(files, valid_files))
        collection_conf = []
        for rep in range(self.num_reps):
            for motion_class in files:
                # entry for collection of rep
                collection_conf.append([motion_class,motion_class.split('.')[0],rep,self.rep_time])
        return collection_conf

    def update_raw_texture(self, tag):
        # self.im.putalpha(255)
        with dpg.texture_registry(show=False):
            self.im.seek(self.frame)
            dpg_image = self.im.resize((720,480))
            dpg_image = dpg_image.convert("RGB")
            dpg_image = np.asarray(dpg_image,dtype=np.uint8)
            dpg_image = dpg_image.ravel()
            dpg_image = np.asfarray(dpg_image, dtype='f')
            dpg_image = np.true_divide(dpg_image, 255.0)
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, dpg_image)
            else:
                dpg.add_raw_texture(width=self.im.width, height=self.im.height, default_value=dpg_image,
                                     format=dpg.mvFormat_Float_rgb, tag=tag)

    def construct_collection(self, media):
        # media is a list of files we need to grab
        # Now spawn the collection window
        
        # open first frame of gif
        self.im  = Image.open(self.media_folder + media[0][0])
        self.frame = 0
        self.im.seek(self.frame)
        self.update_raw_texture("texture_tag")

        with dpg.window(label="SGT - Collection",
                        tag="__collection_window",
                        width=self.width,
                        height=self.height):
            dpg.add_text(default_value="Collection Menu")
            dpg.add_text(media[0][1], tag="__prompt")
            dpg.add_image("texture_tag")
            dpg.add_progress_bar(tag="progress", default_value=0.0,width=self.im.width)

        # this is where we will hold the data
        
        for i, entry  in enumerate(media):
            dpg.set_value("__prompt", value=media[i][1])
            # load the media corresponding to the current class file in
            self.im  = Image.open(self.media_folder + entry[0])
            # print(self.media_folder + media[self.class_number])
            
            self.frame = 0
            self.im.seek(self.frame)
            self.data = []
            self.odh.raw_data.reset_emg()
            # self.odh.raw_data.reset_imu() #TODO: Uncomment when merged to main
            # self.odh.raw_data.reset_others()
            self.timer = time.perf_counter_ns()
            frame_time = time.perf_counter_ns()
            while time.perf_counter_ns() - self.timer < 1e9*self.rep_time:
                self.collection_timestep(rep=True)
                time.sleep(max(0,1/24 - (time.perf_counter_ns() - frame_time)/1e9))
                frame_time = time.perf_counter_ns()

            self.save_data(entry)

            # greyscale next rep to indicate what is next during rest
            if self.rest_time and i+1 < len(media):
                dpg.set_value("__prompt", value="Up next: "+media[i+1][1])
                self.im = Image.open(self.media_folder + media[i+1][0])
                self.frame = 0
                self.im = self.im.convert("L")
                self.update_raw_texture("texture_tag")
                
                self.timer = time.perf_counter_ns()
                frame_time = time.perf_counter_ns()
                while time.perf_counter_ns() - self.timer < 1e9*self.rest_time:
                    self.collection_timestep(rep=False)
                    time.sleep(max(0,1/24 - (time.perf_counter_ns() - frame_time)/1e9))
                    frame_time = time.perf_counter_ns()
            
        dpg.delete_item("__collection_window")

    def collection_timestep(self, rep=True):
        self.update_media()
        self.update_progress(rep)
            
    def update_media(self):
        if hasattr(self.im, "n_frames"):
            if self.frame + 1 < self.im.n_frames:
                self.im.seek(self.frame)
            self.update_raw_texture("texture_tag")
            self.frame = self.frame + 1


    def update_progress(self, rep=True):
        if rep:
            amount = min(1,(time.perf_counter_ns() - self.timer)/(1e9*self.rep_time))
        else:
            amount = min(1,(time.perf_counter_ns() - self.timer)/(1e9*self.rest_time))
        dpg.set_value("progress", value = amount)


    def save_data(self, media_entry):
        self.data = self.odh.raw_data.get_emg()
        with open(self.output_folder + media_entry[1] + "_" + str(media_entry[2]) + ".csv", "w", newline='', encoding='utf-8') as file:
            emg_writer = csv.writer(file)
            for row in self.data:
                emg_writer.writerow(row)
        self.odh.raw_data.reset_emg()

    def get_settings(self):
        self.num_reps     = int(dpg.get_value("__num_reps"))
        self.rep_time     = float(dpg.get_value("__rep_time"))
        self.rest_time    = float(dpg.get_value("__rest_time"))
        self.media_folder = dpg.get_value("__media_folder")
        self.output_folder  = dpg.get_value("__output_folder")
        self.randomize    = bool(dpg.get_value("__randomize"))
        self.continuous   = bool(dpg.get_value("__continuous"))
        self.gifs         = bool(dpg.get_value("__gifs"))