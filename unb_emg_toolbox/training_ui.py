import csv
import time
import os
from tkinter import *
from tkinter.ttk import Progressbar
from threading import Thread
from PIL import ImageTk, Image
from os import walk
import random
random. seed(time.time())

class TrainingUI:
    """Screen Guided Training class.

    Used to create a screen guided training module for acquiring training or testing data.

    Parameters
    ----------
    num_reps: int > 0
        The number of repetitions per class. 
    rep_time: int > 0
        The amount of time for each rep.
    time_between_reps: int > 0
        The amount of time between subsequent classes.
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
    """
    def __init__(self, num_reps=None, rep_time=None, time_between_reps=3, rep_folder=None, output_folder=None, data_handler=None, randomize=False, continuous=False):
        self.window = Tk()
        
        self.num_reps = IntVar(value=num_reps)
        self.rep_time = IntVar(value=rep_time)
        self.rep_folder = StringVar(value=rep_folder)
        self.output_folder = StringVar(value=output_folder)
        self.time_between_reps = IntVar(value=time_between_reps)
        self.randomize = BooleanVar(value=randomize)
        self.continuous = BooleanVar(value=continuous)
        self.inputs = []
        self.data_handler = data_handler
        self.og_inputs = []
        
        # For Data Accumulation Screen:
        self.pb = None
        self.cd_label = None
        self.image_label = None
        self.class_label = None
        self.rep_label = None
        self.next_rep_button = None
        self.rep_number = 0
        self.data_collecting_thread = None

        # For UI
        self._intialize_UI()
        self.window.mainloop()
    
    def _accumulate_training_images(self):
        filenames = next(walk(self.rep_folder.get()), (None, None, []))[2]
        file_types = [".jpg", ".png"]
        for file in filenames:
            if any(sub_str in file for sub_str in file_types):
                self.inputs.append(file)
        self.og_inputs = list(self.inputs)
    
    def _clear_frame(self):
        for widgets in self.window.winfo_children():
            widgets.destroy()
            
    def _intialize_UI(self):
        self._clear_frame()
        self.window.title("Startup Screen")
        self.window.geometry("450x400")
        self.window.resizable(False, False)
        Label(text="EMG Training", font=("Arial", 30)).pack(pady=10)
        label_font = ("Arial bold", 10)
        
        # Create Form
        form_frame = Frame(self.window)
        Label(form_frame, text="Num Reps:", font=label_font).grid(row=0, column=0, sticky=W, padx=(0,10), pady=(10,5))
        self._create_text_input(1, 0, 1, self.num_reps, form_frame)
        Label(form_frame, text="Time Per Rep:", font=label_font).grid(row=0, column=1, sticky=W, padx=(0,0), pady=(10,5))
        self._create_text_input(1, 1, 1, self.rep_time, form_frame)
        Label(form_frame, text="Input Folder:", font=label_font).grid(row=2, column=0, columnspan=2, sticky=W, pady=(10,5))
        self._create_text_input(3, 0, 2, self.rep_folder, form_frame)
        Label(form_frame, text="Output Folder:", font=label_font).grid(row=4, column=0, columnspan=2, sticky=W, pady=(10,5))
        self._create_text_input(5, 0, 2, self.output_folder, form_frame)
        Checkbutton(form_frame, text='Randomize', font=label_font, variable=self.randomize, onvalue=True, offvalue=False).grid(row=6, column=0, pady=(10,5))
        Checkbutton(form_frame, text='Continuous', font=label_font, variable=self.continuous, onvalue=True, offvalue=False).grid(row=6, column=1, pady=(10,5))
        Button(form_frame, text = 'Start Training', command=self._create_data_recording_screen).grid(row=7, column=0, columnspan=2, pady=(10,5))
        form_frame.pack()
    
    def _create_text_input(self, row, col, col_span, default_text, frame):
        text_box_font = ("Arial", 10)
        entry = Entry(frame, font=text_box_font, textvariable=default_text)
        entry.grid(row=row, column=col, columnspan=col_span, sticky=N+S+W+E, padx=(0,10))
        return entry

    def _create_data_recording_screen(self):
        self._clear_frame()
        self.window.title("Data Accumulation")
        self.window.geometry("1000x1000")
        self._accumulate_training_images()
        
        # Create UI Elements
        self.pb = Progressbar(self.window, orient='horizontal', length=500, mode='determinate')
        self.cd_label = Label(text="X", font=("Arial", 25))
        self.image_label = Label(self.window, image = None)
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
                        img = ImageTk.PhotoImage(Image.open(image_file).convert('L').resize((500,500)))
                    else:
                        img = ImageTk.PhotoImage(Image.open(image_file).resize((500,500)))
                        cd_time = self.rep_time.get()
                    self._update_class(str(file.split(".")[0]),img)
                    if val != 0:
                        self.data_handler.raw_data.reset_emg()
                    self._bar_count_down(cd_time)
                    if val != 0:
                        self._write_data(file, self.rep_number)
            self.rep_number = self.rep_number + 1
            self.next_rep_button = Button(self.window, text = 'Next Rep', command=self._next_rep)
            self.next_rep_button.pack(pady = 10)
        else:
            self._intialize_UI()  
            self.rep_number = 0
            return
    
    def _next_rep(self):
        self.next_rep_button.destroy()
        self._collect_data_in_thread()
        
    def _bar_count_down(self, seconds):
        self.pb['value'] = 0
        for i in range (0, seconds):
            self.cd_label['text'] = seconds - i
            self.pb['value'] += (100/seconds)
            self.window.update_idletasks()
            time.sleep(1)
    
    def _update_class(self,label,image):
        self.class_label['text'] = "Class: " + str(label)
        self.image_label['image'] = image
        self.window.update_idletasks()
    
    def _write_data(self, input, rep):
        if not os.path.isdir(self.output_folder.get()):
            os.makedirs(self.output_folder.get()) 
        emg_file = self.output_folder.get() + "R_" + str(rep) + "_C_" + str(self.og_inputs.index(input)) + ".csv"
        with open(emg_file, "w", newline='', encoding='utf-8') as file:
            emg_writer = csv.writer(file)
            for row in self.data_handler.raw_data.get_emg():
                emg_writer.writerow(row)
            # emg_writer.writerows(self.data_handler.raw_data.get_emg_data())
