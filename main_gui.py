import libemg
from tkinter import *
import numpy as np
import os
from sklearn.linear_model import ElasticNet
import sklearn.metrics as metrics

from libemg.data_handler import RegexFilter

SUBJECT_NUMBER = 100
WINDOW_SIZE = 350
WINDOW_INC  = 98
FEATURES    = ["MAV","ZC","SSC","WL"]#["RMSPHASOR"]#

class GUI:
    # lets set up some of the communication stuff here
    def __init__(self):
        # streamer initialization goes here.
        # I am using a myo, but you could sub in the delsys here
        # self.streamer = libemg.streamers.myo_streamer()
        self.streamer = libemg.streamers.sifibridge_streamer(notch_on=True, notch_freq=60,
                                                 emg_fir_on=True,
                                                 emg_fir=[20,450])

        # create an online data handler to listen for the data
        self.odh = libemg.data_handler.OnlineDataHandler()
        # when we start listening we subscribe to the data the device is putting out
        self.odh.start_listening()

        # save_directory:
        self.save_directory = 'data/subject'+str(SUBJECT_NUMBER)+"/"
        if not os.path.isdir(self.save_directory):
            os.makedirs(self.save_directory)
        
        # make the gui
        self.initialize_ui()
        # hang
        self.window.mainloop()
        
    # lets set up some of the GUI stuff here
    def initialize_ui(self):
        # tkinter window (ugly but simple to code up)
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.title("LibEMG GUI")
        self.window.geometry("500x300")

        # add some widgets to the gui

        # get training data button
        Button(self.window, font=("Arial", 10), text="Screen Guided Training", 
               command=self.launch_training).pack(pady=(0,10))
        # start signal visualization button
        #   Note: I'm not sure how nicely things play when data is being consumed in multiple regions
        #         i.e., visualizing signal, classifying on signal, etc.
        #         This is probably fine.
        Button(self.window, font=("Arial", 10), text="Visualize signal", 
               command=self.visualize_signal).pack(pady=(0,10))
        # start live classifier button
        Button(self.window, font=("Arial", 10), text="Visualize Feature Space", 
               command=self.visualize_feature_space).pack(pady=(0,10))
        # start live classifier button
        Button(self.window, font=("Arial", 10), text="Start Live Classifier", 
               command=self.start_classifier).pack(pady=(0,10))
        # visualize live classifier 
        Button(self.window, font=("Arial", 10), text="Visualize classifier", 
               command=self.visualize_classifier).pack(pady=(0,10))
        # do regression things
        # get training data button
        Button(self.window, font=("Arial", 10), text="Screen Guided Training - Regressor", 
               command=self.launch_regressor_training).pack(pady=(0,10))
        # train regressor
        Button(self.window, font=("Arial", 10), text="Regressor stuff", 
               command=self.regression_stuff).pack(pady=(0,10))
        
    def launch_training(self):
        self.window.destroy()
        # get rid of GUI window in favour of libemg training gui
        training_ui = libemg.screen_guided_training.ScreenGuidedTraining()
        # you can find what these numbers in the list correspond to at:
        # https://github.com/LibEMG/LibEMGGestures
        # training will have all the classes you've downloaded into the images folder
        training_ui.download_gestures([1,2,3,4,5,6,7], "images/")
        training_ui.launch_training(self.odh, 4, 5, "images/", self.save_directory, 1)
        # the thread is blocked now until the training process is completed and closed 
        # once the training process ends, relaunch the GUI
        self.initialize_ui()
    

    def launch_regressor_training(self):
        self.window.destroy()
        # get rid of GUI window in favour of libemg training gui
        training_ui = libemg.screen_guided_training.ScreenGuidedTraining()
        # you can find what these numbers in the list correspond to at:
        # https://github.com/LibEMG/LibEMGGestures
        # training will have all the classes you've downloaded into the images folder
        # training_ui.download_gestures([1,2,3,4,5,6,7], "images/")
        training_ui.launch_training(self.odh, 
                            num_reps=5,
                            rep_time=10,
                            rep_folder="animation/",
                            output_folder=self.save_directory,
                            width=720,
                            height=480,
                            continuous=True,
                            gifs=True)
        # the thread is blocked now until the training process is completed and closed 
        # once the training process ends, relaunch the GUI
        self.initialize_ui()
    
    def visualize_signal(self):
        self.window.destroy()
        self.odh.visualize(num_samples=5000)
        # ^ this blocks until its closed
        # v once closed it'll reopen the GUI
        self.initialize_ui()

    def visualize_feature_space(self):
        self.window.destroy()
        offlinedatahandler = self.get_data()
        windows, metadata  = self.extract_windows(offlinedatahandler)
        features = self.extract_features(windows)
        libemg.feature_extractor.FeatureExtractor().visualize_feature_space(features, "PCA", classes=metadata["classes"])
        # ^ this blocks until its closed
        # v once closed it'll reopen the GUI
        self.initialize_ui()


    
    def start_classifier(self):
        offlinedatahandler = self.get_data()
        windows, metadata  = self.extract_windows(offlinedatahandler)
        features = self.extract_features(windows)
        # we need to make an offline classifier to pass to the online classifier
        offlineclassifier = libemg.emg_predictor.EMGClassifier('LDA')
        feature_dictionary = {"training_features": features,
                              "training_labels"  : metadata["classes"]}
        offlineclassifier.fit(feature_dictionary=feature_dictionary)
        self.onlineclassifier = libemg.emg_predictor.OnlineEMGClassifier(offlineclassifier,
                                                                          WINDOW_SIZE,
                                                                          WINDOW_INC,
                                                                          self.odh,
                                                                          FEATURES,
                                                                          std_out=True,
                                                                          output_format="probabilities")
        # start running the online classifier in another thread (block=False)
        self.onlineclassifier.run(block=False)
        

    def visualize_classifier(self):
        self.window.destroy()
        self.onlineclassifier.visualize(legend=["Hand Closed", "Hand Open", "No Motion", "Pronation","Supination","Wrist Extension", "Wrist Flexion"])
        self.initialize_ui()

    def get_data(self):
        classes_values = [str(i) for i in range(10)] # will only grab classes 0,1,2,3,4 currently
        reps_values    = [str(i) for i in range(4)] # will only grab reps 0,1,2 currently
        regex_filters = [
            RegexFilter(left_bound="_C_",right_bound=".csv", values=classes_values, description='classes'),
            RegexFilter(left_bound="/R_", right_bound="_C_", values=reps_values, description='reps')
        ]
        offlinedatahandler = libemg.data_handler.OfflineDataHandler()
        offlinedatahandler.get_data(folder_location=self.save_directory, regex_filters=regex_filters, delimiter=',')
        return offlinedatahandler
    
    def extract_windows(self, offlinedatahandler):
        windows, metadata  = offlinedatahandler.parse_windows(WINDOW_SIZE, WINDOW_INC)
        return windows, metadata
    
    def extract_features(self, windows):
        features           = libemg.feature_extractor.FeatureExtractor().extract_features(FEATURES,
                                                                                         windows)
        return features
    

    def regression_stuff(self):
        offdh = libemg.data_handler.OfflineDataHandler()
        regex_filters = [
            RegexFilter(left_bound="R_", right_bound="_C_", values = ["0","1","2","3","4"], description='reps')
        ]
        offdh.get_data(folder_location=self.save_directory, regex_filters=regex_filters, delimiter=",")
        # offdh.add_regression_labels(file_location="animation/class_file.txt",
        #                             colnames = ["timestamp", "hand","regression0", "regression1"])
        metadata_operations = {
            "timestamp": np.mean,
            "hand": [np.int64, np.bincount, np.argmax],
            "regression0": np.mean,
            "regression1": np.mean
        }

        # Isolate data
        train_odh = offdh.isolate_data("reps", [0,1,2])
        train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE,WINDOW_INC, metadata_operations)
        train_labels = np.hstack((np.expand_dims(train_metadata["regression0"],axis=1), 
                                    np.expand_dims(train_metadata["regression1"],axis=1)))

        test_odh = offdh.isolate_data("reps", [3,4])
        test_windows, test_metadata = test_odh.parse_windows(WINDOW_SIZE,WINDOW_INC, metadata_operations)
        test_labels = np.hstack((np.expand_dims(test_metadata["regression0"],axis=1), 
                                    np.expand_dims(test_metadata["regression1"],axis=1)))

        
        fe = libemg.feature_extractor.FeatureExtractor()
        features = fe.extract_features(FEATURES,train_windows)
        feature_dic = {
            'training_features': features,
            'training_labels': train_labels
        }

        reg = libemg.emg_predictor.EMGRegressor(ElasticNet())
        reg.fit(feature_dic)
        predictions = reg.run(fe.extract_features(FEATURES, test_windows), test_labels)
        dof1_r2 = metrics.r2_score(test_metadata["regression0"], predictions[:,0])
        print(dof1_r2)
        dof2_r2 = metrics.r2_score(test_metadata["regression1"], predictions[:,1])
        print(dof2_r2)


        online_classifier = libemg.emg_predictor.OnlineEMGRegressor(reg, WINDOW_SIZE, WINDOW_INC, self.odh, FEATURES, std_out=True)
        online_classifier.run(block=True)


    # what happens when the GUI is destroyed
    def on_closing(self):
        # Clean up all the processes that have been started
        self.odh.stop_listening()
        self.window.destroy()


if __name__ == "__main__":
    gui = GUI()
