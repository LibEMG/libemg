import numpy as np
import os
import re


class DataHandler:
    def __init__(self):
        # self.training_data = np.empty()
        # self.testing_data = np.empty()
        # self.training_label = np.empty()
        # self.testing_label = np.empty()
        self.data = []

    def get_data(self):
        pass

class OfflineDataHandler(DataHandler):
    def __init__(self, folder_location,
                       filename_dictionary={}):
        super().__init__()

        self.folder_location = folder_location
        self.filename_dictionary = filename_dictionary
    
    def get_data(self, delimiter=","):
        # check that the specified folder exists

        # you can insert custom member variables that will be collected from the filename using the dictionary
        # this gives at least a tiny bit of flexibility around what is recorded aside from the data
        dictionary_keys = self.filename_dictionary.keys()
        keys = [k for k in dictionary_keys if not k.endswith("_regex")]
        for k in keys:
            if not hasattr(self, k):
                setattr(self, k, [])

        if not os.path.exists(self.folder_location):
            print("Invalid dataset directory")
        
        # get all files in directory
        files = os.listdir(self.folder_location)
        for f in files:
        
            # collect the data from the file
            self.data.append(np.genfromtxt(self.folder_location + "/" + f,delimiter=delimiter))
            # also collect the metadata from the filename
            for k in keys:
                k_val = re.findall(self.filename_dictionary[k+"_regex"],f)[0]
                k_id  = self.filename_dictionary[k].index(k_val)
                setattr(self, k, getattr(self,k)+[k_id])


            

def make_regex(left_bound, right_bound, values=[]):
    regex = ""
    left_bound_str = "(?<="+ left_bound +")"
    mid_str = "["
    for i in values:
        mid_str += i + "|"
    mid_str = mid_str[:-1]
    mid_str += "]+"
    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str



    
if __name__ == '__main__':
    # an example for how this can be used
    folder_location = 'unb_emg_toolbox/data/'
    # we also want to extract the classes and rep from the filename
    # the classes are 0-10
    # we can find the classes by using a regex that begins with _ and ends with .txt
    # the reps are just 0
    # we can find the reps by using a regex that begins with e_ and ends with _
    class_values = ["0","1","2","3","4","5","6","7","8","9","10"]
    rep_values = ["0"]
    classes_regex = make_regex(left_bound="_", right_bound=".txt", values=class_values)
    reps_regex = make_regex(left_bound="e_", right_bound="_", values=rep_values)

    filename_dictionary ={
        "classes": class_values,
        "reps":    rep_values,
        "classes_regex": classes_regex, #r"(?<=_)[^#]+(?=.txt)",# the # here means only a number is accepted... need to find a more wildcard solution
        "reps_regex": reps_regex # "(?<=e_)[]+(?=_)"
    }
    odh = OfflineDataHandler(folder_location=folder_location, filename_dictionary=filename_dictionary)
    odh.get_data()

    # show windowing and feature extraction
    A = 1
