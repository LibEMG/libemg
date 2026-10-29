import dearpygui.dearpygui as dpg
import os
import json
import libemg
import numpy as np
from libemg.data_handler import RegexFilter
from libemg._gui import _style as style

class DataImportPanel:
    def __init__(self, 
                 data_folder="data/",
                 gui=None):
        
        self.data_folder = data_folder
        self.widget_tags = {"configuration": ['__di_configuration_window', '__di_data_folder', '__di_folder_validation', \
                                              '__di_import_validation']}
        self.gui = gui

    def cleanup_window(self, window_name):
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)     
    
    def spawn_configuration_window(self):
        self.cleanup_window("configuration")
        with dpg.window(tag="__di_configuration_window",
                        label="Import Data",
                        width=720,
                        height=480):
            style.note("Load a dataset recorded with LibEMG: a folder of CSV files with a "
                       "collection_details.json beside them.", wrap=680)

            style.section("Folder")
            with style.form(label_width=120):
                with style.row("Data folder", help="The folder the dataset was saved to. "
                               "End it with a slash."):
                    dpg.add_input_text(default_value=self.data_folder, tag="__di_data_folder",
                                       width=-1, callback=self.check_data_folder)
            style.note("", tag="__di_folder_validation", wrap=680)
            self.check_data_folder()

            style.section("Import")
            with style.form(label_width=120):
                with style.row("Name", help="What the dataset is called elsewhere in the window."):
                    dpg.add_input_text(default_value="Dataset" + str(np.random.randint(0,100)),
                                       tag="__di_data_alias", width=260)
            style.button("Import", "primary", width=110, callback=self.import_button_callback)
            style.note("", tag="__di_import_validation", wrap=680)

    def check_data_folder(self):
        folder_location   = dpg.get_value(item="__di_data_folder")
        validation_text = ""
        self.ready_to_import = False
        if os.path.exists(folder_location):
            self.details_flag = "collection_details.json" in os.listdir(folder_location)
            
            if self.details_flag:
                validation_text += "Found collections_details \n"
                if os.path.exists(folder_location + "collection_details.json"):
                    with open(folder_location + "collection_details.json") as f:
                        self.collection_details = json.load(f)
                    for key in self.collection_details.keys():
                        # Spaces, not a tab: DearPyGui draws a tab as nothing.
                        validation_text += "    " +  key + ": "+\
                              str(self.collection_details[key]) + "\n"
                    
                    validation_text += "Ready to Import!"
                    self.ready_to_import = True
                else:
                    validation_text += "Folder and collection_details_exists, but file not found\n" +\
                                       "Ensure folder ends with /"
                    
            else:
                validation_text += "Collection_details file missing\n" + \
                                    "If dataset was collected outside of libemg, make sure dataset is interfaceable and has a collection_details.json file!"
        else:
            validation_text += "Folder does not exist."
        style.set_status("__di_folder_validation", validation_text,
                         "ok" if self.ready_to_import else "warn")

    def import_button_callback(self):
        validation_text = ""
        if not self.ready_to_import:
            validation_text += "Import not available -- check data folder."
        else:
            offline_data_handler = libemg.data_handler.OfflineDataHandler()
            regex_filters = [
                RegexFilter("C_", "_R_",[str(i) for i in range(self.collection_details["num_motions"])], description='classes'),
                RegexFilter("_R_",".csv", [str(i) for i in range(self.collection_details["num_reps"])], description='reps')
            ]
            offline_data_handler.get_data(dpg.get_value(item="__di_data_folder"), regex_filters=regex_filters)
            self.gui.offline_data_handlers.append(offline_data_handler)
            self.gui.offline_data_aliases.append(dpg.get_value("__di_data_alias"))
            validation_text += "Import successful"

        style.set_status("__di_import_validation", validation_text,
                         "ok" if self.ready_to_import else "error")
        