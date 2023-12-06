import dearpygui.dearpygui as dpg
import os
import json
import libemg


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
                        label="Data Import Configuration",
                        width=720,
                        height=480):
            
            # dpg.add_spacer(height=50)
            dpg.add_text(label="Import Menu")
            
            with dpg.group(horizontal=True):
                dpg.add_text(label="Data Folder")
                dpg.add_input_text(default_value=self.data_folder, tag="__di_data_folder", callback=self.check_data_folder)
            
            dpg.add_text(label="Preliminary Check Portal")

            dpg.add_text(label=" ", tag="__di_folder_validation")
            self.check_data_folder()

            dpg.add_button(label="Import", callback=self.import_button_callback)
            
            dpg.add_text(label="Import Portal")
            
            dpg.add_text(label=" ", tag="__di_import_validation")

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
                        validation_text += "\t" +  key + ": "+\
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
        dpg.set_value(item="__di_folder_validation", value=validation_text)

    def import_button_callback(self):
        validation_text = ""
        if not self.ready_to_import:
            validation_text += "Import not available -- check data folder."
        else:
            offline_data_handler = libemg.data_handler.OfflineDataHandler()
            offline_data_handler.get_data(dpg.get_value(item="__di_data_folder"),
                                          {"classes":       [str(i) for i in range(self.collection_details["num_motions"])],
                                           "classes_regex": libemg.utils.make_regex("C_", "_R_",[str(i) for i in range(self.collection_details["num_motions"])]),
                                           "reps":          [str(i) for i in range(self.collection_details["num_reps"])],
                                           "reps_regex":    libemg.utils.make_regex("_R_",".csv", [str(i) for i in range(self.collection_details["num_reps"])])
                                           }
                                         )
            self.gui.offline_data_handlers.append(offline_data_handler)
            validation_text += "Import successful"
        

        dpg.set_value(item="__di_import_validation", value=validation_text)
        