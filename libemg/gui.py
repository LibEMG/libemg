import dearpygui.dearpygui as dpg
from libemg._gui._data_collection_panel import DataCollectionPanel
from libemg._gui._data_import_panel import DataImportPanel
from inspect import signature

class GUI:
    def __init__(self, 
                 width=1920,
                 height=1080,
                 args = None, 
                 debug = False):
        self.args = args
        self.window_init(width, height, debug)
        
    
    def window_init(self, width, height, debug=False):
        dpg.create_context()
        dpg.create_viewport(title="LibEMG",
                            width=width,
                            height=height)
        dpg.setup_dearpygui()
        

        self.file_menu_init()

        dpg.show_viewport()

        if debug:
            dpg.configure_app(manual_callback_management=True)
            while dpg.is_dearpygui_running():
                jobs = dpg.get_callback_queue()
                dpg.run_callbacks(jobs)
                dpg.render_dearpygui_frame()
        else:
            dpg.start_dearpygui()
        dpg.destroy_context()

    def file_menu_init(self):

        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Exit")
                
            with dpg.menu(label="Data"):
                dpg.add_menu_item(label="Collect Data", callback=self.data_collection_callback)
                dpg.add_menu_item(label="Import Data",  callback=self.import_data_callback )
                dpg.add_menu_item(label="Export Data",  callback=self.export_data_callback)
                dpg.add_menu_item(label="Inspect Data", callback=self.inspect_data_callback)
            
            with dpg.menu(label="Model"):
                dpg.add_menu_item(label="Train Classifier", callback=self.train_classifier_callback)

            with dpg.menu(label="HCI"):
                dpg.add_menu_item(label="Fitts Law", callback=self.fitts_law_callback)

    def data_collection_callback(self):
        dcp_parameter_list  = list(signature(DataCollectionPanel.__init__).parameters)
        common_set = [element for element in dcp_parameter_list if element in self.args.keys()]
        passable_args = {common_key: self.args[common_key] for common_key in common_set}
        self.dcp = DataCollectionPanel(**passable_args)
        self.dcp.spawn_configuration_window()

    def import_data_callback(self):
        dip_parameter_list  = list(signature(DataImportPanel.__init__).parameters)
        common_set = [element for element in dip_parameter_list if element in self.args.keys()]
        passable_args = {common_key: self.args[common_key] for common_key in common_set}
        self.dip = DataImportPanel(**passable_args)
        self.dip.spawn_configuration_window()

    def export_data_callback(self):
        pass

    def inspect_data_callback(self):
        pass

    def train_classifier_callback(self):
        pass

    def fitts_law_callback(self):
        pass
