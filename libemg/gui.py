import dearpygui.dearpygui as dpg
from libemg._gui._data_collection_panel import DataCollectionPanel

class GUI:
    def __init__(self, 
                 width=1920,
                 height=1080,
                 args = None):
        self.args = args
        self.window_init(width, height)
        
    
    def window_init(self, width, height):
        dpg.create_context()
        dpg.create_viewport(title="LibEMG",
                            width=width,
                            height=height)
        dpg.setup_dearpygui()
        

        self.file_menu_init()

        dpg.show_viewport()

        # dpg.configure_app(manual_callback_management=True)
        # while dpg.is_dearpygui_running():
        #     jobs = dpg.get_callback_queue()
        #     dpg.run_callbacks(jobs)
        #     dpg.render_dearpygui_frame()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def file_menu_init(self):

        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Exit")
                
            with dpg.menu(label="Data"):
                dpg.add_menu_item(label="Collect Data", callback=self.data_collection_callback)
                dpg.add_menu_item(label="Import Data")
                dpg.add_menu_item(label="Export Data")
                dpg.add_menu_item(label="Inspect Data")

    def data_collection_callback(self):
        self.dcp = DataCollectionPanel(**self.args)
        self.dcp.spawn_configuration_window()


