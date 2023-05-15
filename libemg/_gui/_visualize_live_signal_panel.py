import dearpygui.dearpygui as dpg
import numpy as np
import matplotlib.pyplot as plt
from ._utils import Media, set_texture, init_matplotlib_canvas, matplotlib_to_numpy
import time

class VisualizeLiveSignalPanel:
    def __init__(self,
                 visualization_horizon = 10000,
                 visualization_rate    = 24,
                 gui=None):
        self.widget_tags = {"configuration": ['__vls_configuration_window', '__vls_horizon_length', '__vls_update_rate'],
                            "visualization": ['__vls_visualize_window']}
        self.visualization_horizon = visualization_horizon
        self.visualization_rate    = visualization_rate
        self.gui = gui

    def cleanup_window(self, window_name):
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)     

    def spawn_configuration_window(self):
        # reset the emg data
        if self.gui.online_data_handler is not None:
            self.gui.online_data_handler.raw_data.reset_emg()
        self.cleanup_window("configuration")
        with dpg.window(tag="__vls_configuration_window",
                        label="Live Signal Visualization Configuration",
                        width=720,
                        height=480):
            
            dpg.add_text(default_value="Configuration Panel")

            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Online Data Handler State: ")
                dpg.add_text(default_value = 
                    "ODH does not exist" if self.gui.online_data_handler is None else ("ODH is not receiving data - is device streamer running?" if (not len(self.gui.online_data_handler.raw_data.get_emg())) else "ODH is ready")
                        )

            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Horizon Length: ")
                dpg.add_input_text(default_value=self.visualization_horizon, tag="__vls_horizon_length")

            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Update Rate: ")
                dpg.add_input_text(default_value=self.visualization_rate, tag="__vls_update_rate")

            with dpg.group(horizontal=True):
                dpg.add_button(label="Begin", callback=self.begin_visualization_callback)

    def begin_visualization_callback(self):
        self.get_settings()
        dpg.delete_item("__vls_configuration_window")
        self.cleanup_window("configuration")
        self.spawn_visualization_window()

    def get_settings(self):
        self.visualization_horizon      = int(dpg.get_value("__vls_horizon_length"))
        self.visualization_rate         = int(dpg.get_value("__vls_update_rate"))

    def spawn_visualization_window(self):
        # initialize blank data object
        self.data = np.zeros((self.visualization_horizon, np.stack(self.gui.online_data_handler.raw_data.get_emg()).shape[1]))
        self.visualizing = True
        init_matplotlib_canvas(width=800, height=600)
        self.plot_handles = plt.plot(self.data)

        plt.xlabel("Samples")
        plt.ylabel("Values")
        self.update_data()
        img = matplotlib_to_numpy()
        media = Media()
        media.from_numpy(img)
        texture = media.get_dpg_formatted_texture(width=720, height=480)
        set_texture("__vls_plot", texture, width=720, height=480)

        self.cleanup_window("visualization")
        with dpg.window(tag="__vls_visualization_window",
                        label="Live Signal Visualization Configuration",
                        width=800,
                        height=600):
            
            dpg.add_text(label="Visualization Panel")
            dpg.add_image("__vls_plot")
            dpg.add_button(label="Quit", callback=self.quit_visualization_callback)
        
        dpg.configure_app(manual_callback_management=True)
        while self.visualizing:
            
            self.update_data()
            img = matplotlib_to_numpy()
            media = Media()
            media.from_numpy(img)
            texture = media.get_dpg_formatted_texture(width=720, height=480)
            set_texture("__vls_plot", texture, width=720, height=480)
            # time.sleep(1/self.visualization_rate)
            jobs = dpg.get_callback_queue()
            dpg.run_callbacks(jobs)

        dpg.configure_app(manual_callback_management=False)

        dpg.delete_item("__vls_visualization_window")
        self.cleanup_window("visualization")

    def update_data(self):
        new_data = self.gui.online_data_handler.raw_data.get_emg()
        if len(new_data):
            new_data = np.stack(new_data)
            self.data = np.vstack((self.data, new_data))
            self.gui.online_data_handler.raw_data.reset_emg()
            self.data = self.data[-self.visualization_horizon:,:]
            for i, h in enumerate(self.plot_handles):
                h.set_ydata(self.data[:,i])
            plt.ylim((np.min(self.data), np.max(self.data)))
        


    def quit_visualization_callback(self):
        self.visualizing = False