import dearpygui.dearpygui as dpg
from libemg._gui._data_collection_panel import DataCollectionPanel
from libemg._gui._data_import_panel import DataImportPanel
from libemg._gui._visualization_panel import VisualizationPanel
import gc
import inspect
import time
import os
import json
from os import walk

class GUI:
    """
    The Screen Guided Training module. 

    By default, this module has two purposes: 
    (1) Launching a Screen Guided Training window. 
    (2) Downloading gesture sets from our library of gestures located at:
    https://github.com/libemg/LibEMGGestures

    Parameters
    ----------
    online_data_handler: OnlineDataHandler
        Online data handler used for acquiring raw EMG data.
    args: dic, default={'media_folder': 'images/', 'data_folder':'data/', 'num_reps': 3, 'rep_time': 5, 'rest_time': 3, 'auto_advance': True}
        The dictionary that defines the SGT window. Keys are: 'media_folder', 
        'data_folder', 'num_reps', 'rep_time', 'rest_time', 'auto_advance', and 'timestamps'. All media (i.e., images and videos) in 'media_folder' will be played in alphabetical order.
        For video files, a matching labels file of the same name will be searched for and added to the 'data_folder' if found.
        'rep_time' is only used for images since the duration of videos is automatically calculated based on
        the number of frames (assumed to be 24 FPS). 'timestamps' (default False) prepends a timestamp column to
        every logged sample.
        Visualize > Live Signal reads 'num_samples', 'refresh_rate' and 'plot_height' from this same dictionary.
    width: int, default=1920
        The width of the SGT window. 
    height: int, default=1080
        The height of the SGT window.
    gesture_width: int, default=500
        The width of the embedded gesture image/video.
    gesture_height: int, default=500
        The height of the embedded gesture image/video.
    clean_up_on_call: Boolean, default=False
        If true, this will cleanup (and kill) the streamer reference.
    """
    def __init__(self, 
                 online_data_handler=None,
                 args={'media_folder': 'images/', 'data_folder':'data/', 'num_reps': 3, 'rep_time': 5, 'rest_time': 3, 'auto_advance': True},
                 width=1920,
                 height=1080,
                 debug=False,
                 gesture_width = 500,
                 gesture_height = 500,
                 clean_up_on_kill=False):
        
        self.width = width 
        self.height = height 
        self.debug = debug
        self.online_data_handler = online_data_handler
        self.args = args
        self.video_player_width = gesture_width
        self.video_player_height = gesture_height
        self.clean_up_on_kill = clean_up_on_kill
        # Panels that need a live handler are disabled without one. The
        # pipeline editor creates its own sources, so it must be reachable with
        # no handler at all.
        self.panels = []
        self._install_global_fields()

    def start_gui(self):
        """
        Launches the Screen Guided Training UI.
        """
        # Everything past this point runs alongside worker threads that must not
        # be stalled -- the file logger above all. Anything left unfinalized by
        # the script that got us here is collected now, on the main thread, so a
        # worker is not the one to trigger it later. A matplotlib window closed
        # before this call is the usual culprit: its Tk widgets each block a
        # non-main thread for a full second on finalization (see
        # libemg.utils._release_interactive_plot).
        gc.collect()
        self._window_init(self.width, self.height, self.debug)
        
    def _install_global_fields(self):
        # self.global_fields = ['offline_data_handlers', 'online_data_handler']
        self.offline_data_handlers = []   if 'offline_data_handlers' not in self.args.keys() else self.args["offline_data_handlers"]
        self.offline_data_aliases  = []   if 'offline_data_aliases'  not in self.args.keys() else self.args["offline_data_aliases"]

    def _window_init(self, width, height, debug=False):
        dpg.create_context()
        dpg.create_viewport(title="LibEMG",
                            width=width,
                            height=height)
        dpg.setup_dearpygui()
        

        self._file_menu_init()

        dpg.show_viewport()
        dpg.set_exit_callback(self._on_window_close)

        # A manual render loop on every path, not just in debug. Anything that
        # has to be refreshed while the window is open -- a probe drawing live
        # data, a progress bar -- needs a per-frame callback, and
        # start_dearpygui() blocks with nowhere to put one. The loop below is
        # what the debug path already ran.
        if debug:
            dpg.configure_app(manual_callback_management=True)
        while dpg.is_dearpygui_running():
            if debug:
                dpg.run_callbacks(dpg.get_callback_queue())
            self._poll_panels()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def _poll_panels(self):
        """Give every open panel its per-frame slice, on the render thread.

        This is the only thread that may touch DearPyGui items, so it is the
        only place a panel may draw what a running pipeline has produced.
        """
        for panel in list(self.panels):
            poll = getattr(panel, "poll", None)
            if poll is None:
                continue
            try:
                poll()
            except Exception:
                # A panel that fails mid-frame must not take the window down
                # with it, or a transient read error closes the whole GUI.
                self.panels.remove(panel)

    def _file_menu_init(self):

        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Exit")
                
            with dpg.menu(label="Device"):
                dpg.add_menu_item(label="Streamer", callback=self._streamer_callback)

            with dpg.menu(label="Data"):
                dpg.add_menu_item(label="Collect Data", callback=self._data_collection_callback)
                #dpg.add_menu_item(label="Import Data",  callback=self._import_data_callback )
                #dpg.add_menu_item(label="Export Data",  callback=self._export_data_callback)
                #dpg.add_menu_item(label="Inspect Data", callback=self._inspect_data_callback)
            
            with dpg.menu(label="Visualize"):
                dpg.add_menu_item(label="Live Signal", callback=self._visualize_livesignal_callback)

            with dpg.menu(label="Pipeline"):
                dpg.add_menu_item(label="Pipeline Editor", callback=self._pipeline_editor_callback)

            with dpg.menu(label="Environments"):
                dpg.add_menu_item(label="Launch Environment",
                                  callback=self._environments_callback)

            # with dpg.menu(label="Model"):
                # dpg.add_menu_item(label="Train Classifier", callback=self._train_classifier_callback)

            # with dpg.menu(label="HCI"):
                # dpg.add_menu_item(label="Fitts Law", callback=self._fitts_law_callback)

    def _open_panel(self, attribute, build):
        """Show a panel, reusing the one already open.

        Clicking a menu item twice used to build a second panel, and the second
        one's setup deletes the window tag the first one owns. What was left
        was an invisible first panel still in the poll list, writing into
        widgets the second panel now owns, and still holding whatever it had
        started -- a streaming device or a running pipeline -- with no window
        able to stop it. Reusing the panel that is already there is what a menu
        click means anyway: bring it to the front.

        Parameters
        ----------
        attribute: str
            Where the panel is remembered on this object.
        build: callable
            Makes a fresh panel, called only when there is not one open.

        Returns
        ----------
        object
            The panel now on screen.
        """
        panel = getattr(self, attribute, None)
        tag = getattr(panel, "window_tag", None)
        if panel is not None and tag is not None and dpg.does_alias_exist(tag):
            dpg.focus_item(tag)
            return panel
        # The window is gone, so whatever this panel held goes with it rather
        # than outliving the only thing that could stop it.
        if panel is not None:
            try:
                panel.cleanup()
            except Exception:
                pass
            if panel in self.panels:
                self.panels.remove(panel)
        panel = build()
        panel.spawn_window()
        setattr(self, attribute, panel)
        if panel not in self.panels:
            self.panels.append(panel)
        return panel

    def _data_collection_callback(self):
        panel_arguments = list(inspect.signature(DataCollectionPanel.__init__).parameters)
        passed_arguments = {i: self.args[i] for i in self.args.keys() if i in panel_arguments}
        self.dcp = DataCollectionPanel(self.online_data_handler, **passed_arguments, video_player_width=self.video_player_width, video_player_height=self.video_player_height)
        self.dcp.spawn_configuration_window()

    def _visualize_livesignal_callback(self):
        panel_arguments = list(inspect.signature(VisualizationPanel.__init__).parameters)
        passed_arguments = {i: self.args[i] for i in self.args.keys() if i in panel_arguments}
        self.vp = VisualizationPanel(self.online_data_handler, **passed_arguments)
        self.vp.spawn_window()

    def _pipeline_editor_callback(self):
        from libemg._gui._pipeline.editor_panel import PipelineEditorPanel
        self._open_panel("pep", lambda: PipelineEditorPanel(
            width=self.width, height=self.height))

    def _streamer_callback(self):
        from libemg._gui._streamer_panel import StreamerPanel
        self._open_panel("sp", lambda: StreamerPanel(
            on_started=self._streamer_started,
            on_stopped=self._streamer_stopped))

    def _streamer_started(self, online_data_handler, shared_memory_items):
        """Make a device started here available to every other panel.

        The panels that need live data read this attribute when they are
        opened, so publishing it is what lets somebody start a device and then
        collect training data or watch the signal without leaving the window.
        """
        self.online_data_handler = online_data_handler
        self.args["shared_memory_items"] = shared_memory_items

    def _streamer_stopped(self):
        self.online_data_handler = None

    def _environments_callback(self):
        from libemg._gui._environments.panel import EnvironmentsPanel
        panel = self._open_panel("env_panel", lambda: EnvironmentsPanel(
            online_data_handler=self.online_data_handler,
            width=min(self.width, 1280), height=min(self.height, 820)))
        # A device may have been started since this panel was first opened, and
        # an environment launched afterwards should be driven by it.
        panel.online_data_handler = self.online_data_handler

    def _import_data_callback(self):
        panel_arguments = list(inspect.signature(DataImportPanel.__init__).parameters)
        passed_arguments = {i: self.args[i] for i in self.args.keys() if i in panel_arguments}
        self.dip = DataImportPanel(**passed_arguments, gui=self)
        self.dip.spawn_configuration_window()

    def _export_data_callback(self):
        pass

    def _inspect_data_callback(self):
        pass

    def _train_classifier_callback(self):
        pass

    def _fitts_law_callback(self):
        pass

    def _on_window_close(self):
        if self.clean_up_on_kill:
            print("Window is closing. Performing clean-up...")
            if 'streamer' in self.args.keys():
                self.args['streamer'].signal.set()
            time.sleep(3)
    
    def download_gestures(self, gesture_ids, folder, download_imgs=True, download_gifs=False, redownload=False):
        """
        Downloads gesture images (either .png or .gif) from: 
        https://github.com/libemg/LibEMGGestures.
        
        This function dowloads gestures using the "curl" command. 

        Parameters
        ----------
        gesture_ids: list
            A list of indexes corresponding to the gestures you want to download. A list of indexes and their respective 
            gesture can be found at https://github.com/libemg/LibEMGGestures.
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