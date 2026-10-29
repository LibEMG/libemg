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
        'protocol' is a libemg.protocol.CollectionProtocol (or a path to one) describing what to collect: the
        prompt order and its seed, the rest jitter, the design factors to vary between reps, and the acquisition
        settings. Data > Collection Protocol builds one, renders its prompts with the Animator and sets this key.
        'signal_check' ('warn' by default, or 'block' / 'off') looks at the stream for dead channels, clipping and
        mains pickup before the first prompt; 'line_frequency' (default 60) is the mains frequency to look at.
        'subject_number', 'subject_name', 'session_number', 'day_number' and 'condition_number' seed the Subject
        Details section of the configuration window, and whatever that section is left showing when Start is
        pressed is written to 'subject_details.yaml' in the output folder.
        'continuous_log' (default True) records the whole collection into '_session_<modality>.csv' -- rests and
        pauses included, nothing cut out -- and writes '_session_annotations.json' beside it saying where every
        stage of the collection sits in that recording. The per-repetition files are cut from it at those
        indices, and can be cut again afterwards with libemg.session.SessionAnnotations.
        Data > Annotate Contractions opens on 'annotation_folder' and labels its time axis with 'sampling_rate'
        (default 2000 Hz). Which files in that folder are recordings comes from the 'annotation' block of its
        dataset_description.yaml (see libemg.annotations.DatasetIndex); 'annotation_spec' overrides that block.
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
        # One font, spacing and palette for every panel: DearPyGui's own is a
        # 13 px bitmap font with tight padding.
        from libemg._gui import _style
        _style.apply()
        dpg.create_viewport(title="LibEMG",
                            width=width,
                            height=height,
                            clear_color=(22, 24, 27, 255))
        dpg.setup_dearpygui()


        self._file_menu_init()
        self._welcome_init()

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
                dpg.add_menu_item(label="Start here", callback=self._welcome_init)
                # Exit had no callback, so choosing it did nothing at all.
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
                
            with dpg.menu(label="Device"):
                dpg.add_menu_item(label="Streamer", callback=self._streamer_callback)

            with dpg.menu(label="Data"):
                dpg.add_menu_item(label="Collection Protocol",
                                  callback=self._protocol_callback)
                dpg.add_menu_item(label="Collect Data", callback=self._data_collection_callback)
                dpg.add_menu_item(label="Annotate Contractions",
                                  callback=self._annotation_callback)
                #dpg.add_menu_item(label="Import Data",  callback=self._import_data_callback )
                #dpg.add_menu_item(label="Export Data",  callback=self._export_data_callback)
                #dpg.add_menu_item(label="Inspect Data", callback=self._inspect_data_callback)
            
            with dpg.menu(label="Visualize"):
                dpg.add_menu_item(label="Live Signal", callback=self._visualize_livesignal_callback)

            with dpg.menu(label="Pipeline"):
                dpg.add_menu_item(label="Pipeline Editor", callback=self._pipeline_editor_callback)
                dpg.add_menu_item(label="Training History", callback=self._training_history_callback)

            with dpg.menu(label="Environments"):
                dpg.add_menu_item(label="Launch Environment",
                                  callback=self._environments_callback)

            # with dpg.menu(label="Model"):
                # dpg.add_menu_item(label="Train Classifier", callback=self._train_classifier_callback)

            # with dpg.menu(label="HCI"):
                # dpg.add_menu_item(label="Fitts Law", callback=self._fitts_law_callback)

    #: The steps of a typical session, in order: what each is for, and the menu
    #: entry that opens it. The welcome window is drawn from this.
    _WELCOME_STEPS = (
        ("Connect a device", "Stream from a device, or from a synthetic one with no hardware.",
         "Streamer", "_streamer_callback"),
        ("Plan a collection", "Which gestures, in what order, for how long, with what prompts.",
         "Collection Protocol", "_protocol_callback"),
        ("Collect training data", "Prompt each gesture on screen and record every repetition.",
         "Collect Data", "_data_collection_callback"),
        ("Watch the signal", "See the live data from the device that is streaming.",
         "Live Signal", "_visualize_livesignal_callback"),
        ("Build a model", "Filters, windows, features and a classifier or regressor, as blocks.",
         "Pipeline Editor", "_pipeline_editor_callback"),
        ("Run a task", "Fitts' Law and EMG Hero, driven by a model, saved for retraining.",
         "Environments", "_environments_callback"),
    )

    def _welcome_init(self):
        """A first window that says where to start.

        The menu bar alone gives no hint of the order a session goes in, or that
        nothing can be recorded or watched until a device is streaming. This
        lays the steps out in that order, each with the button that opens it,
        and says whether a device is streaming yet. Closing it loses nothing:
        every step is also in the menu, and File > Start here brings it back.
        """
        from libemg._gui import _style as style
        if dpg.does_item_exist("__welcome"):
            dpg.show_item("__welcome")
            dpg.focus_item("__welcome")
            return
        with dpg.window(label="Start here", tag="__welcome", width=660, height=590,
                        pos=(24, 44), no_collapse=True):
            style.note("A session usually goes in this order. Every step is also in the "
                       "menu bar.", wrap=620)
            dpg.add_text("", tag="__welcome_device")
            style.note("", tag="__welcome_message", color=style.WARNING, wrap=620, show=False)
            dpg.add_spacer(height=2)
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit,
                           borders_innerH=True):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=190)
                dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                for number, (title, description, button, method) in enumerate(self._WELCOME_STEPS, 1):
                    with dpg.table_row():
                        style.button(button, "primary" if number == 1 else "default", width=180,
                                     callback=lambda s, a, u: self._welcome_open(u),
                                     user_data=method)
                        with dpg.group():
                            heading = dpg.add_text(f"{number}. {title}")
                            if style.font("bold") is not None:
                                dpg.bind_item_font(heading, style.font("bold"))
                            style.note(description, wrap=420)
        self._welcome_device_status()

    def _welcome_open(self, method):
        """Open a step's panel, or say why it can't be opened yet."""
        if method == "_visualize_livesignal_callback" and self.online_data_handler is None:
            if dpg.does_item_exist("__welcome_message"):
                dpg.set_value("__welcome_message", "Nothing to watch yet: connect a device first.")
                dpg.show_item("__welcome_message")
            return
        if dpg.does_item_exist("__welcome_message"):
            dpg.hide_item("__welcome_message")
        getattr(self, method)()

    def _welcome_device_status(self):
        from libemg._gui import _style as style
        try:
            if not dpg.does_item_exist("__welcome_device"):
                return
        except Exception:
            return      # no window at all, as when a panel is driven from a test
        if self.online_data_handler is None:
            style.set_status("__welcome_device", "No device is streaming yet.", "warn")
        else:
            style.set_status("__welcome_device", "A device is streaming.", "ok")

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

    def _protocol_callback(self):
        from libemg._gui._protocol_panel import ProtocolPanel
        self._open_panel("protocol_panel", lambda: ProtocolPanel(
            protocol=self.args.get("protocol"),
            media_folder=self.args.get("media_folder", "media/"),
            on_ready=self._protocol_ready,
            width=min(self.width, 860), height=min(self.height, 800)))

    def _protocol_ready(self, protocol, media_folder):
        """Point screen guided training at a protocol that was just built here.

        The protocol decides the prompt order, the timings and the design
        factors, and it names the folder its media went into -- so once it is
        built, Collect Data needs nothing typed into it.
        """
        self.args["protocol"] = protocol
        self.args["media_folder"] = media_folder

    def _annotation_callback(self):
        from libemg._gui._annotation_panel import AnnotationPanel
        self._open_panel("annotation_panel", lambda: AnnotationPanel(
            dataset_folder=self.args.get("annotation_folder", ""),
            sampling_rate=self.args.get("sampling_rate", 2000.0),
            spec=self.args.get("annotation_spec"),
            width=self.width, height=self.height))

    def _visualize_livesignal_callback(self):
        # Without a stream there is nothing to draw, and the panel says so by
        # raising -- which from a menu click went to the console and left the
        # window looking as if the click had been ignored.
        if self.online_data_handler is None:
            self._notice("Nothing to watch yet",
                         "No device is streaming. Start one from Device > Streamer, "
                         "then open Visualize > Live Signal again.")
            return
        panel_arguments = list(inspect.signature(VisualizationPanel.__init__).parameters)
        passed_arguments = {i: self.args[i] for i in self.args.keys() if i in panel_arguments}
        self.vp = VisualizationPanel(self.online_data_handler, **passed_arguments)
        try:
            self.vp.spawn_window()
        except ConnectionError:
            self._notice("Nothing to watch yet",
                         "The device is connected but no data has arrived yet. Wait a "
                         "moment, then open Visualize > Live Signal again.")

    def _notice(self, title, text):
        """A small window saying why something the user asked for did not happen."""
        from libemg._gui import _style as style
        if dpg.does_item_exist("__notice"):
            dpg.delete_item("__notice")
        with dpg.window(label=title, tag="__notice", modal=True, no_collapse=True,
                        width=440, pos=(max(20, self.width // 2 - 220), 160)):
            style.note(text, color=style.TEXT, wrap=400)
            dpg.add_spacer(height=4)
            style.button("OK", "primary", width=90,
                         callback=lambda: dpg.delete_item("__notice"))

    def _pipeline_editor_callback(self):
        from libemg._gui._pipeline.editor_panel import PipelineEditorPanel
        self._open_panel("pep", lambda: PipelineEditorPanel(
            width=self.width, height=self.height))

    def _training_history_callback(self):
        """Deep model training runs from any session, loaded from their records."""
        from libemg._gui._pipeline.training_view import TrainingView
        if getattr(self, "training_history", None) is not None:
            self.training_history.cleanup()
        self.training_history = TrainingView("Training History", pos=(60, 60)).spawn_window()

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
        self._welcome_device_status()

    def _streamer_stopped(self):
        self.online_data_handler = None
        self._welcome_device_status()

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