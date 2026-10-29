"""The Environments window: pick a task, set it up, play it in the GUI.

Two screens. The setup screen is generated from the environment's own
configuration, so every setting it accepts appears with the author's own
description beside it. The play screen is a single image, backed by the memory
the environment is drawing into, plus what it is doing and a way to stop it.

The setup screen is laid out as a form in two columns. On the left is how the
task is driven and what is saved of it -- the things that change between
sessions. On the right are the task's own settings, grouped under headings,
each with its label on the left, its unit after it, and a (?) holding the
explanation the environment's author wrote. Only what applies is shown: the
address of a model appears when a model drives the task, the class mapping
when that model is a classifier.

Nothing about the game runs here. The environment is a process of its own, and
this window forwards the keyboard to it and shows the frames it produces. That
separation is what keeps a game that stalls from taking the interface with it.
"""

import os
import threading
import traceback
from datetime import datetime
from pathlib import Path

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style
from libemg._gui._environments.embedded import EmbeddedEnvironment
from libemg._gui._environments.factories import ControllerSpec, build_factory
from libemg._gui._environments.frame_bridge import FORWARDED_KEYS
from libemg._gui._environments.registry import (APPEARANCE, BOOL, COLOR, ENUM, FLOAT,
                                                GROUP_ORDER, INT, default_registry)
from libemg.environments.recording import EnvironmentRecording, RecordingConfig

WINDOW_TAG = "__env_window"
PLAY_TAG = "__env_play"
STATUS_TAG = "__env_status"
MESSAGE_TAG = "__env_message"
TEXTURE_TAG = "__env_texture"
LAUNCH_TAG = "__env_launch"
STOP_TAG = "__env_stop"
FOLDER_DIALOG_TAG = "__env_folder_dialog"

#: The Save section's checkboxes: RecordingConfig field, label, what it adds in
#: a few words, and the full explanation for its tooltip.
_SAVE_FIELDS = (
    ("timestamps", "Timestamps", "when each sample was produced",
     "A timestamp column: when each sample was produced, in time.time() seconds."),
    ("emg", "EMG", "the raw signal, one row per sample",
     "The raw EMG, one row per sample. Without it there is nothing to train on, and the "
     "files hold one row per frame instead."),
    ("model_outputs", "Model outputs", "the class and speed it sent",
     "What the model sent: the class and its speed from a classifier, the outputs of a "
     "regressor."),
    ("game", "Game state", "cursor, target and trial",
     "Cursor, target, whether the cursor is in it, its velocity; for EMG Hero the lane "
     "played and the notes on the line."),
    ("adaptation", "Adaptation (CIIL)", "what the user should have done",
     "The optimal direction and the CIIL label: the class a classifier should have sent, "
     "or the outputs a regressor should have."),
)

#: Where an optional limit starts when it is switched on.
_OPTIONAL_START = {"timeout": 10.0, "game_time": 60.0}

_FORMAT_HELP = ("predictions: each message is '<class> [speed]'. Use this when your own "
                "script sends the class.\n\nprobabilities: one probability per class, then "
                "the speed if velocity is on, then a timestamp. This is what libemg's "
                "OnlineEMGClassifier sends.")

#: How a forwarded key is spelled on each side. DearPyGui names the space bar
#: differently from pygame, which is exactly the kind of mismatch that silently
#: drops a key, so the mapping is written out rather than derived.
_DPG_KEYS = {
    "left": "mvKey_Left", "right": "mvKey_Right", "up": "mvKey_Up",
    "down": "mvKey_Down", "1": "mvKey_1", "2": "mvKey_2", "3": "mvKey_3",
    "4": "mvKey_4", "space": "mvKey_Spacebar", "escape": "mvKey_Escape",
    "w": "mvKey_W", "a": "mvKey_A", "s": "mvKey_S", "d": "mvKey_D",
}


class EnvironmentsPanel:
    """Launch and play LibEMG's environments inside the GUI.

    Parameters
    ----------
    registry: dict or None (optional), default=None
        Environment descriptions. Defaults to the generated registry.
    online_data_handler: OnlineDataHandler or None (optional)
        Only needed to save the EMG of a run.

    Examples
    ---------
    >>> panel = EnvironmentsPanel()
    >>> panel.spawn_window()
    """


    #: The window this panel owns, so a caller can ask if it is open.
    window_tag = WINDOW_TAG

    def __init__(self, registry=None, online_data_handler=None, width=1280,
                 height=800):
        self.registry = registry if registry is not None else default_registry()
        self.online_data_handler = online_data_handler
        self.width, self.height = width, height
        self.selected = next(iter(self.registry))
        self.values = {}
        self.controller = {"kind": "keyboard", "ip": "127.0.0.1", "port": 12346,
                           "num_classes": 5, "output_format": "predictions",
                           "class_map": self.registry[self.selected].default_class_map()}
        # What to save. Kept across tasks: where the data goes is a property of the
        # session, not of which task happens to be selected.
        self.save = {"folder": "", **{name: True for name, _, _, _ in _SAVE_FIELDS}}
        # The number an optional limit had when it was switched off, so switching
        # it back on restores it rather than starting over.
        self._optional_numbers = dict(_OPTIONAL_START)
        self.embedded = None
        self.recorder = None
        # A message from the thread that assembles a finished recording, shown by
        # poll() on the render thread, the only one that may touch the window.
        self._pending_message = None
        self._counter = 0
        self._frame_size = (0, 0)

    @property
    def _column_wrap(self):
        """Where text wraps inside one of the two columns."""
        return max(280, self.width // 2 - 70)

    # ==================================================================
    # setup screen
    # ==================================================================
    def spawn_window(self):
        """Build the setup window for the selected environment."""
        self.cleanup()
        spec = self.registry[self.selected]
        self.values = spec.defaults()
        self._fit_controller(spec)
        with dpg.window(label="Environments", tag=WINDOW_TAG,
                        width=self.width, height=self.height,
                        on_close=lambda: self.cleanup()):
            with dpg.group(horizontal=True):
                dpg.add_text("Task", color=style.MUTED)
                dpg.add_combo([s.title for s in self.registry.values()],
                              default_value=spec.title, tag="__env_choice", width=260,
                              callback=self._choose)
                dpg.add_spacer(width=6)
                style.button("Launch", "primary", tag=LAUNCH_TAG, width=110,
                             callback=self._launch)
                style.button("Stop", "danger", tag=STOP_TAG, width=90, callback=self.stop)
                dpg.add_spacer(width=6)
                style.button("Reset to defaults", "subtle", callback=self._reset)
            dpg.add_text("", tag=MESSAGE_TAG, wrap=self.width - 40)
            dpg.add_separator()
            with dpg.child_window(tag="__env_body", border=False, autosize_x=True,
                                  height=-1):
                self._build_body()
        dpg.add_file_dialog(directory_selector=True, show=False, modal=True,
                            tag=FOLDER_DIALOG_TAG, width=760, height=460,
                            default_path=self.save["folder"] or os.getcwd(),
                            callback=self._folder_chosen)
        style.set_enabled(STOP_TAG, False)
        self._describe()
        return self

    def _build_body(self):
        spec = self.registry[self.selected]
        style.note(spec.help, wrap=self.width - 60)
        style.hover_hint()
        dpg.add_spacer(height=2)
        with dpg.table(header_row=False, resizable=False, borders_innerV=True,
                       policy=dpg.mvTable_SizingStretchSame):
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                with dpg.group():
                    self._build_control(spec)
                    self._build_mapping(spec)
                    self._build_save()
                with dpg.group():
                    self._build_settings(spec)
        self._controller_fields()
        self._check_mapping()

    # -- left column -----------------------------------------------------
    def _build_control(self, spec):
        style.section("Control", first=True)
        with style.form():
            with style.row("Driven by", help="Keyboard: the arrow keys, for trying a task "
                           "out. Classifier and Regressor: a model running elsewhere, "
                           "sending its output to the address below."):
                if len(spec.controllers) == 1:
                    # Nothing to choose, so no control that looks like a choice.
                    dpg.add_text(f"A {spec.controllers[0].lower()}")
                    dpg.add_text("(the only controller this task takes)", color=style.MUTED)
                else:
                    dpg.add_radio_button([_title(kind) for kind in spec.controllers],
                                         default_value=_title(self.controller["kind"]),
                                         horizontal=True, tag="__env_controller",
                                         callback=self._controller_changed)
            with style.row("Listens on", tag="__env_row_address",
                           help="The address and UDP port the running model sends to."):
                dpg.add_input_text(default_value=self.controller["ip"], width=140,
                                   tag="__env_ip", callback=lambda s, a: self._set_controller(ip=a.strip()))
                dpg.add_text(":", color=style.MUTED)
                dpg.add_input_int(default_value=self.controller["port"], width=90, step=0,
                                  min_value=1, max_value=65535, min_clamped=True,
                                  max_clamped=True, tag="__env_port",
                                  callback=lambda s, a: self._set_controller(port=a))
            with style.row("Message format", tag="__env_row_format", help=_FORMAT_HELP):
                dpg.add_combo(["predictions", "probabilities"],
                              default_value=self.controller["output_format"], width=160,
                              tag="__env_format",
                              callback=lambda s, a: self._set_controller(output_format=a))
            with style.row("Classes in the model", tag="__env_row_classes",
                           help="How many probabilities each message starts with. Has to "
                                "match the model."):
                dpg.add_input_int(default_value=self.controller["num_classes"], width=90,
                                  step=0, min_value=1, min_clamped=True, tag="__env_classes",
                                  callback=lambda s, a: self._set_controller(num_classes=a))
        style.note("The arrow keys steer. Click the game once it is running, so it has the "
                   "keyboard.", tag="__env_keyboard_note", wrap=self._column_wrap)
        style.note("The model's first output moves the cursor sideways, its second up and "
                   "down.", tag="__env_regressor_note", wrap=self._column_wrap)

    def _build_mapping(self, spec):
        if not spec.actions:
            return
        with dpg.group(tag="__env_mapping"):
            style.section("Which class does what")
            style.note("The class your model sends for each action.", wrap=self._column_wrap)
            with style.form():
                for action in spec.actions:
                    symbol = style.arrow(action.name)
                    with style.row(f"{symbol}  {action.label}" if symbol else action.label):
                        dpg.add_input_int(
                            default_value=int(self.controller["class_map"].get(
                                action.name, action.default_class)),
                            width=90, step=0, min_value=-1, min_clamped=True,
                            tag=f"__env_map_{action.name}", user_data=action.name,
                            callback=self._class_map_changed)
            idle = spec.action(spec.idle_action) if spec.idle_action else None
            if idle is not None:
                style.note(f"Every other class: {idle.label.lower()}. Use -1 to leave an "
                           f"action unused.", wrap=self._column_wrap)
            style.note("", color=style.WARNING, tag="__env_map_warning",
                       wrap=self._column_wrap, show=False)

    def _build_save(self):
        style.section("Save")
        with style.form():
            with style.row("Folder", help="Each run goes in a folder of its own inside this "
                           "one, named for the task and the time it started. It holds one "
                           "trial_<n>.csv per trial, a row per EMG sample and a column per "
                           "field ticked below; libemg.environments.recording."
                           "load_recording(<run folder>) reads it into an "
                           "OfflineDataHandler."):
                dpg.add_input_text(default_value=self.save["folder"], width=-100,
                                   hint="Leave empty to save nothing",
                                   tag="__env_save_folder", callback=self._folder_typed)
                style.button("Browse...", width=90,
                             callback=lambda: dpg.show_item(FOLDER_DIALOG_TAG))
            with style.row("Include"):
                with dpg.group():
                    for name, label, short, help_text in _SAVE_FIELDS:
                        with dpg.group(horizontal=True):
                            box = dpg.add_checkbox(label=label, default_value=self.save[name],
                                                   tag=f"__env_save_{name}", user_data=name,
                                                   callback=self._save_option_changed)
                            with dpg.tooltip(box):
                                dpg.add_text(help_text, wrap=340)
                            style.note(short)
        style.note("", tag="__env_save_where", wrap=self._column_wrap)
        style.note("No device is streaming in this window, so EMG can't be saved. Start one "
                   "from Device > Streamer, or pass an online data handler to the GUI.",
                   color=style.WARNING, tag="__env_no_device", wrap=self._column_wrap,
                   show=self.online_data_handler is None)

    # -- right column ----------------------------------------------------
    def _build_settings(self, spec):
        first = True
        for group in GROUP_ORDER:
            settings = [s for s in spec.settings if s.group == group]
            if not settings:
                continue
            if group == APPEARANCE:
                dpg.add_spacer(height=4)
                with dpg.collapsing_header(label="Appearance", default_open=False):
                    self._settings_form(settings)
                continue
            style.section(group, first=first)
            first = False
            self._settings_form(settings)

    def _settings_form(self, settings):
        with style.form(label_width=215):
            for setting in settings:
                with style.row(setting.label, help=setting.help or None):
                    self._draw_control(setting)

    def _draw_control(self, setting):
        """One control, chosen by the setting's kind, and its unit.

        The kind is what decides this, so a new setting on an environment gets
        the right control without anything here changing. A number that can be
        switched off -- a timeout, a time limit -- gets a box that says Off
        until it is ticked, rather than reading zero as off, which nobody would
        guess.
        """
        tag = f"__env_set_{setting.name}"
        value = self.values.get(setting.name, setting.default)
        common = dict(tag=tag, user_data=setting.name, callback=self._setting_changed)
        if setting.kind == ENUM:
            dpg.add_combo(list(setting.choices), default_value=value or setting.default,
                          width=160, **common)
        elif setting.kind == BOOL:
            dpg.add_checkbox(default_value=bool(value), **common)
        elif setting.kind in (INT, FLOAT):
            on = True
            if setting.optional:
                on = value is not None
                dpg.add_checkbox(label="" if on else "Off", default_value=on,
                                 tag=f"{tag}_on", user_data=setting.name,
                                 callback=self._optional_switched)
                if value is None:
                    value = self._optional_numbers.get(setting.name, 10)
            bounds = {}
            if setting.minimum is not None:
                bounds.update(min_value=setting.minimum, min_clamped=True)
            if setting.maximum is not None:
                bounds.update(max_value=setting.maximum, max_clamped=True)
            with dpg.group(horizontal=True, tag=f"{tag}_value", show=on):
                if setting.kind == INT:
                    dpg.add_input_int(default_value=int(value or 0), step=0, width=110,
                                      **bounds, **common)
                else:
                    dpg.add_input_float(default_value=float(value or 0.0), step=0, width=110,
                                        format="%.2f", **bounds, **common)
                if setting.unit:
                    style.unit(setting.unit)
        elif setting.kind == COLOR:
            colour = list(value or (255, 255, 255)) + [255]
            dpg.add_color_edit(default_value=colour[:4], no_alpha=True, width=220, **common)
        else:
            dpg.add_input_text(default_value="" if value is None else str(value), width=-1,
                               hint="Leave empty for none" if setting.optional else "",
                               **common)

    # -- what is shown when ----------------------------------------------
    def _controller_fields(self):
        """Show only the fields the chosen controller actually uses."""
        kind = self.controller["kind"]
        classifier = kind == "classifier"
        shown = {
            "__env_row_address": kind in ("classifier", "regressor"),
            "__env_row_format": classifier,
            "__env_row_classes": classifier and self.controller["output_format"] == "probabilities",
            "__env_mapping": classifier,
            "__env_keyboard_note": kind == "keyboard",
            "__env_regressor_note": kind == "regressor",
        }
        for tag, show in shown.items():
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=show)

    def _check_mapping(self):
        """Say when two actions share a class, or a class is one the model cannot send."""
        if not dpg.does_item_exist("__env_map_warning"):
            return
        spec = self.registry[self.selected]
        problems = []
        owner = {}
        for action in spec.actions:
            cls = int(self.controller["class_map"].get(action.name, action.default_class))
            if cls < 0:
                continue
            if cls in owner:
                problems.append(f"Class {cls} is set for both {owner[cls]} and "
                                f"{action.label}; it will do {owner[cls]}.")
            else:
                owner[cls] = action.label
        if self.controller["output_format"] == "probabilities":
            classes = int(self.controller["num_classes"])
            beyond = sorted(c for c in owner if c >= classes)
            if beyond:
                problems.append(f"The model sends {classes} classes (0 to {classes - 1}), so "
                                f"class {', '.join(map(str, beyond))} will never arrive.")
        dpg.set_value("__env_map_warning", " ".join(problems))
        dpg.configure_item("__env_map_warning", show=bool(problems))

    def _describe(self):
        """Say what Launch will do, and where the run will be saved."""
        spec = self.registry[self.selected]
        folder = self.save["folder"].strip()
        if dpg.does_item_exist("__env_save_where"):
            if folder:
                where = Path(folder) / f"{spec.id}_<date>_<time>"
                dpg.set_value("__env_save_where", f"Each run is saved to {where}")
            else:
                dpg.set_value("__env_save_where", "Nothing will be saved.")
        if self.embedded is not None:
            return      # the status line is reporting the run
        width, height = spec.frame_size(self.values)
        kind = self.controller["kind"]
        if kind == "keyboard":
            driven = "the arrow keys"
        else:
            driven = f"a {kind} sending to {self.controller['ip']}:{self.controller['port']}"
        saved = "saved" if folder else "not saved"
        self._say(f"Ready. {spec.title} at {width} × {height}, driven by {driven}; runs "
                  f"are {saved}.")

    # -- callbacks -------------------------------------------------------
    def _fit_controller(self, spec):
        """Keep a controller the task accepts. Curricular Fitts takes only a regressor."""
        offered = [kind.lower() for kind in spec.controllers]
        if self.controller["kind"] not in offered:
            self.controller["kind"] = offered[0]

    def _set_controller(self, **changes):
        self.controller.update(changes)
        self._controller_fields()
        self._check_mapping()
        self._describe()

    def _controller_changed(self, sender, app_data):
        self._set_controller(kind=app_data.strip().lower())

    def _class_map_changed(self, sender, app_data, user_data):
        self.controller["class_map"][user_data] = int(app_data)
        self._check_mapping()

    def _folder_typed(self, sender, app_data):
        self.save["folder"] = app_data.strip()
        self._describe()

    def _folder_chosen(self, sender, app_data):
        folder = (app_data or {}).get("file_path_name", "")
        if not folder:
            return
        self.save["folder"] = folder
        if dpg.does_item_exist("__env_save_folder"):
            dpg.set_value("__env_save_folder", folder)
        self._describe()

    def _save_option_changed(self, sender, app_data, user_data):
        self.save[user_data] = bool(app_data)

    def _choose(self, sender, app_data):
        for key, spec in self.registry.items():
            if spec.title == app_data:
                self.selected = key
                break
        spec = self.registry[self.selected]
        # Keep the chooser showing what is actually selected. Reset settings
        # and any programmatic change come through here too, and a combo left
        # showing the previous task is worse than no label at all.
        if dpg.does_item_exist("__env_choice"):
            dpg.set_value("__env_choice", spec.title)
        self.values = spec.defaults()
        self._fit_controller(spec)
        # Each task has its own actions. One it shares with the last task (Fitts and
        # ISO Fitts share all five) keeps its class; the rest start from the defaults.
        previous = self.controller["class_map"]
        self.controller["class_map"] = {
            name: previous.get(name, default)
            for name, default in spec.default_class_map().items()}
        if dpg.does_item_exist("__env_body"):
            dpg.delete_item("__env_body", children_only=True)
            dpg.push_container_stack("__env_body")
            self._build_body()
            dpg.pop_container_stack()
        self._describe()

    def _reset(self):
        self.values = self.registry[self.selected].defaults()
        self.controller["class_map"] = self.registry[self.selected].default_class_map()
        self._optional_numbers = dict(_OPTIONAL_START)
        self._choose(None, self.registry[self.selected].title)
        if self.embedded is None:
            self._say("Settings are back to the task's own defaults.")

    def _setting_changed(self, sender, app_data, user_data):
        spec = self.registry[self.selected]
        setting = spec.setting(user_data)
        if setting is None:
            return
        if setting.kind == COLOR:
            # The colour control works in 0-1 floats; the environments want
            # 0-255 integers.
            app_data = [c * 255 if c <= 1.0 else c for c in app_data[:3]]
        if setting.optional and setting.kind in (INT, FLOAT):
            self._optional_numbers[setting.name] = app_data
            if not dpg.get_value(f"__env_set_{setting.name}_on"):
                return
        self.values[user_data] = setting.coerce(app_data)
        self._describe()

    def _optional_switched(self, sender, on, name):
        setting = self.registry[self.selected].setting(name)
        number_tag = f"__env_set_{name}"
        dpg.configure_item(f"{number_tag}_value", show=on)
        dpg.configure_item(sender, label="" if on else "Off")
        if on:
            number = dpg.get_value(number_tag)
            if not number or number <= 0:
                number = _OPTIONAL_START.get(name, 10)
                dpg.set_value(number_tag, number)
            self.values[name] = setting.coerce(number)
        else:
            self.values[name] = None
        self._describe()

    # ==================================================================
    # playing
    # ==================================================================
    def _launch(self):
        if self.embedded is not None:
            self._say("A task is already running. Stop it first.", "warn")
            return
        spec = self.registry[self.selected]
        width, height = spec.frame_size(self.values)
        settings = {name: spec.setting(name).coerce(value)
                    for name, value in self.values.items()
                    if spec.setting(name) is not None}
        controller = ControllerSpec(kind=self.controller["kind"],
                                    ip=self.controller["ip"],
                                    port=int(self.controller["port"]),
                                    num_classes=int(self.controller["num_classes"]),
                                    output_format=self.controller["output_format"],
                                    class_map=self.controller["class_map"])
        recording = self._recording_config(spec)
        try:
            factory = build_factory(spec, controller, settings, recording=recording)
        except Exception:
            self._say("Could not prepare that task: "
                      + traceback.format_exc().strip().splitlines()[-1], "error")
            return

        recorder = None
        if recording is not None:
            # Before the environment, so the EMG covers its first frame.
            try:
                recorder = EnvironmentRecording(recording, self.online_data_handler).start()
            except Exception:
                self._say("Could not start saving: "
                          + traceback.format_exc().strip().splitlines()[-1], "error")
                return

        self._counter += 1
        name = f"libemg_env_{spec.id}_{self._counter}"
        try:
            self.embedded = EmbeddedEnvironment(name, factory, width, height).start()
        except Exception:
            self._say("Could not start that task: "
                      + traceback.format_exc().strip().splitlines()[-1], "error")
            self.embedded = None
            if recorder is not None:
                recorder.stop(assemble_files=False)
            return
        self.recorder = recorder
        self._frame_size = (width, height)
        self._open_play_window(spec, width, height)
        style.set_enabled(LAUNCH_TAG, False)
        style.set_enabled(STOP_TAG, True)
        saving = ""
        if recorder is not None:
            saving = f" Saving to {recorder.folder}"
            if recording.emg and not recorder.records_emg:
                saving += " (frames only: no device is streaming)"
            saving += "."
        self._say(f"{spec.title} is running.{saving}", "ok")

    def _recording_config(self, spec):
        """What to save of this run, or None when no folder is set.

        Every run gets a folder of its own, named for the task and when it
        started: the EMG log appends, so reusing a folder would run two
        sessions together, and a second run should never overwrite the first.
        """
        folder = self.save["folder"].strip()
        if not folder:
            return None
        run = Path(folder) / f"{spec.id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        return RecordingConfig(folder=str(run),
                               **{name: bool(self.save[name]) for name, _, _, _ in _SAVE_FIELDS})

    def _open_play_window(self, spec, width, height):
        for tag in (PLAY_TAG, TEXTURE_TAG):
            if dpg.does_alias_exist(tag):
                dpg.delete_item(tag)
        # The texture is backed by the very memory the environment draws into,
        # so there is no per-frame upload and nothing to copy here. What the
        # game paints is what this shows.
        with dpg.texture_registry():
            dpg.add_raw_texture(width, height, self.embedded.pixels(),
                                format=dpg.mvFormat_Float_rgba, tag=TEXTURE_TAG)
        # Offset from the setup window rather than opening on top of it, so the
        # settings stay readable while the task runs and can be compared
        # against what is happening on screen.
        offset = (min(self.width, 1600) - width - 40, 40)
        position = (max(20, offset[0]), offset[1])
        with dpg.window(label=f"{spec.title}", tag=PLAY_TAG,
                        width=width + 28, height=height + 92, pos=position,
                        on_close=lambda: self.stop()):
            with dpg.group(horizontal=True):
                style.button("Stop", "danger", width=90, callback=self.stop)
                dpg.add_text("Starting...", tag=STATUS_TAG, color=style.MUTED)
            style.note("Click the game to give it the keyboard.")
            dpg.add_image(TEXTURE_TAG, tag="__env_image")
        dpg.focus_item(PLAY_TAG)

    def stop(self, message="Stopped.", kind="info"):
        """Stop the running environment and release its memory.

        Parameters
        ----------
        message: str (optional), default='Stopped.'
            What to leave on screen. The default is replaced when the reason
            for stopping is worth keeping: reporting why a task refused its
            settings and then immediately overwriting it with "Stopped." tells
            the user nothing at all.
        kind: str (optional), default='info'
            How the message is coloured: info, ok, busy, warn or error.
        """
        if not isinstance(message, str):
            # Called as a button callback, which passes the button.
            message, kind = "Stopped.", "info"
        if self.embedded is None:
            return
        recorder, self.recorder = self.recorder, None
        try:
            # Long enough for the environment to write its frames on the way
            # out; it returns as soon as the process has exited.
            self.embedded.stop(timeout=30.0 if recorder is not None else 4.0)
        except Exception:
            pass
        self.embedded = None
        if dpg.does_alias_exist(PLAY_TAG):
            dpg.delete_item(PLAY_TAG)
        if dpg.does_alias_exist(TEXTURE_TAG):
            dpg.delete_item(TEXTURE_TAG)
        style.set_enabled(LAUNCH_TAG, True)
        style.set_enabled(STOP_TAG, False)
        if recorder is not None:
            message += f" Saving the run to {recorder.folder}..."
            kind = "busy" if kind != "error" else kind
            # Not a daemon: closing the window mid-write would leave half a
            # file, so the interpreter waits for this to finish.
            threading.Thread(target=self._finish_recording, args=(recorder,),
                             daemon=False).start()
        self._say(message, kind)

    def _finish_recording(self, recorder):
        """Stop the EMG log and write the training files. Runs off the render thread."""
        try:
            written = recorder.stop()
            if written:
                self._pending_message = (f"Saved {len(written)} trial file(s) to "
                                         f"{recorder.folder}.", "ok")
            else:
                self._pending_message = (f"Nothing was saved to {recorder.folder}: the task "
                                         "stopped before it recorded a frame.", "warn")
        except Exception:
            self._pending_message = ("Could not save the run: "
                                     + traceback.format_exc().strip().splitlines()[-1]
                                     + f" The raw files are in {recorder.folder}; "
                                       "libemg.environments.recording.assemble rebuilds "
                                       "the trial files from them.", "error")

    def cleanup(self):
        """Stop anything running and remove the windows."""
        self.stop()
        for tag in (WINDOW_TAG, FOLDER_DIALOG_TAG):
            if dpg.does_alias_exist(tag):
                dpg.delete_item(tag)

    # ==================================================================
    # per-frame, on the render thread
    # ==================================================================
    def poll(self):
        """Forward input and refresh status. Called once per rendered frame.

        The keyboard is read here rather than through key handlers because a
        game wants to know what is held right now, every frame, not to be told
        once when a key goes down.
        """
        if self._pending_message is not None:
            (message, kind), self._pending_message = self._pending_message, None
            self._say(message, kind)
        if self.embedded is None:
            return
        held = {name for name, key in _DPG_KEYS.items()
                if _key_down(key)}
        pointer = None
        if dpg.does_item_exist("__env_image"):
            try:
                x, y = dpg.get_mouse_pos(local=False)
                origin = dpg.get_item_rect_min("__env_image")
                pointer = (int(x - origin[0]), int(y - origin[1]))
            except Exception:
                pointer = None
        self.embedded.send_input(held, mouse=pointer,
                                 mouse_down=dpg.is_mouse_button_down(dpg.mvMouseButton_Left))

        status = self.embedded.status()
        if dpg.does_item_exist(STATUS_TAG):
            if status["running"]:
                dpg.set_value(STATUS_TAG, f"Running   {status['fps']:.0f} fps   "
                                          f"{status['frames']} frames")
                dpg.configure_item(STATUS_TAG, color=style.SUCCESS)
            else:
                dpg.set_value(STATUS_TAG, "Starting..." if status["frames"] == 0 else "Stopping...")
        if not status["running"] and status["frames"] == 0 and not self.embedded.alive:
            # It never drew anything, so it failed to start. The reason comes
            # back through the bridge, because the traceback happened in a
            # process nobody is watching.
            reason = status["error"] or "it stopped before drawing anything"
            self.stop(f"That task could not start: {reason}", "error")
            return
        if status["finished"] or not self.embedded.alive:
            if status["finished"]:
                self.stop("The task finished.", "ok")
            else:
                self.stop(f"The task stopped. {status['error']}".strip(),
                          "error" if status["error"] else "info")

    # ------------------------------------------------------------------
    def _say(self, message, kind="info"):
        style.set_status(MESSAGE_TAG, message, kind)


def _key_down(name):
    key = getattr(dpg, name, None)
    if key is None:
        return False
    try:
        return dpg.is_key_down(key)
    except Exception:
        return False


def _title(kind):
    return str(kind).strip().title()
