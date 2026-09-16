"""The Environments window: pick a task, set it up, play it in the GUI.

Two screens. The setup screen is generated from the environment's own
configuration, so every setting it accepts appears with the author's own
description beside it. The play screen is a single image, backed by the memory
the environment is drawing into, plus what it is doing and a way to stop it.

Nothing about the game runs here. The environment is a process of its own, and
this window forwards the keyboard to it and shows the frames it produces. That
separation is what keeps a game that stalls from taking the interface with it.
"""

import time
import traceback

import dearpygui.dearpygui as dpg

from libemg._gui._environments.embedded import EmbeddedEnvironment
from libemg._gui._environments.factories import ControllerSpec, build_factory
from libemg._gui._environments.frame_bridge import FORWARDED_KEYS
from libemg._gui._environments.registry import (BOOL, COLOR, ENUM, FLOAT, INT,
                                                PATH, STR, default_registry)

WINDOW_TAG = "__env_window"
PLAY_TAG = "__env_play"
STATUS_TAG = "__env_status"
MESSAGE_TAG = "__env_message"
TEXTURE_TAG = "__env_texture"

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
        Not required. Kept so a caller can pass one through without a branch.

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
                           "num_classes": 5, "output_format": "predictions"}
        self.embedded = None
        self._counter = 0
        self._frame_size = (0, 0)

    # ==================================================================
    # setup screen
    # ==================================================================
    def spawn_window(self):
        """Build the setup window for the selected environment."""
        self.cleanup()
        self.values = self.registry[self.selected].defaults()
        with dpg.window(label="Environments", tag=WINDOW_TAG,
                        width=self.width, height=self.height,
                        on_close=lambda: self.cleanup()):
            with dpg.group(horizontal=True):
                dpg.add_text("Task")
                dpg.add_combo([s.title for s in self.registry.values()],
                              default_value=self.registry[self.selected].title,
                              tag="__env_choice", width=240,
                              callback=self._choose)
                dpg.add_button(label="Launch", callback=self._launch)
                dpg.add_button(label="Stop", callback=self.stop)
                dpg.add_button(label="Reset settings", callback=self._reset)
            dpg.add_text("", tag=MESSAGE_TAG, wrap=self.width - 40)
            dpg.add_separator()
            with dpg.child_window(tag="__env_body", autosize_x=True, autosize_y=True):
                self._build_body()
        self._describe()
        return self

    def _build_body(self):
        spec = self.registry[self.selected]
        dpg.add_text(spec.help, wrap=self.width - 60, color=(170, 190, 210))
        dpg.add_separator()

        dpg.add_text("Control")
        with dpg.group(horizontal=True):
            dpg.add_combo(spec.controllers, default_value=_title(self.controller["kind"]),
                          tag="__env_controller", width=160,
                          callback=self._controller_changed)
            dpg.add_input_text(label="Address", default_value=self.controller["ip"],
                               tag="__env_ip", width=120,
                               callback=lambda s, a: self.controller.update(ip=a))
            dpg.add_input_int(label="Port", default_value=self.controller["port"],
                              tag="__env_port", width=110,
                              callback=lambda s, a: self.controller.update(port=a))
            dpg.add_input_int(label="Classes", default_value=self.controller["num_classes"],
                              tag="__env_classes", width=110,
                              callback=lambda s, a: self.controller.update(num_classes=a))
        with dpg.tooltip("__env_controller"):
            dpg.add_text("Keyboard drives the task with the arrow keys, for trying it "
                         "out. Classifier and Regressor listen on the address below "
                         "for a running model's output.", wrap=320)
        self._controller_fields()

        dpg.add_separator()
        dpg.add_text("Settings")
        plain = [s for s in spec.settings if not _is_appearance(s)]
        appearance = [s for s in spec.settings if _is_appearance(s)]
        # One column, not two. A second column has to be given a width, and
        # whatever is chosen is wrong for some window size, which shows up as
        # settings that are simply not on screen. A single list always fits,
        # and the window scrolls.
        for setting in plain:
            self._draw_setting(setting)
        if appearance:
            with dpg.tree_node(label="Appearance", default_open=False):
                for setting in appearance:
                    self._draw_setting(setting)

    def _draw_setting(self, setting):
        """One control, chosen by the setting's kind.

        The kind is what decides this, so a new setting on an environment gets
        the right control without anything here changing.
        """
        tag = f"__env_set_{setting.name}"
        value = self.values.get(setting.name, setting.default)
        data = setting.name
        common = dict(tag=tag, user_data=data, callback=self._setting_changed,
                      label=setting.label, width=150)
        with dpg.group():
            if setting.kind == ENUM:
                dpg.add_combo(list(setting.choices),
                              default_value=value or setting.default, **common)
            elif setting.kind == BOOL:
                dpg.add_checkbox(default_value=bool(value), tag=tag, user_data=data,
                                 callback=self._setting_changed, label=setting.label)
            elif setting.kind == INT:
                dpg.add_input_int(default_value=int(value or 0), step=1, **common)
            elif setting.kind == FLOAT:
                dpg.add_input_float(default_value=float(value or 0.0), step=0.0,
                                    format="%.3f", **common)
            elif setting.kind == COLOR:
                colour = list(value or (255, 255, 255)) + [255]
                dpg.add_color_edit(default_value=colour[:4], tag=tag, user_data=data,
                                   callback=self._setting_changed,
                                   label=setting.label, width=150,
                                   no_alpha=True)
            else:
                dpg.add_input_text(default_value="" if value is None else str(value),
                                   hint="leave empty for none" if setting.optional else "",
                                   **common)
            if setting.help:
                with dpg.tooltip(tag):
                    dpg.add_text(setting.help, wrap=360)

    def _controller_fields(self):
        """Show only the fields the chosen controller actually uses."""
        socket_based = self.controller["kind"] in ("classifier", "regressor")
        for tag in ("__env_ip", "__env_port"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=socket_based)
        if dpg.does_item_exist("__env_classes"):
            dpg.configure_item("__env_classes",
                               show=self.controller["kind"] == "classifier")

    # ------------------------------------------------------------------
    def _choose(self, sender, app_data):
        for key, spec in self.registry.items():
            if spec.title == app_data:
                self.selected = key
                break
        # Keep the chooser showing what is actually selected. Reset settings
        # and any programmatic change come through here too, and a combo left
        # showing the previous task is worse than no label at all.
        if dpg.does_item_exist("__env_choice"):
            dpg.set_value("__env_choice", self.registry[self.selected].title)
        self.values = self.registry[self.selected].defaults()
        if dpg.does_item_exist("__env_body"):
            dpg.delete_item("__env_body", children_only=True)
            dpg.push_container_stack("__env_body")
            self._build_body()
            dpg.pop_container_stack()
        self._describe()

    def _reset(self):
        self.values = self.registry[self.selected].defaults()
        self._choose(None, self.registry[self.selected].title)
        self._say("Settings reset to the environment's own defaults.")

    def _controller_changed(self, sender, app_data):
        self.controller["kind"] = app_data.strip().lower()
        self._controller_fields()

    def _setting_changed(self, sender, app_data, user_data):
        spec = self.registry[self.selected]
        setting = spec.setting(user_data)
        if setting is None:
            return
        if setting.kind == COLOR:
            # The colour control works in 0-1 floats; the environments want
            # 0-255 integers.
            app_data = [c * 255 if c <= 1.0 else c for c in app_data[:3]]
        self.values[user_data] = setting.coerce(app_data)
        self._describe()

    def _describe(self):
        spec = self.registry[self.selected]
        width, height = spec.frame_size(self.values)
        self._say(f"{spec.title} will draw at {width} by {height} pixels, "
                  f"driven by the {_title(self.controller['kind'])} controller.")

    # ==================================================================
    # playing
    # ==================================================================
    def _launch(self):
        if self.embedded is not None:
            self._say("Already running. Stop it first.")
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
                                    output_format=self.controller["output_format"])
        try:
            factory = build_factory(spec, controller, settings)
        except Exception:
            self._say("Could not prepare that task:\n"
                      + traceback.format_exc().strip().splitlines()[-1])
            return

        self._counter += 1
        name = f"libemg_env_{spec.id}_{self._counter}"
        try:
            self.embedded = EmbeddedEnvironment(name, factory, width, height).start()
        except Exception:
            self._say("Could not start that task:\n"
                      + traceback.format_exc().strip().splitlines()[-1])
            self.embedded = None
            return
        self._frame_size = (width, height)
        self._open_play_window(spec, width, height)
        self._say(f"{spec.title} is running. Click the game to give it the keyboard.")

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
                        width=width + 20, height=height + 80, pos=position,
                        on_close=lambda: self.stop()):
            with dpg.group(horizontal=True):
                dpg.add_text("", tag=STATUS_TAG)
                dpg.add_spacer(width=20)
                dpg.add_button(label="Stop", callback=self.stop)
            dpg.add_image(TEXTURE_TAG, tag="__env_image")
        dpg.focus_item(PLAY_TAG)

    def stop(self, message="Stopped."):
        """Stop the running environment and release its memory.

        Parameters
        ----------
        message: str (optional), default='Stopped.'
            What to leave on screen. The default is replaced when the reason
            for stopping is worth keeping: reporting why a task refused its
            settings and then immediately overwriting it with "Stopped." tells
            the user nothing at all.
        """
        if self.embedded is None:
            return
        try:
            self.embedded.stop()
        except Exception:
            pass
        self.embedded = None
        if dpg.does_alias_exist(PLAY_TAG):
            dpg.delete_item(PLAY_TAG)
        if dpg.does_alias_exist(TEXTURE_TAG):
            dpg.delete_item(TEXTURE_TAG)
        self._say(message)

    def cleanup(self):
        """Stop anything running and remove the windows."""
        self.stop()
        for tag in (WINDOW_TAG,):
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
        if self.embedded is None:
            return
        held = {name for name, key in _DPG_KEYS.items()
                if _key_down(key)}
        pointer = None
        if dpg.does_item_exist("__env_image"):
            try:
                x, y = dpg.get_drawing_mouse_pos() if False else dpg.get_mouse_pos(local=False)
                origin = dpg.get_item_rect_min("__env_image")
                pointer = (int(x - origin[0]), int(y - origin[1]))
            except Exception:
                pointer = None
        self.embedded.send_input(held, mouse=pointer,
                                 mouse_down=dpg.is_mouse_button_down(dpg.mvMouseButton_Left))

        status = self.embedded.status()
        if dpg.does_item_exist(STATUS_TAG):
            dpg.set_value(STATUS_TAG,
                          "frames %d   %.0f fps   %s"
                          % (status["frames"], status["fps"],
                             "running" if status["running"] else "stopping"))
        if not status["running"] and status["frames"] == 0 and not self.embedded.alive:
            # It never drew anything, so it failed to start. The reason comes
            # back through the bridge, because the traceback happened in a
            # process nobody is watching.
            reason = status["error"] or "it stopped before drawing anything"
            self.stop(f"That task could not start: {reason}")
            return
        if status["finished"] or not self.embedded.alive:
            self.stop("The task finished." if status["finished"]
                      else f"The task stopped. {status['error']}".strip())

    # ------------------------------------------------------------------
    def _say(self, message):
        if dpg.does_item_exist(MESSAGE_TAG):
            dpg.set_value(MESSAGE_TAG, message)


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


def _is_appearance(setting):
    """Whether a setting is about how the task looks rather than what it does."""
    return setting.kind == COLOR or "color" in setting.name

