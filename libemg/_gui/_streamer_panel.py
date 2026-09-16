"""Start and stop a device from the GUI.

This is the first thing a session needs and the last thing that still had to be
done in a script. With it, a device can be brought up, watched, and handed to
everything else in the window: screen guided training, the live signal view,
the pipeline editor and the environments all work off the handler this panel
publishes.

As elsewhere, the device list and each device's options are generated rather
than written out. The streamers are found in :mod:`libemg.streamers` and
filtered to those that write into shared memory, because a streamer that talks
over a socket has nothing the rest of the GUI could attach to. Each device's
own arguments become its controls, so a new device appears here with its
options intact and nothing in this file changes.
"""

import time
import traceback

import dearpygui.dearpygui as dpg

from libemg._gui._pipeline.registry import (BOOL, FLOAT, INT, STR,
                                            _params_from_signature,
                                            _streamer_functions)

WINDOW_TAG = "__streamer_window"
STATUS_TAG = "__streamer_status"
MESSAGE_TAG = "__streamer_message"
TABLE_TAG = "__streamer_table"

#: Offered alongside the real devices. Building a pipeline, trying an
#: environment or rehearsing a session should not require hardware on the desk.
SYNTHETIC = "Synthetic (no hardware)"


class StreamerPanel:
    """Bring a device up, watch it, and hand it to the rest of the GUI.

    Parameters
    ----------
    on_started: callable or None (optional), default=None
        Called as ``on_started(online_data_handler, shared_memory_items)`` once
        a device is streaming. The GUI uses this to make the handler available
        to its other panels.
    on_stopped: callable or None (optional), default=None
        Called when the device stops.
    width, height: int (optional)
        Window size.

    Examples
    ---------
    >>> panel = StreamerPanel()
    >>> panel.spawn_window()
    """


    #: The window this panel owns, so a caller can ask if it is open.
    window_tag = WINDOW_TAG

    def __init__(self, on_started=None, on_stopped=None, width=760, height=620):
        self.on_started = on_started
        self.on_stopped = on_stopped
        self.width, self.height = width, height
        self.functions = _streamer_functions()
        self.devices = [SYNTHETIC] + sorted(self.functions)
        self.selected = SYNTHETIC
        self.values = {}
        self.handle = None
        self.items = None
        self.odh = None
        self._modalities = []
        self._previous = {}
        self._previous_at = 0.0
        self._rates = {}
        self._started_at = 0.0
        self._seen_data = False
        self._warned = False

    # ==================================================================
    # window
    # ==================================================================
    def spawn_window(self):
        """Build the streamer window."""
        self.cleanup()
        with dpg.window(label="Streamer", tag=WINDOW_TAG,
                        width=self.width, height=self.height,
                        on_close=lambda: self.cleanup()):
            with dpg.group(horizontal=True):
                dpg.add_text("Device")
                dpg.add_combo(self.devices, default_value=self.selected,
                              tag="__streamer_choice", width=240,
                              callback=self._choose)
                dpg.add_button(label="Start", tag="__streamer_start",
                               callback=self.start)
                dpg.add_button(label="Stop", tag="__streamer_stop",
                               callback=self.stop)
            dpg.add_text("", tag=MESSAGE_TAG, wrap=self.width - 40)
            dpg.add_separator()
            with dpg.child_window(tag="__streamer_body", autosize_x=True,
                                  height=260):
                self._build_settings()
            dpg.add_separator()
            dpg.add_text("Incoming data", tag=STATUS_TAG)
            with dpg.table(tag=TABLE_TAG, header_row=True, borders_innerH=True,
                           borders_outerH=True, borders_innerV=True,
                           policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(label="Modality")
                dpg.add_table_column(label="Samples")
                dpg.add_table_column(label="Rate (Hz)")
                dpg.add_table_column(label="Writes")
        self._say("Pick a device and press Start. Nothing else in the window "
                  "can see data until one is running.")
        return self

    def _build_settings(self):
        if self.selected == SYNTHETIC:
            dpg.add_text("Generates samples at a fixed rate, so the rest of the "
                         "window can be used with no hardware attached.",
                         wrap=self.width - 60, color=(170, 190, 210))
            for name, kind, default, minimum, choices in (
                    ("sampling_rate", INT, 1000, 1, None),
                    ("num_channels", INT, 8, 1, None),
                    ("pattern", "enum", "bursts", None,
                     ["noise", "sine", "bursts"]),
                    ("amplitude", FLOAT, 1.0, 0.0, None)):
                self.values.setdefault(name, default)
                self._control(name, kind, default, minimum, choices)
            return

        function = self.functions[self.selected]
        summary = (function.__doc__ or "").strip().split("\n")[0]
        dpg.add_text(summary[:200], wrap=self.width - 60, color=(170, 190, 210))
        specs = _params_from_signature(function)
        if not specs:
            dpg.add_text("This device takes no options.")
        for spec in specs:
            self.values.setdefault(spec.name, spec.default)
            kind = {INT: INT, FLOAT: FLOAT, BOOL: BOOL}.get(spec.kind, STR)
            self._control(spec.name, kind, spec.default, None, None,
                          help=spec.help)

    def _control(self, name, kind, default, minimum, choices, help=""):
        tag = f"__streamer_set_{name}"
        value = self.values.get(name, default)
        label = name.replace("_", " ").title()
        common = dict(tag=tag, user_data=name, callback=self._changed,
                      label=label, width=160)
        if kind == "enum":
            dpg.add_combo(list(choices), default_value=value or default, **common)
        elif kind == BOOL:
            dpg.add_checkbox(default_value=bool(value), tag=tag, user_data=name,
                             callback=self._changed, label=label)
        elif kind == INT:
            dpg.add_input_int(default_value=int(value or 0), step=1, **common)
        elif kind == FLOAT:
            dpg.add_input_float(default_value=float(value or 0.0), step=0.0,
                                format="%.3f", **common)
        else:
            dpg.add_input_text(default_value="" if value is None else str(value),
                               **common)
        if help:
            with dpg.tooltip(tag):
                dpg.add_text(help, wrap=320)

    def _choose(self, sender, app_data):
        if self.handle is not None:
            self._say("Stop the running device before switching to another.")
            dpg.set_value("__streamer_choice", self.selected)
            return
        self.selected = app_data
        self.values = {}
        if dpg.does_item_exist("__streamer_body"):
            dpg.delete_item("__streamer_body", children_only=True)
            dpg.push_container_stack("__streamer_body")
            self._build_settings()
            dpg.pop_container_stack()

    def _changed(self, sender, app_data, user_data):
        self.values[user_data] = app_data

    # ==================================================================
    # running
    # ==================================================================
    def start(self):
        """Bring the selected device up and publish its handler."""
        if self.handle is not None:
            self._say("Already streaming. Stop it first.")
            return
        try:
            handle, items = self._launch()
        except Exception:
            # A device that is not plugged in, a driver that is missing, a port
            # already in use: all of these surface here, and the last line of
            # the traceback is the part worth showing.
            self._say("Could not start that device:\n"
                      + traceback.format_exc().strip().splitlines()[-1])
            return

        from libemg.data_handler import OnlineDataHandler
        self.handle, self.items = handle, items
        try:
            self.odh = OnlineDataHandler(items)
        except Exception:
            self._say("The device started but its data could not be attached:\n"
                      + traceback.format_exc().strip().splitlines()[-1])
            self.stop()
            return

        # Start from zero. Shared-memory segments outlive the process that made
        # them, and every device writes to the same modality names, so a new
        # device attaches to whatever the last one left behind. Without this,
        # counters from a previous session read as live data and a device that
        # is not connected looks like it is working.
        try:
            self.odh.reset()
        except Exception:
            pass

        self._modalities = list(self.odh.modalities)
        self._previous, self._rates = {}, {}
        self._previous_at = time.perf_counter()
        self._started_at = self._previous_at
        self._seen_data = False
        self._warned = False
        self._rebuild_table()
        if self.on_started is not None:
            self.on_started(self.odh, items)
        # Deliberately not "streaming" yet. A device streamer spawns a process
        # that connects on its own, so a device that is unplugged, asleep or
        # paired to something else starts perfectly well and simply never
        # produces a sample. Waiting to see one before claiming success is what
        # stops the window from reporting a working device that is not there.
        self._say(f"{self.selected} started. Waiting for the first samples.")

    def _launch(self):
        """Call the chosen streamer, whatever kind it is."""
        if self.selected == SYNTHETIC:
            from libemg._gui._pipeline.synthetic import synthetic_streamer
            return synthetic_streamer(
                sampling_rate=int(self.values.get("sampling_rate", 1000)),
                num_channels=int(self.values.get("num_channels", 8)),
                pattern=self.values.get("pattern", "bursts"),
                amplitude=float(self.values.get("amplitude", 1.0)))

        function = self.functions[self.selected]
        keywords = {}
        for spec in _params_from_signature(function):
            value = self.values.get(spec.name, spec.default)
            if value is None or value == "":
                continue
            keywords[spec.name] = value
        result = function(**keywords)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        raise RuntimeError(
            f"{self.selected} did not return a streamer and its shared memory. "
            "A device the GUI can use has to return both.")

    def stop(self):
        """Stop the device and tell the rest of the window it has gone."""
        if self.handle is None:
            return
        try:
            # Streamers stop by their own signal where they have one, and are
            # terminated where they do not; both kinds exist in the library.
            if hasattr(self.handle, "signal"):
                self.handle.signal.set()
                if hasattr(self.handle, "join"):
                    self.handle.join(timeout=3)
            if hasattr(self.handle, "is_alive") and self.handle.is_alive():
                self.handle.terminate()
        except Exception:
            pass
        self.handle = None
        self.items = None
        # The handler opened an OS handle per shared-memory segment. Dropping
        # the reference does not close them, so a start-stop-start cycle would
        # accumulate handles for as long as the window stayed open. Closing
        # without unlinking is the right half: this process is not the one that
        # created the segments, and unlinking them would pull the ground out
        # from under anything else still attached.
        if self.odh is not None:
            try:
                self.odh.smm.cleanup(parent=False)
            except Exception:
                pass
        self.odh = None
        self._modalities = []
        self._rebuild_table()
        if self.on_stopped is not None:
            self.on_stopped()
        self._say("Stopped.")

    def cleanup(self):
        """Stop the device and remove the window."""
        self.stop()
        if dpg.does_alias_exist(WINDOW_TAG):
            dpg.delete_item(WINDOW_TAG)

    # ==================================================================
    # per-frame, on the render thread
    # ==================================================================
    def poll(self):
        """Refresh what the device is delivering.

        Reads each modality's state block rather than its data. That is a
        handful of integers, so showing a live rate costs nothing measurable
        and cannot slow the device down.
        """
        if self.odh is None or not self._modalities:
            return
        now = time.perf_counter()
        elapsed = now - self._previous_at
        if elapsed < 0.25:
            return
        try:
            states = self.odh.get_state()
        except Exception:
            return
        self._previous_at = now
        arriving = any(state.total_samples > 0 for state in states.values())
        if arriving and not self._seen_data:
            self._seen_data = True
            self._say(f"{self.selected} is streaming. Collect Data, Live Signal, "
                      "the pipeline editor and the environments can all use it now.")
        elif not arriving and not self._warned and now - self._started_at > 5.0:
            self._warned = True
            self._say(f"{self.selected} started, but no samples have arrived in "
                      "five seconds. Check that the device is on, paired and not "
                      "in use by another program. It is left running in case it "
                      "is still connecting.")
        for modality, state in states.items():
            previous = self._previous.get(modality)
            if previous is not None and elapsed > 0:
                instant = (state.total_samples - previous) / elapsed
                # Smoothed, because a rate recomputed from a quarter second
                # jumps around too much to read.
                held = self._rates.get(modality)
                self._rates[modality] = instant if held is None \
                    else 0.7 * held + 0.3 * instant
            self._previous[modality] = state.total_samples
            self._set_row(modality, state)
        if dpg.does_item_exist(STATUS_TAG):
            total = sum(self._rates.values())
            count = len(self._modalities)
            dpg.set_value(STATUS_TAG,
                          f"Incoming data    {total:.0f} samples per second across "
                          f"{count} modalit{'y' if count == 1 else 'ies'}")

    def _rebuild_table(self):
        if not dpg.does_item_exist(TABLE_TAG):
            return
        for child in dpg.get_item_children(TABLE_TAG, 1) or []:
            dpg.delete_item(child)
        for modality in self._modalities:
            with dpg.table_row(parent=TABLE_TAG, tag=f"__streamer_row_{modality}"):
                dpg.add_text(modality)
                dpg.add_text("0", tag=f"__streamer_n_{modality}")
                dpg.add_text("0", tag=f"__streamer_r_{modality}")
                dpg.add_text("0", tag=f"__streamer_w_{modality}")
        if dpg.does_item_exist(STATUS_TAG):
            dpg.set_value(STATUS_TAG, "Incoming data" if self._modalities
                          else "Incoming data    nothing streaming")

    def _set_row(self, modality, state):
        for prefix, value in (("n", f"{state.total_samples}"),
                              ("r", f"{self._rates.get(modality, 0.0):.0f}"),
                              ("w", f"{state.commits}")):
            tag = f"__streamer_{prefix}_{modality}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)

    # ------------------------------------------------------------------
    def _say(self, message):
        if dpg.does_item_exist(MESSAGE_TAG):
            dpg.set_value(MESSAGE_TAG, message)
