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

What *is* written out is how those arguments read. ``emg_notch_freq`` is a
fine argument name and a poor label, so a small table below gives the common
ones a plain name, a unit and a section, and each control's explanation is
taken from the device's own docstring. An argument the table does not know
still appears, under "Other options", with a label made from its name.
"""

import inspect
import re
import time
import traceback
from collections import namedtuple

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style
from libemg._gui._pipeline.registry import (BOOL, FLOAT, INT, STR,
                                            _params_from_signature,
                                            _streamer_functions)

WINDOW_TAG = "__streamer_window"
STATUS_TAG = "__streamer_status"
MESSAGE_TAG = "__streamer_message"
TABLE_TAG = "__streamer_table"
CHOICE_TAG = "__streamer_choice"
START_TAG = "__streamer_start"
STOP_TAG = "__streamer_stop"
BODY_TAG = "__streamer_body"
IDLE_TAG = "__streamer_idle"

#: Offered alongside the real devices. Building a pipeline, trying an
#: environment or rehearsing a session should not require hardware on the desk.
SYNTHETIC = "Synthetic (no hardware)"

#: What each device is called in the list. A streamer missing from here is
#: still offered, under a name made from its function's.
_DEVICE_TITLES = {
    "delsys_api_streamer": "Delsys (.NET API)",
    "delsys_streamer": "Delsys Trigno / Avanti",
    "emager_streamer": "EMaGer armband",
    "leap_streamer": "Leap Motion",
    "myo_streamer": "Myo armband",
    "oymotion_streamer": "OYMotion armband",
    "sifi_bioarmband_streamer": "SiFi BioArmband",
    "sifi_biopoint_streamer": "SiFi BioPoint",
}

# The sections a device's options are sorted into, in the order they are shown.
_CONNECTION = "Connection"
_SIGNALS = "Signals"
_RATES = "Sampling rates"
_FILTERING = "Filtering"
_DEVICE = "Device"
_OTHER = "Other options"
_GROUPS = (_CONNECTION, _SIGNALS, _RATES, _FILTERING, _DEVICE, _OTHER)

#: How an argument is shown: its label, the unit after it, its section, help
#: for when the device's docstring says nothing better, the greyed hint in an
#: empty text box, and a width for text boxes that should not be the default.
_Shown = namedtuple("_Shown", "label unit group help hint width",
                    defaults=("", _OTHER, "", "", 0))

_SHOWN = {
    "name": _Shown("Device name", "", _CONNECTION,
                   "Which device to connect to, such as BioPoint_v1_3 or BioArmband. "
                   "Leave empty to connect to the first one found.",
                   hint="Leave empty for the first found"),
    "mac": _Shown("MAC address", "", _CONNECTION, hint="Optional"),
    "delsys_ip": _Shown("IP address", "", _CONNECTION),
    "cmd_port": _Shown("Command port", "", _CONNECTION),
    "emg_port": _Shown("EMG port", "", _CONNECTION),
    "aux_port": _Shown("IMU port", "", _CONNECTION),
    "timeout": _Shown("Timeout", "s", _CONNECTION),
    "license": _Shown("License", "", _CONNECTION, width=-1),
    "key": _Shown("Key", "", _CONNECTION, width=-1),
    "dll_folder": _Shown("DLL folder", "", _CONNECTION, width=-1),

    "emg": _Shown("EMG", "", _SIGNALS),
    "imu": _Shown("IMU", "", _SIGNALS),
    "ecg": _Shown("ECG", "", _SIGNALS),
    "eda": _Shown("EDA", "", _SIGNALS),
    "ppg": _Shown("PPG", "", _SIGNALS),
    "temperature": _Shown("Temperature", "", _SIGNALS),
    "sampling_rate": _Shown("Sampling rate", "Hz", _SIGNALS,
                            "Samples per second, on every channel."),
    "num_channels": _Shown("Channels", "", _SIGNALS, "How many channels the device has."),
    "channel_list": _Shown("Channels used", "", _SIGNALS, hint="e.g. 0,1,2,3"),
    "pattern": _Shown("Pattern", "", _SIGNALS,
                      "noise: random noise on every channel.\n"
                      "sine: a sine wave, a different frequency on each channel.\n"
                      "bursts: four seconds quiet, four seconds active, the shape that "
                      "makes a classifier visibly do something."),
    "amplitude": _Shown("Amplitude", "", _SIGNALS, "What the generated signal is scaled by."),

    "emg_fs": _Shown("EMG", "Hz", _RATES),
    "ecg_fs": _Shown("ECG", "Hz", _RATES),
    "eda_fs": _Shown("EDA", "Hz", _RATES),
    "imu_fs": _Shown("IMU", "Hz", _RATES),
    "ppg_sps": _Shown("PPG", "Hz", _RATES),
    "ppg_avg": _Shown("PPG averaging", "samples", _RATES),
    "temperature_fs": _Shown("Temperature", "Hz", _RATES),

    "filtering": _Shown("On-device filtering", "", _FILTERING),
    "filtered": _Shown("Filtered data", "", _FILTERING),
    "emg_notch_freq": _Shown("EMG notch", "Hz", _FILTERING),
    "emg_bandpass": _Shown("EMG band-pass", "Hz", _FILTERING, hint="low,high"),
    "eda_bandpass": _Shown("EDA band-pass", "Hz", _FILTERING, hint="low,high"),
    "eda_freq": _Shown("EDA excitation", "Hz", _FILTERING),

    "streaming": _Shown("Low-latency mode", "", _DEVICE),
    "night_mode": _Shown("LEDs off", "", _DEVICE),
    "high_gain": _Shown("High gain", "", _DEVICE),
}

#: Arguments that switch a signal on. Drawn together as one row of ticks,
#: because which signals to record is one question, not six.
_MODALITIES = ("emg", "imu", "ecg", "eda", "ppg", "temperature")

#: Words kept in capitals when a label is made from an argument's name.
_ACRONYMS = {"emg", "imu", "ecg", "eda", "ppg", "ip", "mac", "dll", "id", "usb"}

#: One control to draw: the argument, its kind, its default, and help from the
#: device's docstring.
_Field = namedtuple("_Field", "name kind default minimum choices help",
                    defaults=(None, None, ""))

_SYNTHETIC_FIELDS = (
    _Field("sampling_rate", INT, 1000, 1),
    _Field("num_channels", INT, 8, 1),
    _Field("pattern", "enum", "bursts", None, ["noise", "sine", "bursts"]),
    _Field("amplitude", FLOAT, 1.0, 0.0),
)


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

    def __init__(self, on_started=None, on_stopped=None, width=780, height=660):
        self.on_started = on_started
        self.on_stopped = on_stopped
        self.width, self.height = width, height
        self.functions = _streamer_functions()
        self.devices = [SYNTHETIC] + sorted(self.functions)
        # The list shows titles; code, tests and the rest of the panel work in
        # the function's own name.
        self._by_title = {_device_title(device): device for device in self.devices}
        self.selected = SYNTHETIC
        self.values = {}
        self._controls = []
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
                dpg.add_text("Device", color=style.MUTED)
                dpg.add_combo(list(self._by_title), default_value=_device_title(self.selected),
                              tag=CHOICE_TAG, width=260, callback=self._choose)
                dpg.add_spacer(width=6)
                style.button("Start", "primary", tag=START_TAG, width=90,
                             callback=self.start)
                style.button("Stop", "danger", tag=STOP_TAG, width=90,
                             callback=self.stop)
            style.note("", tag=MESSAGE_TAG, wrap=self.width - 40)
            dpg.add_separator()
            # As tall as the device's settings, so a short list leaves no gap
            # above the incoming data. A long one (Leap Motion has 33 switches)
            # makes the window scroll instead.
            with dpg.child_window(tag=BODY_TAG, border=False, autosize_x=True,
                                  auto_resize_y=True):
                self._build_settings()
            # The heading is the status line: it says what is arriving, so the
            # numbers sit with the table they summarise.
            style.section("Incoming data", tag=STATUS_TAG)
            with dpg.table(tag=TABLE_TAG, header_row=True, borders_innerH=True,
                           borders_outerH=True, borders_innerV=True,
                           policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(label="Modality")
                dpg.add_table_column(label="Samples")
                dpg.add_table_column(label="Rate (Hz)")
                dpg.add_table_column(label="Writes")
            style.note("Start a device to see what it delivers.", tag=IDLE_TAG)
        self._rebuild_table()
        self._sync_controls()
        self._say("Pick a device and press Start. Nothing else in the window "
                  "can see data until one is running.")
        return self

    def _build_settings(self):
        self._controls = []
        if self.selected == SYNTHETIC:
            summary = ("Generates samples at a fixed rate, so the rest of the "
                       "window can be used with no hardware attached.")
            fields = list(_SYNTHETIC_FIELDS)
        else:
            function = self.functions[self.selected]
            summary = (function.__doc__ or "").strip().split("\n")[0][:200]
            documented = _documented(function)
            fields = [_Field(spec.name,
                             {INT: INT, FLOAT: FLOAT, BOOL: BOOL}.get(spec.kind, STR),
                             spec.default, help=documented.get(spec.name, ""))
                      for spec in _params_from_signature(function)]
        for field in fields:
            self.values.setdefault(field.name, field.default)

        if summary:
            style.note(summary, wrap=self.width - 60)
        if not fields:
            style.note("This device takes no options.")
            return
        if any(_shown(field.name).help or field.help for field in fields):
            style.hover_hint()

        grouped = {}
        for field in fields:
            grouped.setdefault(_shown(field.name).group, []).append(field)
        first = True
        for group in _GROUPS:
            members = grouped.get(group)
            if not members:
                continue
            # A device none of whose options are known has only this section,
            # and calling it "Other" would suggest there was something else.
            style.section("Options" if list(grouped) == [_OTHER] else group, first=first)
            first = False
            self._build_group(group, members)

    def _build_group(self, group, fields):
        """One section: a form of labelled rows, with the on/off flags gathered up.

        The flags that pick signals become one "Record" row. Flags nothing here
        knows about -- the Leap has thirty-odd -- become a grid of ticks, which
        reads far better than thirty rows of a label and a lone box.
        """
        if group == _OTHER:
            ticks = [field for field in fields if field.kind == BOOL]
        else:
            ticks = [field for field in fields
                     if field.kind == BOOL and field.name in _MODALITIES]
        rows = [field for field in fields if field not in ticks]
        if rows or group != _OTHER:
            with style.form(label_width=160):
                if ticks and group != _OTHER:
                    with style.row("Record", help="The signals the device sends. Each one "
                                   "ticked is a modality the rest of the window can use."):
                        self._ticks(ticks, columns=6)
                for field in rows:
                    shown = _shown(field.name)
                    with style.row(shown.label, help=shown.help or field.help or None):
                        self._control(field, shown)
        if ticks and group == _OTHER:
            self._ticks(ticks, columns=3, width=(self.width - 80) // 3)

    def _control(self, field, shown):
        tag = f"__streamer_set_{field.name}"
        self._controls.append(tag)
        value = self.values.get(field.name, field.default)
        common = dict(tag=tag, user_data=field.name, callback=self._changed)
        bounds = {} if field.minimum is None else dict(min_value=field.minimum,
                                                        min_clamped=True)
        if field.kind == "enum":
            dpg.add_combo(list(field.choices), default_value=value or field.default,
                          width=140, **common)
        elif field.kind == BOOL:
            dpg.add_checkbox(default_value=bool(value), **common)
        elif field.kind == INT:
            dpg.add_input_int(default_value=int(value or 0), step=0, width=110,
                              **bounds, **common)
        elif field.kind == FLOAT:
            dpg.add_input_float(default_value=float(value or 0.0), step=0, width=110,
                                format="%.3f", **bounds, **common)
        else:
            width = shown.width or (140 if shown.unit else 260)
            dpg.add_input_text(default_value="" if value is None else str(value),
                               width=width, hint=shown.hint, **common)
        if shown.unit:
            style.unit(shown.unit)

    def _ticks(self, fields, columns, width=0):
        """On/off flags as labelled ticks, ``columns`` to a line."""
        with dpg.table(header_row=False, resizable=False, borders_innerH=False,
                       borders_outerH=False, borders_innerV=False, borders_outerV=False,
                       policy=dpg.mvTable_SizingFixedFit):
            for _ in range(min(columns, len(fields))):
                if width:
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=width)
                else:
                    dpg.add_table_column()
            for start in range(0, len(fields), columns):
                with dpg.table_row():
                    for field in fields[start:start + columns]:
                        shown = _shown(field.name)
                        self._controls.append(f"__streamer_set_{field.name}")
                        box = dpg.add_checkbox(
                            label=shown.label, tag=f"__streamer_set_{field.name}",
                            default_value=bool(self.values.get(field.name, field.default)),
                            user_data=field.name, callback=self._changed)
                        if shown.help or field.help:
                            with dpg.tooltip(box):
                                dpg.add_text(shown.help or field.help, wrap=380)

    def _choose(self, sender, app_data):
        app_data = self._by_title.get(app_data, app_data)
        if self.handle is not None:
            self._say("Stop the running device before switching to another.", "warn")
            dpg.set_value(CHOICE_TAG, _device_title(self.selected))
            return
        self.selected = app_data
        self.values = {}
        # Keep the list showing what is selected when the choice was made from
        # code rather than by clicking it.
        if dpg.does_item_exist(CHOICE_TAG):
            dpg.set_value(CHOICE_TAG, _device_title(self.selected))
        if dpg.does_item_exist(BODY_TAG):
            dpg.delete_item(BODY_TAG, children_only=True)
            dpg.push_container_stack(BODY_TAG)
            self._build_settings()
            dpg.pop_container_stack()

    def _changed(self, sender, app_data, user_data):
        self.values[user_data] = app_data

    def _sync_controls(self):
        """Offer only what can be done now: Start while stopped, Stop while running.

        The device list and its settings are locked while one runs. Switching
        is refused then anyway, and a setting changed mid-run only takes effect
        at the next Start, so leaving it editable suggests it did something.
        """
        running = self.handle is not None
        style.set_enabled(START_TAG, not running)
        style.set_enabled(STOP_TAG, running)
        for item in [CHOICE_TAG] + self._controls:
            style.set_enabled(item, not running)

    # ==================================================================
    # running
    # ==================================================================
    def start(self):
        """Bring the selected device up and publish its handler."""
        if self.handle is not None:
            self._say("Already streaming. Stop it first.", "warn")
            return
        try:
            handle, items = self._launch()
        except Exception:
            # A device that is not plugged in, a driver that is missing, a port
            # already in use: all of these surface here, and the last line of
            # the traceback is the part worth showing.
            self._say("Could not start that device:\n"
                      + traceback.format_exc().strip().splitlines()[-1], "error")
            return

        from libemg.data_handler import OnlineDataHandler
        self.handle, self.items = handle, items
        try:
            self.odh = OnlineDataHandler(items)
        except Exception:
            reason = traceback.format_exc().strip().splitlines()[-1]
            self.stop()
            # After stopping, which says "Stopped." -- the reason is what is
            # worth leaving on screen.
            self._say("The device started but its data could not be attached:\n"
                      + reason, "error")
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
        self._sync_controls()
        if self.on_started is not None:
            self.on_started(self.odh, items)
        # Deliberately not "streaming" yet. A device streamer spawns a process
        # that connects on its own, so a device that is unplugged, asleep or
        # paired to something else starts perfectly well and simply never
        # produces a sample. Waiting to see one before claiming success is what
        # stops the window from reporting a working device that is not there.
        self._say(f"{_device_title(self.selected)} started. Waiting for the first samples.",
                  "busy")

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
        self._sync_controls()
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
        name = _device_title(self.selected)
        arriving = any(state.total_samples > 0 for state in states.values())
        if arriving and not self._seen_data:
            self._seen_data = True
            self._say(f"{name} is streaming. Collect Data, Live Signal, the pipeline "
                      "editor and the environments can all use it now.", "ok")
        elif not arriving and not self._warned and now - self._started_at > 5.0:
            self._warned = True
            self._say(f"{name} started, but no samples have arrived in "
                      "five seconds. Check that the device is on, paired and not "
                      "in use by another program. It is left running in case it "
                      "is still connecting.", "warn")
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
        # An empty table is a row of headings over nothing; a sentence says
        # the same thing and says what to do about it.
        dpg.configure_item(TABLE_TAG, show=bool(self._modalities))
        if dpg.does_item_exist(IDLE_TAG):
            dpg.configure_item(IDLE_TAG, show=not self._modalities)
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
    def _say(self, message, kind="info"):
        style.set_status(MESSAGE_TAG, message, kind)


def _device_title(device):
    """What a device is called in the list: its title, or one made from its name."""
    if device == SYNTHETIC:
        return SYNTHETIC
    if device in _DEVICE_TITLES:
        return _DEVICE_TITLES[device]
    stem = device[:-len("_streamer")] if device.endswith("_streamer") else device
    return " ".join(word.upper() if word in _ACRONYMS else word.capitalize()
                    for word in stem.split("_") if word)


def _shown(name):
    """How the argument ``name`` is shown, made up from its name when not listed."""
    if name in _SHOWN:
        return _SHOWN[name]
    words = [word.upper() if word in _ACRONYMS else word for word in name.split("_") if word]
    label = " ".join(words)
    return _Shown(label[:1].upper() + label[1:] if label else name)


def _documented(function):
    """What a function's numpy-style docstring says about each argument, by name.

    The registry's own help only says which function an argument is passed to;
    the author's description is what someone deciding on a value wants.
    """
    lines = (inspect.getdoc(function) or "").splitlines()
    found = {}
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Parameters") + 2
    except StopIteration:
        return found
    name, words = None, []
    for index in range(start, len(lines)):
        line = lines[index]
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if line.strip() and following and set(following) == {"-"}:
            break       # the next section's heading, over its rule
        if not line.strip():
            continue
        if not line[0].isspace():
            if name and words:
                found[name] = " ".join(words)
            match = re.match(r"\w+", line)
            name, words = (match.group(0) if match else None), []
        elif name:
            words.append(line.strip().replace("``", "'"))
    if name and words:
        found[name] = " ".join(words)
    return found
