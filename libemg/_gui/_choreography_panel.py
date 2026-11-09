"""Dear PyGui front end for the 3D hand choreography collection mode.

Three pieces, used by :class:`~libemg._gui._data_collection_panel.DataCollectionPanel`:

:class:`ChoreographyConfigSection`
    The settings, drawn inside the Collect Data window when the choreography
    mode is selected: gestures (ticked, ordered, with per-gesture repetition and
    timing overrides), fixed or seeded-shuffle order, the run's timings, presets,
    and a live estimate of what Start would collect.
:class:`ChoreographyRunWindow`
    The window the participant watches: what to do, which phase, how long is
    left, progress, the rendered hand, and Start / Pause / Resume / Stop / Reset.
    When the run asks for intensity, an :class:`IntensityBar` beside the hand
    shows the target effort rising and falling with it.
    It is ticked once per GUI frame through :meth:`ChoreographyRunWindow.poll`,
    on the GUI thread -- the only thread that touches DearPyGui items. The hand
    is rendered on its own thread (:class:`~libemg.choreography.renderer.RenderWorker`).
:class:`PanelCollectionLink`
    How a recording run reaches the collection service: the image-folder
    collection's own continuous session log, annotation file and cutter.

All scheduling, annotation and recording logic is in :mod:`libemg.choreography`;
this module only draws it.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from libemg.choreography.assets import REST_ID
from libemg.choreography.config import (FIXED, IN_TURN, SHUFFLE, TIMING_KEYS,
                                        ChoreographyConfig, new_seed)
from libemg.choreography.controller import (FINALIZING, PREVIEW, RECORDING, STARTING,
                                            ChoreographyController, CollectionLink)
from libemg.choreography.schedule import (CANCELLED, COMPLETED, FAILED, HOLD, PAUSED_STATE,
                                          READY, RESTING_PHASES, RUNNING, build_plan)

from . import _style as style
from ._utils import get_texture_registry

CHOREOGRAPHY_MODE = "choreography"
IMAGE_FOLDER_MODE = "image_folder"
MODE_LABELS = {CHOREOGRAPHY_MODE: "3D Hand Choreography",
               IMAGE_FOLDER_MODE: "Image Folder / Optional Protocol"}

#: Longest a frame on screen may lag the schedule before the run is paused.
RENDER_STALL_SECONDS = 0.75
VIEWPORT_SIZES = {"Small (360 px)": 360, "Medium (480 px)": 480, "Large (640 px)": 640}
DEFAULT_VIEWPORT = "Medium (480 px)"

_OVERRIDE_WIDTH = 58
_NUMBER_WIDTH = 120
#: Window height beyond the viewport: labels above it, buttons and status below.
_CHROME_HEIGHT = 380


def format_levels(levels):
    """``[0.3, 0.6]`` as ``'30, 60'`` (percent)."""
    return ", ".join(f"{100 * float(level):g}" for level in levels)


def parse_levels(text):
    """``'30, 60 90%'`` as ``([0.3, 0.6, 0.9], problems)``. Percent in, fractions out."""
    levels, problems = [], []
    for part in re.split(r"[,;\s]+", str(text or "").strip()):
        part = part.strip().rstrip("%")
        if not part:
            continue
        try:
            levels.append(float(part) / 100.0)
        except ValueError:
            problems.append(f"Intensity level '{part}' is not a number.")
    return levels, problems


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"


# =============================================================================
# configuration
# =============================================================================

class ChoreographyConfigSection:
    """The choreography settings, drawn into the Collect Data window.

    Parameters
    ----------
    assets: ChoreographyAssets
    config: ChoreographyConfig, optional
        What the section opens on. Every gesture, ticked, in manifest order, by default.
    on_change: callable, optional
        Called with no arguments whenever a setting changes.
    """

    def __init__(self, assets, config=None, on_change=None, wrap=900):
        self.assets = assets
        self.on_change = on_change
        self.wrap = wrap
        config = config or ChoreographyConfig(gestures=list(assets.gesture_ids))
        self._load_state(config)
        self.tags = []

    # -- state ----------------------------------------------------------------------

    def _load_state(self, config):
        known = self.assets.gesture_ids
        chosen = [g for g in config.gestures if g in known]
        self.order = chosen + [g for g in known if g not in chosen]
        self.checked = set(chosen)
        self.rep_text = {g: str(config.repetition_overrides[g]) if g in config.repetition_overrides
                         else "" for g in known}
        self.timing_text = {g: {k: (f"{config.timing_overrides[g][k]:g}"
                                    if k in config.timing_overrides.get(g, {}) else "")
                                for k in TIMING_KEYS} for g in known}
        self.defaults = config

    def _tag(self, name):
        tag = f"__ch_{name}"
        self.tags.append(tag)
        return tag

    # -- drawing --------------------------------------------------------------------

    def build(self):
        c = self.defaults
        style.section("Gestures")
        style.note("Tick the motions to collect. Their order is the run order (or the starting "
                   "point of a shuffle). Leave an override blank to use the default.",
                   wrap=self.wrap)
        with dpg.group(horizontal=True):
            style.button("Select all", tag=self._tag("select_all"), callback=self._select_all)
            style.button("Clear all", tag=self._tag("clear_all"), callback=self._clear_all)
            style.note("", tag=self._tag("selected_count"))
        with dpg.child_window(tag=self._tag("gesture_box"), height=290, border=True,
                              autosize_x=True):
            self._build_rows()

        style.section("Order")
        with style.form():
            with style.row("Order", help="Fixed: every cycle runs the gestures in the order "
                           "above. Seeded shuffle: each cycle is shuffled, reproducibly from "
                           "the seed, which is saved with the session."):
                dpg.add_radio_button(["Fixed order", "Seeded shuffle"], horizontal=True,
                                     tag=self._tag("order"),
                                     default_value="Seeded shuffle" if c.order_mode == SHUFFLE
                                     else "Fixed order", callback=self._changed)
            with style.row("Shuffle seed", help="The same seed and gestures give the same "
                           "order every time."):
                dpg.add_input_int(default_value=int(c.shuffle_seed), tag=self._tag("seed"),
                                  width=_NUMBER_WIDTH * 1.4, step=0, callback=self._changed)
                style.button("New seed", tag=self._tag("new_seed"), callback=self._new_seed)

        style.section("Timing")
        with style.form(label_width=210):
            for key, label, help_text in (
                    ("warmup", "Warm-up", "The relaxed hand, shown before the first gesture."),
                    ("ramp_in", "Ramp-in", "Moving from rest to the gesture. Must be above 0."),
                    ("hold", "Hold", "Holding the gesture. 0 goes straight from ramp-in to "
                     "ramp-out."),
                    ("ramp_out", "Ramp-out", "Moving back to rest. Must be above 0.")):
                with style.row(label, help=help_text):
                    dpg.add_input_float(default_value=float(getattr(c, key)), tag=self._tag(key),
                                        width=_NUMBER_WIDTH, step=0, format="%g",
                                        callback=self._changed)
                    style.unit("s")
            with style.row("Rest between gestures", help="Ticked: the relaxed hand is held this "
                           "long between one gesture's ramp-out and the next ramp-in. Unticked: "
                           "no wait -- each ramp-out goes straight into the next ramp-in, so the "
                           "recording holds direct transitions between gestures. Rest is then "
                           "collected in the warm-up and the pauses between cycles, one of which "
                           "must be kept."):
                dpg.add_checkbox(default_value=bool(c.rest_between_gestures),
                                 tag=self._tag("rest_between_gestures"), callback=self._changed)
                dpg.add_input_float(default_value=float(c.rest), tag=self._tag("rest"),
                                    width=_NUMBER_WIDTH, step=0, format="%g",
                                    callback=self._changed)
                style.unit("s")
            with style.row("Repetitions per gesture", help="How many times each gesture is "
                           "collected. A gesture's own count above overrides it."):
                dpg.add_input_int(default_value=int(c.repetitions), tag=self._tag("repetitions"),
                                  width=_NUMBER_WIDTH, step=0, callback=self._changed)
            with style.row("Pause between cycles", help="A longer relaxed-hand pause between "
                           "cycles, instead of the normal rest (never both)."):
                dpg.add_checkbox(default_value=bool(c.cycle_pause_enabled),
                                 tag=self._tag("cycle_pause_enabled"), callback=self._changed)
                dpg.add_input_float(default_value=float(c.cycle_pause), tag=self._tag("cycle_pause"),
                                    width=_NUMBER_WIDTH, step=0, format="%g",
                                    callback=self._changed)
                style.unit("s")
            with style.row("Repetition regions", help="Also annotate each repetition as one "
                           "region, ramp-in to ramp-out, on its own track."):
                dpg.add_checkbox(default_value=bool(c.parent_regions),
                                 tag=self._tag("parent_regions"), callback=self._changed)

        style.section("Intensity (optional)")
        style.note("Ask for each repetition at a target effort. A bar beside the hand shows it "
                   "rising and falling with the hand, and the target (position x level) is "
                   "saved as a regression label next to the class and position.", wrap=self.wrap)
        with style.form(label_width=210):
            with style.row("Intensity targets", help="Show the effort bar and record an "
                           "intensity label. Off: the bar is hidden and no intensity is saved."):
                dpg.add_checkbox(default_value=bool(c.intensity_enabled),
                                 tag=self._tag("intensity_enabled"), callback=self._changed)
            with style.row("Levels", help="Target effort as a percent of maximum, separated by "
                           "commas (e.g. 30, 60, 90). Each gesture's repetitions are dealt these "
                           "levels as evenly as its repetition count allows."):
                dpg.add_input_text(default_value=format_levels(c.intensity_levels),
                                   tag=self._tag("intensity_levels"), hint="30, 60, 90",
                                   width=_NUMBER_WIDTH * 2, callback=self._changed)
                style.unit("% of max")
            with style.row("Level order", help="In turn: a gesture's first repetition gets the "
                           "first level, the second the next, and so on. Seeded shuffle: the "
                           "same levels in a random order per gesture, from the shuffle seed."):
                dpg.add_radio_button(["In turn", "Seeded shuffle"], horizontal=True,
                                     tag=self._tag("intensity_order"),
                                     default_value="Seeded shuffle"
                                     if c.intensity_order == SHUFFLE else "In turn",
                                     callback=self._changed)

        with dpg.group(horizontal=True):
            style.button("Save preset...", tag=self._tag("save_preset"),
                         callback=lambda: dpg.show_item("__ch_save_dialog"))
            style.button("Load preset...", tag=self._tag("load_preset"),
                         callback=lambda: dpg.show_item("__ch_load_dialog"))
            style.note("", tag=self._tag("preset_status"))
        style.note("", tag=self._tag("summary"), wrap=self.wrap, color=style.TEXT)
        style.note("", tag=self._tag("problems"), wrap=self.wrap, color=style.ERROR)
        for tag, save in (("__ch_save_dialog", True), ("__ch_load_dialog", False)):
            self.tags.append(tag)
            with dpg.file_dialog(directory_selector=False, show=False, modal=True, tag=tag,
                                 width=760, height=460, user_data=save,
                                 callback=self._preset_chosen):
                dpg.add_file_extension(".json")
        self.describe()

    def _build_rows(self):
        table = "__ch_gesture_table"
        if dpg.does_item_exist(table):
            dpg.delete_item(table)
        elif table not in self.tags:
            self.tags.append(table)
        up, down = style.arrow("up", "Up"), style.arrow("down", "Dn")
        with dpg.table(tag=table, parent="__ch_gesture_box", header_row=True,
                       policy=dpg.mvTable_SizingFixedFit, row_background=True,
                       borders_innerH=False):
            dpg.add_table_column(label="Gesture", width_stretch=True, init_width_or_weight=1.0)
            dpg.add_table_column(label="Order")
            dpg.add_table_column(label="Reps")
            for key in TIMING_KEYS:
                dpg.add_table_column(label={"ramp_in": "Ramp-in", "hold": "Hold",
                                            "ramp_out": "Ramp-out", "rest": "Rest"}[key])
            for position, gesture in enumerate(self.order):
                with dpg.table_row():
                    dpg.add_checkbox(label=self.assets.label(gesture), tag=f"__ch_use_{gesture}",
                                     default_value=gesture in self.checked, user_data=gesture,
                                     callback=self._toggle)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=up, width=28, user_data=(gesture, -1),
                                       enabled=position > 0, callback=self._move)
                        dpg.add_button(label=down, width=28, user_data=(gesture, 1),
                                       enabled=position < len(self.order) - 1, callback=self._move)
                    dpg.add_input_text(tag=f"__ch_reps_{gesture}", default_value=self.rep_text[gesture],
                                       hint="default", width=_OVERRIDE_WIDTH, user_data=(gesture, None),
                                       callback=self._override_typed)
                    for key in TIMING_KEYS:
                        dpg.add_input_text(tag=f"__ch_{key}_{gesture}",
                                           default_value=self.timing_text[gesture][key],
                                           hint="default", width=_OVERRIDE_WIDTH,
                                           user_data=(gesture, key), callback=self._override_typed)

    # -- callbacks ------------------------------------------------------------------

    def _changed(self, *_args):
        self.describe()
        if self.on_change:
            self.on_change()

    def _toggle(self, sender, value, gesture):
        (self.checked.add if value else self.checked.discard)(gesture)
        self._changed()

    def _select_all(self):
        self.checked = set(self.order)
        for gesture in self.order:
            if dpg.does_item_exist(f"__ch_use_{gesture}"):
                dpg.set_value(f"__ch_use_{gesture}", True)
        self._changed()

    def _clear_all(self):
        self.checked = set()
        for gesture in self.order:
            if dpg.does_item_exist(f"__ch_use_{gesture}"):
                dpg.set_value(f"__ch_use_{gesture}", False)
        self._changed()

    def _move(self, sender, app_data, user_data):
        gesture, step = user_data
        index = self.order.index(gesture)
        other = index + step
        if 0 <= other < len(self.order):
            self.order[index], self.order[other] = self.order[other], self.order[index]
            self._build_rows()
            self._changed()

    def _override_typed(self, sender, value, user_data):
        gesture, key = user_data
        if key is None:
            self.rep_text[gesture] = value
        else:
            self.timing_text[gesture][key] = value
        self._changed()

    def _new_seed(self):
        dpg.set_value("__ch_seed", new_seed())
        self._changed()

    def _preset_chosen(self, sender, app_data, save):
        path = (app_data or {}).get("file_path_name", "")
        if not path:
            return
        try:
            if save:
                config, problems = self.read_config()
                if not path.lower().endswith(".json"):
                    path += ".json"
                config.save(path)
                style.set_status("__ch_preset_status", f"Saved {Path(path).name}.", "ok")
            else:
                self.apply_config(ChoreographyConfig.load(path))
                style.set_status("__ch_preset_status", f"Loaded {Path(path).name}.", "ok")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            style.set_status("__ch_preset_status", f"Preset not {'saved' if save else 'loaded'}: "
                             f"{error}", "error")

    def apply_config(self, config):
        """Show ``config`` in the section (a loaded preset)."""
        unknown = [g for g in config.gestures if g not in self.assets.gesture_ids]
        if unknown:
            raise ValueError(f"the preset names motions this asset lacks: {', '.join(unknown)}")
        self._load_state(config)
        values = {"__ch_order": "Seeded shuffle" if config.order_mode == SHUFFLE else "Fixed order",
                  "__ch_seed": int(config.shuffle_seed),
                  "__ch_repetitions": int(config.repetitions),
                  "__ch_cycle_pause_enabled": bool(config.cycle_pause_enabled),
                  "__ch_rest_between_gestures": bool(config.rest_between_gestures),
                  "__ch_cycle_pause": float(config.cycle_pause),
                  "__ch_parent_regions": bool(config.parent_regions),
                  "__ch_intensity_enabled": bool(config.intensity_enabled),
                  "__ch_intensity_levels": format_levels(config.intensity_levels),
                  "__ch_intensity_order": "Seeded shuffle" if config.intensity_order == SHUFFLE
                  else "In turn",
                  **{f"__ch_{k}": float(getattr(config, k))
                     for k in ("warmup", "ramp_in", "hold", "ramp_out", "rest")}}
        for tag, value in values.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)
        self._build_rows()
        self._changed()

    # -- reading it -----------------------------------------------------------------

    def read_config(self):
        """The configuration the section shows, and any overrides that are not numbers.

        Returns
        ----------
        (ChoreographyConfig, list of str)
        """
        problems = []

        def number(tag, fallback, kind=float):
            try:
                value = dpg.get_value(tag)
                return kind(value) if value is not None else fallback
            except (TypeError, ValueError):
                return fallback

        repetition_overrides, timing_overrides = {}, {}
        for gesture in self.order:
            label = self.assets.label(gesture)
            text = str(self.rep_text.get(gesture, "")).strip()
            if text:
                try:
                    value = float(text)
                    repetition_overrides[gesture] = int(value) if value.is_integer() else value
                except ValueError:
                    problems.append(f"{label}: repetitions '{text}' is not a number.")
            for key in TIMING_KEYS:
                text = str(self.timing_text[gesture][key]).strip()
                if not text:
                    continue
                try:
                    timing_overrides.setdefault(gesture, {})[key] = float(text)
                except ValueError:
                    problems.append(f"{label}: {key.replace('_', '-')} '{text}' is not a number.")
        order = dpg.get_value("__ch_order") if dpg.does_item_exist("__ch_order") else None
        intensity = (bool(dpg.get_value("__ch_intensity_enabled"))
                     if dpg.does_item_exist("__ch_intensity_enabled")
                     else self.defaults.intensity_enabled)
        levels = list(self.defaults.intensity_levels)
        if dpg.does_item_exist("__ch_intensity_levels"):
            levels, level_problems = parse_levels(dpg.get_value("__ch_intensity_levels"))
            if intensity:
                problems += level_problems
        level_order = (dpg.get_value("__ch_intensity_order")
                       if dpg.does_item_exist("__ch_intensity_order") else None)
        config = ChoreographyConfig(
            gestures=[g for g in self.order if g in self.checked],
            order_mode=SHUFFLE if order == "Seeded shuffle" else FIXED,
            shuffle_seed=number("__ch_seed", self.defaults.shuffle_seed, int),
            warmup=number("__ch_warmup", self.defaults.warmup),
            ramp_in=number("__ch_ramp_in", self.defaults.ramp_in),
            hold=number("__ch_hold", self.defaults.hold),
            ramp_out=number("__ch_ramp_out", self.defaults.ramp_out),
            rest=number("__ch_rest", self.defaults.rest),
            repetitions=number("__ch_repetitions", self.defaults.repetitions, int),
            repetition_overrides={g: v for g, v in repetition_overrides.items() if g in self.checked},
            cycle_pause_enabled=bool(dpg.get_value("__ch_cycle_pause_enabled"))
            if dpg.does_item_exist("__ch_cycle_pause_enabled") else self.defaults.cycle_pause_enabled,
            cycle_pause=number("__ch_cycle_pause", self.defaults.cycle_pause),
            rest_between_gestures=bool(dpg.get_value("__ch_rest_between_gestures"))
            if dpg.does_item_exist("__ch_rest_between_gestures")
            else self.defaults.rest_between_gestures,
            timing_overrides={g: v for g, v in timing_overrides.items() if g in self.checked},
            parent_regions=bool(dpg.get_value("__ch_parent_regions"))
            if dpg.does_item_exist("__ch_parent_regions") else self.defaults.parent_regions,
            intensity_enabled=intensity,
            intensity_levels=levels if (levels or intensity) else list(self.defaults.intensity_levels),
            intensity_order=(SHUFFLE if level_order == "Seeded shuffle" else IN_TURN)
            if level_order is not None else self.defaults.intensity_order)
        return config, problems

    def validate(self):
        """``(config, plan or None, problems)`` for what the section currently shows."""
        config, problems = self.read_config()
        problems = problems + config.validate(self.assets.gesture_ids)
        plan = None
        if not problems:
            try:
                plan = build_plan(config, self.assets.gesture_ids)
            except ValueError as error:
                problems.append(str(error))
        return config, plan, problems

    def describe(self):
        """Refresh the estimate and the problems list. Returns True if Start could run."""
        try:
            config, plan, problems = self.validate()
        except Exception as error:      # noqa: BLE001 - never let a summary break the window
            problems, plan, config = [f"Could not read the settings: {error}"], None, None
        if config is not None and dpg.does_item_exist("__ch_rest"):
            style.set_enabled("__ch_rest", bool(config.rest_between_gestures))
        selected = len(config.gestures) if config else 0
        style.set_status("__ch_selected_count", f"{selected} of {len(self.order)} selected")
        if plan is not None:
            reps = plan.total_repetitions
            text = (f"{selected} gesture{'s' if selected != 1 else ''}, {reps} repetition"
                    f"{'s' if reps != 1 else ''} in {len(plan.cycles)} cycle"
                    f"{'s' if len(plan.cycles) != 1 else ''}. Estimated duration "
                    f"{format_duration(plan.total_duration)} ({plan.total_duration:.1f} s).")
            requested = ", ".join(f"{self.assets.label(g)} x{n}" for g, n in plan.requested.items()
                                  if n != config.repetitions)
            if requested:
                text += f" Overridden counts: {requested}."
            if not config.rest_between_gestures:
                transitions = sum(1 for p in plan.phases if p.transition_from)
                text += (f" No rest between gestures: {transitions} direct transition"
                         f"{'s' if transitions != 1 else ''} from one gesture into the next.")
                if any("rest" in own for own in config.timing_overrides.values()):
                    text += " The per-gesture rest overrides are not used."
            if config.intensity_enabled:
                levels = config.intensity_levels
                text += (f" Intensity targets {format_levels(levels)}% of maximum, "
                         f"{'shuffled' if config.intensity_order == SHUFFLE else 'in turn'}.")
                uneven = sorted({n for n in plan.requested.values() if n % len(levels)})
                if uneven:
                    text += (f" {' and '.join(map(str, uneven))} repetition"
                             f"{'s are' if uneven != [1] else ' is'} not a multiple of "
                             f"{len(levels)} levels, so some levels come up more often.")
            style.set_status("__ch_summary", text, "ok")
        else:
            style.set_status("__ch_summary", "")
        style.set_status("__ch_problems", "\n".join(problems), "error")
        return plan is not None


# =============================================================================
# the recording link
# =============================================================================

class PanelCollectionLink(CollectionLink):
    """Records a choreography run with the image-folder collection's own machinery.

    The same continuous ``_session_<modality>.csv`` log, the same
    ``_session_annotations.json`` (saved as the run goes), the same clock fit and
    the same cutter -- so every downstream reader of a session keeps working.
    """

    def __init__(self, panel, plan, assets):
        self.panel = panel
        self.plan = plan
        self.assets = assets
        self.session = None

    @property
    def handler(self):
        return self.panel.online_data_handler

    def open(self, header):
        panel = self.panel
        self.handler.prepare_smm()
        class_ids = {name: number for number, name in self.assets.class_map.items()}
        selected = list(self.plan.config.gestures)
        class_map = {str(class_ids[g]): g for g in selected}
        panel.write_choreography_details(self.plan, self.assets, header)
        panel.write_subject_details()
        mode_record = {"mode": CHOREOGRAPHY_MODE, "label": MODE_LABELS[CHOREOGRAPHY_MODE],
                       "config": self.plan.config.to_dict(),
                       "run_id": header.get("run_id"),
                       "asset": self.assets.describe(),
                       "class_map": self.assets.class_map_record(),
                       "schedule": {"cycles": self.plan.cycles,
                                    "shuffle_seed": self.plan.config.shuffle_seed,
                                    "order_mode": self.plan.config.order_mode,
                                    "total_duration": self.plan.total_duration,
                                    "requested_repetitions": self.plan.requested},
                       "phase_labels": "choreography phases: warmup, ramp_in, hold, ramp_out, "
                                       "rest, cycle_pause, paused (see _session_choreography.json)"}
        prefixes = [f"C_{class_ids[r.motion_id]}_R_{r.gesture_repetition_index}_"
                    for r in self.plan.repetitions]
        self.session = panel._open_session_log(class_map, prefixes,
                                               metadata={"collection_mode": mode_record})
        return {"collection_session_id": self.session.annotations.metadata["collection_session_id"],
                "modalities": list(self.handler.modalities)}

    def counts(self):
        try:
            return dict(self.handler.log_counts())
        except Exception:       # noqa: BLE001 - a handler that cannot answer is a lost stream
            return {}

    def logger_alive(self):
        thread = getattr(self.handler, "_log_thread", None)
        if thread is None:
            # A handler without a thread logger (a test double, say) is alive as
            # long as it is counting.
            return hasattr(self.handler, "log_counts")
        return thread.is_alive()

    def save_progress(self, stages):
        if self.session is None:
            return
        self.session.annotations.stages = stages
        self.session.annotations.save()

    def close(self, stages, metadata):
        if self.session is None:
            return {}
        annotations = self.session.annotations
        annotations.stages = stages
        annotations.metadata.update(metadata)
        self.panel._finish_session_log(self.session)
        annotations.save()
        return dict(annotations.clocks)


# =============================================================================
# the intensity bar
# =============================================================================

class IntensityBar:
    """A vertical effort gauge beside the hand: the target level and the live target.

    The fill is the instantaneous intensity target (position x level): it rises
    over the ramp-in, holds at the level, and falls over the ramp-out, in step
    with the hand. The marker is the repetition's level -- bright while it is
    being asked for, dimmed during a rest, where it previews the next one.
    """

    TAG = "__ch_intensity_bar"
    WIDTH = 118
    TOP, BOTTOM = 34, 34            # room for the title above and the value below
    X0, X1 = 46, 80                 # the gauge's column
    TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
    _TRACK = (46, 50, 58)

    def __init__(self):
        self.height = None
        self._fill = self._marker = self._marker_text = self._value_text = None
        self._last = None

    def build(self, height):
        self.height = int(height)
        dpg.add_drawlist(width=self.WIDTH, height=self.height, tag=self.TAG)
        self._draw_static()
        return self

    def resize(self, height):
        if int(height) == self.height:
            return
        self.height = int(height)
        if dpg.does_item_exist(self.TAG):
            dpg.configure_item(self.TAG, height=self.height)
            self._draw_static()

    def y(self, value):
        """Pixel row of an intensity (0..1) on the gauge."""
        top, bottom = self.TOP, self.height - self.BOTTOM
        return bottom - min(max(float(value), 0.0), 1.0) * (bottom - top)

    def _draw_static(self):
        if dpg.does_item_exist(self.TAG):
            dpg.delete_item(self.TAG, children_only=True)
        parent = self.TAG
        top, bottom = self.y(1.0), self.y(0.0)
        dpg.draw_text((self.X0 - 12, 6), "Effort", size=18, color=style.TEXT, parent=parent)
        dpg.draw_rectangle((self.X0, top), (self.X1, bottom), color=style.MUTED,
                           fill=self._TRACK, parent=parent)
        for value in self.TICKS:
            row = self.y(value)
            dpg.draw_line((self.X0 - 6, row), (self.X0, row), color=style.MUTED, parent=parent)
            dpg.draw_text((2, row - 8), f"{100 * value:.0f}%", size=14, color=style.MUTED,
                          parent=parent)
        self._fill = dpg.draw_rectangle((self.X0 + 1, bottom), (self.X1 - 1, bottom),
                                        color=style.ACCENT, fill=style.ACCENT, parent=parent)
        self._marker = dpg.draw_line((self.X0 - 5, bottom), (self.X1 + 5, bottom),
                                     color=style.WARNING, thickness=3, parent=parent, show=False)
        self._marker_text = dpg.draw_text((self.X1 + 7, bottom - 9), "", size=15,
                                          color=style.WARNING, parent=parent)
        self._value_text = dpg.draw_text((self.X0 - 4, bottom + 8), "", size=17,
                                         color=style.TEXT, parent=parent)
        self._last = None

    @staticmethod
    def appearance(view, state):
        """``(fill, level, level_is_current, fill_color)`` for a view; pure, for testing."""
        if state not in (RUNNING, PAUSED_STATE) or view.intensity_level is None:
            return 0.0, None, False, style.ACCENT
        current = view.phase not in RESTING_PHASES
        fill = float(view.intensity or 0.0)
        if state == PAUSED_STATE:
            color = style.MUTED
        elif view.phase == HOLD:
            color = style.SUCCESS
        else:
            color = style.ACCENT
        return fill, float(view.intensity_level), current, color

    def update(self, view, state):
        fill, level, current, color = self.appearance(view, state)
        key = (round(fill, 3), level, current, color)
        if key == self._last or self.height is None:
            return
        self._last = key
        bottom = self.y(0.0)
        dpg.configure_item(self._fill, pmin=(self.X0 + 1, self.y(fill)),
                           pmax=(self.X1 - 1, bottom), color=color, fill=color)
        if level is None:
            dpg.configure_item(self._marker, show=False)
            dpg.configure_item(self._marker_text, text="")
            dpg.configure_item(self._value_text, text="")
            return
        row = self.y(level)
        shade = style.WARNING if current else style.MUTED
        dpg.configure_item(self._marker, p1=(self.X0 - 5, row), p2=(self.X1 + 5, row),
                           color=shade, show=True)
        dpg.configure_item(self._marker_text, pos=(self.X1 + 7, row - 9),
                           text=f"{100 * level:.0f}%", color=shade)
        dpg.configure_item(self._value_text, text=f"{100 * fill:3.0f}%")


# =============================================================================
# the run window
# =============================================================================

class ChoreographyRunWindow:
    """The window a choreography run is shown in, in preview or recording mode.

    Parameters
    ----------
    assets: ChoreographyAssets
    config: ChoreographyConfig
        Snapshot when the window opens; the run never sees later edits.
    mode: str
        ``'preview'`` or ``'recording'``.
    link: CollectionLink, optional
        Required when recording.
    output_folder: str, optional
        Where a recording goes.
    on_closed: callable, optional
        Called once the window has gone.
    renderer_factory: callable, optional
        Builds the renderer on the render thread. Defaults to the packaged hand.
    """

    WINDOW = "__ch_run_window"

    def __init__(self, assets, config, mode=PREVIEW, link=None, output_folder=None,
                 on_closed=None, renderer_factory=None, viewport=DEFAULT_VIEWPORT,
                 metadata=None):
        self.assets = assets
        self.mode = mode
        self.on_closed = on_closed
        self.output_folder = output_folder
        self.controller = ChoreographyController(assets, config, mode=mode, link=link,
                                                 output_folder=output_folder, metadata=metadata)
        self.size = VIEWPORT_SIZES.get(viewport, 480)
        self._viewport_label = viewport if viewport in VIEWPORT_SIZES else DEFAULT_VIEWPORT
        if renderer_factory is None:
            renderer_factory = self._default_renderer
        from libemg.choreography.renderer import RenderWorker
        self.worker = RenderWorker(renderer_factory, self.size, self.size)
        self._texture = None
        self._texture_size = None
        self._texture_counter = 0
        self._shown = {}
        self._finalizer = None
        self._starter = None
        self._reset_pending = False
        self._close_requested = False
        self._closed = False
        self._renderer_ready = False
        self._render_failed = False
        self._message = None
        self._last_frame = None
        self.bar = IntensityBar() if self.controller.plan.config.intensity_enabled else None

    def _window_width(self, image_width):
        bar = IntensityBar.WIDTH + 12 if self.bar is not None else 0
        return max(image_width + bar + 40, 640)

    def _default_renderer(self):
        from libemg.choreography.assets import CLIP_TARGET
        from libemg.choreography.gltf import GltfModel
        from libemg.choreography.renderer import HandRenderer
        # Framed on the motions that can actually be shown, not the preview
        # timeline, so the hand fills the view without any gesture leaving it.
        poses = [(REST_ID, 0.0)] + [(g, CLIP_TARGET) for g in self.assets.gesture_ids]
        return HandRenderer(GltfModel.load(self.assets.glb_path), self.size, self.size,
                            framing_poses=poses)

    # -- state ------------------------------------------------------------------------

    @property
    def active(self):
        return (self.controller.active or self._finalizer is not None
                or self._starter is not None)

    @property
    def recording(self):
        return self.mode == RECORDING

    @property
    def window_tag(self):
        return self.WINDOW

    # -- building -------------------------------------------------------------------

    def spawn(self):
        if dpg.does_item_exist(self.WINDOW):
            dpg.delete_item(self.WINDOW)
        self._make_texture(self.size, self.size, np.full(self.size * self.size * 4, 0.22,
                                                         dtype=np.float32))
        plan = self.controller.plan
        width = self._window_width(self.size)
        # Beside the Collect Data window when the viewport has room, so the
        # settings stay readable; over it otherwise.
        try:
            room = int(dpg.get_viewport_client_width()) - width - 20
            position = (max(20, min(1000, room)), 40)
        except (TypeError, ValueError):
            position = (80, 40)
        with dpg.window(label="Hand Choreography" + (" - Recording" if self.recording
                                                      else " - Preview"),
                        tag=self.WINDOW, width=width, height=self.size + _CHROME_HEIGHT,
                        pos=position, no_collapse=True, on_close=self._on_window_close):
            if self.recording:
                badge = f"RECORDING  -  data and annotations go to {self.output_folder}"
                color = style.ERROR
            else:
                badge = "PREVIEW  -  nothing is recorded and no annotations are written"
                color = style.ACCENT
            dpg.add_text(badge, tag="__ch_badge", color=color)
            motion = dpg.add_text("Ready", tag="__ch_motion")
            font = style.font("bold", 30)
            if font is not None:
                dpg.bind_item_font(motion, font)
            with dpg.group(horizontal=True):
                phase = dpg.add_text("Current phase: -", tag="__ch_phase")
                bold = style.font("bold", 21)
                if bold is not None:
                    dpg.bind_item_font(phase, bold)
                dpg.add_text("", tag="__ch_countdown", color=style.ACCENT)
                if bold is not None:
                    dpg.bind_item_font("__ch_countdown", bold)
            dpg.add_text("", tag="__ch_next", color=style.MUTED)
            if self.bar is not None:
                dpg.add_text("", tag="__ch_intensity_text", color=style.WARNING)
            dpg.add_progress_bar(tag="__ch_phase_progress", default_value=0.0, width=-1)
            dpg.add_text(f"{plan.total_repetitions} repetitions in {len(plan.cycles)} cycles, "
                         f"about {format_duration(plan.total_duration)}.", tag="__ch_progress")
            with dpg.group(horizontal=True):
                dpg.add_image(self._texture, tag="__ch_image", width=self.size, height=self.size)
                if self.bar is not None:
                    self.bar.build(self.size)
            with dpg.group(horizontal=True):
                style.button("Start", "primary", tag="__ch_start", width=90, callback=self._start)
                style.button("Pause", tag="__ch_pause", width=90, callback=self._pause)
                style.button("Resume", tag="__ch_resume", width=90, callback=self._resume)
                style.button("Stop", "danger", tag="__ch_stop", width=90, callback=self._stop)
                style.button("Reset", tag="__ch_reset", width=90, callback=self._reset)
                dpg.add_combo(list(VIEWPORT_SIZES), default_value=self._viewport_label,
                              tag="__ch_size", width=150, callback=self._resize)
            style.note("Preparing the 3D hand...", tag="__ch_status", wrap=width - 30)
        self._refresh_buttons()
        return self

    def _make_texture(self, width, height, data):
        self._texture_counter += 1
        tag = f"__ch_texture_{self._texture_counter}"
        dpg.add_raw_texture(width=width, height=height, default_value=data, tag=tag,
                            format=dpg.mvFormat_Float_rgba, parent=get_texture_registry())
        old, self._texture = self._texture, tag
        self._texture_size = (width, height)
        return old

    # -- the per-frame update ----------------------------------------------------------

    def poll(self):
        """One GUI frame: advance the run, hand the renderer its pose, show what it drew."""
        if self._closed:
            return
        try:
            self._poll()
        except Exception as error:      # noqa: BLE001 - a GUI fault must not orphan a recording
            traceback.print_exc()
            if self.controller.state in (RUNNING, PAUSED_STATE):
                self.controller.fail(f"gui_failure: {error}")
            self._say(f"The run was stopped by an error in the window: {error}", "error")

    def _poll(self):
        controller = self.controller
        if self.worker.error is not None and not self._render_failed:
            self._render_failed = True
            if controller.state in (RUNNING, PAUSED_STATE):
                controller.fail(f"render_failure: {self.worker.error}")
            self._say(f"The 3D hand could not be drawn ({self.worker.error}). Any recording is "
                      f"saved and marked failed. Use the Image Folder mode as a backup.", "error")
        if not self._renderer_ready and self.worker.ready.is_set() and self.worker.error is None:
            self._renderer_ready = True
            controller.render_stall_threshold = RENDER_STALL_SECONDS
            describe = getattr(self.worker.renderer, "motion_record", None)
            if callable(describe):
                # How the hand was timed goes in the run record with everything else.
                controller.metadata["display"] = describe()
            self._say("Ready. Press Start to begin" + (" recording." if self.recording
                                                      else " the preview."), "info")
        if self._starter is not None and not self._starter.is_alive():
            self._starter = None
        if controller.needs_finalize and self._finalizer is None:
            self._finalizer = threading.Thread(target=self._finalize, daemon=True,
                                               name="choreography-finalize")
            self._say("Saving the recording..." if self.recording else "Finishing...", "busy")
            self._finalizer.start()
        if self._finalizer is not None and not self._finalizer.is_alive():
            self._finalizer = None
            self._report_result()
            if self._reset_pending:
                self._reset_pending = False
                controller.reset()
                self._say("Reset. The hand is at rest; press Start for a new run.", "info")
        view = controller.tick()
        if self._renderer_ready and not self._render_failed:
            clip, clip_time = controller.pose(view)
            self.worker.request(clip, clip_time, time.monotonic())
            frame = self.worker.take()
            if frame is not None:
                self._show(frame)
                controller.frame_presented(frame.request_monotonic)
        if view.state == PAUSED_STATE and self._message is None:
            self._describe_pause(view)
        self._update_labels(view)
        self._refresh_buttons()
        if self._close_requested and not self.active:
            self.destroy()

    def _show(self, frame):
        if (frame.width, frame.height) != self._texture_size:
            # A raw texture's size is fixed, so a new size is a new texture; the
            # image is pointed at it before the old one goes.
            old = self._make_texture(frame.width, frame.height, frame.buffer)
            dpg.configure_item("__ch_image", texture_tag=self._texture, width=frame.width,
                               height=frame.height)
            if old and dpg.does_item_exist(old):
                dpg.delete_item(old)
            dpg.configure_item(self.WINDOW, width=self._window_width(frame.width),
                               height=frame.height + _CHROME_HEIGHT)
            if self.bar is not None:
                self.bar.resize(frame.height)
        else:
            dpg.set_value(self._texture, frame.buffer)
        self._last_frame = frame        # keep the buffer DPG is reading alive

    def _set(self, tag, value):
        if self._shown.get(tag) != value and dpg.does_item_exist(tag):
            dpg.set_value(tag, value)
            self._shown[tag] = value

    def _update_labels(self, view):
        label = self.assets.label
        state = self.controller.state
        if state in (READY, STARTING):
            motion, phase, countdown, upcoming = "Ready", "-", "", ""
            if state == STARTING:
                phase = "Starting..."
        elif state in (COMPLETED, CANCELLED, FAILED, FINALIZING):
            motion = {COMPLETED: "Finished", CANCELLED: "Stopped", FAILED: "Failed",
                      FINALIZING: "Saving..."}[state]
            phase, countdown, upcoming = "-", "", ""
        else:
            resting = view.phase in RESTING_PHASES
            motion = "Rest" if resting else label(view.motion_class)
            phase = view.phase_name
            countdown = "" if view.display_phase == "paused" else f"{view.remaining:4.1f} s"
            upcoming = f"Up next: {label(view.next_motion)}" if resting and view.next_motion else ""
            if upcoming and view.intensity_level is not None:
                upcoming += f" at {100 * view.intensity_level:.0f}% effort"
            if view.then_motion:
                # No rest coming: warn of the next gesture before the hand gets there.
                upcoming = f"Then straight into: {label(view.then_motion)}"
                if view.then_intensity_level is not None:
                    upcoming += f" at {100 * view.then_intensity_level:.0f}% effort"
        effort = ""
        if (state in (RUNNING, PAUSED_STATE) and view.intensity_level is not None
                and view.phase not in RESTING_PHASES):
            effort = f"Target effort: {100 * view.intensity_level:.0f}% of maximum"
        self._set("__ch_intensity_text", effort)
        if self.bar is not None:
            self.bar.update(view, state)
        self._set("__ch_motion", motion)
        self._set("__ch_phase", f"Current phase: {phase}")
        self._set("__ch_countdown", countdown)
        self._set("__ch_next", upcoming)
        progress = view.progress if state in (RUNNING, PAUSED_STATE) else (
            1.0 if state == COMPLETED else 0.0)
        self._set("__ch_phase_progress", round(progress, 3))
        if state in (COMPLETED, CANCELLED, FAILED):
            self._set("__ch_progress", f"{view.repetitions_done} of {view.repetitions_total} "
                                       f"repetitions completed.")
        elif state in (RUNNING, PAUSED_STATE) and view.cycle_index is not None:
            parts = [f"Cycle {view.cycle_index + 1} of {view.cycle_count}"]
            if view.gesture_repetition_index is not None and view.gesture_repetition_count:
                parts.append(f"{label(view.motion_class)} repetition "
                             f"{view.gesture_repetition_index + 1} of {view.gesture_repetition_count}")
            parts.append(f"{view.repetitions_done} of {view.repetitions_total} repetitions done")
            parts.append(f"{format_duration(view.total - view.elapsed)} left")
            self._set("__ch_progress", "   |   ".join(parts))

    def _refresh_buttons(self):
        state = self.controller.state
        busy = self._finalizer is not None or self._starter is not None
        ready = self._renderer_ready and not self._render_failed
        enabled = {
            "__ch_start": state == READY and ready and not busy,
            "__ch_pause": state == RUNNING,
            "__ch_resume": state == PAUSED_STATE,
            "__ch_stop": state in (RUNNING, PAUSED_STATE),
            "__ch_reset": (state in (RUNNING, PAUSED_STATE) or (state in (COMPLETED, CANCELLED, FAILED)
                                                               and not busy)) and ready,
            "__ch_size": True,
        }
        for tag, on in enabled.items():
            if self._shown.get(("enabled", tag)) != on:
                style.set_enabled(tag, on)
                self._shown[("enabled", tag)] = on
        closable = not self.active
        if self._shown.get("closable") != closable and dpg.does_item_exist(self.WINDOW):
            dpg.configure_item(self.WINDOW, no_close=not closable)
            self._shown["closable"] = closable

    def _say(self, text, kind="info"):
        style.set_status("__ch_status", text, kind)

    def _describe_pause(self, view):
        reason = view.pause_reason
        if reason == "collection_disconnected":
            alive = self.controller.stream_alive()
            self._say("Paused: the device stopped delivering data. " + (
                "Data is arriving again; press Resume." if alive
                else "Waiting for the device; Resume is refused until data arrives."), "warn")
        elif reason == "ui_stall":
            self._say("Paused: the window stopped updating, so the participant could not see "
                      "the hand. That stretch is marked as interrupted. Press Resume to continue.",
                      "warn")
        elif reason == "render_stall":
            self._say("Paused: the 3D hand fell behind the schedule. The unseen stretch is "
                      "marked and left out of training. Press Resume to continue.", "warn")
        else:
            self._say("Paused. The hand is frozen and the schedule waits; press Resume.", "info")

    # -- buttons ---------------------------------------------------------------------

    def _start(self):
        if self.controller.state != READY or self._starter is not None:
            return
        self._message = None
        if self.recording:
            self._say("Starting the recording...", "busy")
            self._starter = threading.Thread(target=self._start_worker, daemon=True,
                                             name="choreography-start")
            self._starter.start()
        else:
            self._start_worker()

    def _start_worker(self):
        try:
            self.controller.start()
        except Exception as error:      # noqa: BLE001 - reported in the window
            traceback.print_exc()
            self._say(f"Could not start: {error}", "error")
            return
        self._say("Recording." if self.recording else "Previewing: nothing is being recorded.",
                  "error" if self.recording else "info")

    def _pause(self):
        self._message = None
        if self.controller.pause(reason="user"):
            self._describe_pause(self.controller.view())

    def _resume(self):
        if self.controller.resume():
            self._message = None
            self._say("Recording." if self.recording else "Previewing.",
                      "error" if self.recording else "info")
        else:
            self._message = "refused"
            self._say(self.controller.last_error or "Cannot resume yet.", "warn")

    def _stop(self):
        self._message = None
        self.controller.stop(reason="user_stop")

    def _reset(self):
        state = self.controller.state
        if state in (RUNNING, PAUSED_STATE):
            self.controller.stop(reason="reset")
            self._reset_pending = True
        elif state in (COMPLETED, CANCELLED, FAILED) and self._finalizer is None:
            try:
                self.controller.reset()
                self._say("Reset. The hand is at rest; press Start for a new run.", "info")
            except RuntimeError as error:
                self._say(str(error), "warn")

    def _resize(self, sender, value):
        self._viewport_label = value
        size = VIEWPORT_SIZES.get(value, 480)
        self.size = size
        self.worker.resize(size, size)

    def _finalize(self):
        try:
            self.controller.finalize()
        except Exception as error:      # noqa: BLE001 - reported in the window
            traceback.print_exc()
            self._message = "finalize-error"
            self._say(f"Saving failed: {error}", "error")

    def _report_result(self):
        result = self.controller.result
        if result is None:
            return
        self._message = "result"
        reps = [iv for iv in result.intervals if iv["track"] == "repetition"]
        done = sum(1 for iv in reps if iv["completion_status"] == COMPLETED)
        words = {COMPLETED: "Completed", CANCELLED: "Stopped", FAILED: "Failed"}[result.status]
        text = f"{words}: {done} of {self.controller.plan.total_repetitions} repetitions completed."
        if result.reason and result.status != COMPLETED:
            text += f" Reason: {result.reason}."
        if result.record_path is not None:
            text += f" Saved to {Path(result.record_path).parent}."
        if result.error:
            text += f" {result.error}"
        self._say(text, {COMPLETED: "ok", CANCELLED: "warn", FAILED: "error"}[result.status])

    # -- closing ---------------------------------------------------------------------

    def _on_window_close(self):
        self._close_requested = True
        if self.controller.state in (RUNNING, PAUSED_STATE):
            self.controller.stop(reason="window_closed")
        if not self.active:
            self.destroy()
        else:
            dpg.show_item(self.WINDOW)      # stays up until the recording is saved

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        self.worker.close()
        for tag in (self.WINDOW, self._texture):
            if tag and dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        if self.on_closed:
            self.on_closed()

    def shutdown(self):
        """The application is closing: end and save any run, then release everything."""
        try:
            if self._starter is not None:
                self._starter.join(timeout=10)
            if self._finalizer is not None:
                self._finalizer.join(timeout=60)
            self.controller.shutdown()
        finally:
            self._closed = True
            self.worker.close()


__all__ = ["ChoreographyConfigSection", "ChoreographyRunWindow", "PanelCollectionLink",
           "IntensityBar", "CHOREOGRAPHY_MODE", "IMAGE_FOLDER_MODE", "MODE_LABELS",
           "format_duration", "format_levels", "parse_levels"]
