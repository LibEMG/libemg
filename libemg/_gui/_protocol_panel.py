"""Design a collection protocol, and build its prompts, in the window.

Everything this panel edits used to be a script. The trajectory a regression
collection follows was a block of numpy someone wrote once and pasted between
projects; the animator it was drawn with was a constructor call beside it; the
order the prompts ran in was whatever order the media folder sorted into; and
the acquisition settings were in a lab notebook, if anywhere. None of it was
part of the data.

Here it is one document -- :class:`libemg.protocol.CollectionProtocol` -- which
this panel edits, renders to media, and hands to screen guided training. The
protocol is saved beside the data, so a recording carries the description of the
experiment that produced it.

The panel deliberately does not invent its own defaults. The trajectory
templates, the ordering strategies, the design factors and the acquisition
fields all come from :mod:`libemg.protocol`, where the reasoning for each of
them is written down, so adding one there makes it appear here.

The window is a form. Where the protocol and its media live, and the button
that builds them, sit at the top and stay there; the protocol itself scrolls
beneath them, a section for each part of it, every setting with its name on
the left and its unit after it.
"""

import threading
import traceback
from pathlib import Path

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style
from libemg.protocol import (ACQUISITION_FIELDS, ANIMATORS, CLASSIFICATION,
                             ORDERINGS, REGRESSION, SEGMENT_KINDS,
                             TRAJECTORY_TEMPLATES, CollectionProtocol, Segment)

WINDOW_TAG = "__protocol_window"
STATUS_TAG = "__protocol_status"
SEGMENTS_TAG = "__protocol_segments"
SUMMARY_TAG = "__protocol_summary"
BUILD_TAG = "__protocol_build"
CLEAR_TAG = "__protocol_clear"

#: Width of the label column in the protocol's sections. One width for all of
#: them, so the controls line up down the whole window, not just within one.
_LABEL_WIDTH = 210
#: The Load and Save buttons; Build media spans both, so the two file rows end
#: at the same edge.
_FILE_BUTTON = 80

#: Units an acquisition prompt ends in. Drawn after the box rather than left in
#: the label, as everywhere else in the window.
_ACQUISITION_UNITS = ("mm", "Hz", "bits")

_ORDER_HELP = ("fixed: the same order every repetition.\n\nrandom: shuffled within "
               "each repetition.\n\ncounterbalanced: rotated one place each "
               "repetition, so across repetitions every class takes every position.")


class ProtocolPanel:
    """Edit a collection protocol, render its media, and hand it to a collection.

    Parameters
    ----------
    protocol: CollectionProtocol or None (optional)
        The protocol to open on. A fresh one is made when None.
    on_ready: callable or None (optional)
        Called as ``on_ready(protocol, media_folder)`` whenever the media are
        built or a protocol is loaded, so the GUI can point screen guided
        training at them without the user re-typing the folder.
    media_folder: str (optional), default='media/'
        Where generated prompts are written -- the same folder a collection
        reads them back from.
    width, height: int (optional)
        Window size.

    Examples
    ---------
    >>> panel = ProtocolPanel()
    >>> panel.spawn_window()
    """

    #: The window this panel owns, so the GUI can tell whether it is open.
    window_tag = WINDOW_TAG

    def __init__(self, protocol=None, on_ready=None, media_folder='media/',
                 width=820, height=760):
        self.protocol = protocol or CollectionProtocol()
        self.on_ready = on_ready
        self.media_folder = media_folder
        self.protocol_path = ''
        self.width, self.height = width, height
        # Rendering an animation is minutes of matplotlib, so it runs off the
        # render thread and posts back here. Only poll() may touch the window.
        self._worker = None
        self._messages = []
        self._lock = threading.Lock()

    # ==================================================================
    # window
    # ==================================================================
    def spawn_window(self):
        """Build the protocol window."""
        self.cleanup()
        with dpg.window(label="Collection Protocol", tag=WINDOW_TAG,
                        width=self.width, height=self.height,
                        on_close=lambda: self.cleanup()):
            self._build_files()
            dpg.add_separator()
            # The files and the build stay put while the protocol scrolls under
            # them, so the button that does the work is never out of sight.
            with dpg.child_window(border=False, autosize_x=True, height=-1):
                style.hover_hint()
                dpg.add_spacer(height=2)
                self._build_collected()
                self._build_trajectory()
                self._build_drawing()
                self._build_timing()
                self._build_factors()
                self._build_acquisition()
        self._refresh_summary()
        self._sync_buttons()
        self._say("Describe the collection, build its prompts, then open "
                  "Data > Collect Data. Saving writes the protocol beside the data.")
        return self

    @property
    def _wrap(self):
        """Where explanatory text wraps inside the scrolling part of the window."""
        return max(300, self.width - 70)

    def _build_files(self):
        fill = -(2 * _FILE_BUTTON + 20)
        with style.form(label_width=120):
            with style.row("Protocol file",
                           help="A protocol is a YAML file. Load reads one into the "
                                "window; Save writes the window to it. Given a folder, "
                                "the protocol file inside it is used. Left empty, Save "
                                "writes <name>.yaml here."):
                dpg.add_input_text(tag="__protocol_path", default_value=self.protocol_path,
                                   width=fill, hint="A .yaml file, or the folder holding one")
                style.button("Load", width=_FILE_BUTTON, callback=self.load)
                style.button("Save", width=_FILE_BUTTON, callback=self.save)
            with style.row("Media folder",
                           help="Where Build media writes the prompt video and its "
                                "per-frame trajectory, and where Data > Collect Data "
                                "reads them back from."):
                dpg.add_input_text(tag="__protocol_media", default_value=self.media_folder,
                                   width=fill)
                style.button("Build media", "primary", tag=BUILD_TAG,
                             width=2 * _FILE_BUTTON + 10, callback=self.build)
        style.note("", tag=STATUS_TAG, wrap=self.width - 40)

    def _build_collected(self):
        with dpg.collapsing_header(label="What is being collected", default_open=True):
            with style.form(label_width=_LABEL_WIDTH):
                with style.row("Name", help="Names the prompt video, and the protocol "
                                            "file when no path is given."):
                    self._text("name", self.protocol.name, width=260)
                with style.row("Kind", help="Classification prompts one gesture at a "
                                            "time and labels whole repetitions. "
                                            "Regression prompts a moving target and "
                                            "labels every frame of it."):
                    dpg.add_combo(sorted((CLASSIFICATION, REGRESSION)),
                                  default_value=self.protocol.kind,
                                  tag="__protocol_kind", width=160,
                                  callback=self._changed)
                with style.row("DOF labels", help="One name per degree of freedom, "
                                                  "separated by commas. How many there "
                                                  "are sets how many numbers a target "
                                                  "takes; the bar animators draw one "
                                                  "bar per name."):
                    self._text("dof_labels", ", ".join(self.protocol.dof_labels),
                               hint="e.g. Flex/Ext, Pro/Sup")
                with style.row("Notes", help="Anything else worth keeping with the data."):
                    self._text("notes", self.protocol.notes)
            dpg.add_spacer(height=4)

    def _build_trajectory(self):
        with dpg.collapsing_header(label="Prompt trajectory", default_open=True):
            style.note("What the subject is asked to do, as a sequence of segments. "
                       "An animated prompt paces the contraction instead of leaving "
                       "its speed and depth to the subject, and the trajectory is "
                       "written out beside the video as per-frame ground truth.",
                       wrap=self._wrap)
            with style.form(label_width=_LABEL_WIDTH):
                with style.row("Start from", help=_template_help()):
                    dpg.add_combo(list(TRAJECTORY_TEMPLATES), tag="__protocol_template",
                                  width=280, default_value=next(iter(TRAJECTORY_TEMPLATES)))
                    style.button("Apply template", callback=self._apply_template)
            with dpg.child_window(tag=SEGMENTS_TAG, autosize_x=True,
                                  height=self._segments_height()):
                self._build_segments()
            with dpg.group(horizontal=True):
                style.button("Add segment", callback=self._add_segment)
                style.button("Clear all", "danger", tag=CLEAR_TAG,
                             callback=self._clear_segments)
            style.note("", tag=SUMMARY_TAG, wrap=self._wrap)
            style.note("A target is one number per DOF, from -1 to 1 (the edge of the "
                       "display). A sinusoid swings either side of where it starts, as "
                       "far as its target; random targets are scattered within it.",
                       wrap=self._wrap)
            dpg.add_spacer(height=4)

    def _build_drawing(self):
        media = self.protocol.media
        times = "×" if style.symbols() else "x"
        with dpg.collapsing_header(label="How it is drawn", default_open=True):
            with style.form(label_width=_LABEL_WIDTH):
                with style.row("Animator", help="What draws the prompt. bar and "
                                                "single_bar draw one bar per DOF label; "
                                                "the others move a marker across the "
                                                "plane of the first two DOFs."):
                    dpg.add_combo(sorted(ANIMATORS), default_value=media.animator,
                                  tag="__protocol_animator", width=160,
                                  callback=self._changed)
                with style.row("Frame rate"):
                    self._int("fps", media.fps, unit="fps")
                with style.row("Time per unit distance",
                               help="How long the marker takes to travel a distance "
                                    "of 1. Longer is slower, and easier to follow."):
                    self._float("tpd", media.tpd, unit="s")
                with style.row("Show"):
                    self._bool("Direction", "show_direction", media.show_direction,
                               help="A faded marker showing where the prompt is "
                                    "heading next.")
                    self._bool("Countdown", "show_countdown", media.show_countdown,
                               help="A countdown below the target while it holds "
                                    "still.")
                    self._bool("Boundary", "show_boundary", media.show_boundary,
                               help="A circle of radius 1: the furthest a target goes.")
                with style.row("Figure size", help="Width by height. With the "
                                                   "resolution, this sets the video's "
                                                   "size in pixels."):
                    self._int("figure_width", int(media.figsize[0]), width=70)
                    style.unit(times)
                    self._int("figure_height", int(media.figsize[1]), width=70, unit="in")
                with style.row("Resolution"):
                    self._int("dpi", media.dpi, unit="dpi")
                with style.row("Format", help=".mp4 needs ffmpeg; .gif needs nothing "
                                              "extra but makes a larger file."):
                    dpg.add_combo([".mp4", ".gif"], default_value=media.video_format,
                                  tag="__protocol_video_format", width=100,
                                  callback=self._changed)
            dpg.add_spacer(height=4)

    def _build_timing(self):
        timing = self.protocol.timing
        with dpg.collapsing_header(label="Timing and order", default_open=True):
            style.note("Shuffling prompts within a repetition stops class order from "
                       "tracking fatigue and electrode drift; the seed keeps the "
                       "session reproducible. Jittered rest stops the subject "
                       "anticipating the cue and contracting before it.",
                       wrap=self._wrap)
            with style.form(label_width=_LABEL_WIDTH):
                with style.row("Repetitions", help="How many times each prompt is "
                                                   "collected."):
                    self._int("num_reps", timing.num_reps)
                with style.row("Repetition time", help="How long each prompt runs."):
                    self._float("rep_time", timing.rep_time, unit="s")
                with style.row("Rest", help="The pause between prompts."):
                    self._float("rest_time", timing.rest_time, unit="s")
                with style.row("Rest jitter", help="Up to this much is added to each "
                                                   "rest at random, so the subject "
                                                   "cannot time the next cue."):
                    self._float("rest_jitter", timing.rest_jitter, unit="s")
                with style.row("Reaction time", help="How long after the cue a "
                                                     "contraction can be assumed to "
                                                     "have started. Written down for "
                                                     "analysis, not trimmed from the "
                                                     "recording."):
                    self._float("reaction_time", timing.reaction_time, unit="s")
                with style.row("Order", help=_ORDER_HELP):
                    dpg.add_combo(list(ORDERINGS), default_value=self.protocol.order,
                                  tag="__protocol_order", width=160,
                                  callback=self._changed)
                with style.row("Seed", help="Fixes the shuffle and the random "
                                            "segments, so the session can be "
                                            "repeated exactly."):
                    self._int("seed", self.protocol.seed, width=140)
            dpg.add_spacer(height=4)

    def _build_factors(self):
        with dpg.collapsing_header(label="Design factors"):
            style.note("Levels to vary between repetitions, one per line as "
                       "'name: level, level'. Accuracy falls when the limb is in a "
                       "position, or the contraction at an intensity, that was not "
                       "in the training set -- collecting across levels is the fix, "
                       "and it only helps if which level each rep used is recorded.",
                       wrap=self._wrap)
            dpg.add_input_text(tag="__protocol_factors", multiline=True,
                               height=90, width=-1,
                               default_value=self._factors_text(),
                               callback=self._changed)
            dpg.add_spacer(height=4)

    def _build_acquisition(self):
        with dpg.collapsing_header(label="Acquisition (ISEK reporting)"):
            style.note("None of this can be recovered from the recording, and all of "
                       "it decides whether the data can be compared with anyone "
                       "else's.", wrap=self._wrap)
            with style.form(label_width=_LABEL_WIDTH):
                for name, prompt in ACQUISITION_FIELDS:
                    label, unit, hint = _split_prompt(prompt)
                    with style.row(label):
                        dpg.add_input_text(
                            tag=f"__protocol_acq_{name}",
                            default_value=str(self.protocol.acquisition.get(name, "")),
                            width=140 if unit else -1, hint=hint,
                            callback=self._changed)
                        if unit:
                            style.unit(unit)
            dpg.add_spacer(height=4)

    # -- small control helpers -------------------------------------------
    # Each adds one control, and its unit, to the row it is called in.

    def _text(self, name, value, width=-1, hint=""):
        dpg.add_input_text(tag=f"__protocol_{name}", default_value=str(value),
                           width=width, hint=hint, callback=self._changed)

    def _int(self, name, value, width=90, unit=None):
        dpg.add_input_int(tag=f"__protocol_{name}", default_value=int(value),
                          width=width, step=0, callback=self._changed)
        if unit:
            style.unit(unit)

    def _float(self, name, value, width=90, unit=None):
        dpg.add_input_float(tag=f"__protocol_{name}", default_value=float(value),
                            width=width, step=0, format="%.2f", callback=self._changed)
        if unit:
            style.unit(unit)

    def _bool(self, label, name, value, help=None):
        box = dpg.add_checkbox(label=label, tag=f"__protocol_{name}",
                               default_value=bool(value), callback=self._changed)
        if help:
            with dpg.tooltip(box):
                dpg.add_text(help, wrap=340)

    def _value(self, name, default=None):
        tag = f"__protocol_{name}"
        return dpg.get_value(tag) if dpg.does_item_exist(tag) else default

    def _say(self, message, kind="info"):
        style.set_status(STATUS_TAG, message, kind)

    def _sync_buttons(self):
        """Grey out what cannot be pressed right now.

        Build media, while a build is running or there is nothing to build;
        Clear, while there is nothing to clear.
        """
        building = self._worker is not None and self._worker.is_alive()
        style.set_enabled(BUILD_TAG, bool(self.protocol.trajectory) and not building)
        style.set_enabled(CLEAR_TAG, bool(self.protocol.trajectory))

    # ==================================================================
    # trajectory editing
    # ==================================================================
    def _build_segments(self):
        if not self.protocol.trajectory:
            style.note("No segments yet. Start from a template, or add one.")
            return
        remove = "×" if style.symbols() else "x"
        # A table rather than a row of controls per segment, so a sinusoid's
        # period or a random segment's count does not push the rest of its row
        # out of line with the others. The heading stays while the list scrolls.
        with dpg.table(header_row=True, scrollY=True, freeze_rows=1, height=-1,
                       row_background=True, policy=dpg.mvTable_SizingFixedFit,
                       borders_innerH=False, borders_outerH=False,
                       borders_innerV=False, borders_outerV=False):
            for label, width in (("", 26), ("Kind", 124), ("Duration", 112),
                                 ("Target", 160), ("", 180), ("", 34)):
                dpg.add_table_column(label=label, width_fixed=True,
                                     init_width_or_weight=width)
            for index, segment in enumerate(self.protocol.trajectory):
                with dpg.table_row():
                    style.note(str(index + 1))
                    dpg.add_combo(list(SEGMENT_KINDS), default_value=segment.kind,
                                  width=116, tag=f"__protocol_seg_kind_{index}",
                                  user_data=index, callback=self._segment_changed)
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(default_value=float(segment.seconds),
                                            width=76, step=0, format="%.2f",
                                            tag=f"__protocol_seg_seconds_{index}",
                                            user_data=index,
                                            callback=self._segment_changed)
                        style.unit("s")
                    if segment.kind == "rest":
                        # A rest returns to zero whatever its target says.
                        dpg.add_text("")
                    else:
                        dpg.add_input_text(
                            default_value=", ".join(f"{value:g}"
                                                    for value in segment.target),
                            width=152, hint="one per DOF",
                            tag=f"__protocol_seg_target_{index}",
                            user_data=index, callback=self._segment_changed)
                    with dpg.group(horizontal=True):
                        if segment.kind == "sinusoid":
                            dpg.add_input_float(default_value=float(segment.period),
                                                width=76, step=0, format="%.2f",
                                                tag=f"__protocol_seg_period_{index}",
                                                user_data=index,
                                                callback=self._segment_changed)
                            style.unit("s per cycle")
                        elif segment.kind == "random":
                            dpg.add_input_int(default_value=int(segment.count),
                                              width=76, step=0,
                                              tag=f"__protocol_seg_count_{index}",
                                              user_data=index,
                                              callback=self._segment_changed)
                            style.unit("targets")
                    button = style.button(remove, "subtle", width=28, user_data=index,
                                          callback=self._remove_segment)
                    with dpg.tooltip(button):
                        dpg.add_text("Remove this segment")

    def _segments_height(self):
        """The list's height: its rows, up to a screenful, so a short one is not a large empty box."""
        if not self.protocol.trajectory:
            return 60
        return min(300, 50 + 35 * len(self.protocol.trajectory))

    def _redraw_segments(self):
        if not dpg.does_item_exist(SEGMENTS_TAG):
            return
        dpg.configure_item(SEGMENTS_TAG, height=self._segments_height())
        dpg.delete_item(SEGMENTS_TAG, children_only=True)
        dpg.push_container_stack(SEGMENTS_TAG)
        self._build_segments()
        dpg.pop_container_stack()
        self._refresh_summary()
        self._sync_buttons()

    def _segment_changed(self, sender, app_data, user_data):
        segment = self.protocol.trajectory[user_data]
        if sender == f"__protocol_seg_kind_{user_data}":
            segment.kind = app_data
            # A kind change adds or removes its own extra controls.
            self._redraw_segments()
            return
        if sender == f"__protocol_seg_seconds_{user_data}":
            segment.seconds = max(0.0, float(app_data))
        elif sender == f"__protocol_seg_target_{user_data}":
            segment.target = _numbers(app_data)
        elif sender == f"__protocol_seg_period_{user_data}":
            segment.period = max(0.01, float(app_data))
        elif sender == f"__protocol_seg_count_{user_data}":
            segment.count = max(1, int(app_data))
        self._refresh_summary()

    def _add_segment(self):
        self.protocol.trajectory.append(
            Segment(kind="hold", seconds=2.0,
                    target=[0.0] * self.protocol.n_dof))
        self._redraw_segments()

    def _remove_segment(self, sender, app_data, user_data):
        del self.protocol.trajectory[user_data]
        self._redraw_segments()

    def _clear_segments(self):
        self.protocol.trajectory = []
        self._redraw_segments()

    def _apply_template(self):
        self._read_window()
        name = self._value("template") or next(iter(TRAJECTORY_TEMPLATES))
        builder = TRAJECTORY_TEMPLATES[name]
        try:
            self.protocol.trajectory = builder(n_dof=self.protocol.n_dof)
        except TypeError:
            # A template that does not take a DOF count builds its own.
            self.protocol.trajectory = builder()
        self._redraw_segments()
        self._say(f"Applied '{name}'. Edit the segments below, then build the media.",
                  "ok")

    def _refresh_summary(self):
        if not dpg.does_item_exist(SUMMARY_TAG):
            return
        try:
            frames = len(self.protocol.coordinates())
        except ValueError as error:
            style.set_status(SUMMARY_TAG, str(error), "error")
            return
        fps = max(1, self.protocol.media.fps)
        message = (f"{len(self.protocol.trajectory)} segments, {frames} frames, "
                   f"{frames / fps:.1f} s per repetition.")
        kind = "info"
        # With no segments at all there is nothing for the cues to be missing.
        if self.protocol.trajectory and (self.protocol.media.show_countdown
                                         or self.protocol.media.show_direction):
            if not any(segment.kind in ("hold", "rest", "random")
                       for segment in self.protocol.trajectory):
                message += (" The countdown and direction cues need a steady "
                            "stretch to count down to -- add a hold or a rest.")
                kind = "warn"
        style.set_status(SUMMARY_TAG, message, kind)

    # ==================================================================
    # reading the window back into the protocol
    # ==================================================================
    def _changed(self, *_args):
        self._read_window()
        self._refresh_summary()

    def _read_window(self):
        """Pull every control back into the protocol object."""
        protocol, media, timing = self.protocol, self.protocol.media, self.protocol.timing
        protocol.name = (self._value("name") or "collection").strip() or "collection"
        protocol.kind = self._value("kind", protocol.kind)
        protocol.notes = self._value("notes", protocol.notes)
        labels = [part.strip() for part in (self._value("dof_labels") or "").split(",")]
        protocol.dof_labels = [label for label in labels if label]
        media.animator = self._value("animator", media.animator)
        media.fps = max(1, int(self._value("fps", media.fps)))
        media.tpd = max(0.01, float(self._value("tpd", media.tpd)))
        media.show_direction = bool(self._value("show_direction", media.show_direction))
        media.show_countdown = bool(self._value("show_countdown", media.show_countdown))
        media.show_boundary = bool(self._value("show_boundary", media.show_boundary))
        media.figsize = (max(1, int(self._value("figure_width", media.figsize[0]))),
                         max(1, int(self._value("figure_height", media.figsize[1]))))
        media.dpi = max(10, int(self._value("dpi", media.dpi)))
        media.video_format = self._value("video_format", media.video_format)
        timing.num_reps = max(1, int(self._value("num_reps", timing.num_reps)))
        timing.rep_time = max(0.0, float(self._value("rep_time", timing.rep_time)))
        timing.rest_time = max(0.0, float(self._value("rest_time", timing.rest_time)))
        timing.rest_jitter = max(0.0, float(self._value("rest_jitter", timing.rest_jitter)))
        timing.reaction_time = max(0.0, float(self._value("reaction_time",
                                                          timing.reaction_time)))
        protocol.order = self._value("order", protocol.order)
        protocol.seed = int(self._value("seed", protocol.seed))
        protocol.factors = _parse_factors(self._value("factors", ""))
        protocol.acquisition = {name: str(self._value(f"acq_{name}", "") or "").strip()
                                for name, _ in ACQUISITION_FIELDS}
        self.media_folder = self._value("media", self.media_folder) or self.media_folder

    def _factors_text(self):
        return "\n".join(f"{name}: {', '.join(str(level) for level in levels)}"
                         for name, levels in self.protocol.factors.items())

    def _write_window(self):
        """Push a freshly loaded protocol back out to the controls."""
        protocol, media, timing = self.protocol, self.protocol.media, self.protocol.timing
        for name, value in (("name", protocol.name), ("kind", protocol.kind),
                            ("notes", protocol.notes),
                            ("dof_labels", ", ".join(protocol.dof_labels)),
                            ("animator", media.animator), ("fps", media.fps),
                            ("tpd", media.tpd),
                            ("show_direction", media.show_direction),
                            ("show_countdown", media.show_countdown),
                            ("show_boundary", media.show_boundary),
                            ("figure_width", int(media.figsize[0])),
                            ("figure_height", int(media.figsize[1])),
                            ("dpi", media.dpi),
                            ("video_format", media.video_format),
                            ("num_reps", timing.num_reps),
                            ("rep_time", timing.rep_time),
                            ("rest_time", timing.rest_time),
                            ("rest_jitter", timing.rest_jitter),
                            ("reaction_time", timing.reaction_time),
                            ("order", protocol.order), ("seed", protocol.seed),
                            ("factors", self._factors_text())):
            tag = f"__protocol_{name}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)
        for name, _ in ACQUISITION_FIELDS:
            tag = f"__protocol_acq_{name}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, str(protocol.acquisition.get(name, "")))
        self._redraw_segments()

    # ==================================================================
    # files and media
    # ==================================================================
    def save(self):
        """Write the protocol where the path field points."""
        self._read_window()
        target = (self._value("path") or "").strip() or f"{self.protocol.name}.yaml"
        try:
            path = self.protocol.save(target)
        except OSError as error:
            self._say(f"Could not save: {error}", "error")
            return None
        self.protocol_path = str(path)
        if dpg.does_item_exist("__protocol_path"):
            dpg.set_value("__protocol_path", self.protocol_path)
        self._say(f"Saved {path}.", "ok")
        return path

    def load(self):
        """Read the protocol the path field points at into the window."""
        target = (self._value("path") or "").strip()
        try:
            self.protocol = CollectionProtocol.load(target)
        except (OSError, ValueError) as error:
            self._say(f"Could not load {target or '(no path given)'}: {error}", "error")
            return None
        self.protocol_path = target
        # A protocol does not carry a media folder -- the same protocol can be
        # rendered anywhere -- so the window's field stays authoritative.
        self.media_folder = self._value("media", self.media_folder) or self.media_folder
        self._write_window()
        self._refresh_summary()
        self._say(f"Loaded {self.protocol.name}. "
                  f"{self.protocol.describe().splitlines()[-1]}", "ok")
        self._publish()
        return self.protocol

    def build(self):
        """Render the protocol's media, off the render thread."""
        if self._worker is not None and self._worker.is_alive():
            self._say("Already building. Wait for it to finish.", "warn")
            return
        self._read_window()
        if not self.protocol.trajectory:
            self._say("Nothing to build: this protocol has no trajectory. Apply a "
                      "template or add a segment.", "warn")
            return
        folder = self.media_folder
        self._post(f"Building {self.protocol.name}{self.protocol.media.video_format} "
                   f"in {folder} ...")
        self._worker = threading.Thread(target=self._build_worker, args=(folder,),
                                        daemon=True)
        self._worker.start()
        self._sync_buttons()

    def _build_worker(self, folder):
        try:
            written = self.protocol.build_media(folder, progress=self._post)
        except Exception:
            # A missing axis image, an unwritable folder, no ffmpeg for an mp4:
            # the last line is the part worth showing.
            self._post("Could not build the media: "
                       + traceback.format_exc().strip().splitlines()[-1], "error")
            return
        self._post(f"Built {', '.join(path.name for path in written)} in {folder}. "
                   f"Open Data > Collect Data to record against it.", "ok")
        self._post(folder, "ready")

    def _post(self, message, kind="busy"):
        """Queue ``message`` for the window. ``kind`` 'ready' carries the built folder."""
        with self._lock:
            self._messages.append((message, kind))

    def poll(self):
        """Drain what the render thread said onto the window. Called per frame."""
        with self._lock:
            pending, self._messages = self._messages, []
        for message, kind in pending:
            if kind == "ready":
                self.media_folder = message
                self._publish()
            else:
                self._say(message, kind)
        if self._worker is not None and not self._worker.is_alive():
            # Finished, one way or the other. Anything it said after the drain
            # above is shown on the next frame.
            self._worker = None
            self._sync_buttons()

    def _publish(self):
        if self.on_ready is not None:
            self.on_ready(self.protocol, self.media_folder)

    def cleanup(self):
        """Tear the window down. The render thread is left to finish on its own."""
        if dpg.does_alias_exist(WINDOW_TAG):
            dpg.delete_item(WINDOW_TAG)


def _numbers(text):
    """Parse '1, -0.5' into floats, ignoring anything that is not a number."""
    values = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def _template_help():
    """What each trajectory template is, in the words of its own docstring."""
    lines = ["Fills the list below for the DOFs named above, replacing what is in it."]
    for name, builder in TRAJECTORY_TEMPLATES.items():
        summary = (builder.__doc__ or "").strip().splitlines()
        lines.append(f"{name}: {summary[0]}" if summary else name)
    return "\n\n".join(lines)


def _split_prompt(prompt):
    """Split an acquisition prompt into a label, a unit and a hint for its box.

    'Notch filter (Hz, blank if none)' becomes a 'Notch filter' label, 'Hz'
    after the box and 'blank if none' inside it while it is empty, rather than
    one label too long for the column.
    """
    label, _, detail = prompt.partition(" (")
    detail = detail.rstrip(")")
    unit, _, rest = detail.partition(", ")
    if unit in _ACQUISITION_UNITS:
        return label, unit, rest
    return label, "", detail


def _parse_factors(text):
    """Parse 'limb_position: neutral, elevated' lines into {name: [levels]}."""
    factors = {}
    for line in str(text or "").splitlines():
        if ":" not in line:
            continue
        name, _, levels = line.partition(":")
        name = name.strip()
        values = [level.strip() for level in levels.split(",") if level.strip()]
        if name and values:
            factors[name] = values
    return factors
