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
"""

import threading
import traceback
from pathlib import Path

import dearpygui.dearpygui as dpg

from libemg.protocol import (ACQUISITION_FIELDS, ANIMATORS, CLASSIFICATION,
                             ORDERINGS, REGRESSION, SEGMENT_KINDS,
                             TRAJECTORY_TEMPLATES, CollectionProtocol, Segment)

WINDOW_TAG = "__protocol_window"
STATUS_TAG = "__protocol_status"
SEGMENTS_TAG = "__protocol_segments"
SUMMARY_TAG = "__protocol_summary"


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
            with dpg.group(horizontal=True):
                dpg.add_input_text(label="Protocol file", tag="__protocol_path",
                                   default_value=self.protocol_path, width=380)
                dpg.add_button(label="Load", callback=self.load)
                dpg.add_button(label="Save", callback=self.save)
            dpg.add_text("", tag=STATUS_TAG, wrap=self.width - 40)
            dpg.add_separator()

            with dpg.collapsing_header(label="What is being collected",
                                       default_open=True):
                self._text("Name", "name", self.protocol.name)
                dpg.add_combo(sorted((CLASSIFICATION, REGRESSION)),
                              default_value=self.protocol.kind, label="Kind",
                              tag="__protocol_kind", width=200,
                              callback=self._changed)
                self._text("DOF / bar labels (comma separated)", "dof_labels",
                           ", ".join(self.protocol.dof_labels), width=420)
                self._text("Notes", "notes", self.protocol.notes, width=420)

            with dpg.collapsing_header(label="Prompt trajectory", default_open=True):
                dpg.add_text(
                    "What the subject is asked to do, as a sequence of segments. "
                    "An animated prompt paces the contraction instead of leaving "
                    "its speed and depth to the subject, and the trajectory is "
                    "written out beside the video as per-frame ground truth.",
                    wrap=self.width - 60, color=(170, 190, 210))
                with dpg.group(horizontal=True):
                    dpg.add_combo(list(TRAJECTORY_TEMPLATES), label="Template",
                                  tag="__protocol_template", width=260,
                                  default_value=next(iter(TRAJECTORY_TEMPLATES)))
                    dpg.add_button(label="Apply template",
                                   callback=self._apply_template)
                    dpg.add_button(label="Add segment", callback=self._add_segment)
                    dpg.add_button(label="Clear", callback=self._clear_segments)
                with dpg.child_window(tag=SEGMENTS_TAG, autosize_x=True, height=220):
                    self._build_segments()
                dpg.add_text("", tag=SUMMARY_TAG, wrap=self.width - 60)

            with dpg.collapsing_header(label="How it is drawn", default_open=True):
                media = self.protocol.media
                dpg.add_combo(sorted(ANIMATORS), default_value=media.animator,
                              label="Animator", tag="__protocol_animator",
                              width=200, callback=self._changed)
                with dpg.group(horizontal=True):
                    self._int("Frames per second", "fps", media.fps)
                    self._float("Seconds per unit distance", "tpd", media.tpd)
                with dpg.group(horizontal=True):
                    self._bool("Show direction", "show_direction", media.show_direction)
                    self._bool("Show countdown", "show_countdown", media.show_countdown)
                    self._bool("Show boundary", "show_boundary", media.show_boundary)
                with dpg.group(horizontal=True):
                    self._int("Figure width (in)", "figure_width", int(media.figsize[0]))
                    self._int("Figure height (in)", "figure_height", int(media.figsize[1]))
                    self._int("DPI", "dpi", media.dpi)
                    dpg.add_combo([".mp4", ".gif"], default_value=media.video_format,
                                  label="Format", tag="__protocol_video_format",
                                  width=90, callback=self._changed)
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="Media folder", tag="__protocol_media",
                                       default_value=self.media_folder, width=300)
                    dpg.add_button(label="Build media", tag="__protocol_build",
                                   callback=self.build)

            with dpg.collapsing_header(label="Timing and order", default_open=True):
                dpg.add_text(
                    "Shuffling prompts within a repetition stops class order from "
                    "tracking fatigue and electrode drift; the seed keeps the "
                    "session reproducible. Jittered rest stops the subject "
                    "anticipating the cue and contracting before it.",
                    wrap=self.width - 60, color=(170, 190, 210))
                timing = self.protocol.timing
                with dpg.group(horizontal=True):
                    self._int("Reps", "num_reps", timing.num_reps)
                    self._float("Rep time (s)", "rep_time", timing.rep_time)
                    self._float("Rest (s)", "rest_time", timing.rest_time)
                    self._float("Rest jitter (s)", "rest_jitter", timing.rest_jitter)
                with dpg.group(horizontal=True):
                    dpg.add_combo(list(ORDERINGS), default_value=self.protocol.order,
                                  label="Order", tag="__protocol_order", width=180,
                                  callback=self._changed)
                    self._int("Seed", "seed", self.protocol.seed, width=140)
                    self._float("Reaction time (s)", "reaction_time",
                                timing.reaction_time)

            with dpg.collapsing_header(label="Design factors"):
                dpg.add_text(
                    "Levels to vary between repetitions, one per line as "
                    "'name: level, level'. Accuracy falls when the limb is in a "
                    "position, or the contraction at an intensity, that was not "
                    "in the training set -- collecting across levels is the fix, "
                    "and it only helps if which level each rep used is recorded.",
                    wrap=self.width - 60, color=(170, 190, 210))
                dpg.add_input_text(tag="__protocol_factors", multiline=True,
                                   height=80, width=self.width - 60,
                                   default_value=self._factors_text(),
                                   callback=self._changed)

            with dpg.collapsing_header(label="Acquisition (ISEK reporting)"):
                dpg.add_text(
                    "None of this can be recovered from the recording, and all of "
                    "it decides whether the data can be compared with anyone "
                    "else's.", wrap=self.width - 60, color=(170, 190, 210))
                for name, prompt in ACQUISITION_FIELDS:
                    dpg.add_input_text(label=prompt, tag=f"__protocol_acq_{name}",
                                       default_value=str(self.protocol.acquisition.get(name, "")),
                                       width=220, callback=self._changed)
        self._refresh_summary()
        self._say("Describe the collection, build its prompts, then open "
                  "Data > Collect Data. Saving writes the protocol beside the data.")
        return self

    # -- small control helpers -------------------------------------------

    def _text(self, label, name, value, width=260):
        dpg.add_input_text(label=label, tag=f"__protocol_{name}",
                           default_value=str(value), width=width,
                           callback=self._changed)

    def _int(self, label, name, value, width=120):
        dpg.add_input_int(label=label, tag=f"__protocol_{name}",
                          default_value=int(value), width=width, step=1,
                          callback=self._changed)

    def _float(self, label, name, value, width=120):
        dpg.add_input_float(label=label, tag=f"__protocol_{name}",
                            default_value=float(value), width=width, step=0.0,
                            format="%.2f", callback=self._changed)

    def _bool(self, label, name, value):
        dpg.add_checkbox(label=label, tag=f"__protocol_{name}",
                         default_value=bool(value), callback=self._changed)

    def _value(self, name, default=None):
        tag = f"__protocol_{name}"
        return dpg.get_value(tag) if dpg.does_item_exist(tag) else default

    def _say(self, message):
        if dpg.does_item_exist(STATUS_TAG):
            dpg.set_value(STATUS_TAG, message)

    # ==================================================================
    # trajectory editing
    # ==================================================================
    def _build_segments(self):
        if not self.protocol.trajectory:
            dpg.add_text("No segments yet. Apply a template, or add one.",
                         color=(170, 190, 210))
            return
        for index, segment in enumerate(self.protocol.trajectory):
            with dpg.group(horizontal=True):
                dpg.add_combo(list(SEGMENT_KINDS), default_value=segment.kind,
                              width=110, tag=f"__protocol_seg_kind_{index}",
                              user_data=index, callback=self._segment_changed)
                dpg.add_input_float(default_value=float(segment.seconds), width=90,
                                    step=0.0, format="%.2f s",
                                    tag=f"__protocol_seg_seconds_{index}",
                                    user_data=index, callback=self._segment_changed)
                dpg.add_input_text(
                    default_value=", ".join(f"{value:g}" for value in segment.target),
                    width=160, hint="target per DOF",
                    tag=f"__protocol_seg_target_{index}",
                    user_data=index, callback=self._segment_changed)
                if segment.kind == "sinusoid":
                    dpg.add_input_float(default_value=float(segment.period), width=90,
                                        step=0.0, format="%.2f period",
                                        tag=f"__protocol_seg_period_{index}",
                                        user_data=index, callback=self._segment_changed)
                if segment.kind == "random":
                    dpg.add_input_int(default_value=int(segment.count), width=90,
                                      step=1, tag=f"__protocol_seg_count_{index}",
                                      user_data=index, callback=self._segment_changed)
                dpg.add_button(label="x", width=24, user_data=index,
                               callback=self._remove_segment)

    def _redraw_segments(self):
        if not dpg.does_item_exist(SEGMENTS_TAG):
            return
        dpg.delete_item(SEGMENTS_TAG, children_only=True)
        dpg.push_container_stack(SEGMENTS_TAG)
        self._build_segments()
        dpg.pop_container_stack()
        self._refresh_summary()

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
        self._say(f"Applied '{name}'. Edit the segments below, then build the media.")

    def _refresh_summary(self):
        if not dpg.does_item_exist(SUMMARY_TAG):
            return
        try:
            frames = len(self.protocol.coordinates())
        except ValueError as error:
            dpg.set_value(SUMMARY_TAG, str(error))
            return
        fps = max(1, self.protocol.media.fps)
        message = (f"{len(self.protocol.trajectory)} segments, {frames} frames, "
                   f"{frames / fps:.1f} s per repetition.")
        if self.protocol.media.show_countdown or self.protocol.media.show_direction:
            if not any(segment.kind in ("hold", "rest", "random")
                       for segment in self.protocol.trajectory):
                message += (" The countdown and direction cues need a steady "
                            "stretch to count down to -- add a hold or a rest.")
        dpg.set_value(SUMMARY_TAG, message)

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
            self._say(f"Could not save: {error}")
            return None
        self.protocol_path = str(path)
        if dpg.does_item_exist("__protocol_path"):
            dpg.set_value("__protocol_path", self.protocol_path)
        self._say(f"Saved {path}.")
        return path

    def load(self):
        """Read the protocol the path field points at into the window."""
        target = (self._value("path") or "").strip()
        try:
            self.protocol = CollectionProtocol.load(target)
        except (OSError, ValueError) as error:
            self._say(f"Could not load {target or '(no path given)'}: {error}")
            return None
        self.protocol_path = target
        # A protocol does not carry a media folder -- the same protocol can be
        # rendered anywhere -- so the window's field stays authoritative.
        self.media_folder = self._value("media", self.media_folder) or self.media_folder
        self._write_window()
        self._refresh_summary()
        self._say(f"Loaded {self.protocol.name}. {self.protocol.describe().splitlines()[-1]}")
        self._publish()
        return self.protocol

    def build(self):
        """Render the protocol's media, off the render thread."""
        if self._worker is not None and self._worker.is_alive():
            self._say("Already building. Wait for it to finish.")
            return
        self._read_window()
        if not self.protocol.trajectory:
            self._say("Nothing to build: this protocol has no trajectory. Apply a "
                      "template or add a segment.")
            return
        folder = self.media_folder
        self._post(f"Building {self.protocol.name}{self.protocol.media.video_format} "
                   f"in {folder} ...")
        self._worker = threading.Thread(target=self._build_worker, args=(folder,),
                                        daemon=True)
        self._worker.start()

    def _build_worker(self, folder):
        try:
            written = self.protocol.build_media(folder, progress=self._post)
        except Exception:
            # A missing axis image, an unwritable folder, no ffmpeg for an mp4:
            # the last line is the part worth showing.
            self._post("Could not build the media: "
                       + traceback.format_exc().strip().splitlines()[-1])
            return
        self._post(f"Built {', '.join(path.name for path in written)} in {folder}. "
                   f"Open Data > Collect Data to record against it.")
        self._post(("__ready__", folder))

    def _post(self, message):
        with self._lock:
            self._messages.append(message)

    def poll(self):
        """Drain what the render thread said onto the window. Called per frame."""
        with self._lock:
            pending, self._messages = self._messages, []
        for message in pending:
            if isinstance(message, tuple) and message and message[0] == "__ready__":
                self.media_folder = message[1]
                self._publish()
            else:
                self._say(message)

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
