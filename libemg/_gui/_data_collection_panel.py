import shutil
from contextlib import contextmanager
from pathlib import Path
import dearpygui.dearpygui as dpg
import numpy as np
import os
from dataclasses import dataclass, field
from itertools import compress
import time
import json
import yaml
from datetime import datetime
from . import _style as style
from ._utils import Media, set_texture
from ._visualization_panel import VisualizationPanel
from ..session import (PAUSE, PROMPT, REST, SESSION_PREFIX, SessionAnnotations,
                       Stage)

import threading


# Who the collection was of, written beside the data it produced. Held as one
# spec rather than five copies of the same three lines, because the
# configuration window, the settings read off it and the file written from them
# all have to agree on the same set of fields.
SUBJECT_FIELDS = (
    ("subject_number", "Subject number", int),
    ("subject_name", "Subject name", str),
    ("session_number", "Session", int),
    ("day_number", "Day", int),
    ("condition_number", "Condition", int),
)
SUBJECT_DETAILS_NAME = "subject_details.yaml"

# What counts as a prompt in the media folder. Shared by the collection and the
# count the configuration window shows, so the window cannot promise a number of
# motions the collection then disagrees with.
MEDIA_SUFFIXES = (".gif", ".png", ".mp4", "jpg")

# The configuration window. The folders run the full width of it, so a path can
# be read without scrolling along it; the rest sits in two columns.
_CONFIG_SIZE = (980, 780)
_NUMBER_WIDTH = 120
_FULL_WRAP = _CONFIG_SIZE[0] - 70
_COLUMN_WRAP = _CONFIG_SIZE[0] // 2 - 60
#: The prompt during collection is read from across a desk, so it is drawn larger.
PROMPT_FONT_SIZE = 30

# How long a modality may go without delivering a single sample before the
# stream is called interrupted, in seconds. One repetition is too short a window
# to judge on: a low-rate modality arriving in bursts can legitimately miss a
# short one entirely, which is what made a per-repetition zero check fire on a
# perfectly healthy IMU.
STREAM_SILENCE_LIMIT = 5.0


@dataclass
class SessionLog:
    """State for one continuous recording: where it is, and what happened in it.

    The annotations are the durable part -- they are written to disk after every
    stage -- and ``last_movement`` is the only thing kept here that is not, since
    it exists to catch a dead stream while the collection is still running.
    """

    annotations: SessionAnnotations
    last_movement: dict = field(default_factory=dict)

    @property
    def prefix(self):
        return Path(self.annotations.folder,
                    self.annotations.prefix).absolute().as_posix()


class DataCollectionPanel:
    def __init__(self,
                 online_data_handler,
                 num_reps=3,
                 rep_time=3,
                 media_folder='media/',
                 data_folder='data/',
                 rest_time=2,
                 auto_advance=True,
                 exclude_files=[],
                 timestamps=False,
                 continuous_log=True,
                 protocol=None,
                 signal_check='warn',
                 line_frequency=60.0,
                 subject_number=1,
                 subject_name='',
                 session_number=1,
                 day_number=1,
                 condition_number=1,
                 visualize_num_samples=500,
                 video_player_width = 720,
                 video_player_height = 480):

        self.online_data_handler = online_data_handler
        self.num_reps = num_reps
        self.rep_time = rep_time
        self.media_folder = media_folder
        self.data_folder  = data_folder
        self.rest_time = rest_time
        self.auto_advance=auto_advance
        self.exclude_files = exclude_files
        self.timestamps = timestamps
        # Record the whole collection in one piece -- rests and pauses included,
        # nothing removed -- and cut the repetitions out of it afterwards, rather
        # than starting and stopping a log around each one. See
        # _begin_session_log for why the cut has to happen after, and
        # libemg.session for the annotations that say what is where in it.
        self.continuous_log = continuous_log
        # What to collect and in what order. Without one, the collection falls
        # back to what it always did: the media folder in alphabetical order,
        # every repetition the same. See libemg.protocol for why that order is
        # worth replacing.
        self.protocol = self._resolve_protocol(protocol)
        if self.protocol is not None:
            # A protocol is the description of the experiment, so its timings
            # are the ones the window opens on.
            self.num_reps = self.protocol.timing.num_reps
            self.rep_time = self.protocol.timing.rep_time
            self.rest_time = rest_time = self.protocol.timing.rest_time
        # 'warn' reports a bad signal and collects anyway, 'block' refuses to
        # start, 'off' does not look. Warning is the default because these are
        # heuristics, and a heuristic that can cancel a booked session hour is
        # worse than one that prints.
        self.signal_check = signal_check
        self.line_frequency = line_frequency
        # The rest before the next prompt, which a protocol jitters per prompt.
        self.current_rest_time = rest_time
        self.current_plan = None
        self.prompt_plan = []
        # Who this collection is of. Defaults stand in for a first session so the
        # window opens on something sensible; what is actually collected under is
        # whatever the window is left showing when Start is pressed.
        self.subject_number = subject_number
        self.subject_name = subject_name
        self.session_number = session_number
        self.day_number = day_number
        self.condition_number = condition_number
        self.visualize_num_samples = visualize_num_samples
        self.video_player_width = video_player_width
        self.video_player_height = video_player_height

        # The folder dialogs are listed because they are windows of their own
        # rather than part of the configuration window, so deleting that window
        # does not take them with it.
        self.widget_tags = {"configuration":['__dc_configuration_window','__dc_num_reps','__dc_rep_time','__dc_rest_time', '__dc_media_folder',\
                                             '__dc_auto_advance', *(f"__dc_{name}" for name, _, _ in SUBJECT_FIELDS),
                                             '__dc_media_dialog', '__dc_output_dialog'],
                            "collection":   ['__dc_collection_window', '__dc_prompt', '__dc_progress', '__dc_redo_button'],
                            "visualization": ['__vls_visualize_window']}
        # The texture registry set_texture parents its textures to
        # (_utils.TEXTURE_REGISTRY_TAG) is deliberately absent from these lists.
        # It is process-global and shared with every other panel, so deleting it
        # on this panel's teardown would take out the textures those panels are
        # still holding. It is a single item that is reused for the life of the
        # process, so leaving it alive leaks nothing.


    @staticmethod
    def _resolve_protocol(protocol):
        """Accept a protocol object, a path to one, or nothing.

        Imported here rather than at module scope so that reading a protocol
        does not drag matplotlib into every process that opens this panel.
        """
        if protocol is None or not isinstance(protocol, (str, Path)):
            return protocol
        from libemg.protocol import CollectionProtocol
        return CollectionProtocol.load(protocol)

    def cleanup_window(self, window_name):
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)      
    
    def spawn_configuration_window(self):
        self.cleanup_window("configuration")
        self.cleanup_window("collection")
        self.cleanup_window("visualization")
        width, height = _CONFIG_SIZE
        with dpg.window(tag="__dc_configuration_window", label="Collect Data",
                        width=width, height=height):
            # The settings scroll and Start does not: a long signal report
            # would otherwise push the one button that matters out of the window.
            with dpg.child_window(border=False, autosize_x=True, height=-46):
                style.note("Each image or video in the media folder is shown in turn, once per "
                        "repetition, while the device records. Hover over a setting's name "
                        "to see what it does.", wrap=_FULL_WRAP)
                dpg.add_spacer(height=2)
                self._build_files()
                with dpg.table(header_row=False, resizable=False, borders_innerV=True,
                               policy=dpg.mvTable_SizingStretchSame):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.group():
                            self._build_repetitions()
                        with dpg.group():
                            self._build_subject()
                self._build_signal()
            dpg.add_separator()
            with dpg.group(horizontal=True):
                style.button("Start", "primary", tag="__dc_start", width=120,
                          callback=self.start_callback)
                style.note("", tag="__dc_status")
        for dialog, field_tag, folder in (("__dc_media_dialog", "__dc_media_folder", self.media_folder),
                                          ("__dc_output_dialog", "__dc_output_folder", self.data_folder)):
            dpg.add_file_dialog(directory_selector=True, show=False, modal=True, tag=dialog,
                                width=760, height=460, user_data=field_tag,
                                default_path=os.path.abspath(folder) if os.path.isdir(folder)
                                else os.getcwd(),
                                callback=self._folder_chosen)
        self._describe_advance(self.auto_advance)
        self._describe(self.media_folder, int(self.num_reps))

        # dpg.set_primary_window("__dc_configuration_window", True)

    def _build_repetitions(self):
        style.section("Repetitions")
        with style.form():
            with style.row("Repetitions", help="How many times each motion is collected."):
                dpg.add_input_int(default_value=int(self.num_reps), tag="__dc_num_reps",
                                  width=_NUMBER_WIDTH, step=0, min_value=1, min_clamped=True,
                                  callback=self._settings_changed)
            with style.row("Time per repetition", help="How long each image stays up. A video "
                        "or GIF plays for its own length instead."):
                dpg.add_input_float(default_value=float(self.rep_time), tag="__dc_rep_time",
                                    width=_NUMBER_WIDTH, step=0, format="%g", min_value=0.0,
                                    min_clamped=True)
                style.unit("s")
            with style.row("Rest between prompts", help="The rest before each prompt, with the "
                        "next one shown greyed out so the subject can get ready for it. "
                        "0 for none."):
                dpg.add_input_float(default_value=float(self.rest_time), tag="__dc_rest_time",
                                    width=_NUMBER_WIDTH, step=0, format="%g", min_value=0.0,
                                    min_clamped=True)
                style.unit("s")
            with style.row("Auto-advance", help="On: the prompts follow one another, and the "
                        "collection waits for Redo or Continue after each repetition of the "
                        "whole set. Off: it waits after every prompt."):
                dpg.add_checkbox(default_value=bool(self.auto_advance), tag="__dc_auto_advance",
                                 callback=lambda sender, on: self._describe_advance(on))
                style.note("", tag="__dc_advance_note")
        if self.protocol is not None:
            # The window opens on the protocol's timings, but the prompt length
            # and the rests are taken from the protocol itself, prompt by
            # prompt, so it is worth saying where they come from.
            timing = self.protocol.timing
            style.note(f"Following the protocol {self.protocol.name}: {self.protocol.order} order, "
                    f"prompts of {timing.rep_time:g} s, rests of {timing.rest_time:g}-"
                    f"{timing.rest_time + timing.rest_jitter:g} s.", wrap=_COLUMN_WRAP)

    def _build_files(self):
        style.section("Files", first=True)
        with style.form():
            with style.row("Media folder", help="One image or video per motion, named for it: "
                        "the file name is the class name. They are prompted in alphabetical "
                        "order unless a protocol sets the order. A .txt or .csv with the same "
                        "name as a video is copied to the output folder as its labels."):
                dpg.add_input_text(default_value=self.media_folder, tag="__dc_media_folder",
                                   width=-100, callback=self._settings_changed)
                style.button("Browse...", width=90,
                          callback=lambda: dpg.show_item("__dc_media_dialog"))
            with style.row(""):
                style.note("", tag="__dc_media_found")
            with style.row("Output folder", help="Where the recordings are written: a file per "
                        "motion, repetition and modality, with the details of the collection "
                        "beside them. Made if it doesn't exist. Files left there by an earlier "
                        "collection under the same names are replaced."):
                dpg.add_input_text(default_value=self.data_folder, tag="__dc_output_folder",
                                   width=-100)
                style.button("Browse...", width=90,
                          callback=lambda: dpg.show_item("__dc_output_dialog"))

    def _build_subject(self):
        style.section("Subject")
        with style.form():
            for name, label, kind in SUBJECT_FIELDS:
                with style.row(label):
                    dpg.add_input_text(default_value=getattr(self, name), tag=f"__dc_{name}",
                                       width=-1 if kind is str else _NUMBER_WIDTH)
        style.note(f"Saved beside the data as {SUBJECT_DETAILS_NAME}. A label such as P01 works "
                "as well as a number.", wrap=_COLUMN_WRAP)

    def _build_signal(self):
        style.section("Signal")
        before = {"warn": " Start checks again, and warns but collects anyway.",
                  "block": " Start checks again, and won't begin if it finds a problem."}
        style.note("Check the signal while the electrodes can still be adjusted: it looks for "
                "dead channels, clipping and mains noise, and saves what it finds in the "
                "output folder." + before.get(self.signal_check, ""), wrap=_FULL_WRAP)
        with dpg.group(horizontal=True):
            # Offered separately from Start because the point of the check is to
            # be run while the electrodes are still being adjusted, not once at
            # the moment it is too late.
            style.button("Check signal", tag="__dc_check_signal",
                      callback=self.signal_check_callback)
            style.button("Show live signal", tag="__dc_visualize",
                      callback=self.visualize_callback)
        style.note("", tag="__dc_signal_report", wrap=_FULL_WRAP)
        if self.online_data_handler is None:
            style.set_enabled("__dc_check_signal", False)
            style.set_enabled("__dc_visualize", False)

    def _describe(self, media_folder, num_reps):
        """Say what Start will collect, and let it be pressed only when it can start."""
        media = _media_in(media_folder)
        if media is None:
            style.set_status("__dc_media_found", "This folder doesn't exist.", "warn")
        elif not media:
            style.set_status("__dc_media_found", "No images or videos (.png, .jpg, .gif, .mp4) "
                          "in this folder.", "warn")
        else:
            style.set_status("__dc_media_found",
                          f"{len(media)} motion{'s' if len(media) != 1 else ''} found.")
        if self.online_data_handler is None:
            self._say("No device is streaming. Start one from Device > Streamer, then open "
                      "Data > Collect Data again.", "warn")
        elif not media:
            self._say("Choose a media folder with at least one image or video in it.", "warn")
        else:
            self._say(f"{len(media)} motions × {num_reps} repetitions: "
                      f"{len(media) * num_reps} prompts.")
        style.set_enabled("__dc_start", bool(media) and self.online_data_handler is not None)

    def _describe_advance(self, on):
        style.set_status("__dc_advance_note", "waits after each repetition" if on
                         else "waits after every prompt")

    def _settings_changed(self, *_args):
        self._describe(dpg.get_value("__dc_media_folder"), dpg.get_value("__dc_num_reps"))

    def _folder_chosen(self, sender, app_data, user_data):
        folder = (app_data or {}).get("file_path_name", "")
        if folder:
            dpg.set_value(user_data, folder)
            self._settings_changed()

    def _say(self, message, kind="info"):
        style.set_status("__dc_status", message, kind)


    def start_callback(self):
        # Every refusal below is said in the window as well as raised: raised
        # from a button, it only ever reaches the terminal, and the person at
        # the window is left pressing Start again.
        if not (self.online_data_handler and sum(list(self.online_data_handler.get_data()[1].values()))):
            self._say("No data is arriving from the device. Check that it is streaming, "
                      "then press Start again.", "error")
            raise ConnectionError('Attempted to start data collection, but data are not being received. Please ensure the OnlineDataHandler is receiving data.')

        self.get_settings()
        if self.signal_check != 'off':
            self._say("Checking the signal before starting...", "busy")
            reports = self.check_signal_quality()
            problems = [problem for report in reports for problem in report.problems]
            if problems and self.signal_check == 'block':
                self._say("Not started: the signal check found a problem.", "error")
                style.set_status("__dc_signal_report", "\n".join(problems), "error")
                raise ConnectionError(
                    'Refusing to start: the signal does not look collectable.\n  '
                    + '\n  '.join(problems)
                    + "\nFix these, or pass signal_check='warn' to collect anyway.")
        dpg.delete_item("__dc_configuration_window")
        self.cleanup_window("configuration")
        media_list = self.gather_media()

        self.spawn_collection_thread = threading.Thread(target=self.spawn_collection_window, args=(media_list,))
        self.spawn_collection_thread.start()
        # self.spawn_collection_window(media_list)

    def get_settings(self):
        self.num_reps      = int(dpg.get_value("__dc_num_reps"))
        self.rep_time      = float(dpg.get_value("__dc_rep_time"))
        self.rest_time     = float(dpg.get_value("__dc_rest_time"))
        self.media_folder  = dpg.get_value("__dc_media_folder")
        self.output_folder = dpg.get_value("__dc_output_folder")
        self.auto_advance  = bool(dpg.get_value("__dc_auto_advance"))
        for name, _, kind in SUBJECT_FIELDS:
            setattr(self, name, self.read_subject_field(f"__dc_{name}", kind))

    @staticmethod
    def read_subject_field(tag, kind):
        """Read one subject detail off the window, keeping what was typed.

        The numbered fields are stored as numbers when they were typed as
        numbers, since that is what anything reading the file back will want to
        sort and filter on. But a study that labels its subjects ``P01`` is not a
        mistake, and neither is a field someone left blank, so text that is not a
        number is written through as text rather than refused. Refusing it here
        would abort a collection at the moment Start was pressed, which is the
        worst possible time to find out.
        """
        value = dpg.get_value(tag)
        if kind is str:
            return value
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return value

    def gather_media(self):
        # find everything in the media folder
        files = os.listdir(self.media_folder)
        files = sorted(files)
        labels_files = [file for file in files if file.endswith(('.txt', '.csv'))]
        files = [file for file in files if file.endswith(MEDIA_SUFFIXES)]
        self.num_motions = len(files)
        collection_conf = []
        # make the collection_details.json file
        collection_details = {}
        collection_details["num_motions"] = self.num_motions
        collection_details["num_reps"]    = self.num_reps
        collection_details["classes"] =   [f.split('.')[0] for f in files]
        collection_details["class_map"] = {index: f.split('.')[0] for index, f in enumerate(files)}
        collection_details["time"]    = datetime.now().isoformat()
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        with open(Path(self.output_folder, "collection_details.json").absolute().as_posix(), 'w') as f:
            json.dump(collection_details, f)
        self.write_subject_details()

        for media_file in files:
            matching_labels_files = [labels_file for labels_file in labels_files if Path(labels_file).stem == Path(media_file).stem]
            if len(matching_labels_files) == 1:
                # Copy labels file to data directory
                labels_file = matching_labels_files[0]
                class_index = [idx for idx, filename in collection_details['class_map'].items() if filename == Path(labels_file).stem]
                assert len(class_index) == 1, f"Expected a single matching filename in collection_details.json, but got {len(class_index)} for {labels_file}."
                class_index = class_index[0]
                labels_new_filename = Path(labels_file).with_stem(f"C_{class_index}").name
                shutil.copy(Path(self.media_folder, labels_file).absolute(), Path(self.data_folder, labels_new_filename).absolute())

        # One loaded Media per class, reused across repetitions. Loading the
        # same video once per repetition decodes it once per repetition.
        loaded = {}
        for class_index, motion_class in enumerate(files):
            media = Media()
            media.from_file(Path(self.media_folder, motion_class).absolute().as_posix())
            if media.type in ('mp4', 'gif'):
                # Automatically calculate length of video
                rep_time = media.n_frames / media.fps
            else:
                rep_time = self.rep_time
            loaded[class_index] = (media, motion_class.split('.')[0], rep_time)

        self.prompt_plan = self.plan_prompts(
            [name for _, name, _ in loaded.values()])
        for prompt in self.prompt_plan:
            media, name, rep_time = loaded[prompt.motion]
            # A video's length is fixed by the video; only an image's prompt
            # duration is the protocol's to set.
            collection_conf.append([media, name, prompt.motion, prompt.rep,
                                    rep_time if media.type in ('mp4', 'gif')
                                    else prompt.rep_time])
        return collection_conf

    def plan_prompts(self, names):
        """Decide what is prompted, in what order, with what rests.

        With no protocol this is what it always was: every class in folder
        order, the same every repetition, with a fixed rest. A protocol
        replaces that with a shuffled or counterbalanced order and a jittered
        rest, so class does not track fatigue and the subject cannot anticipate
        the cue. See :mod:`libemg.protocol`.

        Returns
        ----------
        prompts: list of libemg.protocol.Prompt
        """
        from libemg.protocol import Prompt

        if self.protocol is not None:
            plan = self.protocol.schedule(names, num_reps=self.num_reps)
            # The protocol is what the data was collected under, so it is kept
            # with the data rather than only wherever it was edited.
            self.protocol.save(self.output_folder)
            return plan
        return [Prompt(motion=motion, rep=rep, name=name, label=name,
                       rep_time=self.rep_time, rest_time=self.rest_time)
                for rep in range(self.num_reps)
                for motion, name in enumerate(names)]

    def signal_check_callback(self):
        """Run the pre-flight check and put its summary in the window."""
        if dpg.does_item_exist("__dc_output_folder"):
            # So the report lands in the folder the window is pointing at, the
            # same one Start would use.
            self.get_settings()
        # The rate is measured over half a second before anything can be said,
        # long enough for a click with no answer to look like one that missed.
        style.set_status("__dc_signal_report", "Checking the signal...", "busy")
        try:
            reports = self.check_signal_quality()
        except Exception as error:
            self._report_signal(f"Could not check the signal: {error}", "error")
            return
        self._report_signal("\n".join(report.summary() for report in reports),
                            "ok" if all(report.ok for report in reports) else "warn")

    def _report_signal(self, message, kind="info"):
        style.set_status("__dc_signal_report", message, kind)
        print(f'LibEMG -> DataCollectionPanel (signal check)\n{message}')

    def measure_sampling_rates(self, seconds=1.0):
        """Measure what each modality is actually delivering, in Hz.

        Taken from the stream rather than from a setting, because the setting is
        what the device was asked for and the rate is what arrived. A device
        running at half its configured rate is a real and quiet failure, and it
        is also the number every frequency-domain check below depends on.
        """
        odh = self.online_data_handler

        def counts():
            # The counters live in shared memory as (1, 1) arrays; reduce them
            # to plain ints so the rates below are scalars.
            return {mod: int(np.ravel(value)[0])
                    for mod, value in odh.get_data()[1].items()}

        before = counts()
        start = time.time()
        time.sleep(max(0.05, seconds))
        elapsed = time.time() - start
        after = counts()
        return {mod: max(0.0, (after.get(mod, 0) - before.get(mod, 0)) / elapsed)
                for mod in after}

    def check_signal_quality(self, seconds=2.0, at_rest=True, write=True,
                             rate_seconds=0.5):
        """Look at the stream before collecting, and say what is wrong with it.

        A disconnected channel, a clipping amplifier and heavy mains pickup all
        write plausible-looking files, and all three are found weeks later in
        analysis unless something looks now -- while the electrodes can still be
        re-seated. See :mod:`libemg.signal_quality` for what is measured.

        Parameters
        ----------
        seconds: float (optional), default=2.0
            How much of the stream to look at. Taken from the buffer, which
            already holds it, so asking for more does not mean waiting longer.
        at_rest: bool (optional), default=True
            Whether the subject is relaxed. The baseline and imbalance checks
            only mean anything if they are.
        write: bool (optional), default=True
            Write the reports to ``signal_quality.yaml`` in the output folder,
            so the state of the electrodes at the start of the session is part
            of the record rather than a line someone remembers.
        rate_seconds: float (optional), default=0.5
            How long to watch the counters for to measure the delivery rate.
            This is the only part that waits, and half a second is hundreds of
            samples at any rate worth using, so it does not need to be the same
            as the window being checked.

        Returns
        ----------
        reports: list of SignalQualityReport
        """
        from libemg.signal_quality import check_signal

        rates = self.measure_sampling_rates(rate_seconds)
        data = self.online_data_handler.get_data()[0]
        reports = []
        for mod in self.online_data_handler.modalities:
            samples = np.asarray(data.get(mod, []), dtype=float)
            rate = rates.get(mod, 0.0)
            if rate <= 0:
                report = check_signal(samples[:0], 1.0, modality=mod)
            else:
                # Only the newest `seconds` of the buffer: an older stretch was
                # recorded while the subject was being set up, and says nothing
                # about the electrodes as they are now.
                window = samples[:max(2, int(rate * seconds))]
                report = check_signal(window, rate, modality=mod,
                                      line_frequency=self.line_frequency,
                                      at_rest=at_rest)
            reports.append(report)
        if write:
            self.write_signal_report(reports, rates)
        return reports

    def write_signal_report(self, reports, rates=None):
        """Record the state of the signal at the start of the session."""
        folder = getattr(self, 'output_folder', None) or self.data_folder
        path = Path(folder, "signal_quality.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as handle:
            yaml.safe_dump({"checked": datetime.now().isoformat(),
                            "line_frequency": self.line_frequency,
                            "measured_rates_hz": dict(rates or {}),
                            "modalities": [report.to_dict() for report in reports]},
                           handle, sort_keys=False, default_flow_style=False)
        return path

    def write_subject_details(self):
        """Write who this collection was of into the folder the data goes in.

        Kept as its own file rather than folded into ``collection_details.json``
        because the two answer different questions and go stale at different
        times: that file describes the prompts a run happened to use, this one
        describes the person, the session and the condition, which is what a
        dataset has to be grouped by long after the run. YAML because this is the
        file most likely to be opened, read and corrected by hand.

        Returns
        ----------
        path: Path
            The file written.
        """
        details = {name: getattr(self, name) for name, _, _ in SUBJECT_FIELDS}
        details["collected"] = datetime.now().isoformat()
        path = Path(self.output_folder, SUBJECT_DETAILS_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as handle:
            # In the order they are asked for, rather than alphabetically, so the
            # file reads the way the window that filled it in does.
            yaml.safe_dump(details, handle, sort_keys=False, default_flow_style=False)
        return path

    def spawn_collection_window(self, media_list):
        # open first frame of gif
        self.online_data_handler.prepare_smm()
        texture = media_list[0][0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height)
        set_texture("__dc_collection_visual", texture, width=self.video_player_width, height=self.video_player_height)
        
        # The subject reads this window, from arm's length and mid-gesture, so it
        # holds the prompt, the picture and how long is left, and nothing to
        # press until there is something to decide.
        with dpg.window(label="Collecting",
                        tag="__dc_collection_window",
                        width=self.video_player_width + 60,
                        height=self.video_player_height + 260):
            with _centred():
                dpg.add_text(f"Repetition 1 of {self.num_reps}", tag="__dc_rep", color=style.MUTED)
            with _centred():
                prompt = dpg.add_text(media_list[0][1], tag="__dc_prompt")
            font = style.font("bold", PROMPT_FONT_SIZE)
            if font is not None:
                dpg.bind_item_font(prompt, font)
            with _centred():
                dpg.add_image("__dc_collection_visual")
            with _centred():
                dpg.add_progress_bar(tag="__dc_progress", default_value=0.0,
                                     width=self.video_player_width)
            dpg.add_spacer(height=4)
            with _centred():
                with dpg.group(horizontal=True):
                    # Not wrapped: in a column that is as wide as its contents,
                    # text that wraps to the column has no width to wrap to.
                    style.note("That repetition is done." if self.auto_advance
                            else "That prompt is done.", tag="__dc_pause_note", wrap=-1,
                            show=False)
                    # One height for both, since the primary button pads more.
                    style.button("Redo", tag="__dc_redo_button", width=110, height=30,
                              callback=self.redo_collection_callback)
                    style.button("Continue", "primary", tag="__dc_continue_button", width=110,
                              height=30, callback=self.continue_collection_callback)
            dpg.hide_item(item="__dc_redo_button")
            dpg.hide_item(item="__dc_continue_button")


        # dpg.set_primary_window("__dc_collection_window", True)

        self.run_sgt(media_list)
        # clean up the window
        
        dpg.delete_item("__dc_collection_window")
        self.cleanup_window("collection")
        # open config back up
        self.spawn_configuration_window()
    
    def run_sgt(self, media_list):
        self.i = 0
        self.advance = True
        self.online_data_handler.reset()
        session = self._begin_session_log(media_list) if self.continuous_log else None
        try:
            self._collect_media_list(media_list, session)
        finally:
            if session is not None:
                self._finish_session_log(session)

    def _collect_media_list(self, media_list, session):
        while self.i < len(media_list):
            media = media_list[self.i]
            # What this prompt is being collected under: its jittered rest, and
            # the level of each design factor. Kept beside the media rather than
            # in it so the class name stays the class name.
            self.current_plan = (self.prompt_plan[self.i]
                                 if self.i < len(self.prompt_plan) else None)
            self.current_rest_time = (self.current_plan.rest_time
                                      if self.current_plan else self.rest_time)
            # do the rest
            if self.rest_time and self.i < len(media_list):
                # The rest is recorded like anything else. It is in the file
                # either way -- the recording is never stopped -- so the only
                # question is whether anything downstream can tell what it was
                # looking at, and an annotated rest is a usable class rather than
                # an unexplained stretch between repetitions.
                self._record_stage(session, REST, media,
                                   lambda: self.play_collection_visual(media, active=False))
                media[0].reset()

            file_prefix = Path(self.output_folder, "C_" + str(media[2]) + "_R_" + str(media[3]) + "_").absolute().as_posix()
            if session is None:
                logged, dropped = self._record_rep_to_own_log(media, file_prefix)
                self.verify_rep_logged(logged, file_prefix, dropped)
            else:
                self._record_stage(session, PROMPT, media,
                                   lambda: self.play_collection_visual(media, active=True))

            last_rep = media[3]
            self.i = self.i+1
            is_final_media = self.i == len(media_list)
            if is_final_media:
                # At the end of the list, so we must be finished a rep
                rep_is_finished = True
                current_rep = self.num_reps - 1
            else:
                # Check if we've finished a rep
                current_rep = media_list[self.i][3]
                rep_is_finished = last_rep != current_rep

            # pause / redo goes here!
            if rep_is_finished  or (not self.auto_advance):
                self._await_advance(session, media)
                if not is_final_media:
                    dpg.set_value('__dc_rep', value=f"Repetition {media_list[self.i][3] + 1} of {self.num_reps}")

    def _await_advance(self, session, media):
        """Hold on the redo/continue buttons, recording how long the hold lasted.

        This is the one stage nobody chose the length of, and it can be minutes
        long. It is annotated for exactly that reason: a stretch of recording
        that no prompt accounts for is otherwise indistinguishable from a rest
        that ran long, and something reading the session end to end has to know
        which it was.
        """
        def hold():
            self.advance = False
            dpg.show_item(item="__dc_pause_note")
            dpg.show_item(item="__dc_redo_button")
            dpg.show_item(item="__dc_continue_button")
            while not self.advance:
                time.sleep(0.1)
                dpg.configure_app(manual_callback_management=True)
                jobs = dpg.get_callback_queue()
                dpg.run_callbacks(jobs)
            dpg.configure_app(manual_callback_management=False)

        self._record_stage(session, PAUSE, media, hold)

    def _record_rep_to_own_log(self, media, file_prefix):
        """Record one repetition into files of its own, cutting the stream as it runs.

        Kept for ``continuous_log=False``. The cut lands on arrival time, which
        is not when the samples were produced -- see _begin_session_log.
        """
        self.clear_rep_files(file_prefix)
        self.online_data_handler.reset()
        # Arm the logger before the prompt goes up: start_log only returns
        # once shared memory is being watched, so the file covers the whole
        # collection phase and nothing else.
        self.online_data_handler.start_log(file_path=file_prefix, timestamps=self.timestamps)
        try:
            self.play_collection_visual(media, active=True)
        finally:
            logged = self.online_data_handler.stop_log()
        return logged, dict(getattr(self.online_data_handler, 'log_dropped', {}))

    def _begin_session_log(self, media_list):
        """Open one continuous recording covering the whole collection.

        Starting and stopping a log around each repetition cuts the stream on
        the time samples *arrived*, and a device does not deliver on a schedule:
        packets come in bursts, tens to hundreds of milliseconds apart. So the
        tail of a repetition is still in flight when its log stops -- never
        written, then wiped by the reset at the top of the next repetition --
        while a burst landing early in a window carries samples produced during
        the preceding rest. Neither of those is lost data. Both are samples
        credited to the wrong repetition, and both disappear if the stream is
        never cut while it is running.

        So it is not cut at all here, and it is not cut down either: the rests
        and the pauses between repetitions stay in the file with everything else,
        because a model reading a longer stretch than one repetition needs the
        stretch between them to exist. What each part of the recording *was* is
        recorded alongside it, in the annotation file this opens; the repetitions
        are taken out of it afterwards, by sample index, in
        :meth:`split_session_into_reps`.
        """
        prefix = Path(self.output_folder, SESSION_PREFIX).absolute().as_posix()
        # The logger appends, and the split below overwrites only the
        # repetitions this run actually collects. Both would otherwise inherit a
        # previous collection's data into this one.
        for mod in self.online_data_handler.modalities:
            path = Path(prefix + mod + ".csv")
            if path.exists():
                path.unlink()
        for media in media_list:
            self.clear_rep_files(
                Path(self.output_folder, f"C_{media[2]}_R_{media[3]}_").absolute().as_posix())

        annotations = SessionAnnotations(
            folder=Path(self.output_folder).absolute(),
            prefix=SESSION_PREFIX,
            modalities=list(self.online_data_handler.modalities),
            class_map={str(media[2]): media[1] for media in media_list},
            timestamps=self.timestamps,
            created=datetime.now().isoformat())
        session = SessionLog(annotations=annotations)
        self.online_data_handler.reset()
        # Timestamps are not optional here, whatever self.timestamps says: they
        # are the only thing tying a stage's wall-clock boundaries to a position
        # in the stream. They are dropped again when the repetition files are
        # written, if that is how they were asked for.
        self.online_data_handler.start_log(file_path=prefix, timestamps=True)
        now = time.time()
        session.last_movement = {mod: now for mod in self.online_data_handler.modalities}
        annotations.save()
        return session

    def _record_stage(self, session, kind, media, play):
        """Run one stage of the collection and mark where it falls in the recording.

        Nothing starts or stops -- ``play`` runs against a logger that has been
        running since the collection began and will keep running after it. The
        sample counters are read either side only so a stream that has died is
        still caught while the repetition can be redone; what a stage *contains*
        is decided later, once the sample clock has been measured.

        The annotations are written out after every stage rather than at the end,
        so a collection that is interrupted -- or that dies on the stream check
        below -- still leaves a file describing everything that reached the disk
        before it stopped.
        """
        if session is None:
            play()
            return None
        odh = self.online_data_handler
        before = odh.log_counts()
        t_start = time.time()
        try:
            play()
        finally:
            t_end = time.time()
            after = odh.log_counts()
            stage = self._append_stage(session, kind, media, t_start, t_end)
        self.verify_stream_alive(session, before, after, t_end)
        return stage

    def _append_stage(self, session, kind, media, t_start, t_end):
        """Add one stage to the annotations and save them."""
        # A pause is only ever bounded by someone pressing a button, so it is the
        # one stage whose asked duration is its observed one.
        duration = media[-1] if kind == PROMPT else (
            self.current_rest_time if kind == REST else t_end - t_start)
        factors = dict(self.current_plan.factors) if self.current_plan else {}
        stage = Stage(kind=kind, label=media[1] if kind != PAUSE else "",
                      motion=media[2] if kind != PAUSE else None, rep=media[3],
                      t_start=t_start, t_end=t_end, duration=duration,
                      factors=factors)
        self._supersede_earlier_attempts(session, stage)
        session.annotations.stages.append(stage)
        session.annotations.save()
        return stage

    @staticmethod
    def _supersede_earlier_attempts(session, stage):
        """Mark a redone stage's earlier takes, and number this one.

        A redo is collected under the same name, so the last take is the one that
        gets cut -- the same rule :meth:`clear_rep_files` enforces when each
        repetition writes its own log. The earlier takes are still in the
        recording, though, and reading the session continuously runs straight
        through them, so they stay in the annotations marked for what they are
        rather than being deleted from the record.

        Pauses are exempt: two pauses after the same repetition are two different
        stretches of recording, not two attempts at one.
        """
        if stage.kind == PAUSE:
            return
        earlier = [s for s in session.annotations.stages
                   if s.kind == stage.kind and s.motion == stage.motion
                   and s.rep == stage.rep]
        for previous in earlier:
            previous.superseded = True
        stage.attempt = len(earlier)

    def verify_stream_alive(self, session, before, after, now):
        """Confirm every modality is still delivering, across repetitions.

        Deliberately not a per-repetition zero check. A modality that arrives in
        bursts can miss a short repetition entirely while the link is perfectly
        healthy, so what is watched here is how long a modality has been silent,
        not whether it was silent for one repetition.
        """
        for mod in self.online_data_handler.modalities:
            if after.get(mod, 0) > before.get(mod, 0):
                session.last_movement[mod] = now
        silent = {mod: now - since for mod, since in session.last_movement.items()
                  if now - since > STREAM_SILENCE_LIMIT}
        if silent:
            detail = ", ".join(f"{mod} ({gap:.1f}s)" for mod, gap in sorted(silent.items()))
            raise ConnectionError(
                f'Attempting to store data, but no samples have arrived for {detail}, '
                'suggesting that the data stream from the device has been interrupted. '
                'Please check the device connection and verify that previous files are '
                'not missing samples.')

    def _finish_session_log(self, session):
        """Close the continuous recording, index it, and cut the repetitions out of it."""
        self.online_data_handler.stop_log()
        dropped = {mod: n for mod, n
                   in getattr(self.online_data_handler, 'log_dropped', {}).items() if n}
        if dropped:
            detail = ", ".join(f"{n} {mod}" for mod, n in sorted(dropped.items()))
            print(f'LibEMG -> DataCollectionPanel (shared memory overwrote {detail} samples '
                  f'before the logger could read them. The gaps are not marked in '
                  f'{session.prefix}*.csv and nothing below can see them, so the repetitions '
                  f'cut from this session are misaligned after the first gap. Recollect.)')
        if not session.annotations.stages:
            return
        self.split_session_into_reps(session)

    def split_session_into_reps(self, session):
        """Index the continuous recording and write each repetition's files from it.

        Within one acquisition the stream is contiguous and ordered, so a
        sample's index *is* a clock. The arrival timestamps are used once, to
        anchor that clock across the whole session, which is what keeps one late
        burst from landing on a stage boundary.

        The indices go into the annotation file before anything is cut, and the
        cut is then a slice at those indices. So what is written here can be
        written again later -- with the rests included, with a lead-in either
        side, with a class left out -- by loading the same file and calling
        :meth:`libemg.session.SessionAnnotations.cut` again.
        """
        annotations = session.annotations
        unmeasured = annotations.resolve()
        annotations.save()
        for mod in unmeasured:
            source = annotations.source_path(mod)
            if not source.exists():
                print(f'LibEMG -> DataCollectionPanel (no {mod} data was recorded this '
                      f'session, so no {mod} files were written.)')
            else:
                print(f'LibEMG -> DataCollectionPanel ({mod} holds too few samples to '
                      f'recover a sample rate from, so its repetitions were not cut.)')

        written = annotations.cut()
        for mod, clock in sorted(annotations.clocks.items()):
            mine = [cut for cut in written if cut.modality == mod]
            if not mine:
                continue
            wanted = [cut.wanted for cut in mine]
            print(f'LibEMG -> DataCollectionPanel ({mod}: {clock.n_samples} samples at a '
                  f'measured {clock.fs:.1f} Hz, cut into {len(mine)} repetitions of '
                  f'{min(wanted)}-{max(wanted)} samples. Delivery latency varied by '
                  f'{1e3 * clock.lag_spread:.0f} ms, which is how far each boundary would '
                  f'have moved had the stream been cut as it ran.)')
            short = [cut.path.name for cut in mine if cut.short]
            if short:
                print(f'LibEMG -> DataCollectionPanel ({mod}: {len(short)} repetitions ran '
                      f'past the end of the recording and are short: {", ".join(short[:5])}'
                      f'{" ..." if len(short) > 5 else ""}.)')
        self.report_fatigue(annotations)
        print(f'LibEMG -> DataCollectionPanel (the whole session, rests and pauses '
              f'included, is still in {session.prefix}*.csv; {annotations.path.name} says '
              f'where every stage of it sits. Re-cut it with '
              f'libemg.session.SessionAnnotations.load(...).cut(...).)')
        return written

    def report_fatigue(self, annotations, modality='emg'):
        """Say whether the last repetitions were the same task as the first.

        The median frequency of the EMG spectrum falls as a muscle fatigues, so
        comparing the first repetition against the last says whether the session
        ran long enough to change what was being measured. Nothing is done about
        it -- the data is what it is -- but it belongs in the record, and it is
        the thing to shorten next time.

        Returns
        ----------
        shift: float or None
            Fractional change in median frequency, negative for the shift a
            fatiguing muscle makes. None when the session is too short, or has
            too few repetitions, to compare.
        """
        from libemg.signal_quality import fatigue_shift

        if modality not in annotations.clocks:
            return None
        prompts = annotations.select(kinds=(PROMPT,))
        reps = sorted({stage.rep for stage in prompts})
        if len(reps) < 2:
            return None
        try:
            _, samples = annotations.read(modality)
        except (OSError, ValueError):
            return None

        def gather(rep):
            spans = [stage.span(modality) for stage in prompts if stage.rep == rep]
            chunks = [samples[start:end] for start, end in spans if end > start]
            return np.vstack(chunks) if chunks else np.empty((0, samples.shape[1]))

        early, late = gather(reps[0]), gather(reps[-1])
        if not len(early) or not len(late):
            return None
        shift, message = fatigue_shift(early, late, annotations.clocks[modality].fs)
        if message:
            print(f'LibEMG -> DataCollectionPanel ({message})')
        else:
            print(f'LibEMG -> DataCollectionPanel ({modality} median frequency moved '
                  f'{100 * shift:+.0f}% from the first repetition to the last, which is '
                  f'not enough to call fatigue.)')
        return shift

    def redo_collection_callback(self):
        if self.auto_advance:
            self.i      = self.i - self.num_motions
        else:
            self.i      = self.i - 1
        dpg.hide_item(item="__dc_pause_note")
        dpg.hide_item(item="__dc_redo_button")
        dpg.hide_item(item="__dc_continue_button")
        self.advance = True

    def continue_collection_callback(self):
        dpg.hide_item(item="__dc_pause_note")
        dpg.hide_item(item="__dc_redo_button")
        dpg.hide_item(item="__dc_continue_button")
        self.advance = True

    def play_collection_visual(self, media, active=True):
        # The prompt says which level of each design factor this repetition is
        # under -- "Wrist Flexion (elevated)" -- because a factor the subject is
        # not told about is one they will not hold.
        label = self.current_plan.label if self.current_plan else media[1]
        # Dimmed during the rest, with the picture greyed out, so what is coming
        # can be read without being mistaken for the cue to start.
        if active:
            timer_duration = media[-1]
            dpg.set_value("__dc_prompt", value=label)
            dpg.configure_item("__dc_prompt", color=style.TEXT)
        else:
            timer_duration = self.current_rest_time
            dpg.set_value("__dc_prompt", value="Up next: "+label)
            dpg.configure_item("__dc_prompt", color=style.MUTED)


        texture = media[0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height, grayscale=not(active))
        set_texture("__dc_collection_visual", texture, self.video_player_width, self.video_player_height)
        # initialize motion and frame timers
        motion_timer = time.perf_counter_ns()
        while (time.perf_counter_ns() - motion_timer)/1e9 < timer_duration:
            time.sleep(1/media[0].fps) # never refresh faster than media fps
            # update visual
            media[0].advance_to((time.perf_counter_ns() - motion_timer)/1e9)
            texture = media[0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height, grayscale=not(active))
            set_texture("__dc_collection_visual", texture, self.video_player_width, self.video_player_height)
            # update progress bar
            progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*timer_duration))
            dpg.set_value("__dc_progress", value = progress)        
    
    def clear_rep_files(self, file_prefix):
        """Remove a previous take of this rep, since the logger appends to its files."""
        for mod in self.online_data_handler.modalities:
            path = Path(file_prefix + mod + ".csv")
            if path.exists():
                path.unlink()

    def verify_rep_logged(self, logged, file_prefix='', dropped=None):
        """Confirm every modality contributed samples to the rep that just finished.

        A rep can also come back complete-looking but short: if the logger is
        held off the shared memory long enough for the device to lap the buffer,
        the samples it missed are simply absent from the file, with the samples
        either side of the gap written adjacent to each other. Nothing in the
        file marks that, so it is called out here while the rep can still be
        redone.
        """
        empty = [mod for mod in self.online_data_handler.modalities if not logged.get(mod, 0)]
        if empty:
            raise ConnectionError(f'Attempting to store data, but received 0 samples during repetition for {", ".join(empty)}, suggesting that the data stream from the device has been interrupted. Please check the device connection and verify that previous files are not missing samples.')
        if dropped is None:
            dropped = getattr(self.online_data_handler, 'log_dropped', {})
        incomplete = {mod: n for mod, n in dropped.items() if n}
        if incomplete:
            detail = ", ".join(f"{n} {mod}" for mod, n in sorted(incomplete.items()))
            print(f'LibEMG -> DataCollectionPanel (this repetition is missing samples: {detail} '
                  f'were overwritten in shared memory before they could be written to '
                  f'{file_prefix}*.csv. The gap is not marked in the file. Redo the repetition '
                  f'if the recording has to be continuous.)')

    def visualize_callback(self):
        self.visualization_panel = VisualizationPanel(self.online_data_handler,
                                                      num_samples=self.visualize_num_samples)
        self.visualization_panel.spawn_window()


def _media_in(folder):
    """The prompts the collection would find in ``folder``, or None if it can't be read."""
    try:
        return [file for file in os.listdir(folder) if file.endswith(MEDIA_SUFFIXES)]
    except OSError:
        return None


@contextmanager
def _centred():
    """A line whose contents sit in the middle of the window, however wide they are.

    A table with a column either side that takes up the slack and one in the
    middle that fits what it holds. The middle one is measured again every
    frame, so a prompt that changes length stays centred without anything here
    having to know how wide the font draws it.
    """
    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingFixedFit,
                   borders_innerH=False, borders_outerH=False, borders_innerV=False,
                   borders_outerV=False):
        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
        dpg.add_table_column(width_fixed=True)
        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
        with dpg.table_row():
            dpg.add_spacer()
            with dpg.group() as cell:
                yield cell
            dpg.add_spacer()
