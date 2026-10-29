"""Mark where the contractions are in a dataset of recordings, file by file.

Two windows. The index lists every recording under a dataset root -- one
``subject/task/aff_0.txt`` per row -- with a tick or a cross for whether it has
been annotated yet, and a progress bar over the whole dataset. Clicking a row
opens the recording in the annotation window, where the contraction regions are
marked on the plot and saved beside the file (see :mod:`libemg.annotations`).
Opening a file that already has annotations, from this session or an earlier
one, shows them again.

Nothing here decides what a contraction is; the panel only makes marking them
quick. The plot is decimated to what the screen can show and re-drawn as it is
zoomed, so a recording of a few hundred thousand samples pans as smoothly as a
short one, and the RMS envelope view makes the onset easier to see than the raw
signal does.
"""

import os
import threading
import traceback
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from libemg._gui import _style as style
from libemg.annotations import (DATASET_DESCRIPTION, ContractionAnnotation,
                                DatasetIndex, _natural_key, load_emg_text)

INDEX_WINDOW = "__annot_index_window"
PLOT_WINDOW = "__annot_plot_window"
CONFIRM_WINDOW = "__annot_confirm_window"
HANDLERS = "__annot_handlers"
INDEX_TABLE = "__annot_index_table"
TABLE_HOLDER = "__annot_table_holder"
FILTERS = "__annot_filters"
INDEX_STATUS = "__annot_index_status"
PROGRESS = "__annot_progress"
PLOT = "__annot_plot"
X_AXIS = "__annot_x_axis"
Y_AXIS = "__annot_y_axis"
REGION_TABLE = "__annot_region_table"
PLOT_TITLE = "__annot_plot_title"
PLOT_STATUS = "__annot_plot_status"
PENDING_LINE = "__annot_pending_line"
ADD_BUTTON = "__annot_add_button"
BUTTON_ROW_THEME = "__annot_button_row_theme"
NEXT_BUTTON = "__annot_next_unannotated"
FOLDER_DIALOG = "__annot_folder_dialog"

ALL = "All"
STATUS_FILTERS = (ALL, "Annotated", "Not annotated")
DISPLAY_MODES = ("Raw", "RMS envelope")
#: Window of the RMS envelope view, in seconds.
ENVELOPE_SECONDS = 0.1
#: How far the mouse may move between press and release and still be a click
#: rather than a pan, in pixels.
CLICK_TOLERANCE = 4
#: Filters on a line of the index, each a label and its list.
FILTER_COLUMNS = 3
#: Height of everything under the plot: the region controls and table, the
#: notes, and the save row. The plot takes the rest of the window.
EDITOR_FOOTER = 250

GOOD = (90, 200, 120, 255)
BAD = (220, 80, 80, 255)
REGION_COLOR = (90, 200, 120, 255)
PENDING_COLOR = (240, 200, 80, 255)
OPEN_ROW = (50, 90, 140, 255)


def _title(name):
    """``"task_name"`` -> ``"Task name"``, for a column or filter label."""
    return name.replace("_", " ").capitalize()


def _tooltip(item, text):
    with dpg.tooltip(item):
        dpg.add_text(text, wrap=340)


def _build_button_row_theme():
    """Plain buttons padded like the coloured ones, for a row that has both.

    A coloured button is padded wider than a plain one, so a row mixing them
    looks uneven without this. Made once per DearPyGui context.
    """
    if dpg.does_alias_exist(BUTTON_ROW_THEME):
        return
    with dpg.theme(tag=BUTTON_ROW_THEME):
        for enabled in (True, False):
            with dpg.theme_component(dpg.mvButton, enabled_state=enabled):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 14, 4,
                                    category=dpg.mvThemeCat_Core)


def decimate_for_display(signal, start, stop, max_points):
    """Reduce ``signal[start:stop]`` to at most ``max_points`` rows per channel.

    Each bin keeps its minimum and maximum, so a burst of activity narrower
    than a pixel is still drawn at its full height -- plain subsampling would
    skip it, and a contraction would vanish when zoomed out.

    Parameters
    ----------
    signal: ndarray
        ``(samples, channels)``.
    start, stop: int
        The sample range to show.
    max_points: int
        Rows to return at most.

    Returns
    ----------
    x: ndarray
        Sample index of each returned row.
    y: ndarray
        ``(rows, channels)``.
    """
    start = max(int(start), 0)
    stop = min(int(stop), len(signal))
    segment = signal[start:stop]
    count = len(segment)
    if count <= max_points:
        return np.arange(start, stop, dtype=float), segment
    bins = max(max_points // 2, 1)
    edges = np.linspace(0, count, bins + 1).astype(int)[:-1]
    lows = np.minimum.reduceat(segment, edges, axis=0)
    highs = np.maximum.reduceat(segment, edges, axis=0)
    y = np.empty((2 * bins, segment.shape[1]), dtype=segment.dtype)
    y[0::2], y[1::2] = lows, highs
    x = np.repeat(edges + start, 2).astype(float)
    return x, y


def moving_rms(signal, window):
    """Centred moving RMS of each channel, the same length as ``signal``."""
    window = max(int(window), 1)
    centred = signal - signal.mean(axis=0)
    squares = np.cumsum(np.vstack([np.zeros((1, signal.shape[1])), centred ** 2]), axis=0)
    half = window // 2
    index = np.arange(len(signal))
    lo = np.clip(index - half, 0, len(signal))
    hi = np.clip(index - half + window, 0, len(signal))
    return np.sqrt((squares[hi] - squares[lo]) / np.maximum(hi - lo, 1)[:, None])


class AnnotationPanel:
    """Annotate contraction regions across a dataset of recordings.

    Parameters
    ----------
    dataset_folder: str (optional)
        The dataset root to open on. Which files under it are recordings, and
        what their folders mean, comes from its ``dataset_description.yaml``
        -- see :class:`libemg.annotations.DatasetIndex`.
    sampling_rate: float (optional), default=2000
        What the time axis assumes. The files do not record it; annotations
        are saved in samples, so it only changes what the axis reads.
    spec: dict or None (optional)
        ``data_files``, ``exclude_files`` and ``path_template`` overriding the
        description's ``annotation`` block, for a dataset whose description
        cannot be edited.
    width, height: int (optional)
        The space the two windows share.

    Examples
    ---------
    >>> panel = AnnotationPanel("B:/data/Shriner_Participants")
    >>> panel.spawn_window()
    """

    #: The window this panel owns, so the GUI can tell whether it is open.
    window_tag = INDEX_WINDOW

    def __init__(self, dataset_folder="", sampling_rate=2000.0,
                 spec=None, width=1600, height=900):
        self.dataset_folder = str(dataset_folder or "")
        self.sampling_rate = float(sampling_rate)
        self.spec = dict(spec or {})
        self._filter_fields = []   # fields with a filter combo
        self.width, self.height = width, height

        self.index = None
        self.visible = []          # entries the table is showing
        self._rows = {}            # entry path -> (drawlist, selectable, row)
        self._table_fields = []    # fields with a table column
        self._lit_row = None       # table row highlighted as open
        self.current = None        # the entry in the annotation window
        self.data = None           # (samples, channels), as loaded
        self._display = None       # what is drawn: normalised and stacked
        self.regions = []          # [start, stop] in samples, as edited
        self.notes = ""
        self.dirty = False
        self._pending_start = None # first click of a region being added
        self._adding = False
        self._press = None         # (screen pos, plot x) of the last press
        self._rendered = None      # sample range the series currently show
        self._after_confirm = None
        self._built_regions = 0    # drag rects currently on the plot
        # set_axis_limits pins an axis until it is released. False, or True
        # for "release next frame", then "pending" for the frame after.
        self._release_x = False

        # Loading a recording is a second or so of parsing, so it runs off the
        # render thread and posts back here. Only poll() may touch the window.
        self._messages = []
        self._lock = threading.Lock()
        self._load_generation = 0

    # ==================================================================
    # windows
    # ==================================================================
    def spawn_window(self):
        """Build the index window, and load the dataset if one is named."""
        self.cleanup()
        _build_button_row_theme()
        index_width = min(720, max(self.width // 2 - 40, 480))
        # Top-level, as file dialogs have to be. Picking a folder loads it:
        # choosing one is the only reason to open this.
        dpg.add_file_dialog(directory_selector=True, show=False, modal=True,
                            tag=FOLDER_DIALOG, width=760, height=460,
                            default_path=self.dataset_folder or os.getcwd(),
                            callback=self._folder_chosen)
        with dpg.window(label="Annotation Index", tag=INDEX_WINDOW,
                        width=index_width, height=self.height - 80, pos=(10, 30),
                        on_close=lambda: self.cleanup()):
            with style.form(label_width=70):
                with style.row("Dataset", help=f"The folder holding {DATASET_DESCRIPTION}. "
                               "Which files under it are recordings, and what their "
                               "folders mean, comes from that file's annotation block."):
                    dpg.add_input_text(tag="__annot_folder",
                                       default_value=self.dataset_folder,
                                       hint=f"Folder holding {DATASET_DESCRIPTION}",
                                       width=-170, on_enter=True,
                                       callback=lambda: self.load_dataset())
                    style.button("Browse...", width=85,
                                 callback=lambda: dpg.show_item(FOLDER_DIALOG))
                    style.button("Load", width=70, callback=lambda: self.load_dataset())
                with style.row(""):
                    dpg.add_checkbox(label=f"Only subjects listed as used in {DATASET_DESCRIPTION}",
                                     tag="__annot_used_only", default_value=True,
                                     callback=lambda: self.load_dataset())
            style.note("", tag=INDEX_STATUS, wrap=index_width - 30)
            with dpg.group(horizontal=True) as progress_row:
                dpg.add_progress_bar(tag=PROGRESS, width=-215, height=29,
                                     overlay="No dataset loaded")
                style.button("Open next unannotated", "primary", width=-1,
                             tag=NEXT_BUTTON, enabled=False,
                             callback=lambda: self.open_next(unannotated=True))
            style.section("Recordings")
            # A filter per field the dataset's path template names, and one
            # for whether it is annotated, so they are built once the dataset
            # is loaded.
            dpg.add_group(tag=FILTERS)
            dpg.add_group(tag=TABLE_HOLDER)
        dpg.bind_item_theme(progress_row, BUTTON_ROW_THEME)
        self._build_table([])

        with dpg.handler_registry(tag=HANDLERS):
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=lambda: self._on_mouse_press())
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=lambda: self._on_mouse_release())
            for key in (dpg.mvKey_R, dpg.mvKey_A):
                dpg.add_key_press_handler(key, callback=lambda: self._on_key_add())
            dpg.add_key_press_handler(dpg.mvKey_Escape,
                                      callback=lambda: self._set_adding(False))

        if self.dataset_folder:
            self.load_dataset()
        else:
            self._say_index("Name the dataset folder and press Load.")
            self._build_filters()
        return self

    def _spawn_plot_window(self):
        if dpg.does_item_exist(PLOT_WINDOW):
            dpg.show_item(PLOT_WINDOW)
            return
        index_width = dpg.get_item_width(INDEX_WINDOW) if dpg.does_item_exist(INDEX_WINDOW) else 700
        # Clear of the viewport's right edge, which the window used to run
        # under, taking the ends of its rows with it.
        width = max(self.width - index_width - 46, 600)
        height = self.height - 80
        with dpg.window(label="Annotate Contractions", tag=PLOT_WINDOW,
                        width=width, height=height, pos=(index_width + 20, 30),
                        on_close=lambda: self._set_adding(False)):
            with dpg.group(horizontal=True) as navigation:
                style.button(f"{style.arrow('left', '<')}  Previous",
                             callback=lambda: self.open_next(step=-1))
                style.button(f"Next  {style.arrow('right', '>')}",
                             callback=lambda: self.open_next(step=1))
                dpg.add_spacer(width=4)
                title = dpg.add_text("", tag=PLOT_TITLE, color=style.HEADING)
                if style.font("bold") is not None:
                    dpg.bind_item_font(title, style.font("bold"))
            with dpg.group(horizontal=True):
                dpg.add_text("View")
                dpg.add_combo(DISPLAY_MODES, default_value=DISPLAY_MODES[0],
                              tag="__annot_view", width=140,
                              callback=lambda: self._redraw_signal(refit=False))
                dpg.add_spacer(width=6)
                rate = dpg.add_text("Sampling rate")
                _tooltip(rate, "What the time axis assumes; the files do not record it. "
                               "Regions are saved in samples, so this changes only what "
                               "the axis and the region times read.")
                dpg.add_input_float(tag="__annot_fs",
                                    default_value=self.sampling_rate, width=90,
                                    step=0, format="%.1f", on_enter=True,
                                    callback=lambda: self._sampling_rate_changed())
                style.unit("Hz")
                dpg.add_spacer(width=6)
                fit = style.button("Show all", callback=lambda: self._fit())
                _tooltip(fit, "Zoom out to the whole recording.")
            style.note("Scroll to zoom time, drag to pan. Press Add region (or R or A over "
                       "the plot), then click where the contraction starts and where it "
                       "ends. Esc cancels.", wrap=width - 30)
            with dpg.plot(tag=PLOT, height=-EDITOR_FOOTER, width=-1,
                          no_box_select=True, no_menus=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=X_AXIS)
                dpg.add_plot_axis(dpg.mvYAxis, tag=Y_AXIS, no_gridlines=True)
            with dpg.group(horizontal=True) as region_row:
                style.button("Add region (R)", tag=ADD_BUTTON,
                             callback=lambda: self._set_adding(not self._adding))
                dpg.add_checkbox(label="Drag to adjust regions", tag="__annot_drag",
                                 callback=lambda: self._rebuild_regions())
                dpg.add_spacer(width=6)
                style.button("Clear regions", "subtle", callback=lambda: self._clear_regions())
                revert = style.button("Revert", "subtle",
                                      callback=lambda: self.open(self.current, force=True))
                _tooltip(revert, "Drop the unsaved changes and show what was last saved.")
            with dpg.table(tag=REGION_TABLE, header_row=True, borders_innerH=True,
                           borders_outerH=True, scrollY=True, height=110,
                           row_background=True, policy=dpg.mvTable_SizingStretchProp):
                for label in ("#", "Start (s)", "Stop (s)", "Duration (s)", ""):
                    dpg.add_table_column(label=label)
            with style.form(label_width=70):
                with style.row("Notes", help="Saved with the regions, for whoever "
                               "reads this recording next."):
                    dpg.add_input_text(tag="__annot_notes", width=-1,
                                       hint="Anything worth knowing about this recording",
                                       callback=lambda: self._notes_changed())
            with dpg.group(horizontal=True) as save_row:
                style.button("Save", "primary", width=90, callback=lambda: self.save())
                style.button("Save and open next unannotated",
                             callback=lambda: self.save(then_next=True))
                dpg.add_spacer(width=4)
                style.note("", tag=PLOT_STATUS)
        for row in (navigation, region_row, save_row):
            dpg.bind_item_theme(row, BUTTON_ROW_THEME)

    def cleanup(self):
        """Remove everything this panel built."""
        self._load_generation += 1
        for tag in (CONFIRM_WINDOW, PLOT_WINDOW, INDEX_WINDOW, HANDLERS, FOLDER_DIALOG):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            if dpg.does_alias_exist(tag):
                dpg.remove_alias(tag)
        self._rows = {}
        self._lit_row = None

    # ==================================================================
    # index
    # ==================================================================
    def _folder_chosen(self, sender, app_data):
        folder = (app_data or {}).get("file_path_name", "")
        if folder:
            dpg.set_value("__annot_folder", folder)
            self.load_dataset()

    def load_dataset(self):
        """Scan the dataset folder into the index."""
        folder = dpg.get_value("__annot_folder").strip().strip('"')
        self.dataset_folder = folder
        try:
            self.index = DatasetIndex.scan(folder, self.spec,
                                           used_only=bool(dpg.get_value("__annot_used_only")))
        except Exception as error:
            self.index = None
            self._say_index(f"Could not load the dataset: {error}", "error")
            self._build_filters()
            self._build_table([])
            self._rebuild_table()
            return
        index = self.index
        message = f"{len(index.entries)} recordings found."
        if index.skipped_subjects:
            message += (f" {len(index.skipped_subjects)} subjects are left out because"
                        f" {DATASET_DESCRIPTION} does not list them as used.")
        if index.unmatched:
            message += (f" {len(index.unmatched)} files do not fit the path template"
                        f" {index.spec.get('path_template')!r} and are left out,"
                        f" e.g. {index.unmatched[0]}.")
        self._say_index(message, "warn" if index.unmatched or not index.entries else "info")
        self._build_filters()
        self._build_table(index.field_names)
        self._rebuild_table()

    def _build_filters(self):
        """A combo per field that has more than one value in the dataset, then Status.

        Laid out a few to a line, each with its name to its left. Status keeps
        what it was set to across a reload; the field filters start over,
        since the fields may not be the same ones.
        """
        status = (dpg.get_value("__annot_filter_status")
                  if dpg.does_item_exist("__annot_filter_status") else ALL)
        dpg.delete_item(FILTERS, children_only=True)
        self._filter_fields = []
        filters = []
        for name in (self.index.field_names if self.index is not None else []):
            values = {e.fields.get(name, "") for e in self.index.entries}
            if len(values) < 2:
                continue
            filters.append((_title(name), f"__annot_filter_{name}",
                            [ALL] + sorted(values, key=_natural_key), ALL))
            self._filter_fields.append(name)
        filters.append(("Status", "__annot_filter_status", list(STATUS_FILTERS),
                        status or ALL))

        with dpg.table(parent=FILTERS, header_row=False, resizable=False,
                       policy=dpg.mvTable_SizingStretchProp, borders_innerH=False,
                       borders_outerH=False, borders_innerV=False, borders_outerV=False):
            for _ in range(FILTER_COLUMNS):
                dpg.add_table_column(width_fixed=True)
                dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
            for start in range(0, len(filters), FILTER_COLUMNS):
                with dpg.table_row():
                    for label, tag, choices, value in filters[start:start + FILTER_COLUMNS]:
                        dpg.add_text(label)
                        dpg.add_combo(choices, default_value=value, tag=tag, width=-1,
                                      callback=lambda: self._rebuild_table())

    def _build_table(self, field_names):
        """The index table, with a column per field and one for the file name."""
        if dpg.does_item_exist(INDEX_TABLE):
            dpg.delete_item(INDEX_TABLE)
        self._table_fields = list(field_names)
        with dpg.table(tag=INDEX_TABLE, parent=TABLE_HOLDER, header_row=True,
                       clipper=True, row_background=True, borders_innerH=True,
                       borders_outerH=True, scrollY=True, freeze_rows=1,
                       policy=dpg.mvTable_SizingStretchProp, height=-1):
            dpg.add_table_column(label="", width_fixed=True, init_width_or_weight=22)
            for name in self._table_fields:
                dpg.add_table_column(label=_title(name),
                                     init_width_or_weight=3.0 if name == "subject" else 1.2)
            dpg.add_table_column(label="File", init_width_or_weight=1.2)

    def _filtered(self):
        if self.index is None:
            return []
        wanted = {name: dpg.get_value(f"__annot_filter_{name}") for name in self._filter_fields}
        wanted = {name: value for name, value in wanted.items() if value not in (ALL, None)}
        status = dpg.get_value("__annot_filter_status")
        chosen = []
        for entry in self.index.entries:
            if any(entry.fields.get(name, "") != value for name, value in wanted.items()):
                continue
            if status == STATUS_FILTERS[1] and not entry.is_annotated:
                continue
            if status == STATUS_FILTERS[2] and entry.is_annotated:
                continue
            chosen.append(entry)
        return chosen

    def _rebuild_table(self):
        dpg.delete_item(INDEX_TABLE, children_only=True, slot=1)
        self._rows = {}
        self._lit_row = None
        self.visible = self._filtered()
        for row, entry in enumerate(self.visible):
            with dpg.table_row(parent=INDEX_TABLE):
                # The empty text sets the row's text baseline, which the
                # selectable after it lines up to as the other cells' text
                # does; with only the mark before it, the first column sat a
                # few pixels above the rest of its row.
                with dpg.group(horizontal=True, horizontal_spacing=0):
                    dpg.add_text("")
                    drawlist = dpg.add_drawlist(width=14, height=25)
                self._draw_status(drawlist, entry.is_annotated)
                cells = [entry.fields.get(n, "") for n in self._table_fields]
                cells.append(entry.path.name)
                # The first cell is a row-wide selectable, so a click anywhere
                # on the row opens it.
                selectable = dpg.add_selectable(
                    label=cells[0], span_columns=True, user_data=entry,
                    callback=lambda sender, value, user: self._row_clicked(sender, user))
                for cell in cells[1:]:
                    dpg.add_text(cell)
            self._rows[entry.path] = (drawlist, selectable, row)
        self._highlight(self.current)
        self._refresh_progress()

    @staticmethod
    def _draw_status(drawlist, annotated):
        dpg.delete_item(drawlist, children_only=True)
        # Centred on the line of text beside it, which sits a frame's padding
        # below the top of the cell.
        if annotated:
            dpg.draw_polyline([(2, 13), (6, 17), (12, 9)], color=GOOD,
                              thickness=2, parent=drawlist)
        else:
            dpg.draw_line((3, 9), (11, 17), color=BAD, thickness=2, parent=drawlist)
            dpg.draw_line((11, 9), (3, 17), color=BAD, thickness=2, parent=drawlist)

    def _refresh_progress(self):
        if self.index is None or not self.index.entries:
            dpg.set_value(PROGRESS, 0.0)
            dpg.configure_item(PROGRESS, overlay="No recordings")
            style.set_enabled(NEXT_BUTTON, False)
            return
        done, total = self.index.progress()
        # Blue means "press this next", which is only true while there is
        # something left to annotate.
        style.set_enabled(NEXT_BUTTON, done < total)
        shown = len(self.visible)
        shown_done = sum(e.is_annotated for e in self.visible)
        overlay = f"{done} / {total} annotated ({100 * done / total:.0f}%)"
        if shown != total:
            overlay += f"   -   {shown_done} / {shown} shown"
        dpg.set_value(PROGRESS, done / total)
        dpg.configure_item(PROGRESS, overlay=overlay)

    def _refresh_row(self, entry):
        row = self._rows.get(entry.path)
        if row is not None and dpg.does_item_exist(row[0]):
            self._draw_status(row[0], entry.is_annotated)
        self._refresh_progress()

    def _highlight(self, entry):
        """Light the table row of the open recording.

        The row is coloured rather than left to the selectable's own state,
        which a span_columns selectable in a clipped table does not draw.
        """
        if not dpg.does_item_exist(INDEX_TABLE):
            return
        if self._lit_row is not None:
            dpg.unhighlight_table_row(INDEX_TABLE, self._lit_row)
            self._lit_row = None
        row = self._rows.get(entry.path) if entry is not None else None
        if row is not None:
            dpg.highlight_table_row(INDEX_TABLE, row[2], OPEN_ROW)
            self._lit_row = row[2]

    def _row_clicked(self, sender, entry):
        # A selectable toggles itself on click; the open row is shown by its
        # colour instead, which a cancelled switch leaves where it was.
        dpg.set_value(sender, False)
        self.open(entry)

    def open_next(self, step=1, unannotated=False):
        """Open the next (or previous) recording in the table's order.

        Parameters
        ----------
        step: int
            +1 for the next row, -1 for the previous one.
        unannotated: bool
            Skip rows that are already annotated, searching the whole dataset
            when nothing below the open row in the table is left.
        """
        rows = self.visible or (self.index.entries if self.index else [])
        if not rows:
            return
        paths = [e.path for e in rows]
        position = paths.index(self.current.path) if (
            self.current is not None and self.current.path in paths) else -step
        order = [rows[(position + step * k) % len(rows)] for k in range(1, len(rows) + 1)]
        if unannotated:
            order = [e for e in order if not e.is_annotated
                     and (self.current is None or e.path != self.current.path)]
            if not order and self.index is not None:
                order = [e for e in self.index.entries if not e.is_annotated
                         and (self.current is None or e.path != self.current.path)]
            if not order:
                self._say_index("Every recording is annotated.", "ok")
                return
        self.open(order[0])

    # ==================================================================
    # opening a recording
    # ==================================================================
    def open(self, entry, force=False):
        """Show ``entry`` in the annotation window, with any saved annotations.

        Unsaved edits to the recording already open are not dropped silently:
        unless ``force`` is set, the user is asked first.
        """
        if entry is None:
            return
        if (not force and self.current is not None and entry.path == self.current.path
                and self.data is not None):
            # Already open: bring it back rather than reloading over the edits.
            self._spawn_plot_window()
            dpg.focus_item(PLOT_WINDOW)
            self._highlight(entry)
            return
        if self.dirty and not force:
            self._confirm_discard(lambda: self.open(entry, force=True))
            return
        self._spawn_plot_window()
        self._set_adding(False)
        self.current = entry
        self.data = None
        self.dirty = False
        self._highlight(entry)
        dpg.set_value(PLOT_TITLE, entry.label)
        self._say_plot(f"Loading {entry.path.name}...", "busy")
        self._clear_plot()

        self._load_generation += 1
        generation = self._load_generation

        def load():
            try:
                data = load_emg_text(entry.path)
                annotation = entry.load_annotation()
                message = ("loaded", generation, entry, data, annotation)
            except Exception as error:
                traceback.print_exc()
                message = ("failed", generation, entry, error, None)
            with self._lock:
                self._messages.append(message)

        threading.Thread(target=load, daemon=True).start()

    def _loaded(self, entry, data, annotation):
        self.data = data
        if annotation is not None:
            self.regions = [list(r) for r in annotation.regions]
            self.notes = annotation.notes
            state = f"Annotated ({len(self.regions)} region(s)), saved {annotation.updated}."
            kind = "ok"
            if annotation.num_samples not in (None, len(data)):
                state += (f" Warning: annotated when the file had {annotation.num_samples}"
                          f" samples; it now has {len(data)}.")
                kind = "warn"
        else:
            self.regions = []
            self.notes = ""
            state = "Not annotated yet."
            kind = "info"
        dpg.set_value("__annot_notes", self.notes)
        self.dirty = False
        self._prepare_display()
        self._build_series()
        self._rebuild_regions()
        self._fit()
        self._say_plot(state, kind)

    # ==================================================================
    # drawing the signal
    # ==================================================================
    @property
    def _fs(self):
        return self.sampling_rate if self.sampling_rate > 0 else 1.0

    def _prepare_display(self):
        """Normalise each channel into its own lane, one lane per channel.

        Channel 1 is the top lane. Each is scaled by a robust estimate of its
        amplitude so a quiet channel is as readable as a loud one.
        """
        data = self.data
        channels = data.shape[1]
        offsets = np.arange(channels)[::-1].astype(float)
        if dpg.get_value("__annot_view") == DISPLAY_MODES[1]:
            envelope = moving_rms(data, ENVELOPE_SECONDS * self._fs)
            scale = np.percentile(envelope, 99.9, axis=0)
            scale[scale <= 0] = 1.0
            self._display = envelope / (1.1 * scale) - 0.45 + offsets
        else:
            centred = data - np.median(data, axis=0)
            scale = np.percentile(np.abs(centred), 99.9, axis=0)
            scale[scale <= 0] = 1.0
            self._display = centred / (2.2 * scale) + offsets
        self._rendered = None

    def _series_tag(self, channel):
        return f"__annot_series_{channel}"

    def _clear_plot(self):
        dpg.delete_item(Y_AXIS, children_only=True)
        self._delete_rects()
        if dpg.does_item_exist(PENDING_LINE):
            dpg.delete_item(PENDING_LINE)
        dpg.delete_item(REGION_TABLE, children_only=True, slot=1)

    def _build_series(self):
        dpg.delete_item(Y_AXIS, children_only=True)
        channels = self._display.shape[1]
        for channel in range(channels):
            dpg.add_line_series([], [], parent=Y_AXIS, tag=self._series_tag(channel))
        dpg.set_axis_ticks(Y_AXIS, tuple((f"Ch {c + 1}", float(channels - 1 - c))
                                         for c in range(channels)))
        dpg.set_axis_limits(Y_AXIS, -0.7, channels - 0.3)
        self._render_range(0, len(self._display))

    def _render_range(self, start, stop):
        width = dpg.get_item_rect_size(PLOT)[0] if dpg.does_item_exist(PLOT) else 0
        max_points = int(max(width, 800) * 2)
        x, y = decimate_for_display(self._display, start, stop, max_points)
        seconds = (x / self._fs).tolist()
        for channel in range(y.shape[1]):
            dpg.set_value(self._series_tag(channel), [seconds, y[:, channel].tolist()])
        self._rendered = (max(int(start), 0), min(int(stop), len(self._display)))

    def _redraw_signal(self, refit=True):
        if self.data is None:
            return
        self._prepare_display()
        self._build_series()
        if refit:
            self._fit()

    def _fit(self):
        if self.data is None:
            return
        dpg.set_axis_limits(X_AXIS, 0.0, len(self.data) / self._fs)
        # Released next frame, so the user can zoom from the fitted view.
        self._release_x = True

    def _sampling_rate_changed(self):
        rate = dpg.get_value("__annot_fs")
        if not rate or rate <= 0:
            dpg.set_value("__annot_fs", self.sampling_rate)
            return
        self.sampling_rate = float(rate)
        self._redraw_signal()
        self._rebuild_regions()

    # ==================================================================
    # regions
    # ==================================================================
    def _rebuild_regions(self):
        """Redraw every region on the plot and in the region table."""
        self._delete_rects()
        dpg.delete_item(REGION_TABLE, children_only=True, slot=1)
        if self.data is None:
            return
        self.regions.sort()
        channels = self.data.shape[1]
        draggable = bool(dpg.get_value("__annot_drag"))
        for i, (start, stop) in enumerate(self.regions):
            dpg.add_drag_rect(parent=PLOT, tag=f"__annot_rect_{i}", color=REGION_COLOR,
                              default_value=(start / self._fs, -0.6,
                                             stop / self._fs, channels - 0.4),
                              no_inputs=not draggable, no_fit=True, user_data=i,
                              callback=lambda sender, value, user: self._rect_moved(user))
            self._built_regions = i + 1
            with dpg.table_row(parent=REGION_TABLE):
                dpg.add_text(str(i + 1))
                dpg.add_input_float(tag=f"__annot_start_{i}", default_value=start / self._fs,
                                    width=-1, step=0, format="%.3f", on_enter=True,
                                    user_data=(i, 0),
                                    callback=lambda s, v, user: self._region_typed(*user, v))
                dpg.add_input_float(tag=f"__annot_stop_{i}", default_value=stop / self._fs,
                                    width=-1, step=0, format="%.3f", on_enter=True,
                                    user_data=(i, 1),
                                    callback=lambda s, v, user: self._region_typed(*user, v))
                dpg.add_text(f"{(stop - start) / self._fs:.3f}", tag=f"__annot_dur_{i}")
                with dpg.group(horizontal=True):
                    style.button("Go to", user_data=i,
                                 callback=lambda s, a, user: self._zoom_to_region(user))
                    style.button("Delete", user_data=i,
                                 callback=lambda s, a, user: self._delete_region(user))

    def _delete_rects(self):
        for i in range(self._built_regions):
            if dpg.does_item_exist(f"__annot_rect_{i}"):
                dpg.delete_item(f"__annot_rect_{i}")
        self._built_regions = 0

    def _rect_moved(self, i):
        x1, _, x2, _ = dpg.get_value(f"__annot_rect_{i}")[:4]
        start, stop = sorted((x1 * self._fs, x2 * self._fs))
        start = int(round(max(start, 0)))
        stop = int(round(min(stop, len(self.data))))
        self.regions[i] = [start, stop]
        channels = self.data.shape[1]
        # Regions span every channel; only their extent in time is theirs to drag.
        dpg.set_value(f"__annot_rect_{i}",
                      (start / self._fs, -0.6, stop / self._fs, channels - 0.4))
        dpg.set_value(f"__annot_start_{i}", start / self._fs)
        dpg.set_value(f"__annot_stop_{i}", stop / self._fs)
        dpg.set_value(f"__annot_dur_{i}", f"{(stop - start) / self._fs:.3f}")
        self._mark_dirty()

    def _region_typed(self, i, edge, seconds):
        sample = int(round(min(max(seconds, 0.0) * self._fs, len(self.data))))
        self.regions[i][edge] = sample
        self.regions[i].sort()
        if self.regions[i][1] <= self.regions[i][0]:
            self._say_plot("A region must end after it starts.", "warn")
        self._mark_dirty()
        self._rebuild_regions()

    def _delete_region(self, i):
        del self.regions[i]
        self._mark_dirty()
        self._rebuild_regions()

    def _clear_regions(self):
        if self.regions:
            self.regions = []
            self._mark_dirty()
            self._rebuild_regions()

    def _zoom_to_region(self, i):
        start, stop = self.regions[i]
        margin = max((stop - start) * 0.5, self._fs)
        dpg.set_axis_limits(X_AXIS, max(start - margin, 0) / self._fs,
                            min(stop + margin, len(self.data)) / self._fs)
        self._release_x = True

    def add_region(self, start, stop):
        """Add a region in samples. Empty regions are ignored."""
        start, stop = sorted((int(round(start)), int(round(stop))))
        start, stop = max(start, 0), min(stop, len(self.data))
        if stop <= start:
            return
        self.regions.append([start, stop])
        self._mark_dirty()
        self._rebuild_regions()

    def _set_adding(self, adding):
        self._adding = bool(adding) and self.data is not None
        self._pending_start = None
        if dpg.does_item_exist(PENDING_LINE):
            dpg.delete_item(PENDING_LINE)
        if dpg.does_item_exist(ADD_BUTTON):
            dpg.configure_item(ADD_BUTTON, label="Cancel adding (Esc)" if self._adding else "Add region (R)")
        if self._adding:
            self._say_plot("Click where the contraction starts.", "busy")

    def _plot_hovered(self):
        return (self.data is not None and dpg.does_item_exist(PLOT)
                and dpg.is_item_shown(PLOT_WINDOW) and dpg.is_item_hovered(PLOT))

    def _on_key_add(self):
        if not self._plot_hovered():
            return
        # Editing notes or a numeric field still owns the keyboard when the
        # pointer is moved back over the plot.
        focused = dpg.get_focused_item()
        if focused and dpg.does_item_exist(focused) and dpg.is_item_active(focused):
            return
        if any(dpg.is_key_down(key) for key in (
                dpg.mvKey_LControl, dpg.mvKey_RControl,
                dpg.mvKey_LAlt, dpg.mvKey_RAlt)):
            return
        self._set_adding(not self._adding)

    def _on_mouse_press(self):
        self._press = None
        if self._adding and self._plot_hovered():
            self._press = (dpg.get_mouse_pos(local=False), dpg.get_plot_mouse_pos()[0])

    def _on_mouse_release(self):
        press, self._press = self._press, None
        if press is None or not self._adding or not self._plot_hovered():
            return
        (x0, y0), seconds = press
        x1, y1 = dpg.get_mouse_pos(local=False)
        if abs(x1 - x0) > CLICK_TOLERANCE or abs(y1 - y0) > CLICK_TOLERANCE:
            return  # that was a pan, not a click
        sample = int(round(min(max(seconds, 0.0) * self._fs, len(self.data))))
        if self._pending_start is None:
            self._pending_start = sample
            dpg.add_drag_line(parent=PLOT, tag=PENDING_LINE, color=PENDING_COLOR,
                              default_value=sample / self._fs, no_inputs=True,
                              show_label=False, thickness=2)
            self._say_plot("Now click where it ends.", "busy")
        else:
            start = self._pending_start
            self._set_adding(False)
            self.add_region(start, sample)
            self._say_plot(f"{len(self.regions)} region(s), unsaved.", "warn")

    # ==================================================================
    # saving
    # ==================================================================
    def _notes_changed(self):
        self.notes = dpg.get_value("__annot_notes")
        self._mark_dirty()

    def _mark_dirty(self):
        self.dirty = True
        if self.current is not None and dpg.does_item_exist(PLOT_TITLE):
            dpg.set_value(PLOT_TITLE, self.current.label + "   (unsaved changes)")

    def save(self, then_next=False):
        """Write the regions beside the open recording.

        Saving with no regions records that the recording was looked at and
        holds no usable contraction, which is why it still counts as annotated.
        """
        if self.current is None or self.data is None:
            return
        annotation = ContractionAnnotation(
            regions=[tuple(r) for r in self.regions], num_samples=len(self.data),
            sampling_rate=self.sampling_rate, notes=dpg.get_value("__annot_notes") or "")
        try:
            path = annotation.save(self.current.path)
        except Exception as error:
            self._say_plot(f"Could not save: {error}", "error")
            return
        self.regions = [list(r) for r in annotation.regions]
        self.dirty = False
        dpg.set_value(PLOT_TITLE, self.current.label)
        self._rebuild_regions()
        self._refresh_row(self.current)
        detail = f"{len(self.regions)} region(s)" if self.regions else "no contractions"
        self._say_plot(f"Saved {detail} to {path.name}.", "ok")
        if then_next:
            self.open_next(unannotated=True)

    def _confirm_discard(self, proceed):
        """Ask what to do with unsaved edits before ``proceed`` replaces them."""
        if dpg.does_item_exist(CONFIRM_WINDOW):
            dpg.delete_item(CONFIRM_WINDOW)
        self._after_confirm = proceed

        def choose(choice):
            dpg.delete_item(CONFIRM_WINDOW)
            action, self._after_confirm = self._after_confirm, None
            if choice == "save":
                self.save()
                if self.dirty:  # the save failed; keep the edits on screen
                    return
            if choice in ("save", "discard") and action is not None:
                self.dirty = False
                action()
            else:
                self._highlight(self.current)

        with dpg.window(label="Unsaved changes", tag=CONFIRM_WINDOW, modal=True,
                        no_resize=True, autosize=True,
                        pos=(self.width // 2 - 180, self.height // 2 - 60)):
            name = self.current.label if self.current is not None else "this recording"
            dpg.add_text(f"{name} has unsaved annotations.")
            style.note("Save them before moving on, or discard them?")
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True) as buttons:
                style.button("Save", "primary", width=90, callback=lambda: choose("save"))
                style.button("Discard", "danger", width=90, callback=lambda: choose("discard"))
                style.button("Cancel", width=90, callback=lambda: choose("cancel"))
            dpg.bind_item_theme(buttons, BUTTON_ROW_THEME)

    # ==================================================================
    # per-frame
    # ==================================================================
    def poll(self):
        """Apply finished loads and re-draw the signal at the zoom shown."""
        with self._lock:
            messages, self._messages = self._messages, []
        for kind, generation, entry, payload, annotation in messages:
            if generation != self._load_generation or not dpg.does_item_exist(PLOT_WINDOW):
                continue  # superseded by a later click, or the window is gone
            if kind == "loaded":
                self._loaded(entry, payload, annotation)
            else:
                self._say_plot(f"Could not load {entry.path.name}: {payload}", "error")

        if self._release_x == "pending":
            # One frame pinned is enough for the plot to take the new range;
            # released, it keeps that range and can be zoomed from it.
            if dpg.does_item_exist(X_AXIS):
                dpg.set_axis_limits_auto(X_AXIS)
            self._release_x = False
        elif self._release_x:
            self._release_x = "pending"

        try:
            self._follow_zoom()
        except Exception:
            # The GUI stops polling a panel whose poll raises, which would
            # leave every later load undelivered; a bad frame is only a frame.
            traceback.print_exc()

    def _follow_zoom(self):
        if self._display is None or self.data is None or not dpg.does_item_exist(PLOT):
            return
        if not dpg.is_item_shown(PLOT_WINDOW):
            return
        lo, hi = dpg.get_axis_limits(X_AXIS)
        span = max(hi - lo, 1e-9) * self._fs
        start, stop = int(lo * self._fs - span * 0.25), int(hi * self._fs + span * 0.25)
        start, stop = max(start, 0), min(stop, len(self._display))
        if self._rendered is None:
            self._render_range(start, stop)
            return
        r_start, r_stop = self._rendered
        rendered_span = max(r_stop - r_start, 1)
        visible_lo, visible_hi = max(int(lo * self._fs), 0), min(int(hi * self._fs), len(self._display))
        # Re-draw when the view leaves what is drawn, or zooms far enough in
        # that the drawn resolution is visibly coarse.
        if (visible_lo < r_start or visible_hi > r_stop
                or (stop - start) < rendered_span * 0.5):
            self._render_range(start, stop)

    # ==================================================================
    # status lines
    # ==================================================================
    def _say_index(self, text, kind="info"):
        style.set_status(INDEX_STATUS, text, kind)

    def _say_plot(self, text, kind="info"):
        style.set_status(PLOT_STATUS, text, kind)
