"""What a model predicted, drawn over the signal it predicted from.

A Prediction Viewer block takes a model's output and the labels, and once a
stored-data run has read the recording it opens a window on every recording:
the EMG, the model's probability for every class over time, and a band with
the class it predicted against the class that was performed. Two views:

    Recording   one file -- one repetition of one motion, as a collection
                writes them.
    Stream      every recording that shares a value of a metadata field --
                a set, a repetition number -- laid end to end in time, the
                way the session was collected.

Recordings can be narrowed to the ones a model trained on or the ones it held
out. A deep model trained by the pipeline remembers which recordings it held
out for validation; any model can also be split by a field in the filenames,
such as the set or repetition a recording belongs to.

**Save figure...** writes the whole session as one image: a panel per
recording, a row per class and a column per repetition, each with its EMG, the
true class as the background and the predicted class as the line's colour.

Each recording is predicted on its own, so a model with memory starts fresh
at each one, as it was trained to -- not carrying a state across the join
between two files.
"""

import textwrap
import threading

import numpy as np

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style
from libemg._gui._pipeline.feature_viewer import CLASS_COLOURS

WINDOW_TAG = "__prediction_viewer"
PLOTS = "__pv_plots"
STATUS = "__pv_status"
SAVE_DIALOG = "__pv_save_dialog"

RECORDING, STREAM = "Recording", "Stream"
ALL, TRAINING, HELD_OUT = "All", "Training", "Held out"


# ======================================================================
# the maths
# ======================================================================
def predict_recordings(predictor, inputs, classification=True):
    """A model's outputs for each recording on its own.

    Parameters
    ----------
    predictor: EMGClassifier, EMGRegressor or a model
    inputs: list of arrays
        One per recording, one row per window.

    Returns
    ----------
    list of (outputs, predicted)
        ``outputs`` is class probabilities (windows, classes) for a
        classifier, or values (windows, dofs) for a regressor; ``predicted``
        is the class per window, or the values again.
    """
    model = getattr(predictor, "model", predictor)
    results = []
    for x in inputs:
        if hasattr(model, "reset"):
            model.reset()           # a fresh stream: no memory of the last recording
        if classification and hasattr(predictor, "_predict_proba"):
            probabilities = np.atleast_2d(np.asarray(predictor._predict_proba(x), dtype=float))
            classes = getattr(model, "classes_", None)
            index = probabilities.argmax(axis=1)
            predicted = np.asarray(classes)[index] if classes is not None else index
            results.append((probabilities, predicted))
        else:
            values = np.asarray(predictor._predict(x), dtype=float)
            values = values.reshape(len(x), -1)
            results.append((values, values))
    return results


def class_runs(values):
    """Contiguous runs of one value: [(start, end, value)], end exclusive."""
    values = np.asarray(values).ravel()
    if not len(values):
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(values)]])
    return [(int(a), int(b), values[a]) for a, b in zip(starts, ends)]


def window_times(count, window, increment, rate):
    """When each window's output is available: at its last sample, in seconds."""
    return (np.arange(count) * increment + window) / float(rate)


def recording_class(recording):
    """The class a recording mostly is, or None without labels."""
    truth = recording.get("truth")
    if truth is None or not len(truth) or np.ndim(truth) > 1:
        return None
    values, counts = np.unique(np.asarray(truth), return_counts=True)
    return values[np.argmax(counts)]


def accuracy(recordings):
    """Windows predicted right, over windows with a label: a fraction, or None."""
    right = total = 0
    for r in recordings:
        if r.get("truth") is None or np.ndim(r["truth"]) > 1:
            continue
        right += int((np.asarray(r["predicted"]) == np.asarray(r["truth"])).sum())
        total += len(r["truth"])
    return right / total if total else None


def stream(recordings, window, increment, rate, gap=0.25):
    """Recordings laid end to end in time, ``gap`` seconds apart.

    Returns a dict: ``time`` and ``signal`` for the samples, ``window_time``,
    ``outputs``, ``predicted`` and ``truth`` for the windows (truth is None
    if any recording has none), ``bounds``, a (start, end, recording)
    per recording in seconds, and ``sample_starts`` and ``window_starts``, where
    each recording begins in the sample and window arrays.
    """
    offset, bounds = 0.0, []
    times, signals, window_times_, outputs, predicted, truth = [], [], [], [], [], []
    for r in recordings:
        n = len(r["signal"])
        times.append(offset + np.arange(n) / rate)
        signals.append(r["signal"])
        window_times_.append(offset + window_times(len(r["predicted"]), window, increment, rate))
        outputs.append(r["outputs"])
        predicted.append(np.asarray(r["predicted"]))
        truth.append(r.get("truth"))
        bounds.append((offset, offset + n / rate, r))
        offset += n / rate + gap
    has_truth = all(t is not None for t in truth) and bool(truth)
    return {"time": np.concatenate(times) if times else np.empty(0),
            "signal": np.vstack(signals) if signals else np.empty((0, 0)),
            "window_time": np.concatenate(window_times_) if window_times_ else np.empty(0),
            "outputs": np.vstack(outputs) if outputs else np.empty((0, 0)),
            "predicted": np.concatenate(predicted) if predicted else np.empty(0),
            "truth": np.concatenate(truth) if has_truth else None,
            "bounds": bounds,
            "sample_starts": np.cumsum([0] + [len(r["signal"]) for r in recordings])[:-1],
            "window_starts": np.cumsum([0] + [len(r["predicted"]) for r in recordings])[:-1]}


def broken(x, y, starts):
    """``x`` and ``y`` as lists for a line series, cut where each recording starts.

    A NaN between two recordings leaves a gap in the line, so the gap between
    them is not drawn as a ramp the signal never made.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    cuts = [int(i) for i in starts if 0 < i < len(x)]
    return np.insert(x, cuts, np.nan).tolist(), np.insert(y, cuts, np.nan).tolist()


def describe(recording):
    """A recording's name: its metadata, its split."""
    meta = ", ".join(f"{k} {v}" for k, v in recording.get("meta", {}).items())
    split = recording.get("split") or ""
    return f"#{recording['index'] + 1}" + (f"  {meta}" if meta else "") + \
        (f"  ({split})" if split else "")


# ======================================================================
# the window
# ======================================================================
class PredictionViewer:
    """A window drawing a model's predictions over the signals it read.

    Parameters
    ----------
    view: dict
        What a stored-data run hands over: ``task``, ``classes``,
        ``sampling_rate``, ``window_size``, ``window_increment`` and
        ``recordings``, each a dict with ``index``, ``meta``, ``signal``,
        ``truth``, ``outputs``, ``predicted`` and ``split``.
    """

    window_tag = WINDOW_TAG

    def __init__(self, view, width=1180, height=760, pos=(40, 50)):
        self.view = view
        self.recordings = view["recordings"]
        self.classification = view.get("task", "classification") == "classification"
        self.classes = list(view.get("classes") or [])
        self.rate = float(view.get("sampling_rate") or 1000)
        self.window, self.increment = int(view["window_size"]), int(view["window_increment"])
        self.width, self.height, self.pos = width, height, pos
        self.mode = RECORDING
        self.split = ALL
        self.current = 0
        fields = sorted({k for r in self.recordings for k in r.get("meta", {})})
        self.fields = fields
        self.stream_field = next((f for f in fields if f not in ("classes",)), fields[0]) \
            if fields else None
        self.stream_value = None
        self._fit_next = []         # axes to fit to their data once drawn
        self._saved = None

    # -- life cycle --------------------------------------------------------
    def spawn_window(self):
        self.cleanup()
        with dpg.window(label="Prediction Viewer", tag=WINDOW_TAG, width=self.width,
                        height=self.height, pos=self.pos, on_close=self.cleanup):
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingFixedFit,
                           borders_innerV=True):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=330)
                dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                with dpg.table_row():
                    with dpg.group():
                        self._build_controls()
                    dpg.add_group(tag=PLOTS)
        with dpg.file_dialog(directory_selector=False, show=False, modal=True, tag=SAVE_DIALOG,
                             width=760, height=460, default_filename="predictions",
                             callback=self._save_chosen):
            dpg.add_file_extension(".png")
        self._choices()
        self._draw()
        return self

    def cleanup(self):
        for tag in (WINDOW_TAG, SAVE_DIALOG):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    # -- controls ----------------------------------------------------------
    def _build_controls(self):
        style.section("View", first=True)
        splits = [ALL] + [s for s in (TRAINING, HELD_OUT)
                          if any(r.get("split") == s.lower() for r in self.recordings)]
        with style.form(label_width=95):
            with style.row("Show", help="Recording: one file. Stream: every recording that "
                           "shares a value of a field, end to end in time, as collected."):
                dpg.add_radio_button([RECORDING, STREAM], default_value=self.mode, horizontal=True,
                                     tag="__pv_mode", callback=lambda s, a: self._set(mode=a))
            with style.row("Data", help="Training: recordings the model learned from. Held "
                           "out: recordings it was validated on and never trained on. Only "
                           "offered when the model recorded which were which.",
                           tag="__pv_row_split"):
                dpg.add_combo(splits, default_value=self.split, width=-1, tag="__pv_split",
                              enabled=len(splits) > 1,
                              callback=lambda s, a: self._set(split=a))
            with style.row("Recording", tag="__pv_row_recording"):
                dpg.add_combo([], width=-1, tag="__pv_recording",
                              callback=self._recording_chosen)
            with style.row("Stream by", tag="__pv_row_field",
                           help="The field whose value the stream's recordings share: a set, "
                                "a repetition number."):
                dpg.add_combo(self.fields, default_value=self.stream_field or "", width=-1,
                              tag="__pv_field", callback=lambda s, a: self._set(stream_field=a,
                                                                               stream_value=None))
            with style.row("Value", tag="__pv_row_value"):
                dpg.add_combo([], width=-1, tag="__pv_value",
                              callback=lambda s, a: self._set(stream_value=a))
        with dpg.group(horizontal=True):
            style.button("< Previous", width=110, callback=lambda: self._step(-1))
            style.button("Next >", width=110, callback=lambda: self._step(1))
        style.section("Accuracy")
        style.note("", tag="__pv_accuracy", wrap=310)
        dpg.add_spacer(height=4)
        style.button("Save figure...", width=140, callback=lambda: dpg.show_item(SAVE_DIALOG))
        style.note("Every recording as one image: a row per class, a column per repetition.",
                   wrap=310)
        dpg.add_spacer(height=4)
        style.note("", tag=STATUS, wrap=310)

    def _shown(self):
        """Recordings the Data filter lets through."""
        if self.split == ALL:
            return list(self.recordings)
        return [r for r in self.recordings if r.get("split") == self.split.lower()]

    def _choices(self):
        """Fill the Recording and Value lists from what the filter lets through."""
        shown = self._shown()
        if dpg.does_item_exist("__pv_recording"):
            names = [describe(r) for r in shown]
            dpg.configure_item("__pv_recording", items=names)
            if shown:
                self.current = min(self.current, len(shown) - 1)
                dpg.set_value("__pv_recording", names[self.current])
        values = sorted({str(r["meta"].get(self.stream_field)) for r in shown
                         if self.stream_field in r.get("meta", {})}, key=_natural)
        if self.stream_value not in values:
            self.stream_value = values[0] if values else None
        if dpg.does_item_exist("__pv_value"):
            dpg.configure_item("__pv_value", items=values)
            dpg.set_value("__pv_value", self.stream_value or "")
        for tag, value in (("__pv_mode", self.mode), ("__pv_split", self.split)):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)
        for tag, show in (("__pv_row_recording", self.mode == RECORDING),
                          ("__pv_row_field", self.mode == STREAM and bool(self.fields)),
                          ("__pv_row_value", self.mode == STREAM and bool(self.fields))):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=show)
        lines = [f"Shown: {_percent(accuracy(shown))} of windows right"]
        for split in (TRAINING, HELD_OUT):
            subset = [r for r in self.recordings if r.get("split") == split.lower()]
            if subset:
                lines.append(f"{split}: {_percent(accuracy(subset))}")
        if dpg.does_item_exist("__pv_accuracy"):
            dpg.set_value("__pv_accuracy", "\n".join(lines) if self.classification else
                          "A regressor's outputs are drawn against its targets.")

    def _set(self, **changes):
        for name, value in changes.items():
            setattr(self, name, value)
        if "split" in changes:
            self.current = 0
        self._choices()
        self._draw()

    def _recording_chosen(self, sender, name):
        names = [describe(r) for r in self._shown()]
        self.current = names.index(name) if name in names else 0
        self._draw()

    def _step(self, by):
        """The next recording, or the next stream value."""
        if self.mode == RECORDING:
            count = len(self._shown())
            if count:
                self.current = (self.current + by) % count
        else:
            values = dpg.get_item_configuration("__pv_value")["items"] or []
            if values:
                at = values.index(self.stream_value) if self.stream_value in values else 0
                self.stream_value = values[(at + by) % len(values)]
        self._choices()
        self._draw()

    # -- drawing -------------------------------------------------------------
    def _selection(self):
        shown = self._shown()
        if not shown:
            return [], "No recordings to show."
        if self.mode == RECORDING:
            r = shown[self.current]
            return [r], describe(r)
        chosen = [r for r in shown if str(r["meta"].get(self.stream_field)) == self.stream_value]
        chosen.sort(key=lambda r: (_natural(str(recording_class(r))), r["index"]))
        return chosen, f"{self.stream_field} {self.stream_value}: {len(chosen)} recordings"

    def _draw(self):
        if not dpg.does_item_exist(PLOTS):
            return
        dpg.delete_item(PLOTS, children_only=True)
        self._fit_next = []
        chosen, title = self._selection()
        if not chosen:
            dpg.add_text(title, parent=PLOTS, color=style.MUTED)
            return
        data = stream(chosen, self.window, self.increment, self.rate)
        dpg.add_text(title + (f"   {_percent(accuracy(chosen))} of windows right"
                              if self.classification and accuracy(chosen) is not None else ""),
                     parent=PLOTS, color=style.HEADING)
        ratios = [3, 3, 1] if self.classification else [3, 3]
        with dpg.subplots(len(ratios), 1, parent=PLOTS, width=-1, height=-1, link_all_x=True,
                          row_ratios=ratios):
            self._signal_plot(data)
            self._output_plot(data)
            if self.classification:
                self._band_plot(data)

    def _signal_plot(self, data):
        signal, time = data["signal"], data["time"]
        with dpg.plot(no_menus=True):
            self._fit_next.append(dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True))
            with dpg.plot_axis(dpg.mvYAxis, label="EMG", no_tick_labels=True) as axis:
                self._fit_next.append(axis)
                self._truth_shading(data, -1.0, float(signal.shape[1]) if signal.size else 1.0)
                if signal.size:
                    # One trace per channel, each scaled to its own spread and
                    # stacked, so a quiet channel is not flattened by a loud one.
                    spread = np.percentile(np.abs(signal), 99, axis=0)
                    spread = np.where(spread > 0, spread, 1.0)
                    step = max(1, len(time) // 40000)
                    kept = np.arange(0, len(time), step)
                    starts = np.searchsorted(kept, data["sample_starts"])
                    for c in range(signal.shape[1]):
                        series = dpg.add_line_series(*broken(
                            time[kept], signal[kept, c] / spread[c] * 0.45 + c, starts))
                        dpg.bind_item_theme(series, _line_theme((175, 180, 190), 1.0))
                for start, end, r in data["bounds"]:
                    dpg.add_inf_line_series([start], parent=axis)

    def _output_plot(self, data):
        times, outputs = data["window_time"], data["outputs"]
        starts = data["window_starts"]
        with dpg.plot(no_menus=True):
            # Above the plot, not beside it, so its time axis lines up with the others.
            dpg.add_plot_legend(location=dpg.mvPlot_Location_North, outside=True,
                                horizontal=True)
            dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=self.classification,
                              label="" if self.classification else "Time (s)")
            label = "Probability" if self.classification else "Output"
            with dpg.plot_axis(dpg.mvYAxis, label=label) as axis:
                if self.classification:
                    self._truth_shading(data, 0.0, 1.0)
                    for k in range(outputs.shape[1]):
                        name = f"Class {self.classes[k]}" if k < len(self.classes) else f"Class {k}"
                        series = dpg.add_line_series(*broken(times, outputs[:, k], starts),
                                                     label=name)
                        dpg.bind_item_theme(series, _line_theme(_colour(k), 2.0))
                    dpg.set_axis_limits(axis, -0.02, 1.05)
                else:
                    truth = data["truth"]
                    for d in range(outputs.shape[1]):
                        colour = CLASS_COLOURS[d % len(CLASS_COLOURS)]
                        series = dpg.add_line_series(*broken(times, outputs[:, d], starts),
                                                     label=f"DOF {d + 1} predicted")
                        dpg.bind_item_theme(series, _line_theme(colour, 2.0))
                        if truth is not None:
                            target = np.asarray(truth).reshape(len(times), -1)[:, d]
                            series = dpg.add_line_series(*broken(times, target, starts),
                                                         label=f"DOF {d + 1} target")
                            dpg.bind_item_theme(series, _line_theme(tuple(int(c * 0.55)
                                                                          for c in colour), 1.5))
                    self._fit_next.append(axis)

    def _band_plot(self, data):
        """The predicted class as a row of colour, under the class that was performed."""
        times = data["window_time"]
        half = (self.increment / self.rate) / 2
        with dpg.plot(no_menus=True):
            # The x axes are linked: the signal's, fitted, sets them all.
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
            with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True) as axis:
                for row, values in ((0.0, data["predicted"]), (1.15, data["truth"])):
                    if values is None:
                        continue
                    for start, end, value in class_runs(values):
                        colour = _colour(self._index(value))
                        shade = dpg.add_shade_series(
                            [times[start] - half, times[end - 1] + half],
                            [row + 1.0, row + 1.0], y2=[row, row])
                        dpg.bind_item_theme(shade, _fill_theme(colour, 200))
                dpg.set_axis_ticks(axis, (("Predicted", 0.5), ("True", 1.65)))
                dpg.set_axis_limits(axis, -0.1, 2.25)

    def _truth_shading(self, data, low, high):
        """The class performed, as a tint behind a plot."""
        truth = data["truth"]
        if truth is None or np.ndim(truth) > 1:
            return
        times, half = data["window_time"], (self.increment / self.rate) / 2
        starts = list(data["window_starts"]) + [len(times)]
        for (begin, finish, _), first, last in zip(data["bounds"], starts[:-1], starts[1:]):
            # Recording by recording, so a tint reaches the edges of the signal
            # it belongs to rather than stopping at the first and last window.
            for start, end, value in class_runs(truth[first:last]):
                left = begin if start == 0 else times[first + start] - half
                right = finish if first + end == last else times[first + end - 1] + half
                shade = dpg.add_shade_series([left, right], [high, high], y2=[low, low])
                dpg.bind_item_theme(shade, _fill_theme(_colour(self._index(value)), 38))

    def _index(self, value):
        """A class's position in the model's list, which picks its colour."""
        try:
            return self.classes.index(value)
        except ValueError:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

    # -- saving ------------------------------------------------------------------
    def _save_chosen(self, sender, app_data):
        path = (app_data or {}).get("file_path_name", "")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        style.set_status(STATUS, "Drawing the figure...", "busy")

        def work():
            try:
                save_grid_figure(self.view, path, recordings=self._shown())
                self._saved = (f"Saved {path}.", "ok")
            except Exception as error:
                self._saved = (f"Could not save the figure: {type(error).__name__}: {error}",
                               "error")
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        """Fit freshly drawn plots, and report a figure saved on the worker thread.

        Call once a frame. A plot's axes can only be fitted once it has been
        drawn, so a redraw asks for the fit here, a frame later.
        """
        if self._fit_next:
            axes, self._fit_next = self._fit_next, []
            for axis in axes:
                if dpg.does_item_exist(axis):
                    dpg.fit_axis_data(axis)
        saved = self._saved
        if saved is not None:
            self._saved = None
            style.set_status(STATUS, *saved)


# ======================================================================
# the whole session, as one image
# ======================================================================
def save_grid_figure(view, path, recordings=None, columns_by=None):
    """Every recording as a panel: a row per class, a column per repetition.

    Each panel shows the recording's EMG, the class performed as its background,
    and a line whose height is the model's probability for the class it
    predicted and whose colour is that class. Held-out recordings are framed.

    Parameters
    ----------
    view: dict
        As :class:`PredictionViewer` takes it.
    path: str
        A .png to write.
    recordings: list or None
        Which recordings; all of them by default.
    columns_by: str or None
        The metadata field that orders a row's panels (a repetition or set
        number). Defaults to the first field that is not the class.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    recordings = list(view["recordings"] if recordings is None else recordings)
    if not recordings:
        raise ValueError("There are no recordings to draw.")
    classification = view.get("task", "classification") == "classification"
    classes = list(view.get("classes") or [])
    rate = float(view.get("sampling_rate") or 1000)
    window, increment = int(view["window_size"]), int(view["window_increment"])
    fields = sorted({k for r in recordings for k in r.get("meta", {})})
    columns_by = columns_by or next((f for f in fields if f != "classes"), None)

    by_class = {}
    for r in recordings:
        by_class.setdefault(recording_class(r), []).append(r)
    rows = sorted(by_class, key=lambda c: _natural(str(c)))
    for row in rows:
        by_class[row].sort(key=lambda r: (_natural(str(r["meta"].get(columns_by, ""))),
                                          r["index"]))
    columns = max(len(v) for v in by_class.values())
    width, height = 1.6 * columns + 1.0, 1.5 * len(rows) + 1.3
    fig, axes = plt.subplots(len(rows) * 2, columns, figsize=(width, height), squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1.4] * len(rows)})
    # Fixed margins in inches, so the title and legend have room however many
    # rows there are.
    fig.subplots_adjust(left=0.5 / width, right=1 - 0.1 / width, top=1 - 1.0 / height,
                        bottom=0.45 / height, hspace=0.12, wspace=0.06)
    rgb = lambda colour: tuple(c / 255 for c in colour)
    for i, row in enumerate(rows):
        for j in range(columns):
            top, bottom = axes[2 * i, j], axes[2 * i + 1, j]
            for ax in (top, bottom):
                ax.set_xticks([]), ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if j >= len(by_class[row]):
                continue
            r = by_class[row][j]
            signal = np.asarray(r["signal"])
            t = np.arange(len(signal)) / rate
            spread = np.percentile(np.abs(signal), 99) or 1.0
            top.plot(t, signal / spread, color="0.35", linewidth=0.3)
            top.set_ylim(-1.2, 1.2)
            for ax in (top, bottom):
                ax.set_xlim(0, len(signal) / rate)
            if i == 0 and columns_by:
                top.set_title(f"{columns_by} {r['meta'].get(columns_by, '')}", fontsize=7, pad=8)
            times = window_times(len(r["predicted"]), window, increment, rate)
            tint = rgb(_colour(_index_in(classes, row))) if row is not None else (0.9, 0.9, 0.9)
            bottom.set_facecolor((*tint, 0.18))
            if classification:
                confidence = np.asarray(r["outputs"]).max(axis=1)
                for start, end, value in class_runs(r["predicted"]):
                    stop = min(end + 1, len(times))
                    bottom.plot(times[start:stop], confidence[start:stop],
                                color=rgb(_colour(_index_in(classes, value))), linewidth=1.2)
                bottom.set_ylim(0, 1.05)
                right = accuracy([r])
                if right is not None:
                    top.text(0.0, 1.0, f"{right:.0%}", transform=top.transAxes, fontsize=6,
                             ha="left", va="bottom", color="0.25" if right >= 0.9 else "#b42318")
            else:
                outputs = np.asarray(r["outputs"])
                for d in range(outputs.shape[1]):
                    bottom.plot(times, outputs[:, d], linewidth=1.0,
                                color=rgb(CLASS_COLOURS[d % len(CLASS_COLOURS)]))
            tag = (r.get("split") or "").upper()
            if tag:
                bottom.text(0.98, 0.04, tag, transform=bottom.transAxes, fontsize=5,
                            ha="right", va="bottom", color="0.3")
            if r.get("split") == "held out":
                for spine in bottom.spines.values():
                    spine.set_visible(True), spine.set_color("#1f3a68"), spine.set_linewidth(1.2)
            if j == 0:
                bottom.set_ylabel(f"class {row}", fontsize=7)
    shown_accuracy = accuracy(recordings)
    title = (f"{len(recordings)} recordings  |  {_percent(shown_accuracy)} of windows right"
             if classification else f"{len(recordings)} recordings")
    split_scores = []
    for split in ("training", "held out"):
        subset = [r for r in recordings if r.get("split") == split]
        if subset and classification:
            split_scores.append(f"{split}: {_percent(accuracy(subset))}")
    if split_scores:
        title += "   (" + ", ".join(split_scores) + ")"
    key = ("Background: the class performed. Line: the predicted class's colour, at the "
           "model's probability for it." + (" Held-out recordings are framed."
                                            if any(r.get("split") == "held out"
                                                   for r in recordings) else ""))
    fig.suptitle(title + "\n" + textwrap.fill(key, width=int(width * 15)),
                 fontsize=9, x=0.01, y=1 - 0.08 / height, ha="left", va="top")
    if classification:
        fig.legend(handles=[Patch(color=rgb(_colour(k)), label=f"class {c}")
                            for k, c in enumerate(classes)], loc="lower center", fontsize=7,
                   ncol=min(10, max(1, len(classes))), frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _colour(index):
    return CLASS_COLOURS[int(index) % len(CLASS_COLOURS)]


def _index_in(classes, value):
    try:
        return list(classes).index(value)
    except ValueError:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


def _percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def _natural(text):
    """Sort '2' before '10'."""
    try:
        return (0, float(text), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(text))


def _line_theme(colour, weight):
    alias = "__pv_line_{}_{}_{}_{}".format(*colour[:3], int(weight * 10))
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight,
                                category=dpg.mvThemeCat_Plots)
    return alias


def _fill_theme(colour, alpha):
    alias = "__pv_fill_{}_{}_{}_{}".format(*colour[:3], alpha)
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvShadeSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (*colour[:3], alpha),
                                category=dpg.mvThemeCat_Plots)
    return alias
