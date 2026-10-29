"""Deep model training, drawn as it happens, and kept to look at again later.

The same window serves two purposes.

Live, while the pipeline editor trains a Deep Model block. Training runs on a
worker thread and appends an :class:`~libemg.deep.training.EpochResult` to the
pipeline's ``epochs`` list as each epoch ends; :meth:`TrainingView.update`
reads that list once a frame, on the render thread, and redraws when it has
grown. When training finishes the run can be exported.

Later, in any session, as the Training History window (Pipeline > Training
History, or the editor's History button). It loads training records --
the ``<model>.training.json`` a pipeline writes beside every model it trains,
a record exported from here, or a saved model, which carries its own history
-- and draws them. Several at once are overlaid, one colour each, so runs can
be compared: an architecture against another, a learning rate against another.

One run: training and validation loss, a line at the best epoch (the one whose
weights were kept), and the validation score beneath. Several runs: each run's
validation loss and score in its own colour, and a table of what each was.
"""

import os
from pathlib import Path

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style

#: The live view's tags start with this; the Training History window's with
#: HISTORY, so both can be open at once.
LIVE = "__training_view"
HISTORY = "__training_history"

_TRAINING = style.ACCENT
_VALIDATION = (251, 146, 60)
_SCORE = style.SUCCESS
#: One per run when comparing, in order.
_RUN_COLOURS = [(96, 165, 250), (251, 146, 60), (74, 222, 128), (248, 113, 113),
                (192, 132, 252), (250, 204, 21), (45, 212, 191), (244, 114, 182)]


class _Run:
    """One run on screen: its record when there is one, else epochs still arriving."""

    def __init__(self, name, epochs=(), outcome=None):
        self.name, self.epochs, self.outcome = name, list(epochs), outcome


class TrainingView:
    """A window drawing training runs, epoch by epoch.

    Parameters
    ----------
    title: str
    total_epochs: int or None
        For a live run: the most epochs it will take, for "epoch 12 of 50".
    on_stop: callable or None
        For a live run: what Stop does. Without it there is no Stop button.
    """

    def __init__(self, title="Training History", total_epochs=None, on_stop=None,
                 width=900, height=660, pos=(80, 80)):
        self.title, self.total_epochs, self.on_stop = title, total_epochs, on_stop
        base = LIVE if on_stop is not None else HISTORY
        self.window_tag = base
        (self._status, self._loss_x, self._loss_y, self._score_x, self._score_y, self._runs,
         self._export_dialog, self._load_dialog, self._stop, self._export,
         self._clear) = (f"{base}_{part}" for part in (
            "status", "loss_x", "loss_y", "score_x", "score_y", "runs", "export_dialog",
            "load_dialog", "stop", "export", "clear"))
        self.width, self.height, self.pos = width, height, pos
        self.live = on_stop is not None
        self.runs = [_Run("This run")] if self.live else []
        self._shown = 0

    # -- life cycle ----------------------------------------------------------
    def spawn_window(self):
        self.cleanup()
        with dpg.window(label=self.title, tag=self.window_tag, width=self.width, height=self.height,
                        pos=self.pos, no_collapse=True, on_close=self.cleanup):
            with dpg.group(horizontal=True):
                if self.live:
                    style.button("Stop", "danger", tag=self._stop, width=90,
                                 callback=lambda: self.on_stop())
                style.button("Export...", tag=self._export, width=100,
                             enabled=False, callback=lambda: dpg.show_item(self._export_dialog))
                style.button("Load...", width=90, callback=lambda: dpg.show_item(self._load_dialog))
                style.button("Clear", "subtle", tag=self._clear, width=80,
                             enabled=False, callback=self.clear)
            dpg.add_text("Reading the recording..." if self.live else
                         "Load a training record to look at it: the .training.json a pipeline "
                         "writes beside each model it trains, one exported from here, or a "
                         "saved deep model. Load several to compare them.",
                         tag=self._status, color=style.ACCENT if self.live else style.MUTED,
                         wrap=self.width - 40)
            with dpg.plot(height=250, width=-1, tag=f"{self.window_tag}_loss_plot"):
                dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag=self._loss_x)
                dpg.add_plot_axis(dpg.mvYAxis, label="Loss", tag=self._loss_y)
            with dpg.plot(height=170, width=-1, tag=f"{self.window_tag}_score_plot"):
                dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag=self._score_x)
                dpg.add_plot_axis(dpg.mvYAxis, label="Score", tag=self._score_y)
            dpg.add_group(tag=self._runs)
        # File dialogs are top-level items, so they are made after the window.
        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             tag=self._export_dialog, width=760, height=460,
                             default_filename="training", callback=self._export_chosen):
            dpg.add_file_extension(".json", color=(150, 186, 238, 255))
        with dpg.file_dialog(directory_selector=False, show=False, modal=True, file_count=8,
                             tag=self._load_dialog, width=760, height=460,
                             callback=self._load_chosen):
            dpg.add_file_extension("Training records and models{.json,.pkl}",
                                   color=(150, 186, 238, 255))
            dpg.add_file_extension(".*")
        self._draw()
        return self

    def cleanup(self):
        for tag in (self.window_tag, self._export_dialog, self._load_dialog):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._shown = 0

    # -- a live run ------------------------------------------------------------
    def update(self, epochs):
        """Draw the live run's epochs so far. Cheap when nothing new has finished."""
        if not self.live or not dpg.does_item_exist(self.window_tag) or len(epochs) == self._shown:
            return
        self.runs[0].epochs = list(epochs)
        self._shown = len(epochs)
        self._draw()
        last, best = self.runs[0].epochs[-1], _best(self.runs[0].epochs)
        seconds = sum(e.seconds for e in self.runs[0].epochs) / len(self.runs[0].epochs)
        of = f" of {self.total_epochs}" if self.total_epochs else ""
        style.set_status(self._status, f"Epoch {last.epoch}{of}   {_score(last)}   best: epoch "
                                 f"{best.epoch} ({_score(best)})   {seconds:.2g} s per epoch",
                         "busy")

    def finish(self, message, kind="ok", outcome=None):
        """Say how the live run ended, keep its record, and allow exporting it."""
        if outcome is not None and self.runs:
            self.runs[0].outcome = outcome
            self.runs[0].epochs = list(outcome.epochs)
            self.runs[0].name = outcome.name
            self._draw()
        if dpg.does_item_exist(self.window_tag):
            style.set_status(self._status, message, kind)
            style.set_enabled(self._stop, False)

    # -- loading and exporting --------------------------------------------------
    def load(self, path):
        """Add a training record (or a saved deep model's history) to the view."""
        from libemg.deep import TrainingOutcome
        outcome = TrainingOutcome.load(path)
        if any(run.outcome is not None and run.outcome.to_dict() == outcome.to_dict()
               for run in self.runs):
            return outcome          # already on screen
        self.runs.append(_Run(outcome.name, outcome.epochs, outcome))
        self._draw()
        return outcome

    def clear(self):
        """Remove every loaded run, keeping a live one."""
        self.runs = self.runs[:1] if self.live else []
        self._draw()

    def _load_chosen(self, sender, app_data):
        paths = list((app_data or {}).get("selections", {}).values()) or \
            [(app_data or {}).get("file_path_name", "")]
        loaded, failed = [], []
        for path in paths:
            if not path or not os.path.isfile(path):
                continue
            try:
                loaded.append(self.load(path))
            except Exception as error:
                failed.append(f"{Path(path).name} ({type(error).__name__}: {error})")
        if failed:
            style.set_status(self._status, "Could not load " + "; ".join(failed), "error")
        elif loaded:
            style.set_status(self._status, f"Loaded {len(loaded)} training record(s). "
                                     f"{len(self.runs)} run(s) on screen.", "ok")

    def _export_chosen(self, sender, app_data):
        path = (app_data or {}).get("file_path_name", "")
        outcome = self._exportable()
        if not path or outcome is None:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            written = outcome.save(path)
            style.set_status(self._status, f"Exported {outcome.name} to {written}.", "ok")
        except Exception as error:
            style.set_status(self._status, f"Could not export: {type(error).__name__}: {error}",
                             "error")

    def _exportable(self):
        """The record Export writes: the live run once finished, else the newest loaded."""
        for run in (self.runs[:1] if self.live else []) + list(reversed(self.runs)):
            if run.outcome is not None:
                return run.outcome
        return None

    # -- drawing -----------------------------------------------------------------
    def _draw(self):
        if not dpg.does_item_exist(self.window_tag):
            return
        for axis in (self._loss_y, self._score_y):
            dpg.delete_item(axis, children_only=True)
        style.set_enabled(self._export, self._exportable() is not None)
        style.set_enabled(self._clear, len(self.runs) > (1 if self.live else 0))
        runs = [run for run in self.runs if run.epochs]
        if len(runs) == 1:
            self._draw_one(runs[0])
        else:
            for i, run in enumerate(runs):
                self._draw_compared(run, _RUN_COLOURS[i % len(_RUN_COLOURS)])
        scored = [e for run in runs for e in run.epochs if e.score is not None]
        if scored:
            name = scored[-1].score_name
            dpg.configure_item(self._score_y, label="Accuracy" if name == "accuracy" else "RMSE")
        for axis in (self._loss_x, self._loss_y, self._score_x, self._score_y):
            dpg.fit_axis_data(axis)
        self._draw_table(runs)

    def _draw_one(self, run):
        epochs = run.epochs
        x = [float(e.epoch) for e in epochs]
        self._line(self._loss_y, x, [e.train_loss for e in epochs], "Training", _TRAINING)
        held = [e for e in epochs if e.validation_loss is not None]
        if held:
            self._line(self._loss_y, [float(e.epoch) for e in held],
                       [e.validation_loss for e in held], "Validation", _VALIDATION)
        best = _best(epochs)
        line = dpg.add_inf_line_series([float(best.epoch)], label="Best epoch", parent=self._loss_y)
        dpg.bind_item_theme(line, _line_theme(style.MUTED))
        scored = [e for e in epochs if e.score is not None]
        if scored:
            self._line(self._score_y, [float(e.epoch) for e in scored], [e.score for e in scored],
                       ("Validation " if held else "Training ") + scored[-1].score_name, _SCORE)

    def _draw_compared(self, run, colour):
        epochs = run.epochs
        held = [e for e in epochs if e.validation_loss is not None]
        source = held or epochs
        self._line(self._loss_y, [float(e.epoch) for e in source],
                   [e.validation_loss if held else e.train_loss for e in source], run.name,
                   colour)
        scored = [e for e in epochs if e.score is not None]
        if scored:
            self._line(self._score_y, [float(e.epoch) for e in scored], [e.score for e in scored],
                       run.name, colour)

    def _draw_table(self, runs):
        dpg.delete_item(self._runs, children_only=True)
        if not runs:
            return
        with dpg.table(parent=self._runs, header_row=True, row_background=True,
                       policy=dpg.mvTable_SizingStretchProp, borders_innerH=True):
            for label, weight in (("Run", 3.0), ("Model", 3.0), ("Epochs", 1.0),
                                  ("Best", 2.2), ("Trained on", 2.5)):
                dpg.add_table_column(label=label, init_width_or_weight=weight)
            compared = len(runs) > 1
            for i, run in enumerate(runs):
                outcome, best = run.outcome, _best(run.epochs)
                with dpg.table_row():
                    dpg.add_text(run.name, color=_RUN_COLOURS[i % len(_RUN_COLOURS)]
                                 if compared else style.TEXT)
                    if outcome is not None:
                        options = outcome.options
                        dpg.add_text(f"{outcome.architecture.upper()}, {outcome.timing} "
                                     f"{outcome.task}, lr {options.get('learning_rate', '?')}, "
                                     f"seq {options.get('sequence_length', '?')}",
                                     color=style.MUTED)
                    else:
                        dpg.add_text("training...", color=style.MUTED)
                    dpg.add_text(f"{len(run.epochs)}" + (" (stopped)" if outcome is not None
                                                         and outcome.stopped else ""))
                    dpg.add_text(f"epoch {best.epoch}, {_score(best)}")
                    data = outcome.data if outcome is not None else {}
                    dpg.add_text(f"{data.get('recordings', '?')} recordings, "
                                 f"{data.get('windows', '?')} windows" if data else "",
                                 color=style.MUTED)

    def _line(self, axis, x, y, label, colour):
        series = dpg.add_line_series([float(v) for v in x], [float(v) for v in y], label=label,
                                     parent=axis)
        dpg.bind_item_theme(series, _line_theme(colour))
        return series


def _best(epochs):
    return min(epochs, key=lambda e: e.validation_loss if e.validation_loss is not None
               else e.train_loss)


def _score(result):
    if result.score is None:
        return f"loss {result.train_loss:.3f}"
    if result.score_name == "accuracy":
        return f"accuracy {result.score:.1%}"
    return f"{result.score_name} {result.score:.3f}"


def _line_theme(colour):
    alias = "__training_line_{}_{}_{}".format(*colour[:3])
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        for component in (dpg.mvLineSeries, dpg.mvInfLineSeries):
            with dpg.theme_component(component):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 255),
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3.0,
                                    category=dpg.mvThemeCat_Plots)
    return alias
