"""How a trained model did: the window a model block's Statistics button opens.

Every supervised model is reported the same way, from the statistics
:func:`~libemg._gui._pipeline.fitting.fit_block` computes and keeps on the
fitted model: how much it was trained on, and how it scores on the recordings
it learned from and on those held out from it. A classifier adds its accuracy
per class and a confusion matrix; a regressor its error per degree of freedom;
a deep model the loss of every epoch.

The statistics travel inside the saved model, so loading a model in a later
session brings its report back with it.
"""

import os

import numpy as np

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style
from libemg._gui._pipeline.feature_viewer import CLASS_COLOURS

#: Height of the per-class table and confusion matrix, side by side.
CLASS_PANEL = 400


class ModelReport:
    """A window of one model block's training statistics.

    Parameters
    ----------
    node_id: str
        The block it reports on; it names the window's items.
    title: str
    statistics: dict
        As :func:`~libemg._gui._pipeline.fitting.fit_block` makes them.
    epochs: list or None
        A deep model's EpochResults, for its loss curves.
    source: str
        Where the model came from, said at the top: trained here, or a file.
    """

    def __init__(self, node_id, title, statistics, epochs=None, source="", width=760,
                 height=None, pos=(80, 80)):
        self.node_id, self.title = node_id, title
        self.statistics = statistics or {}
        self.epochs = list(epochs or [])
        self.source = source
        classification = self.statistics.get("task") == "classification"
        if height is None:
            height = 330 + (CLASS_PANEL + 10 if classification else 0) + \
                (270 if self.epochs else 0)
        self.width, self.height, self.pos = width, height, pos
        self.tag = f"__model_report_{node_id}"

    @property
    def shown(self):
        return dpg.does_item_exist(self.tag)

    def spawn_window(self):
        self.cleanup()
        stats = self.statistics
        with dpg.window(label=f"Statistics: {self.title}", tag=self.tag, width=self.width,
                        height=self.height, pos=self.pos, on_close=self.cleanup):
            if not stats:
                style.note("This model carries no training statistics: it was fitted outside "
                           "the pipeline editor, or by an older version.", wrap=self.width - 40)
                return self
            self._summary(stats)
            classification = stats.get("task") == "classification"
            style.section("Scores")
            self._scores(stats, classification)
            if classification:
                # Side by side, so the whole report fits without scrolling.
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=260, height=CLASS_PANEL, border=False):
                        self._per_class(stats)
                    with dpg.child_window(width=-1, height=CLASS_PANEL, border=False):
                        self._confusion(stats)
            if self.epochs:
                self._losses()
        return self

    def cleanup(self):
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)

    # -- parts -------------------------------------------------------------
    def _summary(self, stats):
        held = len(stats.get("validation_files") or [])
        lines = [f"{stats.get('model', '?')}, learning '{stats.get('label_field', '?')}'"
                 + (f" over classes {stats['classes']}" if stats.get("classes") else "")
                 + (f" ({stats['dofs']} degrees of freedom)" if stats.get("dofs") else "") + ".",
                 f"{stats.get('windows', 0)} windows from {stats.get('recordings', 0)} "
                 f"recordings; {held} recording{'s' if held != 1 else ''} held out for "
                 "validation and never trained on."
                 if held else f"{stats.get('windows', 0)} windows from "
                              f"{stats.get('recordings', 0)} recordings, all trained on: there "
                              "is no validation score. Set Validation Share above 0 for one."]
        if stats.get("epochs"):
            lines.append(f"{stats['epochs']} epochs" + (", stopped early; the best epoch was "
                                                         "kept." if stats.get("stopped") else "."))
        if self.source:
            lines.append(self.source)
        style.note("\n".join(lines), wrap=self.width - 40)

    def _scores(self, stats, classification):
        sides = [(name, stats.get(key)) for name, key in (("Training", "training"),
                                                          ("Validation", "validation"))]
        with dpg.table(header_row=True, row_background=True, borders_innerH=True,
                       policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="")
            dpg.add_table_column(label="Recordings")
            dpg.add_table_column(label="Windows")
            if classification:
                dpg.add_table_column(label="Accuracy")
            else:
                dpg.add_table_column(label="RMSE")
                dpg.add_table_column(label="MAE")
                dpg.add_table_column(label="R²")
            for name, result in sides:
                with dpg.table_row():
                    dpg.add_text(name, color=style.HEADING)
                    if not result:
                        dpg.add_text("-", color=style.MUTED)
                        dpg.add_text("-", color=style.MUTED)
                        dpg.add_text("not held out" if name == "Validation" else "-",
                                     color=style.MUTED)
                        continue
                    dpg.add_text(str(result.get("recordings", "")))
                    dpg.add_text(str(result.get("windows", "")))
                    if classification:
                        dpg.add_text(f"{result['accuracy']:.1%}")
                    else:
                        dpg.add_text(_per_dof(result["rmse"]))
                        dpg.add_text(_per_dof(result["mae"]))
                        dpg.add_text(_per_dof(result["r2"]))

    def _per_class(self, stats):
        training = (stats.get("training") or {}).get("per_class", {})
        validation = (stats.get("validation") or {}).get("per_class", {})
        classes = sorted(set(training) | set(validation), key=lambda c: (str(type(c)), c))
        if not classes:
            return
        style.section("Per class")
        with dpg.table(header_row=True, row_background=True, borders_innerH=True,
                       policy=dpg.mvTable_SizingStretchProp, height=CLASS_PANEL - 50,
                       scrollY=True):
            dpg.add_table_column(label="Class")
            dpg.add_table_column(label="Training")
            dpg.add_table_column(label="Validation")
            for c in classes:
                with dpg.table_row():
                    dpg.add_text(str(c))
                    dpg.add_text(_percent(training.get(c)))
                    dpg.add_text(_percent(validation.get(c)))

    def _confusion(self, stats):
        side, result = ("validation", stats.get("validation")) if stats.get("validation") \
            else ("training", stats.get("training"))
        if not result or not result.get("confusion"):
            return
        matrix = np.asarray(result["confusion"], dtype=float)
        classes = [str(c) for c in result["classes"]]
        count = len(classes)
        style.section(f"Confusion matrix ({side})")
        style.note("A row per class performed, a column per class predicted; each row as a "
                   "share of that class's windows.", wrap=self.width - 320)
        rows = matrix.sum(axis=1, keepdims=True)
        share = np.divide(matrix, rows, out=np.zeros_like(matrix), where=rows > 0)
        size = CLASS_PANEL - 110
        with dpg.group(horizontal=True):
            with dpg.plot(width=size, height=size, no_menus=True, no_mouse_pos=True,
                          equal_aspects=True) as plot:
                dpg.bind_colormap(plot, dpg.mvPlotColormap_Viridis)
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Predicted", lock_min=True,
                                           lock_max=True, no_gridlines=True)
                with dpg.plot_axis(dpg.mvYAxis, label="Performed", lock_min=True,
                                   lock_max=True, no_gridlines=True) as y_axis:
                    # Row 0 is drawn at the top, so the y ticks count down.
                    dpg.add_heat_series(share.ravel().tolist(), rows=count, cols=count,
                                        scale_min=0.0, scale_max=1.0, format="%.2f",
                                        bounds_min=(0, 0), bounds_max=(count, count))
                dpg.set_axis_ticks(x_axis, tuple((c, i + 0.5) for i, c in enumerate(classes)))
                dpg.set_axis_ticks(y_axis, tuple((c, count - i - 0.5)
                                                 for i, c in enumerate(classes)))
                dpg.set_axis_limits(x_axis, 0, count)
                dpg.set_axis_limits(y_axis, 0, count)
            dpg.add_colormap_scale(min_scale=0.0, max_scale=1.0, height=size, width=60,
                                   colormap=dpg.mvPlotColormap_Viridis)

    def _losses(self):
        style.section("Loss per epoch")
        x = [r.epoch for r in self.epochs]
        with dpg.plot(height=220, width=-1, no_menus=True):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
            with dpg.plot_axis(dpg.mvYAxis, label="Loss") as y_axis:
                series = dpg.add_line_series(x, [r.train_loss for r in self.epochs],
                                             label="Training")
                dpg.bind_item_theme(series, _line_theme(CLASS_COLOURS[0]))
                validation = [r.validation_loss for r in self.epochs]
                if all(v is not None for v in validation):
                    series = dpg.add_line_series(x, validation, label="Validation")
                    dpg.bind_item_theme(series, _line_theme(CLASS_COLOURS[1]))
            dpg.set_axis_limits_auto(x_axis)
            dpg.set_axis_limits_auto(y_axis)


def source_text(how, path=""):
    """Where a model came from, as the report's last line says it."""
    name = os.path.basename(path) if path else ""
    if how == "loaded":
        return f"Loaded from {path}." if path else "Loaded from a file."
    return "Trained in this session" + (f", saved to {name}." if name else "; not saved yet.")


def _per_dof(values):
    values = [v for v in values]
    if len(values) == 1:
        return "-" if values[0] is None else f"{values[0]:.3g}"
    return ", ".join("-" if v is None else f"{v:.3g}" for v in values)


def _percent(value):
    return "-" if value is None else f"{value:.1%}"


def _line_theme(colour):
    alias = "__mr_line_{}_{}_{}".format(*colour[:3])
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0,
                                category=dpg.mvThemeCat_Plots)
    return alias
