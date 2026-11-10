"""See what a feature selection chose, and why it chose it over everything else.

A Feature Selection block runs :meth:`libemg.feature_selector.FeatureSelector.run_study`
over the windows a pipeline reads, and this window draws the
:class:`~libemg.feature_selector.SelectionStudy` it returns -- or one saved
earlier and loaded from Pipeline > Feature Selection Results. Three views:

    Overview.        The criterion and the held-out accuracy as each feature is
                     added, for every subject or fold; and which features each
                     one chose, in what order.
    Decisions.       One iteration at a time: every candidate that was scored,
                     best first, with the one chosen marked, how far ahead of
                     the runner-up it was, and -- for a shared feature set --
                     what each subject's own score was.
    All iterations.  Every score of every iteration at once.

As in the other viewers, everything drawn is computed by plain functions at the
top of the module, so it can be checked without a display.
"""

import math
import os

import numpy as np

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style
from libemg._gui._pipeline.feature_viewer import _plot_theme
from libemg._feature_selection_metrics import METRICS, resolve_metric

#: The feature bank choice that offers every feature the extractor has.
ALL_FEATURES = "(all features)"

#: What each metric is called on screen. The registry's names run the words
#: together ('meanwithinrepetitionrepeatability'), which nobody can read.
METRIC_TITLES = {
    "CA": "Accuracy", "AE": "Active error", "MSA": "Mean semi-principal axis",
    "FE": "Feature efficiency", "RI": "Repeatability", "SI": "Separability",
    "mwRI": "Mean within-repetition repeatability",
    "swRI": "Std of within-repetition repeatability",
    "swSI": "Std of within-trial separability", "CD": "Centroid drift",
    "MAV": "Mean absolute value", "mSI": "Modified separability",
    "mwSI": "Mean within-trial separability", "BD": "Bhattacharyya distance",
    "KLD": "Kullback-Leibler divergence", "HD": "Hellinger distance",
    "FDR": "Fisher discriminant ratio", "VOR": "Volume of overlap region",
    "TSM": "Trace of scatter matrices", "DS": "Desirability score",
    "CDM": "Class discriminability", "PU": "Purity", "rPU": "Rescaled purity",
    "NS": "Neighbourhood separability", "rNS": "Rescaled neighbourhood separability",
    "CE": "Collective entropy", "rCE": "Rescaled collective entropy", "C": "Compactness",
    "rC": "Rescaled compactness", "ACA": "Active classification accuracy",
    "UD": "Usable data", "ICF": "Interclass fraction", "IIF": "Intra/inter fraction",
    "TT": "Training time",
}

#: Readable on the dark background and distinct from one another, in order.
UNIT_COLOURS = [
    (96, 165, 250), (251, 146, 60), (74, 222, 128), (248, 113, 113),
    (192, 132, 252), (250, 204, 21), (45, 212, 191), (244, 114, 182),
    (163, 230, 53), (148, 163, 184), (56, 189, 248), (251, 191, 36),
]
CHOSEN = (59, 130, 246)
OTHER = (110, 118, 132)
MEAN = (236, 239, 244)


# ======================================================================
# what is drawn
# ======================================================================
def metric_title(code):
    return METRIC_TITLES.get(code, METRICS[code].name if code in METRICS else str(code))


def metric_choices():
    """Every metric as the block's dropdown offers it: 'Accuracy (CA)'."""
    return [f"{metric_title(code)} ({code})" for code in METRICS]


def metric_code(choice):
    """The abbreviation a dropdown entry, a name or an abbreviation stands for."""
    text = str(choice or "CA").strip()
    if text.endswith(")") and "(" in text:
        text = text[text.rindex("(") + 1:-1]
    return resolve_metric(text)


def better(objective):
    return "lower is better" if objective == "min" else "higher is better"


def format_score(value, digits=4):
    """A score as text: 'inf' for infinity, a dash where there is none."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}"


def chosen_at(trace, iteration):
    """The group a sequential iteration chose, or None for a filter ranking."""
    if trace.strategy == "filter" or iteration >= len(trace.selected):
        return None
    return trace.selected[iteration]


def _gap(value, reference, objective):
    """How far ``value`` is ahead of ``reference``: positive is better, in score units."""
    if value == reference:
        return 0.0
    with np.errstate(invalid="ignore"):
        gap = (value - reference) if objective == "max" else (reference - value)
    return float(gap)


def decision_rows(trace, iteration):
    """Every candidate scored at ``iteration``, best first, as rows of a table.

    Returns
    ----------
    list of dict
        ``rank``, ``name``, ``score``, ``gap`` (how far ahead of the chosen
        group it is, in score units: zero for the chosen one, negative for
        those it beat), ``chosen`` and ``parts`` (each subject's score, for an
        aggregated criterion, else None). For a filter, every selected group
        is ``chosen`` and the gap is to the best.
    """
    ranking = trace.ranking(iteration)
    if not ranking:
        return []
    chosen = chosen_at(trace, iteration)
    reference = ranking[0][1] if chosen is None else dict(ranking)[chosen]
    selected = set(trace.selected) if trace.strategy == "filter" else {chosen}
    rows = []
    for rank, (name, value) in enumerate(ranking, 1):
        index = trace.candidates.index(name)
        parts = None if trace.parts is None else trace.parts[iteration, index].tolist()
        rows.append({"rank": rank, "name": name, "score": value,
                     "gap": _gap(value, reference, trace.objective),
                     "chosen": name in selected, "parts": parts})
    return rows


def part_winners(trace, iteration):
    """For an aggregated criterion, which candidate each subject alone would have chosen."""
    if trace.parts is None:
        return []
    winners = []
    pick = np.nanargmin if trace.objective == "min" else np.nanargmax
    for part in range(trace.parts.shape[2]):
        column = trace.parts[iteration, :, part]
        winners.append(None if np.isnan(column).all() else trace.candidates[int(pick(column))])
    return winners


def explain(trace, iteration, aggregate="mean"):
    """A few sentences on why ``iteration`` went the way it did."""
    rows = decision_rows(trace, iteration)
    title = metric_title(trace.metric).lower()
    if not rows:
        return "Nothing could be scored at this iteration."
    if trace.strategy == "filter":
        kept = len(trace.selected)
        text = (f"Every group was scored alone by {title} ({better(trace.objective)}) and "
                f"ranked. The top {kept} were kept, best first: "
                f"{', '.join(trace.selected[:5])}{'...' if kept > 5 else ''}.")
    else:
        before = trace.chosen_before(iteration)
        chosen = next(r for r in rows if r["chosen"])
        start = (f"With {', '.join(before)} already chosen, each" if before
                 else "With nothing chosen yet, each")
        text = (f"Iteration {iteration + 1}. {start} of the {len(rows)} remaining groups was "
                f"added in turn and the set scored by {title} ({better(trace.objective)}). "
                f"{chosen['name']} gave the best score, {format_score(chosen['score'])}")
        others = [r for r in rows if not r["chosen"]]
        if not others:
            text += ", and was the only group left."
        else:
            runner = others[0]
            if runner["gap"] == 0:
                text += (f", tied with {runner['name']}; the one earlier in the bank was "
                         "taken.")
            else:
                text += (f", {format_score(-runner['gap'])} ahead of the runner-up "
                         f"{runner['name']} ({format_score(runner['score'])}).")
    winners = part_winners(trace, iteration)
    if winners:
        top = rows[0]["name"]
        agree = sum(1 for w in winners if w == top)
        text += (f" Each candidate's score is the {aggregate} of every subject's own score; "
                 f"{agree} of {len(winners)} subjects would have chosen {top} on their own.")
    return text


def trace_cells(trace):
    """Every score of every iteration, as rows for a table.

    Returns
    ----------
    list of (str, list of (str, str))
        One row per candidate -- those chosen first, in the order chosen, then
        the rest -- each with a ``(text, state)`` per iteration. ``state`` is
        'chosen' where the iteration chose it, 'in set' after that, 'scored'
        or 'none' where it has no score.
    """
    order = list(trace.selected) + [c for c in trace.candidates if c not in trace.selected]
    rows = []
    for name in order:
        index = trace.candidates.index(name)
        cells = []
        for iteration in range(trace.iterations):
            value = trace.scores[iteration, index]
            if trace.strategy != "filter" and name in trace.selected[:iteration]:
                cells.append(("in set", "in set"))
            elif np.isnan(value):
                cells.append(("-", "none"))
            elif (trace.strategy == "filter" and name in trace.selected) or \
                    chosen_at(trace, iteration) == name:
                cells.append((format_score(value), "chosen"))
            else:
                cells.append((format_score(value), "scored"))
        rows.append((name, cells))
    return rows


def unit_curves(study):
    """Per unit: its name, its criterion after each addition, and its held-out accuracy."""
    return [(unit.name, np.asarray(unit.trace.selected_scores, dtype=float),
             None if unit.test_accuracy is None else np.asarray(unit.test_accuracy, dtype=float))
            for unit in study.units]


def test_curves(study):
    """The held-out accuracy curves worth drawing, and the mean of them.

    One curve per unit when there are several units; with a single unit (a
    shared or pooled feature set) one per tested subject instead, since that
    is where the spread is.

    Returns
    ----------
    (list of (str, ndarray), ndarray or None)
    """
    if len(study.units) == 1:
        unit = study.units[0]
        curves = [(f"Subject {name}", np.asarray(curve, dtype=float))
                  for name, curve in unit.test_accuracy_by_subject.items()]
        mean = unit.test_accuracy
    else:
        curves = [(name, test) for name, _, test in unit_curves(study) if test is not None]
        mean = study.mean_test_accuracy()
    return curves, (None if mean is None else np.asarray(mean, dtype=float))


def best_count(curve):
    """How many groups gave the best accuracy on a curve, and that accuracy, or None."""
    if curve is None or not len(curve) or np.isnan(curve).all():
        return None
    index = int(np.nanargmax(curve))
    return index + 1, float(curve[index])


def bar_limits(lengths):
    """Where a bar chart's value axis should start and end.

    A decision usually turns on small differences between large scores --
    accuracies of 91 and 92 -- which bars drawn from zero would hide. So the
    axis is narrowed to the spread of the scores, never past zero.
    """
    lengths = np.asarray(lengths, dtype=float)
    low, high = float(lengths.min()), float(lengths.max())
    span = high - low
    if span == 0:
        span = abs(high) * 0.1 or 1.0
    start, end = low - 0.6 * span, high + 0.08 * span
    if low >= 0:
        start = max(0.0, start)
    if high <= 0:
        end = min(0.0, end)
    return start, end


def finite_bars(values):
    """Bar lengths for scores that may be infinite: infinity drawn a little past the rest."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    top = float(np.max(np.abs(finite))) if len(finite) else 1.0
    top = top if top > 0 else 1.0
    return np.where(np.isposinf(values), top * 1.15,
                    np.where(np.isneginf(values), -top * 1.15, values))


# ======================================================================
# the window
# ======================================================================
class SelectionViewer:
    """A window onto a :class:`~libemg.feature_selector.SelectionStudy`.

    Parameters
    ----------
    study: SelectionStudy or None
        What to show. None opens an empty window to Load results into.
    """

    _instances = 0

    def __init__(self, study=None, width=1180, height=820, pos=(40, 50),
                 title="Feature Selection Results"):
        SelectionViewer._instances += 1
        self.prefix = f"__fs{SelectionViewer._instances}"
        self.window_tag = self._tag("window")
        self.study = study
        self.width, self.height, self.pos, self.title = width, height, pos, title
        self.unit = 0
        self.iteration = 0

    def _tag(self, name):
        return f"{self.prefix}_{name}"

    # ------------------------------------------------------------------
    def spawn_window(self):
        self.cleanup()
        with dpg.window(label=self.title, tag=self.window_tag, width=self.width,
                        height=self.height, pos=self.pos, on_close=self.cleanup):
            with dpg.group(horizontal=True):
                style.button("Load...", callback=lambda: dpg.show_item(self._tag("open")))
                style.button("Save as...", tag=self._tag("save_button"),
                             callback=lambda: dpg.show_item(self._tag("save")))
                style.note("", tag=self._tag("status"))
            dpg.add_group(tag=self._tag("body"))
        for purpose, callback in (("open", self._opened), ("save", self._saved)):
            with dpg.file_dialog(tag=self._tag(purpose), show=False, directory_selector=False,
                                 width=640, height=420, modal=True, callback=callback):
                dpg.add_file_extension(".json")
                dpg.add_file_extension(".*")
        self._build_body()
        return self

    def cleanup(self):
        for name in ("window", "open", "save"):
            if dpg.does_item_exist(self._tag(name)):
                dpg.delete_item(self._tag(name))

    def poll(self):
        """Nothing runs in the background; here so the GUI can poll it like any panel."""

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------
    def set_study(self, study):
        self.study, self.unit, self.iteration = study, 0, 0
        self._build_body()

    def load(self, path):
        from libemg.feature_selector import SelectionStudy
        study = SelectionStudy.load(path)
        study.settings["saved_to"] = path
        self.set_study(study)
        style.set_status(self._tag("status"), f"Loaded {path}.", "ok")
        return study

    def save(self, path):
        if self.study is None:
            return None
        if not os.path.splitext(path)[1]:
            path += ".json"
        self.study.save(path)
        self.study.settings["saved_to"] = path
        style.set_status(self._tag("status"), f"Saved to {path}.", "ok")
        return path

    def _opened(self, sender, app_data):
        path = (app_data or {}).get("file_path_name", "")
        try:
            self.load(path)
        except Exception as error:      # a bad file is reported, not raised into the GUI loop
            style.set_status(self._tag("status"), f"Could not load {path}: {error}", "error")

    def _saved(self, sender, app_data):
        path = (app_data or {}).get("file_path_name", "")
        if path.endswith(".*"):
            path = path[:-2]
        try:
            self.save(path)
        except Exception as error:
            style.set_status(self._tag("status"), f"Could not save {path}: {error}", "error")

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def _build_body(self):
        body = self._tag("body")
        if not dpg.does_item_exist(body):
            return
        dpg.delete_item(body, children_only=True)
        style.set_enabled(self._tag("save_button"), self.study is not None)
        dpg.push_container_stack(body)
        try:
            if self.study is None:
                style.note("No results yet. Run a pipeline with a Feature Selection block, "
                           "or Load results saved from one.", wrap=self.width - 60)
                return
            self._build_header()
            with dpg.tab_bar(tag=self._tag("tabs")):
                with dpg.tab(label="Overview", tag=self._tag("tab_overview")):
                    self._build_overview()
                with dpg.tab(label="Decisions", tag=self._tag("tab_decisions")):
                    self._build_unit_picker("decisions")
                    dpg.add_group(tag=self._tag("decisions"))
                with dpg.tab(label="All iterations", tag=self._tag("tab_trace")):
                    self._build_unit_picker("trace")
                    dpg.add_group(tag=self._tag("trace"))
                notes = self._notes()
                if notes:
                    with dpg.tab(label=f"Notes ({len(notes)})"):
                        for line in notes:
                            style.note(line, wrap=self.width - 60)
        finally:
            dpg.pop_container_stack()
        self._build_decisions()
        self._build_trace()

    def _notes(self):
        settings = self.study.settings
        notes = [f"Left out of the bank: {s}" for s in settings.get("skipped_features") or []]
        seen = set()
        for unit in self.study.units:
            for line in unit.trace.notes:
                if line not in seen:
                    seen.add(line)
                    notes.append(f"{unit.name}: {line}" if len(self.study.units) > 1 else line)
        return notes

    def _build_header(self):
        study, settings = self.study, self.study.settings
        heading = dpg.add_text(f"{study.title}  |  {metric_title(study.metric)} ({study.metric})")
        if style.font("bold") is not None:
            dpg.bind_item_font(heading, style.font("bold"))
        how = ("sequential forward selection" if study.strategy == "sequential"
               else "filter ranking, every group scored alone")
        parts = [f"{len(study.candidates)} candidate features", how, better(study.objective)]
        test = settings.get("test_repetitions")
        if test:
            parts.append(f"tested on repetitions {', '.join(str(v) for v in test)}")
        elif study.scheme == "cross_subject":
            parts.append("tested on each left-out subject")
        else:
            parts.append("nothing held out for testing")
        if settings.get("classifier"):
            parts.append(f"classifier: {settings['classifier']}")
        style.note(";  ".join(parts) + ".", wrap=self.width - 60)
        meta = [f"Run {study.created}" if study.created else ""]
        if settings.get("windows"):
            meta.append(f"{settings['windows']} windows")
        if settings.get("subjects"):
            meta.append(f"{len(settings['subjects'])} subjects")
        if settings.get("saved_to"):
            meta.append(f"file: {settings['saved_to']}")
        style.note("  |  ".join(m for m in meta if m), wrap=self.width - 60)

    def _build_unit_picker(self, area):
        if len(self.study.units) < 2:
            return
        with dpg.group(horizontal=True):
            dpg.add_text("Selection")
            dpg.add_combo([u.name for u in self.study.units], tag=self._tag(f"unit_{area}"),
                          default_value=self.study.units[self.unit].name, width=280,
                          callback=self._unit_chosen)

    def _unit_chosen(self, sender, name):
        names = [u.name for u in self.study.units]
        self.unit, self.iteration = names.index(name), 0
        for area in ("decisions", "trace"):
            picker = self._tag(f"unit_{area}")
            if dpg.does_item_exist(picker):
                dpg.set_value(picker, name)
        self._build_decisions()
        self._build_trace()

    # ------------------------------------------------------------------
    # overview
    # ------------------------------------------------------------------
    def _build_overview(self):
        study = self.study
        half = max(320, (self.width - 70) // 2)
        sequential = study.strategy == "sequential"
        with dpg.group(horizontal=True):
            with dpg.plot(label=("Criterion as features are added" if sequential
                                 else "Each feature's own score, by rank"),
                          width=half, height=300, no_menus=True) as plot:
                dpg.bind_item_theme(plot, _plot_theme())
                dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                dpg.add_plot_axis(dpg.mvXAxis, label=("features in the set" if sequential
                                                      else "rank"), tag=self._tag("cx"))
                y = dpg.add_plot_axis(dpg.mvYAxis, label=f"{metric_title(study.metric)} "
                                                          f"({better(study.objective)})")
                for position, (name, criterion, _) in enumerate(unit_curves(study)):
                    if len(criterion):
                        self._line(y, np.arange(1, len(criterion) + 1),
                                   finite_bars(criterion), name, UNIT_COLOURS[position % 12])
                dpg.fit_axis_data(y)
            _whole_ticks(self._tag("cx"), max([len(c) for _, c, _ in unit_curves(study)] + [1]))
            curves, mean = test_curves(study)
            with dpg.plot(label="Held-out accuracy as features are added", width=half,
                          height=300, no_menus=True) as plot:
                dpg.bind_item_theme(plot, _plot_theme())
                dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                dpg.add_plot_axis(dpg.mvXAxis, label="features in the set", tag=self._tag("tx"))
                y = dpg.add_plot_axis(dpg.mvYAxis, label="accuracy (%)")
                for position, (name, curve) in enumerate(curves):
                    self._line(y, np.arange(1, len(curve) + 1), curve, name,
                               UNIT_COLOURS[position % 12])
                if mean is not None and len(curves) > 1:
                    self._line(y, np.arange(1, len(mean) + 1), mean, "Mean", MEAN, weight=3)
                elif mean is not None and not curves:
                    self._line(y, np.arange(1, len(mean) + 1), mean, "Accuracy", MEAN, weight=3)
                dpg.fit_axis_data(y)
            _whole_ticks(self._tag("tx"), max([len(c) for _, c in curves]
                                              + [0 if mean is None else len(mean), 1]))
        best = best_count(mean)
        if best is None:
            style.note("Nothing was held out, so there is no test accuracy. Set Test "
                       "Repetitions on the block to hold some out.", wrap=self.width - 60)
        else:
            style.note(f"Best {'mean ' if len(curves) > 1 else ''}held-out accuracy: "
                       f"{best[1]:.1f}% with the first {best[0]} feature"
                       f"{'s' if best[0] > 1 else ''}"
                       + (f"; {mean[-1]:.1f}% with all {len(mean)}." if len(mean) > 1 else "."),
                       wrap=self.width - 60)
        if len(study.units) > 1:
            self._build_rank_table()
        else:
            self._build_order_table(study.units[0])

    def _build_rank_table(self):
        study = self.study
        style.section("What each selection chose, and when")
        style.note("A number is the place a feature was chosen in. Features most often chosen "
                   "come first.", wrap=self.width - 60)
        ranks = study.rank_matrix()
        consensus = study.consensus()
        with dpg.table(header_row=True, scrollX=True, scrollY=True, freeze_columns=1,
                       freeze_rows=1, height=-1, policy=dpg.mvTable_SizingFixedFit,
                       borders_innerV=True, borders_innerH=True, row_background=True) as table:
            dpg.add_table_column(label="Feature", width_fixed=True, init_width_or_weight=110)
            dpg.add_table_column(label="Chosen by", width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(label="Mean place", width_fixed=True, init_width_or_weight=86)
            for unit in study.units:
                dpg.add_table_column(label=unit.name.replace("Left out: ", "-"),
                                     width_fixed=True, init_width_or_weight=78)
            count = len(study.candidates)
            for row, (name, chosen_by, mean_rank) in enumerate(consensus):
                column = study.candidates.index(name)
                with dpg.table_row():
                    dpg.add_text(name)
                    dpg.add_text(f"{chosen_by} of {len(study.units)}")
                    dpg.add_text(format_score(mean_rank, 3))
                    for position in range(len(study.units)):
                        rank = int(ranks[position, column])
                        dpg.add_text(str(rank) if rank else "")
                        if rank:
                            strength = 1.0 - (rank - 1) / max(1, count)
                            dpg.highlight_table_cell(table, row, 3 + position,
                                                     (*CHOSEN, int(40 + 150 * strength)))

    def _build_order_table(self, unit):
        style.section("Chosen, in order")
        trace = unit.trace
        sequential = trace.strategy == "sequential"
        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True,
                       height=-1, borders_innerH=True, row_background=True):
            for label, width in (("Place", 50), ("Feature", 110),
                                 ("Criterion" if sequential else "Score", 100),
                                 ("Ahead of runner-up by" if sequential
                                  else "Ahead of next by", 160),
                                 ("Runner-up" if sequential else "Next", 110),
                                 ("Held-out accuracy (%)", 160)):
                dpg.add_table_column(label=label, width_fixed=True, init_width_or_weight=width)
            scores = trace.selected_scores
            for step, name in enumerate(trace.selected):
                margin, runner = "", ""
                if sequential:
                    others = [r for r in decision_rows(trace, step) if not r["chosen"]]
                    if others:
                        margin, runner = format_score(-others[0]["gap"]), others[0]["name"]
                elif step + 1 < len(trace.selected):
                    following = trace.selected[step + 1]
                    margin = format_score(_gap(scores[step], scores[step + 1], trace.objective))
                    runner = following
                accuracy = ("" if unit.test_accuracy is None
                            else format_score(float(unit.test_accuracy[step]), 3))
                with dpg.table_row():
                    for text in (str(step + 1), name, format_score(float(scores[step])), margin,
                                 runner, accuracy):
                        dpg.add_text(text)

    # ------------------------------------------------------------------
    # decisions: one iteration at a time
    # ------------------------------------------------------------------
    def _build_decisions(self):
        area = self._tag("decisions")
        if self.study is None or not dpg.does_item_exist(area):
            return
        dpg.delete_item(area, children_only=True)
        trace = self.study.units[self.unit].trace
        dpg.push_container_stack(area)
        try:
            if trace.iterations > 1:
                with dpg.group(horizontal=True):
                    dpg.add_text("Iteration")
                    style.button("<", width=34, callback=lambda: self._step(-1))
                    dpg.add_slider_int(tag=self._tag("iteration"), min_value=1,
                                       max_value=trace.iterations,
                                       default_value=self.iteration + 1, width=260,
                                       callback=lambda s, a: self._go(int(a) - 1))
                    style.button(">", width=34, callback=lambda: self._step(1))
                    style.note(f"of {trace.iterations}: what was added at each step")
            dpg.add_group(tag=self._tag("decision"))
        finally:
            dpg.pop_container_stack()
        self._build_decision()

    def _step(self, delta):
        trace = self.study.units[self.unit].trace
        self._go(min(max(0, self.iteration + delta), trace.iterations - 1))

    def _go(self, iteration):
        self.iteration = iteration
        if dpg.does_item_exist(self._tag("iteration")):
            dpg.set_value(self._tag("iteration"), iteration + 1)
        self._build_decision()

    def _build_decision(self):
        area = self._tag("decision")
        if not dpg.does_item_exist(area):
            return
        dpg.delete_item(area, children_only=True)
        trace = self.study.units[self.unit].trace
        rows = decision_rows(trace, self.iteration)
        dpg.push_container_stack(area)
        try:
            style.note(explain(trace, self.iteration,
                               self.study.settings.get("aggregate", "mean")),
                       color=style.TEXT, wrap=self.width - 60)
            if not rows:
                return
            plot_width = max(360, int((self.width - 70) * 0.45))
            with dpg.group(horizontal=True):
                with dpg.child_window(width=plot_width, height=-1, border=False):
                    self._build_bars(trace, rows, plot_width - 20)
                with dpg.child_window(width=-1, height=-1, border=False):
                    self._build_decision_table(trace, rows)
        finally:
            dpg.pop_container_stack()

    def _build_bars(self, trace, rows, width):
        """Every candidate's score as a horizontal bar, best at the top, the chosen one lit."""
        count = len(rows)
        positions = np.arange(count)[::-1].astype(float)      # best first, at the top
        lengths = finite_bars([r["score"] for r in rows])
        title = (f"Iteration {self.iteration + 1}: every candidate added to the set"
                 if trace.strategy == "sequential" else "Every candidate scored alone")
        with dpg.plot(label=title, width=width, height=max(220, 24 * count + 90),
                      no_menus=True, no_mouse_pos=True):
            x = dpg.add_plot_axis(dpg.mvXAxis, label=f"{metric_title(trace.metric)} "
                                                      f"({better(trace.objective)})")
            y = dpg.add_plot_axis(dpg.mvYAxis, no_gridlines=True)
            dpg.set_axis_ticks(y, tuple((r["name"], float(p)) for r, p in zip(rows, positions)))
            for chosen, colour, label in ((True, CHOSEN, "chosen"), (False, OTHER, "not chosen")):
                mask = np.array([r["chosen"] == chosen for r in rows])
                if mask.any():
                    series = dpg.add_bar_series(lengths[mask].tolist(), positions[mask].tolist(),
                                                horizontal=True, weight=0.7, label=label,
                                                parent=y)
                    dpg.bind_item_theme(series, _bar_theme(colour))
            dpg.set_axis_limits(y, -0.7, count - 0.3)
            dpg.set_axis_limits(x, *bar_limits(lengths))

    def _build_decision_table(self, trace, rows):
        names = trace.part_names or []
        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, scrollX=True,
                       scrollY=True, freeze_rows=1, freeze_columns=2, height=-1,
                       borders_innerH=True, row_background=True) as table:
            dpg.add_table_column(label="Rank", width_fixed=True, init_width_or_weight=44)
            dpg.add_table_column(label="Feature", width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(label="Score", width_fixed=True, init_width_or_weight=86)
            dpg.add_table_column(label="vs chosen" if trace.strategy == "sequential"
                                 else "vs best", width_fixed=True, init_width_or_weight=86)
            for name in names:
                dpg.add_table_column(label=name.replace("Subject ", "S"), width_fixed=True,
                                     init_width_or_weight=72)
            winners = part_winners(trace, self.iteration)
            for row, entry in enumerate(rows):
                with dpg.table_row():
                    dpg.add_text(str(entry["rank"]))
                    dpg.add_text(entry["name"])
                    dpg.add_text(format_score(entry["score"]))
                    dpg.add_text("" if entry["gap"] == 0 else format_score(entry["gap"], 3))
                    for part, value in enumerate(entry["parts"] or []):
                        dpg.add_text(format_score(value))
                        if part < len(winners) and winners[part] == entry["name"]:
                            dpg.highlight_table_cell(table, row, 4 + part, (*style.SUCCESS, 70))
                if entry["chosen"]:
                    dpg.highlight_table_row(table, row, (*CHOSEN, 70))
        if names:
            style.note("Each subject's own score; green marks the candidate that subject alone "
                       "would have chosen.", wrap=max(300, self.width // 2))

    # ------------------------------------------------------------------
    # every iteration at once
    # ------------------------------------------------------------------
    def _build_trace(self):
        area = self._tag("trace")
        if self.study is None or not dpg.does_item_exist(area):
            return
        dpg.delete_item(area, children_only=True)
        trace = self.study.units[self.unit].trace
        cells = trace_cells(trace)
        dpg.push_container_stack(area)
        try:
            style.note(("Each column is one iteration: every remaining candidate added to the "
                        "set and scored. Highlighted is the one chosen; 'in set' marks those "
                        "chosen earlier." if trace.strategy == "sequential" else
                        "One pass: every candidate scored alone. Highlighted are those kept.")
                       + f" Scores are {metric_title(trace.metric).lower()}, "
                       f"{better(trace.objective)}.", wrap=self.width - 60)
            with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, scrollX=True,
                           scrollY=True, freeze_rows=1, freeze_columns=1, height=-1,
                           borders_innerH=True, borders_innerV=True) as table:
                dpg.add_table_column(label="Feature", width_fixed=True, init_width_or_weight=100)
                for iteration in range(trace.iterations):
                    dpg.add_table_column(label=str(iteration + 1) if trace.iterations > 1
                                         else "Score", width_fixed=True,
                                         init_width_or_weight=74)
                for row, (name, values) in enumerate(cells):
                    with dpg.table_row():
                        dpg.add_text(name)
                        for column, (text, state) in enumerate(values, 1):
                            dpg.add_text(text, color=style.MUTED if state in ("in set", "none")
                                         else style.TEXT)
                            if state == "chosen":
                                dpg.highlight_table_cell(table, row, column, (*CHOSEN, 140))
        finally:
            dpg.pop_container_stack()

    # ------------------------------------------------------------------
    def _line(self, axis, xs, ys, label, colour, weight=1.5):
        series = dpg.add_line_series(np.asarray(xs, float).tolist(),
                                     np.asarray(ys, float).tolist(), label=label, parent=axis)
        dpg.bind_item_theme(series, _line_theme(colour, weight))
        return series


# ----------------------------------------------------------------------
# series themes, one per colour, made once per DearPyGui context
# ----------------------------------------------------------------------
def _bar_theme(colour):
    alias = "__fs_bar_{}_{}_{}".format(*colour[:3])
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (*colour[:3], 200),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 255),
                                category=dpg.mvThemeCat_Plots)
    return alias


def _line_theme(colour, weight):
    alias = "__fs_line_{}_{}_{}_{}".format(*colour[:3], int(weight * 10))
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 230),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3,
                                category=dpg.mvThemeCat_Plots)
    return alias


def _whole_ticks(axis, count):
    """Ticks at whole numbers 1..count, thinned out so they do not overlap."""
    step = max(1, int(math.ceil(count / 12)))
    dpg.set_axis_ticks(axis, tuple((str(i), float(i)) for i in range(1, count + 1, step)))
    dpg.set_axis_limits(axis, 0.7, count + 0.3)
