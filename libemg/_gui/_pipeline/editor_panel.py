"""The node editor: a view onto a pipeline document.

Everything this window does is edit a
:class:`~libemg._gui._pipeline.document.PipelineDocument` and then ask the
compiler to run it. It holds no pipeline state of its own, which is what lets
the same pipeline be built, saved, validated and run with no display at all.

Two things the toolkit does for us
----------------------------------
A node attribute carries a ``category``, and DearPyGui refuses to link two
attributes whose categories differ. Setting the category to the port's type
means a wrong connection cannot be made in the first place, rather than being
made and then explained away. The rules a category cannot express, such as an
input that already has a link, are checked in the link callback and reported in
words.

A node also carries its position, and the position can be read back, so canvas
layout survives a save without the document having to track it separately.

Where the work happens
----------------------
Nothing in the running pipeline lives in this process. The stages run in their
own processes and publish to shared memory; this window reads probe items and
progress on a periodic callback from the render thread and draws them. That is
the only safe arrangement, because DearPyGui will not accept calls from another
process, and it is also why a probe cannot slow down the pipeline it watches.
"""

import os
import time
import traceback

import dearpygui.dearpygui as dpg
import numpy as np

from libemg._gui import _style as style
from libemg._gui._pipeline.compile import CompileError, compile_pipeline
from libemg._gui._pipeline.document import PipelineDocument
from libemg._gui._pipeline.registry import (BOOL, ENUM, FLOAT, FOLDER, INT,
                                            PortType,
                                            MULTI_SELECT, PATH, STR, SOURCE,
                                            default_registry)

WINDOW_TAG = "__pipeline_window"
OPEN_TAG = "__pipeline_open"
SAVEAS_TAG = "__pipeline_saveas"
EDITOR_TAG = "__pipeline_editor"
#: How wide a block draws. Fixed rather than -1, which inside a node editor
#: means the width of the canvas rather than the width of the node.
NODE_WIDTH = 250
STATUS_TAG = "__pipeline_status"
PROGRESS_TAG = "__pipeline_progress"
PROBLEMS_TAG = "__pipeline_problems"
SCOPE_TAG = "__pipeline_scope"
RESULTS_TAG = "__pipeline_results"
START_TAG = "__pipeline_start"
STOP_TAG = "__pipeline_stop"
TRAIN_TAG = "__pipeline_train"
FILE_TAG = "__pipeline_file"
#: Height of the readiness report under the canvas: five lines before it scrolls.
REPORT_HEIGHT = 100

# The blocks of a category share a colour, on the canvas and in the palette, so
# a pipeline reads left to right as source, conditioning, windows, features,
# model, output without the titles having to be read.
_CATEGORIES = ("source", "transform", "window", "features", "model", "sink")
_CATEGORY_NAMES = {"source": "Sources", "transform": "Transforms", "window": "Windowing",
                   "features": "Features", "model": "Models", "sink": "Outputs"}
_CATEGORY_COLOURS = {
    "source": (50, 118, 84), "transform": (40, 110, 122), "window": (100, 78, 148),
    "features": (146, 104, 40), "model": (46, 94, 166), "sink": (144, 62, 80),
    "unresolved": (120, 72, 48),
}
_THEME = "__pipeline_theme_{}"
_MODE_NAMES = {"online": "live", "offline": "stored data", "empty": "no source yet",
               "mixed": "live and stored sources"}

#: A parameter drawn with its label above it at the full width of the block,
#: rather than beside it: a path, a list, or a box of several lines would be
#: too cramped in what is left of a row beside its label.
_WIDE = (PATH, FOLDER, MULTI_SELECT)
#: How far DearPyGui indents what is inside a tree node, such as Advanced.
_TREE_INDENT = 21
#: The space between two items on a line, and a checkbox's width, at the
#: spacing and font size the style module sets.
_GAP = 10
_CHECKBOX_WIDTH = 25
_BROWSE_WIDTH = 80
#: The unit after a parameter's box. The registry does not record units, so
#: they are listed here by name, and any name ending in _fs or _freq is in Hz;
#: a parameter not covered simply shows no unit.
_UNITS = {"sampling_rate": "Hz", "cutoff": "Hz", "bandwidth": "Hz",
          "window_size": "samples", "window_increment": "samples"}


class PipelineEditorPanel:
    """A window for building, saving and running pipelines.

    Parameters
    ----------
    registry: dict or None (optional), default=None
        Block descriptions. Defaults to the generated registry.
    width, height: int (optional)
        The space the window fits in, such as the size of the viewport.

    Examples
    ---------
    >>> panel = PipelineEditorPanel()
    >>> panel.spawn_window()
    """


    #: The window this panel owns, so a caller can ask if it is open.
    window_tag = WINDOW_TAG

    def __init__(self, registry=None, width=1280, height=760):
        self.registry = registry if registry is not None else default_registry()
        self.document = PipelineDocument(self.registry)
        self.width, self.height = width, height
        self.pipeline = None
        self.path = None
        self._link_ids = {}        # dpg link id -> document link key
        self._attr_ids = {}        # (node, port, direction) -> dpg attribute id
        self._probe_plots = {}     # (node, port) -> drawing state
        self._offline_thread = None
        self._offline_progress = 0.0
        self._last_error = ""
        self._training = False
        self._trained = None

    # ==================================================================
    # window
    # ==================================================================
    def spawn_window(self):
        """Build the editor window."""
        self.cleanup()
        _build_themes()
        # Inset from the space it is given, which is the whole viewport: a
        # window that size runs under the menu bar and off the bottom edge,
        # taking the readiness report under the canvas with it.
        with dpg.window(label="Pipeline Editor", tag=WINDOW_TAG,
                        width=self.width - 36, height=self.height - 80, pos=(10, 30),
                        on_close=lambda: self.cleanup()):
            self._build_toolbar()
            dpg.add_separator()
            with dpg.group(horizontal=True):
                self._build_palette()
                self._build_canvas()
            with dpg.child_window(height=-1, border=False):
                style.note("", tag=PROBLEMS_TAG, wrap=self.width - 90)
        self._refresh_status()
        self._sync_run_buttons()
        return self

    def cleanup(self):
        """Stop anything running and remove the window.

        The file dialogs are listed separately because DearPyGui parents them
        to the viewport rather than to the window that declared them. Deleting
        the window leaves them behind, and the next time this panel is opened
        it fails partway through building its toolbar, having already created a
        window it will never finish.
        """
        self.stop()
        if getattr(self, "feature_viewer", None) is not None:
            self.feature_viewer.cleanup()
            self.feature_viewer = None
        for attribute in ("training_view", "history_view", "prediction_viewer"):
            if getattr(self, attribute, None) is not None:
                getattr(self, attribute).cleanup()
                setattr(self, attribute, None)
        for tag in (SCOPE_TAG, RESULTS_TAG, OPEN_TAG, SAVEAS_TAG, WINDOW_TAG):
            if dpg.does_alias_exist(tag):
                dpg.delete_item(tag)

    # ------------------------------------------------------------------
    def _build_toolbar(self):
        with dpg.group(horizontal=True) as toolbar:
            # What happens to the file, then, set apart, what happens to the
            # pipeline in it.
            style.button("New", callback=self._new)
            style.button("Open...", callback=lambda: dpg.show_item(OPEN_TAG))
            style.button("Save", callback=self._save)
            style.button("Save as...", callback=lambda: dpg.show_item(SAVEAS_TAG))
            style.note("Untitled", tag=FILE_TAG)
            dpg.add_spacer(width=24)
            style.button("Start", "primary", tag=START_TAG, width=90, callback=self._start)
            _tooltip(START_TAG, "Build the pipeline and run it: live, from its device, or "
                                "once over the stored data it reads.")
            style.button("Stop", "danger", tag=STOP_TAG, width=80, callback=self.stop)
            style.button("Train", tag=TRAIN_TAG, callback=self._train)
            _tooltip(TRAIN_TAG, "Fit this pipeline's model on the stored data it "
                                "reads, and save it where the model block points. "
                                "Only for a stored-data pipeline.")
            style.button("History...", tag=f"{TRAIN_TAG}_history",
                         callback=self.open_training_history)
            _tooltip(f"{TRAIN_TAG}_history", "Look at deep model training runs from any "
                                             "session: load the .training.json saved beside "
                                             "each model, or a saved model, and compare them.")
            dpg.add_spacer(width=16)
            # The bar is only meaningful for a run whose total is known, which
            # is what a recording has and a live stream does not.
            dpg.add_progress_bar(tag=PROGRESS_TAG, default_value=0.0, width=220,
                                 show=False)
            style.note("", tag=STATUS_TAG)
        # The plain buttons get the coloured ones' padding, so the row is one
        # height rather than two.
        _bind_theme(toolbar, "toolbar")
        with dpg.file_dialog(tag=OPEN_TAG, show=False, directory_selector=False,
                             width=620, height=420, callback=self._open_selected):
            dpg.add_file_extension(".json")
        with dpg.file_dialog(tag=SAVEAS_TAG, show=False, directory_selector=False,
                             width=620, height=420, callback=self._saveas_selected):
            dpg.add_file_extension(".json")

    def _build_palette(self):
        with dpg.child_window(width=220, height=-REPORT_HEIGHT):
            style.section("Blocks", first=True)
            style.note("Click a block to add it. Drag from an output's dot to an "
                       "input's dot to connect them; Ctrl+click a link to remove it.",
                       wrap=190)
            grouped = {}
            for spec in self.registry.values():
                grouped.setdefault(spec.category, []).append(spec)
            for category in _CATEGORIES:
                specs = sorted(grouped.get(category, []), key=lambda s: s.title)
                if not specs:
                    continue
                with dpg.tree_node(label=_CATEGORY_NAMES[category],
                                   default_open=category != "source"):
                    for spec in specs:
                        button = dpg.add_button(label=spec.title, width=-1,
                                                user_data=spec.id,
                                                callback=self._add_from_palette)
                        _bind_theme(button, f"block_{category}")
                        if spec.help:
                            _tooltip(button, spec.help)

    def _build_canvas(self):
        with dpg.child_window(autosize_x=True, height=-REPORT_HEIGHT, border=False) as canvas:
            with dpg.node_editor(tag=EDITOR_TAG, callback=self._link_requested,
                                 delink_callback=self._unlink_requested,
                                 minimap=True,
                                 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                pass
        # On the window around the editor, not the editor: the editor applies
        # its own theme too late for the grid and the nodes to take it.
        _bind_theme(canvas, "canvas")

    # ==================================================================
    # editing
    # ==================================================================
    def _add_from_palette(self, sender, app_data, user_data):
        node_id = self.document.add_node(user_data, position=self._free_position())
        self._draw_node(node_id)
        self._refresh_status()

    def _free_position(self):
        """Somewhere that does not sit on top of an existing node.

        Left to right, three to a row, in the order blocks are added -- which
        is usually the order the signal flows through them.
        """
        count = len(self.document.nodes)
        return (30 + (NODE_WIDTH + 70) * (count % 3), 30 + 460 * (count // 3))

    # The node drawing below calls DearPyGui only through this module's dpg and
    # takes nothing but colours from the style module. The tests draw nodes
    # against a stand-in for that dpg, with no DearPyGui context behind it,
    # where any real DearPyGui call would crash the interpreter.
    def _draw_node(self, node_id):
        node = self.document.nodes[node_id]
        spec = self.registry.get(node.spec_id)
        title = spec.title if spec else f"? {node.spec_id}"
        tag = f"__node_{node_id}"
        with dpg.node(label=f"{title}  [{node_id}]", parent=EDITOR_TAG,
                      tag=tag, pos=node.position):
            if spec is None:
                # An unresolved block is drawn, and says so, rather than being
                # dropped. Losing a user's work to a version difference is
                # worse than showing them something they have to deal with.
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    dpg.add_text("Not recognised by this LibEMG.", color=style.WARNING)
                    dpg.add_text(f"{node.params}", wrap=NODE_WIDTH, color=style.MUTED)
            else:
                self._draw_ports(node_id, spec)
                with dpg.node_attribute(parent=tag, attribute_type=dpg.mvNode_Attr_Static):
                    self._draw_params(node_id, spec)
            with dpg.node_attribute(parent=tag, attribute_type=dpg.mvNode_Attr_Static):
                # A fixed width, not -1: inside a node editor, -1 means "fill
                # the remaining canvas", so this one button stretched its node
                # across the whole editor and buried the links under it.
                remove = dpg.add_button(label="Remove", width=NODE_WIDTH, user_data=node_id,
                                        callback=self._remove_node)
                _bind_theme(remove, "remove")
        _bind_theme(tag, f"node_{spec.category if spec else 'unresolved'}")

    def _draw_ports(self, node_id, spec):
        tag = f"__node_{node_id}"
        for port in spec.inputs:
            attribute = dpg.add_node_attribute(
                parent=tag, label=port.label,
                attribute_type=dpg.mvNode_Attr_Input,
                category=PortType.category_of(port.type))
            self._attr_ids[(node_id, port.name, "input")] = attribute
            name = dpg.add_text(port.label, parent=attribute)
            _tooltip(name, f"Takes {port.type}." + (" Optional." if port.optional else ""))

        for port in spec.outputs:
            attribute = dpg.add_node_attribute(
                parent=tag, label=port.label,
                attribute_type=dpg.mvNode_Attr_Output,
                category=PortType.category_of(port.type))
            self._attr_ids[(node_id, port.name, "output")] = attribute
            with dpg.group(horizontal=True, parent=attribute) as line:
                # A probe is a toggle on the port, not a block to place.
                # Making it a block would clutter the canvas and make the
                # user wire something they only want to look at.
                dpg.add_checkbox(label="", default_value=False,
                                 tag=f"__probe_{node_id}_{port.name}",
                                 user_data=(node_id, port.name),
                                 callback=self._toggle_probe)
                dpg.add_text("Watch", color=style.MUTED)
                # The name goes against the right edge, beside the dot a link
                # leaves from, as an input's name sits beside its own.
                used = _CHECKBOX_WIDTH + _text_width("Watch") + _text_width(port.label)
                dpg.add_spacer(width=max(int(NODE_WIDTH - used - 3 * _GAP), 4))
                dpg.add_text(port.label)
            _tooltip(line, f"Carries {port.type}. Tick Watch to see it live while the "
                           "pipeline runs.")

    def _draw_params(self, node_id, spec):
        plain = [p for p in spec.params if not p.advanced]
        advanced = [p for p in spec.params if p.advanced]
        # One label column for the whole block, wide enough for its longest
        # label, so every box in it starts at the same place -- the ones under
        # Advanced included, which DearPyGui indents.
        widths = [_text_width(p.label) for p in plain if p.kind not in _WIDE]
        widths += [_text_width(p.label) + _TREE_INDENT for p in advanced
                   if p.kind not in _WIDE]
        label_width = int(max(widths, default=0)) + _GAP + 4
        for param in plain:
            self._draw_param(node_id, param, label_width)
        if advanced:
            with dpg.tree_node(label="Advanced", default_open=False):
                for param in advanced:
                    self._draw_param(node_id, param, label_width - _TREE_INDENT,
                                     NODE_WIDTH - _TREE_INDENT)

    def _draw_param(self, node_id, param, label_width=110, width=NODE_WIDTH):
        """One widget, chosen by the parameter's kind, with its name to its left.

        This mapping is the whole reason a parameter declares a kind rather
        than a widget: the rule for what a value should be edited with lives
        here once, instead of being restated on every block.

        A node is too narrow for the window's two-column form, and a table
        inside a node editor stretches its node across the whole canvas, so a
        row is the name in a fixed-width column and the box after it. A path, a
        list or a box of several lines gets the block's full width instead, with
        its name above it.
        """
        value = self.document.nodes[node_id].params.get(param.name, param.default)
        tag = f"__param_{node_id}_{param.name}"
        data = (node_id, param.name)
        common = dict(tag=tag, user_data=data, callback=self._param_changed)
        wide = param.kind in _WIDE or (param.multiline
                                       and param.kind not in (ENUM, BOOL, INT, FLOAT))

        if wide:
            name = dpg.add_text(param.label, color=style.MUTED)
            if param.kind == MULTI_SELECT:
                dpg.add_listbox(list(param.choices), num_items=6,
                                default_value=(value or [None])[0] if value else "",
                                tag=tag, user_data=data, width=width,
                                callback=self._multi_toggled)
                dpg.add_text(_summarise(value), tag=f"{tag}__summary", wrap=width,
                             color=style.HEADING)
            elif param.kind in (PATH, FOLDER):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(default_value=str(value or ""),
                                       width=width - _BROWSE_WIDTH - _GAP, **common)
                    dpg.add_button(label="Browse...", width=_BROWSE_WIDTH,
                                   user_data=(node_id, param.name, param.kind),
                                   callback=self._browse)
            else:
                # Whether a value needs room for several lines is a property of
                # the parameter, declared in the registry, not a list of names
                # kept here -- the second such parameter is how that list goes
                # stale.
                dpg.add_input_text(default_value=str(value or ""), multiline=True,
                                   height=70, width=width, **common)
        else:
            unit = _unit(param.name)
            box = width - label_width - (_text_width(unit) + _GAP if unit else 0)
            box = int(max(box, 60))
            with dpg.group(horizontal=True, xoffset=label_width):
                # The label has a group of its own because its tooltip is an
                # item beside it, and the row places its second item, not
                # whatever comes second, at the offset.
                with dpg.group():
                    name = dpg.add_text(param.label, color=style.MUTED)
                with dpg.group(horizontal=True):
                    if param.kind == ENUM:
                        dpg.add_combo(list(param.choices),
                                      default_value=value or param.default,
                                      width=box, **common)
                    elif param.kind == BOOL:
                        dpg.add_checkbox(default_value=bool(value), **common)
                    elif param.kind == INT:
                        dpg.add_input_int(default_value=int(value or 0), step=0,
                                          width=box, **_bounds(param), **common)
                    elif param.kind == FLOAT:
                        dpg.add_input_float(default_value=float(value or 0.0), step=0,
                                            format="%g", width=box, **_bounds(param),
                                            **common)
                    else:
                        dpg.add_input_text(default_value=str(value or ""), width=box,
                                           **common)
                    if unit:
                        dpg.add_text(unit, color=style.MUTED)
        if param.help:
            _tooltip(name, param.help)
            _tooltip(tag, param.help)

    def _param_changed(self, sender, app_data, user_data):
        node_id, name = user_data
        self.document.set_param(node_id, name, app_data)
        self._refresh_status()

    def _multi_toggled(self, sender, app_data, user_data):
        """A list box selection toggles membership rather than replacing it.

        A features block usually wants several features, and a single-selection
        list would make choosing eight of them impossible.
        """
        node_id, name = user_data
        current = list(self.document.nodes[node_id].params.get(name) or [])
        if app_data in current:
            current.remove(app_data)
        else:
            current.append(app_data)
        self.document.set_param(node_id, name, current)
        summary = f"__param_{node_id}_{name}__summary"
        if dpg.does_item_exist(summary):
            dpg.set_value(summary, _summarise(current))
        self._refresh_status()

    def _browse(self, sender, app_data, user_data):
        node_id, name, kind = user_data
        tag = f"__browse_{node_id}_{name}"
        if dpg.does_alias_exist(tag):
            dpg.delete_item(tag)
        with dpg.file_dialog(tag=tag, directory_selector=(kind == FOLDER),
                             width=620, height=420, modal=True,
                             user_data=(node_id, name), callback=self._browsed):
            dpg.add_file_extension(".*")

    def _browsed(self, sender, app_data, user_data):
        node_id, name = user_data
        chosen = app_data.get("file_path_name") or app_data.get("current_path") or ""
        self.document.set_param(node_id, name, chosen)
        widget = f"__param_{node_id}_{name}"
        if dpg.does_item_exist(widget):
            dpg.set_value(widget, chosen)
        self._refresh_status()

    def _remove_node(self, sender, app_data, user_data):
        node_id = user_data
        for link_id, key in list(self._link_ids.items()):
            if key[0] == node_id or key[2] == node_id:
                self._link_ids.pop(link_id, None)
        self.document.remove_node(node_id)
        if dpg.does_alias_exist(f"__node_{node_id}"):
            dpg.delete_item(f"__node_{node_id}")
        self._refresh_status()

    # ------------------------------------------------------------------
    # linking
    # ------------------------------------------------------------------
    def _link_requested(self, sender, app_data):
        """The user dragged a connection. Accept it only if it could run."""
        from_attr, to_attr = app_data
        source = self._port_of(from_attr, "output")
        target = self._port_of(to_attr, "input")
        if source is None or target is None:
            return
        problem = self.document.why_not_connect(source[0], source[1],
                                                target[0], target[1])
        if problem:
            # Types are already blocked by the toolkit, so anything reaching
            # here is a rule the toolkit cannot express. Saying why beats
            # letting the link silently fail to appear.
            self._say(problem, "warn")
            return
        self.document.connect(source[0], source[1], target[0], target[1])
        link_id = dpg.add_node_link(from_attr, to_attr, parent=EDITOR_TAG)
        self._link_ids[link_id] = (source[0], source[1], target[0], target[1])
        self._refresh_status()

    def _unlink_requested(self, sender, app_data):
        key = self._link_ids.pop(app_data, None)
        if key is not None:
            self.document.disconnect(*key)
        dpg.delete_item(app_data)
        self._refresh_status()

    def _port_of(self, attribute, direction):
        for (node_id, port, side), identifier in self._attr_ids.items():
            if identifier == attribute and side == direction:
                return node_id, port
        return None

    def _toggle_probe(self, sender, app_data, user_data):
        node_id, port = user_data
        if app_data:
            self.document.add_probe(node_id, port)
        else:
            self.document.remove_probe(node_id, port)
        self._refresh_status()

    # ==================================================================
    # files
    # ==================================================================
    def _new(self):
        self.stop()
        self.document = PipelineDocument(self.registry)
        self.path = None
        self._rebuild_canvas()
        self._show_file()

    def _save(self):
        if self.path is None:
            dpg.show_item(SAVEAS_TAG)
            return
        self._capture_positions()
        self.document.save(self.path)
        self._show_file()
        self._say(f"Saved to {self.path}", "ok")

    def _saveas_selected(self, sender, app_data):
        path = app_data.get("file_path_name") or ""
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self.path = path
        self._save()

    def _open_selected(self, sender, app_data):
        path = app_data.get("file_path_name") or ""
        if not path or not os.path.exists(path):
            return
        try:
            self.document = PipelineDocument.load(path, registry=self.registry)
        except Exception as error:
            self._say(f"Could not open that file: {error}", "error")
            return
        self.path = path
        self._rebuild_canvas()
        self._show_file()

    def _capture_positions(self):
        """Read canvas positions back before saving, so layout survives."""
        for node_id in self.document.nodes:
            tag = f"__node_{node_id}"
            if dpg.does_item_exist(tag):
                self.document.nodes[node_id].position = tuple(dpg.get_item_pos(tag))

    def _rebuild_canvas(self):
        """Redraw every node and link from the document."""
        if dpg.does_item_exist(EDITOR_TAG):
            dpg.delete_item(EDITOR_TAG, children_only=True)
        self._attr_ids.clear()
        self._link_ids.clear()
        for node_id in self.document.nodes:
            self._draw_node(node_id)
        for link in self.document.links:
            source = self._attr_ids.get((link.from_node, link.from_port, "output"))
            target = self._attr_ids.get((link.to_node, link.to_port, "input"))
            if source is None or target is None:
                continue
            link_id = dpg.add_node_link(source, target, parent=EDITOR_TAG)
            self._link_ids[link_id] = link.key()
        for probe in self.document.probes:
            tag = f"__probe_{probe.node}_{probe.port}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, True)
        self._refresh_status()

    # ==================================================================
    # running
    # ==================================================================
    def _start(self):
        if self.pipeline is not None:
            self._say("Already running. Stop it first.", "warn")
            return
        self._capture_positions()
        try:
            self.pipeline = compile_pipeline(self.document)
        except CompileError as error:
            self._say(str(error), "error")
            return
        except Exception:
            self._say("Could not build the pipeline:\n"
                      + traceback.format_exc().strip().splitlines()[-1], "error")
            return

        if self.document.mode() == "offline":
            self._start_offline()
        else:
            self._start_online()

    def _start_online(self):
        try:
            self.pipeline.start()
        except Exception:
            self._say("Could not start:\n"
                      + traceback.format_exc().strip().splitlines()[-1], "error")
            self.pipeline = None
            return
        dpg.configure_item(PROGRESS_TAG, show=False)
        self._open_scope()
        self._sync_run_buttons()
        self._say("Running.", "busy")

    def _start_offline(self):
        import threading
        dpg.configure_item(PROGRESS_TAG, show=True)
        dpg.set_value(PROGRESS_TAG, 0.0)
        self._offline_progress = 0.0

        def work():
            try:
                self.pipeline.run(on_progress=self._record_progress)
            except Exception:
                self._last_error = traceback.format_exc().strip().splitlines()[-1]

        # A thread, not a process: the run has to hand its results back, and
        # the only thing it shares with the render thread is a float and a dict
        # that the periodic callback reads.
        self._offline_thread = threading.Thread(target=work, daemon=True,
                                                name="libemg-offline-pipeline")
        self._offline_thread.start()
        self._sync_run_buttons()
        self._say("Running over stored data.", "busy")

    def _record_progress(self, fraction):
        self._offline_progress = float(fraction)

    def stop(self):
        """Stop whatever is running."""
        if self.pipeline is None:
            return
        if getattr(self, "_training", False) and self._offline_thread is not None:
            # Training is asked to stop rather than abandoned: the worker
            # finishes its batch and poll() reports what happened -- for a deep
            # model, that the best epoch so far was saved.
            if hasattr(self.pipeline, "request_stop"):
                self.pipeline.request_stop()
            self._say("Stopping after the current batch...", "busy")
            return
        try:
            if hasattr(self.pipeline, "request_stop"):
                self.pipeline.request_stop()
            if hasattr(self.pipeline, "stop"):
                self.pipeline.stop()
        except Exception:
            pass
        self.pipeline = None
        self._offline_thread = None
        if dpg.does_item_exist(PROGRESS_TAG):
            dpg.configure_item(PROGRESS_TAG, show=False)
        self._sync_run_buttons()
        self._say("Stopped.")

    def open_training_history(self, paths=()):
        """Open the Training History window, optionally with records already loaded."""
        from libemg._gui._pipeline.training_view import TrainingView
        if getattr(self, "history_view", None) is not None:
            self.history_view.cleanup()
        self.history_view = TrainingView("Training History", pos=(60, 60)).spawn_window()
        for path in paths:
            self.history_view.load(path)
        return self.history_view

    def _train(self):
        """Fit the model this pipeline names, on the data it reads.

        The step that used to need a script. A pipeline built for stored data
        already describes everything a fit needs: where the recordings are, how
        they are filtered and windowed, which features to take and which labels
        to learn. Training reuses that description rather than asking for it
        again, so the model is fitted on exactly what it will be scored on.
        """
        if self.pipeline is not None:
            self._say("Stop the running pipeline before training.", "warn")
            return
        if self.document.mode() != "offline":
            self._say("Training reads stored data. Add a Stored Data source, or "
                      "open the pipeline you collected your training data with.", "warn")
            return
        self._capture_positions()
        try:
            pipeline = compile_pipeline(self.document)
        except CompileError as error:
            self._say(str(error), "error")
            return

        import threading
        dpg.configure_item(PROGRESS_TAG, show=True)
        dpg.set_value(PROGRESS_TAG, 0.0)
        self._offline_progress = 0.0
        self._trained = None
        self.pipeline = pipeline

        deep = pipeline.plan.get("deep")
        if deep:
            # The loss curves are drawn from pipeline.epochs, which the worker
            # appends to as each epoch ends; poll() redraws when it grows.
            from libemg._gui._pipeline.training_view import TrainingView
            if getattr(self, "training_view", None) is not None:
                self.training_view.cleanup()
            title = (f"Training: {deep.get('architecture', 'GRU')}, "
                     f"{str(deep.get('timing', 'Continuous')).lower()} "
                     f"{str(deep.get('task', 'Classification')).lower()}")
            self.training_view = TrainingView(
                title, int(deep.get("epochs", 50)), on_stop=self.stop,
                pos=(max(40, self.width - 720), 70)).spawn_window()

        def work():
            try:
                self._trained = pipeline.train(on_progress=self._record_progress)
            except Exception as error:
                self._last_error = f"{type(error).__name__}: {error}"

        self._offline_thread = threading.Thread(target=work, daemon=True,
                                                name="libemg-pipeline-train")
        self._offline_thread.start()
        self._training = True
        self._sync_run_buttons()
        self._say("Training on the stored data.", "busy")

    # ==================================================================
    # per-frame work, called from the render thread
    # ==================================================================
    def poll(self):
        """Refresh anything driven by the running pipeline.

        Called once per rendered frame by the GUI's loop. This is the only
        place the editor reads what the pipeline produced, and it runs on the
        render thread, because DearPyGui items must not be touched from
        anywhere else.
        """
        for attribute in ("feature_viewer", "prediction_viewer"):
            viewer = getattr(self, attribute, None)
            if viewer is not None:
                viewer.poll()
        if self.pipeline is None:
            return
        if hasattr(self.pipeline, "plan"):
            self._poll_offline()
        else:
            self._poll_online()

    def _poll_offline(self):
        if dpg.does_item_exist(PROGRESS_TAG):
            dpg.set_value(PROGRESS_TAG, self._offline_progress)
        view = getattr(self, "training_view", None)
        if view is not None and getattr(self, "_training", False):
            view.update(getattr(self.pipeline, "epochs", []))
        thread = self._offline_thread
        if thread is not None and not thread.is_alive():
            self._offline_thread = None
            training = getattr(self, "_training", False)
            self._training = False
            if self._last_error:
                self._say(("Training failed: " if training else "The run failed: ")
                          + self._last_error, "error")
                if training and view is not None:
                    view.finish("Training failed: " + self._last_error, "error")
                self._last_error = ""
            elif training and (getattr(self, "_trained", None) or {}).get("epochs"):
                trained = self._trained
                if view is not None:
                    view.update(getattr(self.pipeline, "epochs", []))    # the last epoch
                score = trained.get("best_score")
                if score is None:
                    scored = ""
                elif trained.get("score_name") == "accuracy":
                    scored = f", accuracy {score:.1%}"
                else:
                    scored = f", {trained.get('score_name')} {score:.3f}"
                message = (f"Trained for {trained['epochs']} epochs on {trained['windows']} "
                           f"windows from {trained['recordings']} recordings; the best was "
                           f"epoch {trained['best_epoch']}{scored}. "
                           + ("Stopped early: that epoch was saved. " if trained.get("stopped")
                              else "")
                           + f"Saved to {trained['model_path']}.")
                self._say(message, "ok")
                if view is not None:
                    view.finish(message + (f" The training record is in "
                                           f"{trained['outcome_path']}."
                                           if trained.get("outcome_path") else ""),
                                "ok", outcome=getattr(self.pipeline, "outcome", None))
            elif training:
                trained = getattr(self, "_trained", None) or {}
                classes = trained.get("classes")
                self._say(
                    "Trained on {windows} windows of {features} features"
                    .format(**{"windows": trained.get("windows", 0),
                               "features": trained.get("features", 0)})
                    + (f" over classes {classes}" if classes else "")
                    + f". Saved to {trained.get('model_path', '?')}. "
                    "Point a live pipeline at that file to run it.", "ok")
            else:
                view = getattr(self.pipeline, "feature_view", None)
                predictions = getattr(self.pipeline, "prediction_view", None)
                if view is not None:
                    self._show_features(view)
                if predictions is not None:
                    self._show_predictions(predictions)
                if self.pipeline.results or (view is None and predictions is None):
                    self._show_results(self.pipeline.results)
                if self.pipeline.stopped:
                    self._say(f"Stopped after {self.pipeline.progress:.0%}.")
                elif view is not None and view.get("model_note"):
                    self._say(view["model_note"], "warn")
                else:
                    self._say("Finished.", "ok")
            self.pipeline = None
            self._sync_run_buttons()

    def _poll_online(self):
        try:
            status = self.pipeline.status()
        except Exception:
            return
        parts = []
        for tag in sorted(status):
            if tag.startswith("probe_"):
                continue
            parts.append(f"{tag.replace('pipe_', '')}={status[tag].total_samples}")
        if dpg.does_item_exist(STATUS_TAG):
            dpg.set_value(STATUS_TAG, "  ".join(parts[:6]))
        self._draw_probes()

    # ------------------------------------------------------------------
    # probes
    # ------------------------------------------------------------------
    def _open_scope(self):
        if dpg.does_alias_exist(SCOPE_TAG):
            dpg.delete_item(SCOPE_TAG)
        if not self.pipeline.probe_items:
            return
        self._probe_plots.clear()
        with dpg.window(label="Scope", tag=SCOPE_TAG, width=620,
                        height=180 + 200 * len(self.pipeline.probe_items),
                        pos=(self.width - 640, 60)):
            from libemg.shared_memory_manager import SharedMemoryManager
            self._probe_reader = SharedMemoryManager()
            declared = {item[0]: item for item in self.pipeline.shared_memory_items}
            for (node_id, port), info in self.pipeline.probe_items.items():
                item = declared.get(info["tag"])
                if item is None or not self._probe_reader.find_variable(*item):
                    continue
                self._build_probe_plot(node_id, port, info)

    def _build_probe_plot(self, node_id, port, info):
        """One plot, shaped by what the port carries.

        The render mode comes from the port's type rather than from a setting,
        which is what keeps probing to a single click.
        """
        render, width = info["render"], info["width"]
        base = f"__scope_{node_id}_{port}"
        with dpg.group(horizontal=True):
            dpg.add_text(f"{node_id}.{port}", color=style.HEADING)
            style.note(render.replace("_", " "))
        with dpg.plot(height=170, width=-1, no_menus=True, tag=f"{base}_plot"):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="", tag=f"{base}_x")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="", tag=f"{base}_y")
            series = []
            if render in ("timeseries", "window_overlay"):
                for channel in range(width):
                    series.append(dpg.add_line_series(
                        [], [], label=f"ch{channel + 1}", parent=y_axis,
                        tag=f"{base}_s{channel}"))
            else:
                series.append(dpg.add_bar_series([], [], label=port, parent=y_axis,
                                                 tag=f"{base}_bar"))
        self._probe_plots[(node_id, port)] = {
            "info": info, "render": render, "x": x_axis, "y": y_axis, "base": base}

    def _draw_probes(self):
        reader = getattr(self, "_probe_reader", None)
        if reader is None:
            return
        for (node_id, port), state in self._probe_plots.items():
            info = state["info"]
            try:
                block, _ = reader.read_window(info["tag"], info["rows"])
            except Exception:
                continue
            if block.size == 0:
                continue
            base, render = state["base"], state["render"]
            if render in ("timeseries", "window_overlay"):
                x = list(range(block.shape[0]))
                for channel in range(min(info["width"], block.shape[1])):
                    tag = f"{base}_s{channel}"
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, [x, block[:, channel].tolist()])
            else:
                row = block[-1]
                tag = f"{base}_bar"
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, [list(range(len(row))), row.tolist()])
            dpg.fit_axis_data(state["x"])
            dpg.fit_axis_data(state["y"])

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
    def _show_features(self, view):
        """Open the Feature Viewer on what the run read, replacing one already open."""
        from libemg._gui._pipeline.feature_viewer import FeatureViewer
        if getattr(self, "feature_viewer", None) is not None:
            self.feature_viewer.cleanup()
        self.feature_viewer = FeatureViewer(
            view["features"], view["names"], view["labels"], view["params"],
            width=min(1100, self.width - 80), height=min(720, self.height - 100),
            pos=(40, 60), spaces=view.get("spaces")).spawn_window()

    def _show_predictions(self, view):
        """Open the Prediction Viewer on what the run predicted, replacing one already open."""
        from libemg._gui._pipeline.prediction_viewer import PredictionViewer
        if getattr(self, "prediction_viewer", None) is not None:
            self.prediction_viewer.cleanup()
        self.prediction_viewer = PredictionViewer(
            view, width=min(1200, self.width - 60), height=min(780, self.height - 80),
            pos=(30, 50)).spawn_window()

    def _show_results(self, results):
        if dpg.does_alias_exist(RESULTS_TAG):
            dpg.delete_item(RESULTS_TAG)
        if not results:
            self._say("The run finished, but produced no metrics. "
                      "Connect an Offline Metrics block to score it.", "warn")
            return
        with dpg.window(label="Results", tag=RESULTS_TAG, width=520, height=440,
                        pos=(self.width - 560, 80)):
            scalars = {k: v for k, v in results.items() if np.ndim(v) == 0}
            matrices = {k: v for k, v in results.items() if np.ndim(v) >= 2}
            vectors = {k: v for k, v in results.items() if np.ndim(v) == 1}
            if scalars or vectors:
                with dpg.table(header_row=True, borders_innerH=True,
                               borders_outerH=True, borders_innerV=True):
                    dpg.add_table_column(label="Metric")
                    dpg.add_table_column(label="Value")
                    for name, value in sorted(scalars.items()):
                        with dpg.table_row():
                            dpg.add_text(name)
                            dpg.add_text(f"{float(value):.4f}")
                    for name, value in sorted(vectors.items()):
                        with dpg.table_row():
                            dpg.add_text(name)
                            dpg.add_text(", ".join(f"{v:.3f}" for v in np.ravel(value)))
            for name, matrix in sorted(matrices.items()):
                style.section(name)
                matrix = np.asarray(matrix, dtype=float)
                with dpg.plot(height=260, width=-1, no_menus=True):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Predicted")
                    with dpg.plot_axis(dpg.mvYAxis, label="True"):
                        dpg.add_heat_series(matrix.ravel().tolist(),
                                            matrix.shape[0], matrix.shape[1],
                                            scale_min=float(matrix.min()),
                                            scale_max=float(matrix.max()))
            dpg.add_spacer(height=4)
            style.button("Copy as CSV", callback=lambda: dpg.set_clipboard_text(
                _as_csv(results)))

    # ==================================================================
    # status
    # ==================================================================
    def _refresh_status(self):
        if not dpg.does_item_exist(PROBLEMS_TAG):
            return
        mode = _MODE_NAMES.get(self.document.mode(), self.document.mode())
        problems = self.document.validate()
        if problems:
            bullet = "\u2022" if style.symbols() else "-"
            style.set_status(PROBLEMS_TAG, f"Not ready to run ({mode}):\n"
                             + "\n".join(f"  {bullet}  {p}" for p in problems), "warn")
        else:
            style.set_status(PROBLEMS_TAG, f"Ready to run ({mode}). Press Start.", "ok")

    def _say(self, message, kind="info"):
        """Report ``message`` where the readiness report goes, coloured by ``kind``.

        It stays until the next edit, which puts the readiness report back.
        """
        if dpg.does_item_exist(PROBLEMS_TAG):
            style.set_status(PROBLEMS_TAG, message, kind)

    def _sync_run_buttons(self):
        """Start and Train while nothing runs, Stop while something does."""
        if not dpg.does_item_exist(WINDOW_TAG):
            return
        running = self.pipeline is not None
        style.set_enabled(START_TAG, not running)
        style.set_enabled(TRAIN_TAG, not running)
        style.set_enabled(STOP_TAG, running)

    def _show_file(self):
        """Name the file Save writes to beside the file buttons."""
        if dpg.does_item_exist(FILE_TAG):
            dpg.set_value(FILE_TAG, os.path.basename(self.path) if self.path else "Untitled")


def _summarise(values):
    """A short rendering of a multi-selection, for the label under the list."""
    values = list(values or [])
    if not values:
        return "none selected"
    if len(values) <= 6:
        return ", ".join(values)
    return f"{', '.join(values[:6])} and {len(values) - 6} more"


def _as_csv(results):
    lines = ["metric,value"]
    for name, value in sorted(results.items()):
        array = np.asarray(value)
        if array.ndim == 0:
            lines.append(f"{name},{float(array):.6f}")
        else:
            lines.append(f"{name},\"{';'.join(str(v) for v in array.ravel())}\"")
    return "\n".join(lines)


# ======================================================================
# drawing helpers
# ======================================================================
def _unit(name):
    """The unit after a parameter's box, or "" when it has none or it is not known."""
    if name in _UNITS:
        return _UNITS[name]
    if name.endswith(("_fs", "_freq")):
        return "Hz"
    return ""


def _bounds(param):
    """Hold a number box to its parameter's range, as the document will hold the value."""
    cast = int if param.kind == INT else float
    bounds = {}
    if param.minimum is not None:
        bounds.update(min_value=cast(param.minimum), min_clamped=True)
    if param.maximum is not None:
        bounds.update(max_value=cast(param.maximum), max_clamped=True)
    return bounds


def _text_width(text):
    """How wide ``text`` is drawn, to line things up inside a node.

    Measured where DearPyGui can measure it, which is once its font has been
    drawn with; estimated before then.
    """
    try:
        width = float(dpg.get_text_size(text)[0])
    except Exception:
        width = 0.0
    return width if width > 0 else 7.5 * len(text)


def _tooltip(item, text):
    with dpg.tooltip(item):
        dpg.add_text(text, wrap=320)


def _bind_theme(item, name):
    """Give ``item`` one of the themes :func:`_build_themes` made, if it has."""
    theme = _THEME.format(name)
    if dpg.does_alias_exist(theme):
        dpg.bind_item_theme(item, theme)


def _mix(colour, other, amount):
    return tuple(int(a + (b - a) * amount) for a, b in zip(colour, other))


def _build_themes():
    """The canvas, block and toolbar themes, made once per DearPyGui context.

    The style module themes ordinary widgets; the node editor's own colours --
    grid, links, pins, title bars -- are a separate set it does not touch.
    """
    if dpg.does_alias_exist(_THEME.format("canvas")):
        return
    nodes = dpg.mvThemeCat_Nodes
    with dpg.theme(tag=_THEME.format("canvas")):
        with dpg.theme_component(dpg.mvAll):
            for colour, value in (
                    (dpg.mvNodeCol_GridBackground, (25, 27, 31)),
                    (dpg.mvNodeCol_GridLine, (37, 40, 46)),
                    (dpg.mvNodeCol_NodeBackground, (37, 40, 47)),
                    (dpg.mvNodeCol_NodeBackgroundHovered, (40, 44, 52)),
                    (dpg.mvNodeCol_NodeBackgroundSelected, (44, 49, 59)),
                    (dpg.mvNodeCol_NodeOutline, (66, 72, 84)),
                    (dpg.mvNodeCol_TitleBar, (58, 64, 76)),
                    (dpg.mvNodeCol_TitleBarHovered, (66, 72, 86)),
                    (dpg.mvNodeCol_TitleBarSelected, (74, 82, 98)),
                    (dpg.mvNodeCol_Link, (110, 148, 204)),
                    (dpg.mvNodeCol_LinkHovered, style.HEADING),
                    (dpg.mvNodeCol_LinkSelected, (190, 214, 250)),
                    (dpg.mvNodeCol_Pin, style.HEADING),
                    (dpg.mvNodeCol_PinHovered, (210, 226, 250)),
                    (dpg.mvNodeCol_BoxSelector, (*style.ACCENT, 40)),
                    (dpg.mvNodeCol_BoxSelectorOutline, (*style.ACCENT, 150)),
            ):
                dpg.add_theme_color(colour, value, category=nodes)
            for var, value in ((dpg.mvNodeStyleVar_NodeCornerRounding, 6),
                               (dpg.mvNodeStyleVar_NodeBorderThickness, 1),
                               (dpg.mvNodeStyleVar_LinkThickness, 3),
                               (dpg.mvNodeStyleVar_PinCircleRadius, 5),
                               (dpg.mvNodeStyleVar_GridSpacing, 28)):
                dpg.add_theme_style(var, value, category=nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 10, 7, category=nodes)
    for category, colour in _CATEGORY_COLOURS.items():
        with dpg.theme(tag=_THEME.format(f"node_{category}")):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, colour, category=nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,
                                    _mix(colour, (255, 255, 255), 0.08), category=nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,
                                    _mix(colour, (255, 255, 255), 0.16), category=nodes)
        with dpg.theme(tag=_THEME.format(f"block_{category}")):
            with dpg.theme_component(dpg.mvButton):
                for target, amount in ((dpg.mvThemeCol_Button, 0.45),
                                       (dpg.mvThemeCol_ButtonHovered, 0.7),
                                       (dpg.mvThemeCol_ButtonActive, 0.9)):
                    dpg.add_theme_color(target, _mix((40, 43, 50), colour, amount))
    with dpg.theme(tag=_THEME.format("remove")):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (120, 42, 48))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (150, 45, 50))
            dpg.add_theme_color(dpg.mvThemeCol_Text, style.MUTED)
    with dpg.theme(tag=_THEME.format("toolbar")):
        # Disabled too, or Train shrinks while a run has it greyed out.
        for enabled in (True, False):
            with dpg.theme_component(dpg.mvButton, enabled_state=enabled):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 14, 4)
