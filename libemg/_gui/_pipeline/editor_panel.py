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
STATUS_TAG = "__pipeline_status"
PROGRESS_TAG = "__pipeline_progress"
PROBLEMS_TAG = "__pipeline_problems"
SCOPE_TAG = "__pipeline_scope"
RESULTS_TAG = "__pipeline_results"


class PipelineEditorPanel:
    """A window for building, saving and running pipelines.

    Parameters
    ----------
    registry: dict or None (optional), default=None
        Block descriptions. Defaults to the generated registry.
    width, height: int (optional)
        Initial window size.

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
        with dpg.window(label="Pipeline Editor", tag=WINDOW_TAG,
                        width=self.width, height=self.height,
                        on_close=lambda: self.cleanup()):
            self._build_toolbar()
            dpg.add_separator()
            with dpg.group(horizontal=True):
                self._build_palette()
                self._build_canvas()
            dpg.add_separator()
            dpg.add_text("", tag=PROBLEMS_TAG, wrap=self.width - 40)
        self._refresh_status()
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
        for tag in (SCOPE_TAG, RESULTS_TAG, OPEN_TAG, SAVEAS_TAG, WINDOW_TAG):
            if dpg.does_alias_exist(tag):
                dpg.delete_item(tag)

    # ------------------------------------------------------------------
    def _build_toolbar(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label="New", callback=self._new)
            dpg.add_button(label="Open", callback=lambda: dpg.show_item(OPEN_TAG))
            dpg.add_button(label="Save", callback=self._save)
            dpg.add_button(label="Save As", callback=lambda: dpg.show_item(SAVEAS_TAG))
            dpg.add_spacer(width=20)
            dpg.add_button(label="Start", tag="__pipeline_start", callback=self._start)
            dpg.add_button(label="Stop", tag="__pipeline_stop", callback=self.stop)
            dpg.add_button(label="Train", tag="__pipeline_train", callback=self._train)
            with dpg.tooltip("__pipeline_train"):
                dpg.add_text("Fit this pipeline's model on the stored data it "
                             "reads, and save it where the model block points. "
                             "Only for a stored-data pipeline.", wrap=320)
            dpg.add_spacer(width=20)
            # The bar is only meaningful for a run whose total is known, which
            # is what a recording has and a live stream does not.
            dpg.add_progress_bar(tag=PROGRESS_TAG, default_value=0.0, width=220,
                                 show=False)
            dpg.add_text("", tag=STATUS_TAG)
        with dpg.file_dialog(tag=OPEN_TAG, show=False, directory_selector=False,
                             width=620, height=420, callback=self._open_selected):
            dpg.add_file_extension(".json")
        with dpg.file_dialog(tag=SAVEAS_TAG, show=False, directory_selector=False,
                             width=620, height=420, callback=self._saveas_selected):
            dpg.add_file_extension(".json")

    def _build_palette(self):
        with dpg.child_window(width=210, autosize_y=True):
            dpg.add_text("Blocks")
            dpg.add_separator()
            grouped = {}
            for spec in self.registry.values():
                grouped.setdefault(spec.category, []).append(spec)
            for category in ("source", "transform", "window", "features", "model", "sink"):
                specs = sorted(grouped.get(category, []), key=lambda s: s.title)
                if not specs:
                    continue
                with dpg.tree_node(label=category.title(), default_open=category != "source"):
                    for spec in specs:
                        dpg.add_button(label=spec.title, width=-1,
                                       user_data=spec.id, callback=self._add_from_palette)
                        if spec.help:
                            with dpg.tooltip(dpg.last_item()):
                                dpg.add_text(spec.help, wrap=320)

    def _build_canvas(self):
        with dpg.child_window(autosize_x=True, autosize_y=True):
            with dpg.node_editor(tag=EDITOR_TAG, callback=self._link_requested,
                                 delink_callback=self._unlink_requested,
                                 minimap=True,
                                 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                pass

    # ==================================================================
    # editing
    # ==================================================================
    def _add_from_palette(self, sender, app_data, user_data):
        node_id = self.document.add_node(user_data, position=self._free_position())
        self._draw_node(node_id)
        self._refresh_status()

    def _free_position(self):
        """Somewhere that does not sit exactly on top of an existing node."""
        count = len(self.document.nodes)
        return (40 + 190 * (count % 6), 40 + 150 * (count // 6))

    def _draw_node(self, node_id):
        node = self.document.nodes[node_id]
        spec = self.registry.get(node.spec_id)
        title = spec.title if spec else f"? {node.spec_id}"
        with dpg.node(label=f"{title}  [{node_id}]", parent=EDITOR_TAG,
                      tag=f"__node_{node_id}", pos=node.position):
            if spec is None:
                # An unresolved block is drawn, and says so, rather than being
                # dropped. Losing a user's work to a version difference is
                # worse than showing them something they have to deal with.
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    dpg.add_text("Not recognised by this LibEMG.", color=(230, 140, 100))
                    dpg.add_text(f"{node.params}", wrap=220)
                return

            for port in spec.inputs:
                attribute = dpg.add_node_attribute(
                    parent=f"__node_{node_id}", label=port.label,
                    attribute_type=dpg.mvNode_Attr_Input,
                    category=PortType.category_of(port.type))
                self._attr_ids[(node_id, port.name, "input")] = attribute
                dpg.add_text(port.label, parent=attribute)

            for port in spec.outputs:
                attribute = dpg.add_node_attribute(
                    parent=f"__node_{node_id}", label=port.label,
                    attribute_type=dpg.mvNode_Attr_Output,
                    category=PortType.category_of(port.type))
                self._attr_ids[(node_id, port.name, "output")] = attribute
                with dpg.group(horizontal=True, parent=attribute):
                    # A probe is a toggle on the port, not a block to place.
                    # Making it a block would clutter the canvas and make the
                    # user wire something they only want to look at.
                    dpg.add_checkbox(label="", default_value=False,
                                     tag=f"__probe_{node_id}_{port.name}",
                                     user_data=(node_id, port.name),
                                     callback=self._toggle_probe)
                    dpg.add_text(f"{port.label} >")
                with dpg.tooltip(dpg.last_container()):
                    dpg.add_text(f"Carries {port.type}. Tick to watch it live.")

            with dpg.node_attribute(parent=f"__node_{node_id}",
                                    attribute_type=dpg.mvNode_Attr_Static):
                self._draw_params(node_id, spec)
            with dpg.node_attribute(parent=f"__node_{node_id}",
                                    attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_button(label="Remove", width=-1, user_data=node_id,
                               callback=self._remove_node)

    def _draw_params(self, node_id, spec):
        plain = [p for p in spec.params if not p.advanced]
        advanced = [p for p in spec.params if p.advanced]
        for param in plain:
            self._draw_param(node_id, param)
        if advanced:
            with dpg.tree_node(label="Advanced", default_open=False):
                for param in advanced:
                    self._draw_param(node_id, param)

    def _draw_param(self, node_id, param):
        """One widget, chosen by the parameter's kind.

        This mapping is the whole reason a parameter declares a kind rather
        than a widget: the rule for what a value should be edited with lives
        here once, instead of being restated on every block.
        """
        value = self.document.nodes[node_id].params.get(param.name, param.default)
        tag = f"__param_{node_id}_{param.name}"
        data = (node_id, param.name)
        common = dict(tag=tag, user_data=data, callback=self._param_changed,
                      label=param.label, width=150)
        if param.kind == ENUM:
            dpg.add_combo(list(param.choices), default_value=value or param.default,
                          **common)
        elif param.kind == BOOL:
            dpg.add_checkbox(default_value=bool(value), tag=tag, user_data=data,
                             callback=self._param_changed, label=param.label)
        elif param.kind == INT:
            dpg.add_input_int(default_value=int(value or 0), step=1, **common)
        elif param.kind == FLOAT:
            dpg.add_input_float(default_value=float(value or 0.0), step=0.0,
                                format="%.4f", **common)
        elif param.kind == MULTI_SELECT:
            dpg.add_text(param.label)
            dpg.add_listbox(list(param.choices), num_items=6,
                            default_value=(value or [None])[0] if value else "",
                            tag=tag, user_data=data, width=150,
                            callback=self._multi_toggled)
            dpg.add_text(_summarise(value), tag=f"{tag}__summary", wrap=200,
                         color=(150, 190, 210))
        elif param.kind in (PATH, FOLDER):
            dpg.add_input_text(default_value=str(value or ""), **common)
            dpg.add_button(label=f"Browse {param.label}", width=-1,
                           user_data=(node_id, param.name, param.kind),
                           callback=self._browse)
        else:
            multiline = param.name == "regex_filters"
            dpg.add_input_text(default_value=str(value or ""), multiline=multiline,
                               height=70 if multiline else 0, **common)
        if param.help:
            with dpg.tooltip(tag):
                dpg.add_text(param.help, wrap=320)

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
            self._say(problem)
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

    def _save(self):
        if self.path is None:
            dpg.show_item(SAVEAS_TAG)
            return
        self._capture_positions()
        self.document.save(self.path)
        self._say(f"Saved to {self.path}")

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
            self._say(f"Could not open that file: {error}")
            return
        self.path = path
        self._rebuild_canvas()

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
            self._say("Already running. Stop it first.")
            return
        self._capture_positions()
        try:
            self.pipeline = compile_pipeline(self.document)
        except CompileError as error:
            self._say(str(error))
            return
        except Exception:
            self._say("Could not build the pipeline:\n"
                      + traceback.format_exc().strip().splitlines()[-1])
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
                      + traceback.format_exc().strip().splitlines()[-1])
            self.pipeline = None
            return
        dpg.configure_item(PROGRESS_TAG, show=False)
        self._open_scope()
        self._say("Running.")

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
        self._say("Running over stored data.")

    def _record_progress(self, fraction):
        self._offline_progress = float(fraction)

    def stop(self):
        """Stop whatever is running."""
        if self.pipeline is None:
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
        self._say("Stopped.")

    def _train(self):
        """Fit the model this pipeline names, on the data it reads.

        The step that used to need a script. A pipeline built for stored data
        already describes everything a fit needs: where the recordings are, how
        they are filtered and windowed, which features to take and which labels
        to learn. Training reuses that description rather than asking for it
        again, so the model is fitted on exactly what it will be scored on.
        """
        if self.pipeline is not None:
            self._say("Stop the running pipeline before training.")
            return
        if self.document.mode() != "offline":
            self._say("Training reads stored data. Add a Stored Data source, or "
                      "open the pipeline you collected your training data with.")
            return
        self._capture_positions()
        try:
            pipeline = compile_pipeline(self.document)
        except CompileError as error:
            self._say(str(error))
            return

        import threading
        dpg.configure_item(PROGRESS_TAG, show=True)
        dpg.set_value(PROGRESS_TAG, 0.0)
        self._offline_progress = 0.0
        self._trained = None
        self.pipeline = pipeline

        def work():
            try:
                self._trained = pipeline.train(on_progress=self._record_progress)
            except Exception as error:
                self._last_error = f"{type(error).__name__}: {error}"

        self._offline_thread = threading.Thread(target=work, daemon=True,
                                                name="libemg-pipeline-train")
        self._offline_thread.start()
        self._training = True
        self._say("Training on the stored data.")

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
        if self.pipeline is None:
            return
        if hasattr(self.pipeline, "plan"):
            self._poll_offline()
        else:
            self._poll_online()

    def _poll_offline(self):
        if dpg.does_item_exist(PROGRESS_TAG):
            dpg.set_value(PROGRESS_TAG, self._offline_progress)
        thread = self._offline_thread
        if thread is not None and not thread.is_alive():
            self._offline_thread = None
            training = getattr(self, "_training", False)
            self._training = False
            if self._last_error:
                self._say(("Training failed: " if training else "The run failed: ")
                          + self._last_error)
                self._last_error = ""
            elif training:
                trained = getattr(self, "_trained", None) or {}
                classes = trained.get("classes")
                self._say(
                    "Trained on {windows} windows of {features} features"
                    .format(**{"windows": trained.get("windows", 0),
                               "features": trained.get("features", 0)})
                    + (f" over classes {classes}" if classes else "")
                    + f". Saved to {trained.get('model_path', '?')}. "
                    "Point a live pipeline at that file to run it.")
            else:
                self._show_results(self.pipeline.results)
                self._say("Finished." if not self.pipeline.stopped
                          else f"Stopped after {self.pipeline.progress:.0%}.")
            self.pipeline = None

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
        dpg.add_text(f"{node_id}.{port}  ({render})")
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
    def _show_results(self, results):
        if dpg.does_alias_exist(RESULTS_TAG):
            dpg.delete_item(RESULTS_TAG)
        if not results:
            self._say("The run finished, but produced no metrics. "
                      "Connect an Offline Metrics block to score it.")
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
                dpg.add_separator()
                dpg.add_text(name)
                matrix = np.asarray(matrix, dtype=float)
                with dpg.plot(height=260, width=-1, no_menus=True):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Predicted")
                    with dpg.plot_axis(dpg.mvYAxis, label="True"):
                        dpg.add_heat_series(matrix.ravel().tolist(),
                                            matrix.shape[0], matrix.shape[1],
                                            scale_min=float(matrix.min()),
                                            scale_max=float(matrix.max()))
            dpg.add_separator()
            dpg.add_button(label="Copy as CSV", callback=lambda: dpg.set_clipboard_text(
                _as_csv(results)))

    # ==================================================================
    # status
    # ==================================================================
    def _refresh_status(self):
        if not dpg.does_item_exist(PROBLEMS_TAG):
            return
        mode = self.document.mode()
        problems = self.document.validate()
        if problems:
            dpg.set_value(PROBLEMS_TAG,
                          f"[{mode}]  not ready:\n" + "\n".join(f"  - {p}" for p in problems))
            dpg.configure_item(PROBLEMS_TAG, color=(230, 170, 110))
        else:
            dpg.set_value(PROBLEMS_TAG, f"[{mode}]  ready to run.")
            dpg.configure_item(PROBLEMS_TAG, color=(140, 210, 160))

    def _say(self, message):
        if dpg.does_item_exist(PROBLEMS_TAG):
            dpg.set_value(PROBLEMS_TAG, message)
            dpg.configure_item(PROBLEMS_TAG, color=(220, 220, 220))


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
