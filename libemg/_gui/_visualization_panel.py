import threading
import time

import dearpygui.dearpygui as dpg
import numpy as np


class VisualizationPanel:
    """Live signal viewer drawn with dearpygui's native plots.

    One plot per modality, one line series per channel, updated in place from
    the shared memory the OnlineDataHandler already holds open. Nothing is
    rasterised on the way: the samples go straight from the shared-memory buffer
    into the series, so a frame costs a buffer copy and a series update rather
    than a figure render, a PNG encode and a texture upload.

    Parameters
    ----------
    online_data_handler: OnlineDataHandler
        The handler whose shared memory is plotted.
    num_samples: int (optional), default=500
        Initial number of samples shown per modality. Each modality's plot has
        its own control, so this is only the starting value.
    refresh_rate: float (optional), default=60.0
        Target plot updates per second. Building a frame costs well under a
        millisecond, so this — not the work — sets how stale the newest plotted
        sample is. 60 matches the display refresh; going higher does not reach
        the screen any sooner. See get_frame_stats for what was achieved.
    plot_height: int (optional), default=220
        Height in pixels of each modality's plot.
    """

    def __init__(self,
                 online_data_handler,
                 num_samples=500,
                 refresh_rate=60.0,
                 plot_height=220):
        self.online_data_handler = online_data_handler
        self.num_samples = num_samples
        self.refresh_rate = refresh_rate
        self.plot_height = plot_height

        self.modalities = []
        self.channels = {}
        self.buffer_samples = {}
        self._thread = None
        self._stop_event = threading.Event()
        # Rolling frame timings, for get_frame_stats(). Bounded so a long
        # session cannot grow it without limit.
        self._frame_times = []
        self._frame_times_lock = threading.Lock()
        self._max_frame_samples = 600

        self.widget_tags = {"visualization": ["__vls_visualize_window"]}

    # ------------------------------------------------------------------
    # tags
    # ------------------------------------------------------------------
    def _enable_tag(self, mod):
        return f"__vls_enable_{mod}"

    def _samples_tag(self, mod):
        return f"__vls_samples_{mod}"

    def _plot_tag(self, mod):
        return f"__vls_plot_{mod}"

    def _xaxis_tag(self, mod):
        return f"__vls_xaxis_{mod}"

    def _yaxis_tag(self, mod):
        return f"__vls_yaxis_{mod}"

    def _series_tag(self, mod, channel):
        return f"__vls_series_{mod}_{channel}"

    # ------------------------------------------------------------------
    # window
    # ------------------------------------------------------------------
    def cleanup_window(self):
        """Stop the updater and delete the window, in that order.

        The updater writes into the plot items, so it has to be stopped before
        they are deleted or it will address items that no longer exist.
        """
        self.stop_callback()
        for tag in self.widget_tags["visualization"]:
            if dpg.does_alias_exist(tag):
                dpg.delete_item(tag)

    def _detect_modalities(self):
        """Read one snapshot to learn which modalities exist and how wide they are.

        The channel count comes from the data rather than the shared-memory
        declaration so that an installed channel mask is reflected.
        """
        vals, _ = self.online_data_handler.get_data(N=0)
        self.modalities = [mod for mod in self.online_data_handler.modalities if mod in vals]
        self.channels = {mod: int(vals[mod].shape[1]) for mod in self.modalities}
        self.buffer_samples = {mod: int(vals[mod].shape[0]) for mod in self.modalities}

    def spawn_window(self):
        """Build the visualization window. Plots stay idle until Start."""
        self.cleanup_window()
        self._detect_modalities()

        if not self.modalities:
            raise ConnectionError(
                "Attempted to visualize, but no modalities were found in shared memory. "
                "Please ensure the OnlineDataHandler is receiving data."
            )

        with dpg.window(label="Visualize Live Signals",
                        tag="__vls_visualize_window",
                        width=900,
                        height=200 + self.plot_height * len(self.modalities),
                        on_close=lambda: self.stop_callback()):

            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="__vls_start_button",
                               callback=self.start_callback)
                dpg.add_button(label="Stop", tag="__vls_stop_button",
                               callback=self.stop_callback)
                dpg.add_text("Stopped", tag="__vls_status")

            dpg.add_separator()
            dpg.add_text("Modalities")

            # One row per modality: show/hide, and its own sample count. Both are
            # read on every frame, so changes take effect without a callback.
            with dpg.table(header_row=True, policy=dpg.mvTable_SizingStretchProp,
                           borders_outerH=True, borders_innerV=True,
                           borders_innerH=True, borders_outerV=True):
                dpg.add_table_column(label="Show")
                dpg.add_table_column(label="Modality")
                dpg.add_table_column(label="Channels")
                dpg.add_table_column(label="Samples plotted")
                for mod in self.modalities:
                    with dpg.table_row():
                        dpg.add_checkbox(tag=self._enable_tag(mod), default_value=True,
                                         callback=self._toggle_modality_callback,
                                         user_data=mod)
                        dpg.add_text(mod)
                        dpg.add_text(str(self.channels[mod]))
                        dpg.add_input_int(
                            tag=self._samples_tag(mod),
                            default_value=min(self.num_samples, self.buffer_samples[mod]),
                            min_value=2,
                            max_value=self.buffer_samples[mod],
                            min_clamped=True,
                            max_clamped=True,
                            step=100,
                            width=160,
                        )

            dpg.add_separator()

            for mod in self.modalities:
                with dpg.plot(label=mod, tag=self._plot_tag(mod),
                              height=self.plot_height, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Sample (oldest to newest)",
                                      tag=self._xaxis_tag(mod))
                    dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude",
                                      tag=self._yaxis_tag(mod))
                    for channel in range(self.channels[mod]):
                        dpg.add_line_series([], [],
                                            label=f"{mod}_CH{channel + 1}",
                                            parent=self._yaxis_tag(mod),
                                            tag=self._series_tag(mod, channel))

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _toggle_modality_callback(self, sender, app_data, user_data):
        plot = self._plot_tag(user_data)
        if not dpg.does_item_exist(plot):
            return
        if app_data:
            dpg.show_item(plot)
        else:
            dpg.hide_item(plot)

    def start_callback(self):
        """Begin updating the plots."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._frame_times_lock:
            self._frame_times = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if dpg.does_item_exist("__vls_status"):
            dpg.set_value("__vls_status", "Running")

    def stop_callback(self):
        """Stop updating the plots, leaving the last frame on screen."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        self._thread = None
        if dpg.does_item_exist("__vls_status"):
            dpg.set_value("__vls_status", "Stopped")

    # ------------------------------------------------------------------
    # updating
    # ------------------------------------------------------------------
    def _run(self):
        period = 1.0 / self.refresh_rate if self.refresh_rate > 0 else 0.0
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                self._update_plots()
            except Exception:
                # The window can be torn down mid-frame, which leaves the items
                # this writes to gone. Stop rather than spin on the failure.
                break
            elapsed = time.perf_counter() - started
            with self._frame_times_lock:
                self._frame_times.append(elapsed)
                if len(self._frame_times) > self._max_frame_samples:
                    del self._frame_times[:-self._max_frame_samples]
            if self._stop_event.wait(max(0.0, period - elapsed)):
                break

    def _update_plots(self):
        vals, _ = self.online_data_handler.get_data(N=0)
        for mod in self.modalities:
            if not dpg.get_value(self._enable_tag(mod)):
                # Hidden plots are skipped, so narrowing the view to one
                # modality buys back the work the others were costing.
                continue
            requested = int(dpg.get_value(self._samples_tag(mod)))
            data = vals[mod]
            n = max(2, min(requested, data.shape[0]))
            # Shared-memory buffers are newest-first; flip so the newest sample
            # is on the right, the way a scope reads.
            window = np.flip(data[:n, :], axis=0)
            x = list(range(n))
            for channel in range(self.channels[mod]):
                dpg.set_value(self._series_tag(mod, channel),
                              [x, window[:, channel].tolist()])
            dpg.set_axis_limits(self._xaxis_tag(mod), 0, n - 1)
            dpg.fit_axis_data(self._yaxis_tag(mod))

    # ------------------------------------------------------------------
    # instrumentation
    # ------------------------------------------------------------------
    def get_frame_stats(self):
        """Return timing for the frames drawn so far.

        Returns
        ----------
        stats: dict
            ``frames``, and when any were drawn, the mean/p95/max seconds spent
            building a frame plus the ``max_fps`` those timings would sustain.
        """
        with self._frame_times_lock:
            times = list(self._frame_times)
        if not times:
            return {"frames": 0}
        times = np.array(times)
        return {
            "frames": int(times.size),
            "mean": float(times.mean()),
            "p95": float(np.percentile(times, 95)),
            "max": float(times.max()),
            "max_fps": float(1.0 / times.mean()) if times.mean() else float("inf"),
        }
