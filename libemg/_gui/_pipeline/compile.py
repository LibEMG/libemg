"""Turn a pipeline document into something that runs.

Online, the document becomes a reactive graph: a streamer writes into shared
memory and each stage is woken by the one before it. Offline, the same document
is walked once over a recording, which is a sequential run rather than a graph
because there is no stream to react to and a progress bar wants to know how far
through it is.

Which of the two happens is inferred from the document's sources, never
configured. A user who had to declare it could declare it wrongly and get a
pipeline that silently did nothing.

Fusing the window
-----------------
A window block does not become a stage of its own. What it configures is *how
the next stage observes the one before it*: its increment becomes that stage's
criterion and its size becomes the number of rows delivered. So a window
between a filter and a feature block compiles to the feature block observing
the filter with ``OnSamples(increment)`` in window mode. That is why the block
has no online-or-offline mode to set, and why nothing has to copy an enframed
array into shared memory just to hand it along.
"""

import os
import time

import numpy as np

from libemg._gui._pipeline.registry import (FEATURES, MODEL, SINK, SOURCE,
                                            TRANSFORM, WINDOW)
from libemg.reactive import (DELTA, LATEST, WINDOW as WINDOW_MODE, Hook, Input,
                             OnCommit, OnSamples, Output, Periodic,
                             ReactiveGraph, default_notifier_pool)


def tag_for(node_id, port):
    """The shared-memory item a node's output port publishes to."""
    return f"pipe_{node_id}_{port}"


def _as_floats(text, default=None):
    """Parse a comma-separated parameter into numbers."""
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return default
    try:
        return [float(p) for p in parts]
    except ValueError:
        return default


def _as_ints(text, default=None):
    values = _as_floats(text, None)
    return [int(v) for v in values] if values is not None else default


class CompileError(Exception):
    """Raised when a document cannot be turned into a runnable pipeline."""


# ======================================================================
# Hooks the compiler builds
# ======================================================================
class _FilterHook(Hook):
    """Apply a filter to a stream, once, for everything downstream."""

    def __init__(self, name, source, target, filter_config, shape, window, increment):
        super().__init__(
            name,
            inputs=[Input(source, OnSamples(increment), mode=WINDOW_MODE, size=window)],
            outputs=[Output(target, shape, np.double)],
        )
        self.source, self.target = source, target
        self.filter_config = filter_config
        self.increment = increment
        self._filter = None

    def setup(self, context):
        # Built here, not in __init__: a spawned executor receives this hook by
        # pickling, and a configured filter object need not survive that.
        from libemg.filtering import Filter
        self._filter = Filter(sampling_frequency=self.filter_config["sampling_rate"])
        self._filter.install_filters(self.filter_config["dictionary"])

    def step(self, data, snapshots):
        samples = data[self.source]
        if samples.shape[0] == 0:
            return None
        filtered = self._filter.filter(samples)
        # Only the newest increment is new. The rest was emitted by an earlier
        # run and re-publishing it would duplicate rows downstream; it is read
        # at all so the filter sees enough history for its edges to settle.
        return {self.target: filtered[-self.increment:]}


class _ChannelMaskHook(Hook):
    """Narrow a stream to a subset of channels."""

    def __init__(self, name, source, target, channels, shape, increment=1):
        super().__init__(
            name,
            inputs=[Input(source, OnSamples(increment), mode=DELTA)],
            outputs=[Output(target, shape, np.double)],
        )
        self.source, self.target, self.channels = source, target, channels

    def step(self, data, snapshots):
        samples = data[self.source]
        if samples.shape[0] == 0:
            return None
        return {self.target: samples[:, self.channels]}


class _FeatureHook(Hook):
    """Extract features from each window."""

    def __init__(self, name, source, target, feature_list, window_size,
                 window_increment, num_features, rows=200):
        super().__init__(
            name,
            inputs=[Input(source, OnSamples(window_increment),
                          mode=WINDOW_MODE, size=window_size)],
            outputs=[Output(target, (rows, num_features), np.double)],
        )
        self.source, self.target = source, target
        self.feature_list = list(feature_list)
        self.window_size = window_size
        self._extractor = None

    def setup(self, context):
        from libemg.feature_extractor import FeatureExtractor
        self._extractor = FeatureExtractor()

    def step(self, data, snapshots):
        samples = data[self.source]
        if samples.shape[0] < self.window_size:
            return None
        window = samples.transpose()[np.newaxis, :, :]
        return {self.target: self._extractor.extract_features(
            self.feature_list, window, array=True)}


class _ModelHook(Hook):
    """Run a fitted predictor on whatever its input delivers."""

    def __init__(self, name, source, target, model_path, width, rows=200,
                 criterion=None, mode=LATEST, size=None):
        super().__init__(
            name,
            inputs=[Input(source, criterion or OnSamples(1), mode=mode, size=size)],
            outputs=[Output(target, (rows, width), np.double)],
        )
        self.source, self.target = source, target
        self.model_path = model_path
        self.width = width
        self._predictor = None

    def setup(self, context):
        import pickle
        # A fitted model can be large and is not always picklable in the way a
        # spawned process needs, so the hook carries a path and opens it here.
        with open(self.model_path, "rb") as handle:
            self._predictor = pickle.load(handle)
        # A model with memory -- a recurrent network, a transformer over a
        # horizon -- is fed one window per step live, so it has to carry its
        # state from one step to the next instead of starting fresh each time.
        model = getattr(self._predictor, "model", self._predictor)
        if hasattr(model, "streaming"):
            model.streaming = True
            model.reset()

    def step(self, data, snapshots):
        model_input = np.atleast_2d(data[self.source])
        if model_input.size == 0 or self._predictor is None:
            return None
        row = self._predict(model_input)
        if row is None:
            return None
        padded = np.zeros((1, self.width), dtype=np.double)
        row = np.asarray(row, dtype=np.double).ravel()[:self.width]
        padded[0, :row.size] = row
        return {self.target: padded}

    def _predict(self, model_input):
        raise NotImplementedError


class _ClassifierHook(_ModelHook):
    """Predict a class, with its confidence."""

    def _predict(self, model_input):
        probabilities = self._predictor._predict_proba(model_input)
        probabilities = np.atleast_2d(probabilities)
        index = int(np.argmax(probabilities, axis=1)[0])
        return [float(index), float(np.max(probabilities))]


class _RegressorHook(_ModelHook):
    """Predict continuous outputs."""

    def _predict(self, model_input):
        return np.atleast_2d(self._predictor._predict(model_input))[0]


class _SinkHook(Hook):
    """Hand each output to a LibEMG output writer."""

    def __init__(self, name, source, kind, config):
        super().__init__(name, inputs=[Input(source, OnCommit(), mode=LATEST)])
        self.source, self.kind, self.config = source, kind, config
        self._writer = None

    def setup(self, context):
        from libemg.output_writer import (ConsoleOutputWriter, FileOutputWriter,
                                          SocketOutputWriter)
        # Sockets and file handles are exactly the things that cannot be
        # pickled into a spawned process, so they are opened here.
        if self.kind == "socket":
            self._writer = SocketOutputWriter("value", ip=self.config["ip"],
                                              port=self.config["port"],
                                              protocol=self.config["protocol"])
        elif self.kind == "file":
            path = self.config["file_path"]
            folder = os.path.dirname(os.path.abspath(path))
            os.makedirs(folder, exist_ok=True)
            self._writer = FileOutputWriter("value", folder + os.sep,
                                            os.path.basename(path))
        else:
            self._writer = ConsoleOutputWriter("value")

    def step(self, data, snapshots):
        row = np.asarray(data[self.source]).ravel()
        self._writer.write({"timestamp": time.time(),
                            "value": row.tolist(),
                            "prediction": row[0] if row.size else None,
                            "probability": row[1] if row.size > 1 else None,
                            "velocity": None})
        return None


class _ProbeHook(Hook):
    """Publish a rate-limited copy of a port for the editor to draw.

    The probe writes into its own shared-memory item rather than calling back,
    because the thing drawing it is a GUI in another process and a probe must
    never be able to stall what it watches.
    """

    def __init__(self, name, source, target, shape, hz, mode, size):
        super().__init__(
            name,
            inputs=[Input(source, Periodic(hz), mode=mode, size=size)],
            outputs=[Output(target, shape, np.double)],
        )
        self.source, self.target, self.rows = source, target, shape[0]
        self.width = shape[1]

    def step(self, data, snapshots):
        block = np.atleast_2d(np.asarray(data[self.source], dtype=np.double))
        out = np.zeros((min(block.shape[0], self.rows), self.width), dtype=np.double)
        usable = min(block.shape[1], self.width)
        out[:, :usable] = block[:out.shape[0], :usable]
        return {self.target: out}


# ======================================================================
# The compiled results
# ======================================================================
class OnlinePipeline:
    """A running, or runnable, live pipeline.

    Produced by :func:`compile_pipeline`. Owns the streamer process and the
    reactive graph, so stopping it stops both.
    """

    def __init__(self, document, graph, shared_memory_items, streamer_specs,
                 probe_items, source_tags, log=None):
        self.document = document
        self.graph = graph
        self.shared_memory_items = shared_memory_items
        self.streamer_specs = streamer_specs
        self.probe_items = probe_items
        self.source_tags = source_tags
        self.log = log
        self.streamers = []
        self._running = False

    @property
    def running(self):
        return self._running

    def start(self):
        """Start the devices, then the graph."""
        if self._running:
            return self
        from libemg import streamers as streamer_module
        from libemg._gui._pipeline import synthetic
        pool = default_notifier_pool()
        for spec in self.streamer_specs:
            function = getattr(streamer_module, spec["function"], None)
            if function is None:
                function = getattr(synthetic, spec["function"])
            result = function(shared_memory_items=spec["items"], **spec["kwargs"])
            handle = result[0] if isinstance(result, tuple) else result
            if handle is not None:
                handle.notifier_pool = pool
            self.streamers.append(handle)
        self.graph.start()
        self._running = True
        return self

    def stop(self, timeout=5.0):
        """Stop the graph, then the devices."""
        if not self._running:
            return
        self.graph.stop(timeout=timeout)
        for handle in self.streamers:
            try:
                if hasattr(handle, "signal"):
                    handle.signal.set()
                elif hasattr(handle, "terminate"):
                    handle.terminate()
            except Exception:
                pass
        self.streamers = []
        self._running = False

    def status(self):
        """Throughput and lag for each item the pipeline publishes.

        Returns
        ----------
        dict
            Mapping from item tag to its
            :class:`~libemg.shared_memory_manager.Snapshot`.
        """
        from libemg.shared_memory_manager import SharedMemoryManager
        reader = SharedMemoryManager()
        out = {}
        declared = {item[0]: item for item in self.shared_memory_items}
        for tag, item in declared.items():
            if tag.endswith("_count"):
                continue
            if reader.find_variable(*item):
                out[tag] = reader.snapshot(tag)
        reader.cleanup(parent=False)
        return out

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


class OfflinePipeline:
    """A pipeline over stored data, run once from start to finish."""

    def __init__(self, document, plan):
        self.document = document
        self.plan = plan
        self.progress = 0.0
        self.results = {}
        #: True when the last run was cut short. Metrics from a stopped run are
        #: real but computed over only the recordings that were read, so a
        #: caller has to be able to tell the two apart.
        self.stopped = False
        self.files_read = 0
        self.files_total = 0
        self._stop = False
        #: A name per feature column, from the last read of the recording.
        self.feature_names = []
        #: Windows in each file of the last read, in order.
        self.file_windows = []
        #: Every EpochResult of the last deep training run, as they completed.
        self.epochs = []
        #: The last deep training run as a TrainingOutcome, once it has finished.
        self.outcome = None
        #: For a Feature Viewer block: features, names, labels and its settings.
        self.feature_view = None
        #: For a Prediction Viewer block: every recording's signal, labels and
        #: what the model made of it (see :mod:`.prediction_viewer`).
        self.prediction_view = None
        #: Per file of the last read, when a Prediction Viewer wants them: the
        #: filtered signal, the window labels and the filename metadata.
        self.file_details = []

    def request_stop(self):
        """Ask a run in progress to finish early."""
        self._stop = True

    def run(self, on_progress=None):
        """Walk the recording through every stage and score the result.

        Parameters
        ----------
        on_progress: callable or None (optional), default=None
            Called as ``on_progress(fraction)`` as the run advances. A run over
            stored data knows its own total, which is why this can report a
            real fraction rather than a spinner.

        Returns
        ----------
        dict
            Metric name to value, from the metrics block. Empty if there is
            none.
        """
        from libemg.offline_metrics import OfflineMetrics

        plan = self.plan
        windows, features, truth = self._gather(on_progress)
        if windows is None:
            self.results = {}
            return self.results

        predictions = None
        predictor = None
        view = plan.get("feature_view")
        model_missing = bool(plan["model_path"]) and not os.path.exists(plan["model_path"])
        # Scoring cannot happen without the model. A Feature Viewer can still
        # draw the features, so when it is the only thing that wanted the model
        # the run goes on and the viewer says why the model's spaces are absent.
        if model_missing and (plan["metrics"] or plan.get("prediction_view") or view is None):
            raise CompileError(
                f"The model block's Fitted Model file {plan['model_path']} does not exist "
                "yet. Press Train to fit and save it first.")
        if plan["model_path"] and not model_missing:
            import pickle
            with open(plan["model_path"], "rb") as handle_:
                predictor = pickle.load(handle_)
            model_input = self._model_input(windows, features)
            predictions = np.asarray(predictor._predict(model_input))
            # Classification metrics compare indices, and an index that came
            # back as a float would make every comparison and every confusion
            # matrix row wrong. Which one to do follows from what was saved.
            from libemg.emg_predictor import EMGClassifier as _Classifier
            if isinstance(predictor, _Classifier):
                predictions = predictions.ravel().astype(int)
                if truth is not None:
                    truth = np.asarray(truth).astype(int)
            elif predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)

        # What a Feature Viewer block draws: the matrix, a name per column and
        # the labels, kept so the window can change its view without reading
        # the recording again.
        self.feature_view = None
        if view is not None and features is not None:
            self.feature_view = {"features": features, "names": list(self.feature_names),
                                 "labels": truth if view.get("labels") else None,
                                 "params": view, "spaces": [], "model_note": ""}
            if view.get("model"):
                # A connected model adds the spaces it works in: its latent
                # space, its decision scores, its outputs.
                from libemg._gui._pipeline.feature_viewer import model_spaces
                if predictor is None:
                    self.feature_view["model_note"] = (
                        "The connected model has not been trained yet, so only the features "
                        "are shown. Press Train, then Start again.")
                else:
                    self.feature_view["spaces"] = model_spaces(
                        predictor, features if features is not None else windows)

        self.prediction_view = None
        if plan.get("prediction_view") is not None and predictor is not None:
            self.prediction_view = self._prediction_view(predictor, windows, features)

        self.results = {}
        if plan["metrics"] and predictions is not None and truth is not None:
            count = min(len(predictions), len(truth))
            self.results = OfflineMetrics().extract_offline_metrics(
                plan["metrics"], truth[:count], predictions[:count],
                null_label=plan["null_label"])
        # A run that was asked to stop must not claim to have finished. Setting
        # the fraction to one and reporting it would leave a progress bar full
        # and a caller unable to tell a cancelled run from a complete one, even
        # though the metrics below cover only part of the recording.
        if not self.stopped:
            self.progress = 1.0
            if on_progress is not None:
                on_progress(1.0)
        return self.results

    def _gather(self, on_progress=None):
        """Walk the recording into windows, features and labels.

        Shared by scoring and by training, because they read the recording the
        same way and differ only in what they do at the end. One walk means a
        model is fitted on exactly the data it will later be scored on.

        Returns
        ----------
        windows: numpy.ndarray or None
            Enframed samples, or None when the recording produced none.
        features: numpy.ndarray or None
            Extracted features, or None when there is no features block.
        truth: numpy.ndarray or None
            Windowed labels, or None when the recording carries none.
        """
        from libemg.data_handler import OfflineDataHandler, RegexFilter
        from libemg.feature_extractor import FeatureExtractor
        from libemg.filtering import Filter
        from libemg.utils import get_windows

        plan = self.plan
        handler = OfflineDataHandler()
        filters = [RegexFilter(**spec) for spec in plan["regex_filters"]]
        handler.get_data(folder_location=plan["folder"], regex_filters=filters,
                         metadata_fetchers=_metadata_fetchers(plan),
                         delimiter=plan["delimiter"])
        total = max(1, len(handler.data))
        self.progress = 0.0
        self.stopped = False
        self.files_total = len(handler.data)
        self.files_read = 0

        windows, labels = [], []
        # Windows per file, in order, so a model that reads sequences is never
        # trained across the join between two recordings.
        self.file_windows = []
        self.file_details = []
        keep_details = plan.get("prediction_view") is not None
        window_size, increment = plan["window_size"], plan["window_increment"]
        conditioner = None
        if plan["filters"]:
            conditioner = Filter(sampling_frequency=plan["sampling_rate"])
            for dictionary in plan["filters"]:
                if dictionary.get("name") == "standardize":
                    # Standardizing means subtracting a mean and dividing by a
                    # deviation, and those have to be measured from data. Here
                    # the recording is already loaded, so it supplies them.
                    dictionary = dict(dictionary, data=handler)
                conditioner.install_filters(dictionary)

        for index, block in enumerate(handler.data):
            if self._stop:
                self.stopped = True
                break
            if plan["channels"] is not None:
                block = block[:, plan["channels"]]
            if conditioner is not None:
                block = conditioner.filter(block)
            enframed = get_windows(block, window_size, increment)
            if enframed.shape[0]:
                windows.append(enframed)
                self.file_windows.append(int(enframed.shape[0]))
                labels.append(self._labels_for(handler, plan["label_key"],
                                               index, window_size, increment))
                if keep_details:
                    self.file_details.append({
                        "index": len(self.file_details), "signal": np.asarray(block),
                        "truth": labels[-1], "meta": _file_metadata(handler, plan, index)})
            self.files_read = index + 1
            self.progress = (index + 1) / total
            if on_progress is not None:
                on_progress(self.progress)

        if not windows:
            return None, None, None

        windows = np.vstack(windows)
        truth = np.concatenate([l for l in labels if l is not None]) \
            if any(l is not None for l in labels) else None

        features = None
        if plan["features"]:
            # Extracted by name and stacked in the order asked for, which is
            # exactly what array=True builds, so that each column can be named.
            by_name = FeatureExtractor().extract_features(plan["features"], windows)
            features = np.hstack([by_name[name] for name in plan["features"]])
            from libemg._gui._pipeline.feature_viewer import column_names
            self.feature_names = column_names(plan["features"], by_name, windows.shape[1])
        return windows, features, truth

    def _prediction_view(self, predictor, windows, features):
        """Predict every recording on its own, for a Prediction Viewer block.

        Each recording is marked as training or held out when that is known:
        from the block's own Held Out setting, or else from the recordings a
        deep model put aside for validation while it trained here.
        """
        from libemg._gui._pipeline.prediction_viewer import predict_recordings
        from libemg.emg_predictor import EMGClassifier

        params = self.plan["prediction_view"]
        model_input = self._model_input(windows, features)
        edges = np.cumsum([0] + list(self.file_windows))
        pieces = [model_input[a:b] for a, b in zip(edges[:-1], edges[1:])]
        classification = isinstance(predictor, EMGClassifier)
        results = predict_recordings(predictor, pieces, classification)
        model = getattr(predictor, "model", predictor)
        if hasattr(model, "reset"):
            model.reset()
        classes = getattr(model, "classes_", None)

        splits = [""] * len(self.file_details)
        field = (params.get("held_out_field") or "").strip()
        values = {v.strip() for v in str(params.get("held_out_values") or "").split(",")
                  if v.strip()}
        note = ""
        if field and values:
            splits = ["held out" if str(d["meta"].get(field)) in values else "training"
                      for d in self.file_details]
            note = f"Held out: {field} {', '.join(sorted(values))}, as the block sets it."
        else:
            trained = getattr(model, "trained_on", None) or {}
            held = trained.get("validation_indices") or []
            if held and trained.get("recordings") == len(self.file_details):
                splits = ["held out" if i in set(held) else "training"
                          for i in range(len(self.file_details))]
                note = "Held out: the recordings the model was validated on while it trained."

        recordings = []
        for detail, (outputs, predicted), split in zip(self.file_details, results, splits):
            truth = detail["truth"]
            if truth is not None and classification:
                truth = np.asarray(truth).reshape(len(truth), -1)[:, 0].astype(int)
            recordings.append(dict(detail, truth=truth, outputs=outputs,
                                   predicted=predicted, split=split))
        return {"task": "classification" if classification else "regression",
                "classes": [] if classes is None else list(np.asarray(classes).tolist()),
                "sampling_rate": self.plan["sampling_rate"],
                "window_size": self.plan["window_size"],
                "window_increment": self.plan["window_increment"],
                "params": params, "note": note, "recordings": recordings}

    def _model_input(self, windows, features):
        """What the model block reads: features, or the raw windows it is linked to."""
        if self.plan.get("model_input") == "windows" or features is None:
            return windows
        return features

    def train(self, on_progress=None, model_path=None, on_epoch=None):
        """Fit the pipeline's model on this recording and save it.

        The last thing that still needed a script. A model block names a model
        and a file; this reads the recording exactly as a scoring run does,
        fits that model to the features and labels it produced, and writes it
        where the block expects to find it. The same document can then be
        pointed at a device and run live.

        Parameters
        ----------
        on_progress: callable or None (optional), default=None
            Called as ``on_progress(fraction)`` while the recording is read.
        model_path: str or None (optional), default=None
            Where to save. Defaults to the model block's own Fitted Model path,
            which is what makes the trained model immediately usable by the
            pipeline that trained it.
        on_epoch: callable or None (optional), default=None
            For a Deep Model block: called with each
            :class:`~libemg.deep.training.EpochResult` as an epoch completes.
            Every result is also kept in ``self.epochs``.

        Returns
        ----------
        dict
            ``model_path`` where it was written, ``windows`` and ``features``
            it was fitted on, ``classes`` it learned for a classifier, and
            ``dofs`` it learned for a regressor.

        Raises
        ----------
        CompileError
            If the block names no model or no file, if the model is one this
            version cannot fit, if the recording yields no data or carries no
            labels, or if the run was stopped before it finished.
        """
        import pickle
        from libemg.emg_predictor import (CLASSIFIER_MODELS, EMGClassifier,
                                          EMGRegressor, REGRESSOR_MODELS)

        plan = self.plan
        destination = model_path or plan["model_path"]
        if not destination:
            raise CompileError(
                "The model block has no Fitted Model path, so there is nowhere "
                "to save what training produces. Set that parameter first.")
        name = plan.get("model_name") or ""
        if not name:
            raise CompileError("This pipeline has no model block to train.")
        if plan.get("deep"):
            return self._train_deep(destination, on_progress, on_epoch)
        if name not in CLASSIFIER_MODELS and name not in REGRESSOR_MODELS:
            # Checked before the recording is read, not after. A misspelled
            # model would otherwise be reported only once the whole folder had
            # been walked, which for a real session is minutes of waiting for
            # an answer that was available immediately.
            raise CompileError(
                f"'{name}' is not a model this version can fit. Choose one of "
                f"{sorted(set(CLASSIFIER_MODELS) | set(REGRESSOR_MODELS))}.")

        windows, features, truth = self._gather(on_progress)
        if self.stopped:
            # A model fitted on the part of the recording that was read before
            # the stop would be saved under the same name as one fitted on all
            # of it, and nothing afterwards could tell them apart. Stopping
            # therefore leaves whatever file was there untouched.
            raise CompileError(
                "Training was stopped part way, so nothing was saved. The "
                "model file is unchanged.")
        if windows is None:
            raise CompileError(
                "That recording produced no windows. Check the folder, the "
                "regex filters, and that the window is not longer than the "
                "recordings.")
        if truth is None:
            raise CompileError(
                "That recording carries no labels, so there is nothing to learn "
                "from. Set the Stored Data block's Label Key to a metadata "
                "field its regex filters produce.")
        model_input = features if features is not None else \
            windows.reshape(windows.shape[0], -1)
        count = min(len(model_input), len(truth))
        model_input, truth = model_input[:count], truth[:count]

        # Which kind of predictor follows from which model was named, so the
        # user picks a model rather than picking a model and a kind that could
        # disagree with each other.
        if name in CLASSIFIER_MODELS:
            predictor = EMGClassifier(name)
            labels = truth.astype(int)
        elif name in REGRESSOR_MODELS:
            predictor = EMGRegressor(name)
            # A regressor predicts one value per degree of freedom, so its
            # targets are a matrix with a column per DOF. A label field that
            # holds a single value per sample is one DOF, and saying so here is
            # what lets a single-DOF recording train at all: scikit-learn's
            # multi-output wrapper rejects a flat vector outright.
            labels = np.asarray(truth, dtype=float)
            if labels.ndim == 1:
                labels = labels.reshape(-1, 1)

        predictor.fit({"training_features": model_input,
                       "training_labels": labels})
        folder = os.path.dirname(os.path.abspath(destination))
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(destination, "wb") as handle:
            pickle.dump(predictor, handle)
        return {"model_path": destination,
                "windows": int(windows.shape[0]),
                "features": int(model_input.shape[1]),
                "classes": sorted(set(np.asarray(labels).ravel().tolist()))
                if name in CLASSIFIER_MODELS else None,
                "dofs": None if name in CLASSIFIER_MODELS else int(labels.shape[1])}

    def _train_deep(self, destination, on_progress=None, on_epoch=None):
        """Train a Deep Model block's network, reporting every epoch as it ends.

        The recording is read exactly as for any other model, then split back
        into its files: a sequence model is trained on each file in order and
        never across the join between two. A stop asked for mid-training keeps
        the best epoch so far and saves it -- unlike a statistical fit, a
        network's best epoch is a finished model, not half of one.
        """
        import pickle
        from libemg.deep import DeepModel, TrainingSet
        from libemg.emg_predictor import EMGClassifier, EMGRegressor

        params = self.plan["deep"]
        task = str(params.get("task", "Classification")).lower()
        timing = str(params.get("timing", "Continuous")).lower()
        windows, features, truth = self._gather(on_progress)
        if self.stopped:
            raise CompileError("Training was stopped while the recording was being read, "
                               "so nothing was trained or saved.")
        if windows is None:
            raise CompileError(
                "That recording produced no windows. Check the folder, the regex filters, "
                "and that the window is not longer than the recordings.")
        if truth is None:
            raise CompileError(
                "That recording carries no labels, so there is nothing to learn from. Set "
                "the Stored Data block's Label Key to a metadata field its regex filters "
                "produce.")
        inputs = self._model_input(windows, features)
        count = min(len(inputs), len(truth))
        lengths = list(self.file_windows)
        if sum(lengths) != count:
            lengths = [count]
        try:
            training = TrainingSet.from_arrays(inputs[:count], np.asarray(truth)[:count],
                                               lengths=lengths, task=task, timing=timing)
        except ValueError as error:
            raise CompileError(str(error)) from error
        patience = int(params.get("patience", 10))
        model = DeepModel(
            str(params.get("architecture", "GRU")).lower(), task=task, timing=timing,
            pooling=params.get("pooling", "max"), hidden_size=int(params.get("hidden_size", 64)),
            layers=int(params.get("layers", 2)),
            embedding_size=int(params.get("embedding_size", 32)),
            dropout=float(params.get("dropout", 0.1)), horizon=int(params.get("horizon", 10)),
            epochs=int(params.get("epochs", 50)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            batch_size=int(params.get("batch_size", 32)),
            sequence_length=int(params.get("sequence_length", 32)),
            patience=patience or None, seed=int(params.get("seed", 0)))

        self.epochs = []

        def record(result):
            self.epochs.append(result)
            if on_epoch is not None:
                on_epoch(result)

        try:
            model.fit(training, validation=float(params.get("validation", 0.2)),
                      on_epoch=record, should_stop=lambda: self._stop)
        except ValueError as error:
            raise CompileError(str(error)) from error
        if not model.history:
            raise CompileError("Training was stopped before its first epoch finished, so "
                               "there is nothing to save.")
        self.stopped = bool(self._stop)
        predictor = EMGClassifier(model) if task == "classification" else EMGRegressor(model)
        folder = os.path.dirname(os.path.abspath(destination))
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(destination, "wb") as handle:
            pickle.dump(predictor, handle)
        # The run, kept beside the model as JSON: readable without the model or
        # PyTorch, and what the Training History window draws in a later session.
        from libemg.deep import OUTCOME_SUFFIX
        self.outcome = model.outcome(model_path=destination,
                                     data={"folder": self.plan.get("folder", ""),
                                           "features": list(self.plan.get("features") or []),
                                           "window_size": self.plan.get("window_size"),
                                           "window_increment": self.plan.get("window_increment")})
        outcome_path = os.path.splitext(destination)[0] + OUTCOME_SUFFIX
        self.outcome.save(outcome_path)
        best = min(model.history, key=lambda r: r.validation_loss
                   if r.validation_loss is not None else r.train_loss)
        return {"model_path": destination, "windows": int(training.windows),
                "features": int(np.prod(training.input_shape)),
                "recordings": len(training.recordings),
                "classes": training.classes if task == "classification" else None,
                "dofs": training.dofs if task == "regression" else None,
                "epochs": len(model.history), "best_epoch": best.epoch,
                "best_score": best.score, "score_name": best.score_name,
                "stopped": self.stopped, "outcome_path": outcome_path}

    @staticmethod
    def _labels_for(handler, key, index, window_size, increment):
        """Window a metadata field alongside the data it belongs to.

        Returns one row per window: a scalar for a single-column field such as
        a class index, and a row per degree of freedom for a field that carries
        several. Collapsing to the first column here would silently turn a
        two-DOF regression target into a one-DOF one, and every metric computed
        afterwards would be a real number describing the wrong problem.
        """
        from libemg.utils import get_windows
        values = getattr(handler, key, None)
        if values is None or len(values) == 0 or index >= len(values):
            return None
        column = np.asarray(values[index])
        if column.ndim == 1:
            column = column.reshape(-1, 1)
        enframed = get_windows(column, window_size, increment)
        if not enframed.shape[0]:
            return None
        # The label of a window is the label at its end, which is the
        # convention the rest of LibEMG uses -- the same as the
        # ``{'labels': lambda x: x[-1]}`` a script passes to parse_windows. The
        # dtype is left as it was read: a class index arrives as an integer
        # already, and a regression target read from a file is a real number
        # that rounding would destroy.
        ending = enframed[:, :, -1]
        return ending[:, 0] if ending.shape[1] == 1 else ending


# ======================================================================
# Compilation
# ======================================================================
def compile_pipeline(document, log=None, probe_rows=400, buffer_rows=2000):
    """Turn a document into a runnable pipeline.

    Parameters
    ----------
    document: PipelineDocument
        What to compile. It is validated first, so a pipeline that cannot run
        fails here rather than half way through starting.
    log: EventLog or None (optional), default=None
        Passed to the reactive graph, for an online pipeline.
    probe_rows: int (optional), default=400
        Rows retained in each probe's buffer.
    buffer_rows: int (optional), default=2000
        Rows in each intermediate stream buffer.

    Returns
    ----------
    OnlinePipeline or OfflinePipeline

    Raises
    ----------
    CompileError
        If the document does not validate, or names something unbuildable.
    """
    problems = document.validate()
    if problems:
        raise CompileError("This pipeline cannot run yet:\n"
                           + "\n".join(f"  - {p}" for p in problems))
    mode = document.mode()
    if mode == "offline":
        return _compile_offline(document)
    return _compile_online(document, log=log, probe_rows=probe_rows,
                           buffer_rows=buffer_rows)


def _window_feeding(document, node_id):
    """The window block immediately upstream of a node, if any.

    A window is fused into whatever it feeds, so a stage asks what window
    configures it rather than reading one as data.
    """
    for link in document.links_into(node_id):
        spec = document.registry.get(document.nodes[link.from_node].spec_id)
        if spec is not None and spec.category == WINDOW:
            return document.nodes[link.from_node]
    return None


def _upstream_stream(document, node_id):
    """Walk back past any window block to the item actually carrying samples."""
    for link in document.links_into(node_id):
        upstream = document.nodes[link.from_node]
        spec = document.registry.get(upstream.spec_id)
        if spec is not None and spec.category == WINDOW:
            return _upstream_stream(document, upstream.id)
        return tag_for(link.from_node, link.from_port)
    return None


def _filter_dictionary(params):
    """Build the dictionary Filter.install_filters expects."""
    name = params.get("name", "bandpass")
    cutoff = _as_floats(params.get("cutoff"), [20.0, 450.0])
    dictionary = {"name": name}
    if name == "standardize":
        return dictionary
    if name == "notch":
        dictionary["cutoff"] = cutoff[0]
        dictionary["bandwidth"] = float(params.get("bandwidth", 3.0))
        return dictionary
    dictionary["cutoff"] = cutoff if len(cutoff) > 1 else cutoff[0]
    dictionary["order"] = int(params.get("order", 4))
    return dictionary


def _compile_online(document, log, probe_rows, buffer_rows):
    from multiprocessing import Lock

    items, streamer_specs, probe_items = [], [], {}
    source_tags, channels_of = {}, {}

    def declare(tag, shape, dtype=np.double):
        lock = Lock()
        items.append([tag, shape, dtype, lock])
        items.append([tag + "_count", (1, 1), np.int32, lock])

    # --- sources -------------------------------------------------------
    for node_id in document.sources():
        node = document.nodes[node_id]
        spec = document.registry[node.spec_id]
        function_name = node.spec_id.split(".", 1)[1]
        tag = tag_for(node_id, "emg")
        # The streamer writes straight into the item its output port names, so
        # nothing has to copy device samples before the first stage sees them.
        channels = int(node.params.get("num_channels") or 8)
        stream_items = [[tag, (buffer_rows, channels), np.double],
                        [tag + "_count", (1, 1), np.int32]]
        from libemg.shared_memory_manager import assign_shared_memory_locks
        assign_shared_memory_locks(stream_items)
        items.extend(stream_items)
        kwargs = {}
        for param in spec.params:
            value = node.params.get(param.name)
            if value in (None, ""):
                continue
            kwargs[param.name] = value
        streamer_specs.append({"function": function_name, "items": stream_items,
                               "kwargs": kwargs})
        source_tags[node_id] = tag
        channels_of[tag] = channels

    graph = ReactiveGraph(items, log=log, notifier_pool=default_notifier_pool())

    # --- stages, in dependency order ----------------------------------
    for node_id in document.order():
        node = document.nodes[node_id]
        spec = document.registry.get(node.spec_id)
        if spec is None or spec.category in (SOURCE, WINDOW):
            continue
        upstream = _upstream_stream(document, node_id)
        if upstream is None:
            raise CompileError(f"'{spec.title}' ({node_id}) has no input to read.")
        window = _window_feeding(document, node_id)
        window_size = int(window.params["window_size"]) if window else 200
        increment = int(window.params["window_increment"]) if window else 1
        channels = channels_of.get(upstream, 8)

        if spec.id == "transform.filter":
            if node.params.get("name") == "standardize":
                # Standardizing needs a mean and a deviation measured from
                # data, and a live stream has none to measure from before it
                # starts. Refusing here, by name, beats letting it compile and
                # then fail inside a spawned executor where the reason is much
                # harder to see.
                raise CompileError(
                    f"'{spec.title}' ({node_id}) is set to standardize, which needs "
                    "statistics measured from data and so only works on a stored-data "
                    "pipeline. For a live stream, choose another filter type, or "
                    "standardize the features instead by installing a scaler on the "
                    "model you point this pipeline at.")
            target = tag_for(node_id, "output")
            declare(target, (buffer_rows, channels))
            channels_of[target] = channels
            config = {"sampling_rate": int(node.params.get("sampling_rate", 1000)),
                      "dictionary": _filter_dictionary(node.params)}
            # Filtered edges settle over a margin, so more history is read than
            # is republished. Four increments is enough for the orders LibEMG's
            # filters use without holding up the stage.
            margin = max(increment * 4, 64)
            graph.add(_FilterHook(node_id, upstream, target, config,
                                  (buffer_rows, channels), margin, increment),
                      executor=node_id)

        elif spec.id == "transform.channel_mask":
            selected = _as_ints(node.params.get("channels"), None)
            selected = selected if selected else list(range(channels))
            target = tag_for(node_id, "output")
            declare(target, (buffer_rows, len(selected)))
            channels_of[target] = len(selected)
            graph.add(_ChannelMaskHook(node_id, upstream, target, selected,
                                       (buffer_rows, len(selected)), increment),
                      executor=node_id)

        elif spec.category == FEATURES:
            features = _resolve_features(node.params)
            width = len(features) * channels
            target = tag_for(node_id, "output")
            declare(target, (probe_rows, width))
            graph.add(_FeatureHook(node_id, upstream, target, features,
                                   window_size, increment, width, rows=probe_rows),
                      executor=node_id)
            channels_of[target] = width

        elif spec.category == MODEL:
            path = node.params.get("model_path") or ""
            if not path or not os.path.exists(path):
                raise CompileError(
                    f"'{spec.title}' ({node_id}) needs a fitted model to run live. "
                    "Set its Fitted Model parameter to a saved predictor.")
            classifies = spec.id == "model.classifier" or (
                spec.id == "model.deep"
                and str(node.params.get("task", "Classification")) == "Classification")
            width = 2 if classifies else 8
            target = tag_for(node_id, "output")
            declare(target, (probe_rows, width))
            channels_of[target] = width
            # What the model observes, and how often, comes from whatever feeds
            # it. Features arrive one row at a time; raw windows arrive on the
            # window's increment.
            feeds_features = _feeds_category(document, node_id, FEATURES)
            hook_type = _ClassifierHook if classifies else _RegressorHook
            if feeds_features:
                hook = hook_type(node_id, upstream, target, path, width,
                                 rows=probe_rows, criterion=OnSamples(1), mode=LATEST)
            else:
                hook = hook_type(node_id, upstream, target, path, width,
                                 rows=probe_rows, criterion=OnSamples(increment),
                                 mode=WINDOW_MODE, size=window_size)
            graph.add(hook, executor=node_id)

        elif spec.category == SINK:
            kind = spec.id.split(".", 1)[1]
            if kind == "metrics":
                continue
            graph.add(_SinkHook(node_id, upstream, kind, dict(node.params)),
                      executor=node_id)

    # --- probes --------------------------------------------------------
    for probe in document.probes:
        source = tag_for(probe.node, probe.port)
        if not any(item[0] == source for item in items):
            continue
        width = channels_of.get(source, 8)
        target = f"probe_{probe.node}_{probe.port}"
        render = document.probe_render(probe)
        rows = probe_rows if render == "timeseries" else 1
        declare(target, (rows, width))
        graph.add(_ProbeHook(f"probe_{probe.node}_{probe.port}", source, target,
                             (rows, width), probe.hz,
                             WINDOW_MODE if rows > 1 else LATEST,
                             rows if rows > 1 else None),
                  executor=f"probe_{probe.node}")
        probe_items[(probe.node, probe.port)] = {"tag": target, "render": render,
                                                 "rows": rows, "width": width}

    return OnlinePipeline(document, graph, items, streamer_specs, probe_items,
                          source_tags, log=log)


def _feeds_category(document, node_id, category):
    """Whether the block immediately feeding this one is of a category."""
    for link in document.links_into(node_id):
        upstream = document.nodes[link.from_node]
        spec = document.registry.get(upstream.spec_id)
        if spec is None:
            continue
        if spec.category == WINDOW:
            return _feeds_category(document, upstream.id, category)
        return spec.category == category
    return False


def _resolve_features(params):
    """The feature list a features block asks for."""
    from libemg.feature_extractor import FeatureExtractor
    group = params.get("feature_group", "(custom)")
    if group and group != "(custom)":
        groups = FeatureExtractor().get_feature_groups()
        if group in groups:
            return list(groups[group])
    chosen = params.get("features") or []
    return list(chosen) or ["MAV"]


def _compile_offline(document):
    """Flatten a stored-data pipeline into one sequential pass."""
    plan = {"folder": "", "regex_filters": [], "label_files": [],
            "package_by": "same folder", "delimiter": ",",
            "label_key": "classes", "sampling_rate": 1000,
            "filters": [], "channels": None, "window_size": 200,
            "window_increment": 50, "features": [], "model_path": "",
            "model_name": "", "metrics": [], "null_label": None,
            "feature_view": None, "model_input": "features", "deep": None,
            "prediction_view": None}

    for node_id in document.order():
        node = document.nodes[node_id]
        spec = document.registry.get(node.spec_id)
        if spec is None:
            continue
        if spec.id == "source.offline":
            plan["folder"] = node.params.get("folder", "")
            plan["delimiter"] = node.params.get("delimiter", ",")
            plan["label_key"] = node.params.get("label_key", "classes")
            plan["sampling_rate"] = int(node.params.get("sampling_rate", 1000))
            plan["regex_filters"] = _parse_regex_filters(
                node.params.get("regex_filters", ""))
            plan["label_files"] = _parse_regex_filters(
                node.params.get("label_files", ""))
            plan["package_by"] = node.params.get("package_by", "same folder")
            if not plan["folder"]:
                raise CompileError("The Stored Data block needs a folder to read from.")
            if not plan["regex_filters"]:
                raise CompileError(
                    "The Stored Data block needs at least one regex filter, so it "
                    "knows which files belong to what.")
        elif spec.id == "transform.filter":
            plan["filters"].append(_filter_dictionary(node.params))
            plan["sampling_rate"] = int(node.params.get("sampling_rate",
                                                        plan["sampling_rate"]))
        elif spec.id == "transform.channel_mask":
            plan["channels"] = _as_ints(node.params.get("channels"), None)
        elif spec.category == WINDOW:
            plan["window_size"] = int(node.params["window_size"])
            plan["window_increment"] = int(node.params["window_increment"])
        elif spec.category == FEATURES:
            plan["features"] = _resolve_features(node.params)
        elif spec.category == MODEL:
            plan["model_path"] = node.params.get("model_path", "")
            # Carried so the pipeline can fit this model, not only run one that
            # was fitted elsewhere.
            plan["model_name"] = node.params.get("model", "")
            # A model reads features or raw windows, whichever is connected. A
            # pipeline can extract features for a Feature Viewer while a CNN
            # reads the windows, so it is the link that decides, not whether a
            # Features block exists.
            plan["model_input"] = "windows" if any(
                link.to_port == "windows" for link in document.links_into(node_id))                 else "features"
            if spec.id == "model.deep":
                plan["model_name"] = "deep"
                plan["deep"] = dict(node.params)
        elif spec.id == "sink.metrics":
            plan["metrics"] = list(node.params.get("metrics") or [])
            null_label = node.params.get("null_label", -1)
            plan["null_label"] = None if null_label in (-1, None) else int(null_label)
        elif spec.id == "sink.feature_viewer":
            plan["feature_view"] = dict(node.params)
            # Its labels input is optional: without a link the points are
            # drawn in one colour even if the recording carries labels.
            plan["feature_view"]["labels"] = any(
                link.to_port == "labels" for link in document.links_into(node_id))
            plan["feature_view"]["model"] = any(
                link.to_port == "model" for link in document.links_into(node_id))
        elif spec.id == "sink.prediction_viewer":
            plan["prediction_view"] = dict(node.params)

    return OfflinePipeline(document, plan)


def _file_metadata(handler, plan, index):
    """What a file's name says about it: each named filename filter, and its value.

    The handler keeps a filter's match as an index into the filter's values,
    repeated on every sample; this turns it back into the value itself.
    """
    meta = {}
    for spec in plan["regex_filters"]:
        name = spec.get("description") or ""
        column = getattr(handler, name, None) if name else None
        if column is None or index >= len(column):
            continue
        position = int(np.ravel(column[index])[0])
        values = list(spec.get("values") or [])
        meta[name] = values[position] if 0 <= position < len(values) else position
    return meta


def _metadata_fetchers(plan):
    """Build the packagers that read labels out of files of their own.

    A filename filter can only say what a whole recording was -- the class it
    belongs to. A regression target is a value per sample, so it lives in a
    file beside the recording, and reading it needs a
    :class:`~libemg.data_handler.FilePackager` rather than another
    :class:`~libemg.data_handler.RegexFilter`. Without this a stored-data
    pipeline could score a classifier and not a regressor, which is exactly the
    gap between the two offline examples in the documentation.

    Returns
    ----------
    list or None
        Fetchers to hand to ``OfflineDataHandler.get_data``, or None when the
        block names no label files.
    """
    from pathlib import Path

    from libemg.data_handler import FilePackager, RegexFilter

    specs = plan.get("label_files") or []
    if not specs:
        return None
    if plan.get("package_by") == "same stem":
        def belongs(metadata_file, data_file):
            return Path(metadata_file).stem == Path(data_file).stem
    else:
        # The LibEMG convention: one labels file per recording folder.
        def belongs(metadata_file, data_file):
            return (Path(metadata_file).parent.absolute()
                    == Path(data_file).parent.absolute())
    return [FilePackager(RegexFilter(**spec), belongs) for spec in specs]


def _parse_regex_filters(text):
    """Parse the regex-filter parameter into RegexFilter keyword arguments.

    One filter per line, written as ``left|right|values|description``, which is
    the smallest thing that carries everything a RegexFilter needs and still
    reads back as what the user typed.
    """
    filters = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            raise CompileError(
                f"Cannot read the filter '{line}'. Each line should be "
                "left|right|values|description, for example _C_|_EMG.csv|0,1,2|classes.")
        filters.append({"left_bound": parts[0], "right_bound": parts[1],
                        "values": [v.strip() for v in parts[2].split(",") if v.strip()],
                        "description": parts[3]})
    return filters
