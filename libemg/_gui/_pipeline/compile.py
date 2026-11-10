"""Turn a pipeline document into something that runs.

Online, the document becomes a reactive graph: a streamer writes into shared
memory and each stage is woken by the one before it. Offline, the same document
is evaluated block by block over whole recordings: each block computes its
outputs once from its inputs, so a Split block's branches are separate sets of
recordings and a model can be trained on one and run on another.

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


class _Stopped(Exception):
    """A stored-data run was asked to stop between two blocks."""


class OfflinePipeline:
    """A stored-data pipeline, evaluated block by block over whole recordings.

    Every block computes its outputs once, from its inputs, and keeps them;
    a block is computed only when something downstream asks for it. So a
    Split block's two branches are two separate sets of recordings, each
    windowed, featured and scored on its own, and a model can be trained on
    one branch and run on the other in the same document.

    What flows along each link is described in
    :mod:`~libemg._gui._pipeline.offline_data`. Labels are never read from the
    recordings directly: they are a Window block's Metadata output, windowed
    exactly as its signal was, so every label is the label of one window.

    Parameters
    ----------
    document: PipelineDocument
    models: dict or None
        Fitted predictors by model block id -- what the editor's Train and
        Load buttons produce. A block not listed here runs the model saved at
        its Fitted Model path.
    """

    def __init__(self, document, models=None):
        self.document = document
        self.registry = document.registry
        #: Fitted predictors by model block, used before any Fitted Model file.
        self.models = dict(models or {})
        #: FitResults by model block, from :meth:`train`.
        self.fits = {}
        self.progress = 0.0
        #: The metrics of the last run: those of its Offline Metrics block.
        self.results = {}
        #: Metrics by Offline Metrics block, when there are several.
        self.results_by_node = {}
        #: True when the last run or training was cut short.
        self.stopped = False
        self._stop = False
        #: Every EpochResult of the last deep training, as they completed.
        self.epochs = []
        #: The last deep training run as a TrainingOutcome, once it has finished.
        self.outcome = None
        #: For a Feature Viewer block: what it draws.
        self.feature_view = None
        #: For a Prediction Viewer block: every recording and its predictions.
        self.prediction_view = None
        self._values = {}
        self._evaluated = set()
        self._on_progress = None
        self._needed = 1

    def request_stop(self):
        """Ask a run or a training in progress to finish early."""
        self._stop = True

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def value(self, node_id, port):
        """What an output port carries, computing its block if it has not been."""
        if node_id not in self._evaluated:
            self._evaluate(node_id)
        return self._values.get((node_id, port))

    def input(self, node_id, port):
        """What arrives at an input port, or None if nothing is connected."""
        links = self.document.links_into(node_id, port)
        if not links:
            return None
        return self.value(links[0].from_node, links[0].from_port)

    def connected(self, node_id, port):
        return bool(self.document.links_into(node_id, port))

    def predictor(self, node_id):
        """A model block's fitted predictor: trained or loaded here, or from its file."""
        if node_id in self.models:
            return self.models[node_id]
        path = self.document.nodes[node_id].params.get("model_path", "")
        if path and os.path.exists(path):
            self.models[node_id] = load_model(path)
            return self.models[node_id]
        return None

    def _evaluate(self, node_id):
        if self._stop:
            raise _Stopped()
        node = self.document.nodes[node_id]
        spec = self.registry.get(node.spec_id)
        compute = _OFFLINE_BLOCKS.get(spec.id) or _OFFLINE_BLOCKS.get(spec.category)
        outputs = compute(self, node_id, node.params, spec) if compute else {}
        for port in spec.outputs:
            self._values[(node_id, port.name)] = (outputs or {}).get(port.name)
        self._evaluated.add(node_id)
        self.progress = min(1.0, len(self._evaluated) / max(1, self._needed))
        if self._on_progress is not None and not self.stopped:
            self._on_progress(self.progress)

    def _upstream(self, node_ids, ports=None):
        """Every block ``node_ids`` depend on, themselves included."""
        seen, stack = set(), []
        for node_id in node_ids:
            links = self.document.links_into(node_id)
            if ports is not None:
                links = [l for l in links if l.to_port in ports]
                seen.add(node_id)
            else:
                stack.append(node_id)
            stack.extend(l.from_node for l in links)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(l.from_node for l in self.document.links_into(current))
        return seen

    def _begin(self, on_progress, needed):
        self._values, self._evaluated = {}, set()
        self.stopped = False        # a stop asked for before the start still stands
        self.progress = 0.0
        self._on_progress = on_progress
        self._needed = max(1, len(needed))

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def run(self, on_progress=None):
        """Evaluate every output block over the recordings, and score what they score.

        Parameters
        ----------
        on_progress: callable or None (optional), default=None
            Called as ``on_progress(fraction)`` as blocks finish.

        Returns
        ----------
        dict
            Metric name to value, from the Offline Metrics block. Empty if
            there is none.
        """
        sinks = [n for n in self.document.order()
                 if (self.registry.get(self.document.nodes[n].spec_id) or _NO_SPEC).category
                 == SINK]
        self._begin(on_progress, self._upstream(sinks))
        self.results, self.results_by_node = {}, {}
        self.feature_view = self.prediction_view = None
        try:
            for node_id in sinks:
                self.value(node_id, None)
        except _Stopped:
            # A run that was asked to stop must not claim to have finished: the
            # metrics below would cover only part of the recordings.
            self.stopped = True
            return self.results
        self.progress = 1.0
        if on_progress is not None:
            on_progress(1.0)
        return self.results

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def trainable(self):
        """Model blocks with training inputs connected, in pipeline order."""
        return [n for n in self.document.order()
                if (self.registry.get(self.document.nodes[n].spec_id) or _NO_SPEC).category
                == MODEL and self.connected(n, "train_labels")]

    def train(self, on_progress=None, model_path=None, on_epoch=None, node_id=None, save=True):
        """Fit model blocks on their training inputs.

        Parameters
        ----------
        on_progress: callable or None (optional), default=None
            Called as ``on_progress(fraction)`` while the training data is read.
        model_path: str or None (optional), default=None
            Where to save. Defaults to the block's own Fitted Model path.
        on_epoch: callable or None (optional), default=None
            For a Deep Model block: called with each EpochResult as an epoch
            completes. Every result is also kept in ``self.epochs``.
        node_id: str, list or None (optional), default=None
            The block, or blocks, to train. Every block with training inputs,
            by default.
        save: bool (optional), default=True
            Write each model to its Fitted Model path. The editor trains with
            this off and saves when Save is pressed; either way the fitted
            model is kept in ``self.models`` and used by :meth:`run`.

        Returns
        ----------
        dict
            For the (last) block trained: ``model_path``, ``windows``,
            ``features``, ``recordings``, ``classes`` or ``dofs``, the
            ``statistics``, and for a deep model ``epochs``, ``best_epoch``,
            ``best_score``, ``score_name``, ``stopped`` and ``outcome_path``.

        Raises
        ----------
        CompileError
            If no block has training inputs, if one is missing, if the model
            or the save path is unusable -- checked before any data is read --
            or if the data yields no windows or no such label field.
        """
        from libemg.emg_predictor import CLASSIFIER_MODELS, REGRESSOR_MODELS
        from libemg._gui._pipeline.fitting import FitError, fit_block

        nodes = ([node_id] if isinstance(node_id, str) else list(node_id)) if node_id             else self.trainable()
        if not nodes:
            raise CompileError(
                "No model block has training inputs. Connect a Window's Metadata to a model's "
                "Train Labels, and the features (or windows) it should learn from to its "
                "Train Features.")
        for n in nodes:
            node = self.document.nodes[n]
            spec = self.registry[node.spec_id]
            title = f"'{spec.title}' ({n})"
            if not self.connected(n, "train_labels") or not (
                    self.connected(n, "train_input") or self.connected(n, "train_windows")):
                raise CompileError(f"{title} needs both Train Labels and Train Features "
                                   "(or Train Windows) connected to be trained.")
            if save and not (model_path or node.params.get("model_path")):
                raise CompileError(
                    f"{title} has no Fitted Model path, so there is nowhere to save what "
                    "training produces. Set that parameter first.")
            if spec.id != "model.deep":
                # Checked before the recordings are read: a misspelled model would
                # otherwise be reported only once the whole folder had been walked.
                name = node.params.get("model", "")
                known = CLASSIFIER_MODELS if spec.id == "model.classifier" else REGRESSOR_MODELS
                if name not in known:
                    raise CompileError(f"'{name}' is not a model this version can fit. "
                                       f"Choose one of {sorted(known)}.")

        self._begin(on_progress, self._upstream(nodes, ports=("train_input", "train_windows",
                                                              "train_labels")))
        self.epochs = []

        def record(result):
            self.epochs.append(result)
            if on_epoch is not None:
                on_epoch(result)

        summary = {}
        for n in nodes:
            node = self.document.nodes[n]
            params = node.params
            try:
                inputs = self.input(n, "train_input")
                if inputs is None:
                    inputs = self.input(n, "train_windows")
                labels = self.input(n, "train_labels")
            except _Stopped:
                # A model fitted on part of the data would be saved under the same
                # name as one fitted on all of it. Stopping leaves the file alone.
                self.stopped = True
                raise CompileError("Training was stopped part way, so nothing was saved. "
                                   "The model file is unchanged.")
            if inputs is None or not len(inputs):
                raise CompileError(
                    "That recording produced no windows. Check the folder, the regex filters, "
                    "and that the window is not longer than the recordings.")
            try:
                fitted = fit_block(node.spec_id, params, inputs, labels,
                                   params.get("label_field") or "classes",
                                   on_epoch=record, should_stop=lambda: self._stop)
            except FitError as error:
                raise CompileError(str(error)) from error
            self.stopped = fitted.stopped
            self.models[n], self.fits[n] = fitted.predictor, fitted
            if fitted.outcome is not None:
                fitted.outcome.data.update(self._describe_data(n, inputs))
            self.outcome = fitted.outcome
            destination = model_path or params.get("model_path", "")
            outcome_path = save_model(fitted.predictor, destination, fitted.outcome) \
                if save else None
            summary = _summary(fitted, inputs, destination if save else "", outcome_path)
        self.progress = 1.0
        if on_progress is not None:
            on_progress(1.0)
        return summary

    def _describe_data(self, node_id, inputs):
        """What a model was trained on, for its training record: folder, features, window."""
        upstream = self._upstream([node_id], ports=TRAIN_INPUTS)
        folders = [self.document.nodes[u].params.get("folder", "") for u in sorted(upstream)
                   if self.document.nodes[u].spec_id == "source.offline"]
        features = []
        for link in self.document.links_into(node_id, "train_input"):
            features = _resolve_features(self.document.nodes[link.from_node].params)
        windows = getattr(inputs, "windows", inputs)
        windows = windows if hasattr(windows, "window_size") else inputs
        return {"folder": folders[0] if len(folders) == 1 else folders, "features": features,
                "window_size": getattr(windows, "window_size", None),
                "window_increment": getattr(windows, "window_increment", None)}


#: A model block's training inputs.
TRAIN_INPUTS = ("train_input", "train_windows", "train_labels")


def _summary(fitted, inputs, model_path, outcome_path):
    """What :meth:`OfflinePipeline.train` reports about one block."""
    statistics = fitted.statistics
    x = inputs.matrix if hasattr(inputs, "matrix") else inputs.windows
    out = {"model_path": model_path, "windows": statistics["windows"],
           "features": int(np.prod(x.shape[1:])) if len(x) else 0,
           "recordings": statistics["recordings"],
           "classes": statistics["classes"] if statistics["task"] == "classification" else None,
           "dofs": statistics["dofs"], "statistics": statistics,
           "stopped": fitted.stopped, "outcome_path": outcome_path}
    if fitted.epochs:
        best = min(fitted.epochs, key=lambda r: r.validation_loss
                   if r.validation_loss is not None else r.train_loss)
        out.update(epochs=len(fitted.epochs), best_epoch=best.epoch, best_score=best.score,
                   score_name=best.score_name)
    return out


def save_model(predictor, path, outcome=None):
    """Write a fitted predictor, and a deep model's training record beside it.

    Returns
    ----------
    str or None
        Where the training record went, for a deep model.
    """
    import pickle
    if not path:
        raise CompileError("There is no Fitted Model path to save to.")
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(predictor, handle)
    if outcome is None:
        return None
    # The run, kept beside the model as JSON: readable without the model or
    # PyTorch, and what the Training History window draws in a later session.
    from libemg.deep import OUTCOME_SUFFIX
    outcome.model_path = str(path)
    outcome_path = os.path.splitext(path)[0] + OUTCOME_SUFFIX
    outcome.save(outcome_path)
    return outcome_path


def load_model(path):
    """A predictor saved by :func:`save_model`, or by a script with pickle."""
    import pickle
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ----------------------------------------------------------------------
# what each block does to stored data
# ----------------------------------------------------------------------
def _stored_data(pipeline, node_id, params, spec):
    from libemg._gui._pipeline.offline_data import Recordings
    _check_stored_data(params)
    fetchers = _metadata_fetchers({
        "label_files": _parse_regex_filters(params.get("label_files", "")),
        "package_by": params.get("package_by", "same folder")})
    recordings = Recordings.read(params.get("folder", ""),
                                 _parse_regex_filters(params.get("regex_filters", "")),
                                 fetchers, params.get("delimiter", ","),
                                 int(params.get("sampling_rate", 1000)))
    return {"emg": recordings}


def _filter_block(pipeline, node_id, params, spec):
    from libemg.filtering import Filter
    recordings = pipeline.input(node_id, "input")
    if recordings is None:
        return {}
    dictionary = _filter_dictionary(params)
    conditioner = Filter(sampling_frequency=int(params.get("sampling_rate", 1000)))
    if dictionary.get("name") == "standardize":
        # Standardizing divides by a deviation that has to be measured from
        # data. Offline the recordings are all here, so they supply it.
        dictionary = dict(dictionary, data=recordings)
    conditioner.install_filters(dictionary)
    return {"output": recordings.with_data(
        [conditioner.filter(np.asarray(block, dtype=float)) for block in recordings.data])}


def _channel_mask(pipeline, node_id, params, spec):
    recordings = pipeline.input(node_id, "input")
    channels = _as_ints(params.get("channels"), None)
    if recordings is None or not channels:
        return {"output": recordings}
    return {"output": recordings.with_data([np.asarray(b)[:, channels]
                                            for b in recordings.data])}


def _split(pipeline, node_id, params, spec):
    from libemg._gui._pipeline.offline_data import parse_values
    recordings = pipeline.input(node_id, "input")
    if recordings is None:
        return {}
    field = str(params.get("field", "")).strip()
    wanted = parse_values(params.get("values", ""))
    if not field or not wanted:
        raise CompileError(f"'{spec.title}' ({node_id}) needs a field and the values to pick "
                           "out, for example reps and 0,1,2 (or 0-2).")
    try:
        matching, rest = recordings.split(field, wanted)
    except KeyError:
        raise CompileError(f"'{spec.title}' ({node_id}) splits on '{field}', but the recordings "
                           f"carry: {', '.join(recordings.fields()) or 'no fields'}. A field is "
                           "the description of a regex filter or a label file.")
    return {"matching": matching, "rest": rest}


def _window(pipeline, node_id, params, spec):
    from libemg._gui._pipeline.offline_data import window_recordings
    recordings = pipeline.input(node_id, "input")
    if recordings is None:
        return {}
    windows = window_recordings(recordings, int(params["window_size"]),
                                int(params["window_increment"]),
                                params.get("metadata_rule", "last sample"))
    return {"output": windows, "metadata": windows.metadata}


def _features(pipeline, node_id, params, spec):
    from libemg.feature_extractor import FeatureExtractor
    from libemg._gui._pipeline.feature_viewer import column_names
    from libemg._gui._pipeline.offline_data import FeatureSet
    windows = pipeline.input(node_id, "input")
    if windows is None:
        return {}
    feature_list = _resolve_features(params)
    if not len(windows):
        return {"output": FeatureSet(np.empty((0, 0)), [], windows)}
    # Extracted by name and stacked in the order asked for, which is exactly
    # what array=True builds, so that each column can be named.
    by_name = FeatureExtractor().extract_features(feature_list, windows.windows)
    matrix = np.hstack([by_name[name] for name in feature_list])
    names = column_names(feature_list, by_name, windows.windows.shape[1])
    return {"output": FeatureSet(matrix, names, windows)}


def _model(pipeline, node_id, params, spec):
    from libemg.emg_predictor import EMGClassifier
    from libemg._gui._pipeline.offline_data import (FeatureSet, Predictions, by_recording,
                                                    predict_recordings)
    source = pipeline.input(node_id, "input")
    if source is None:
        source = pipeline.input(node_id, "windows")
    if source is None:
        return {}
    predictor = pipeline.predictor(node_id)
    if predictor is None:
        path = params.get("model_path", "")
        raise CompileError(
            f"'{spec.title}' ({node_id}) has no fitted model: "
            + (f"its Fitted Model file {path} does not exist yet. " if path else "")
            + "Press Train on it, or Load a fitted model.")
    if not len(source):
        raise CompileError(
            "That recording produced no windows. Check the folder, the regex filters, and "
            "that the window is not longer than the recordings.")
    x = source.matrix if isinstance(source, FeatureSet) else source.windows
    if spec.id != "model.deep" and x.ndim > 2:
        x = x.reshape(len(x), -1)
    classification = isinstance(predictor, EMGClassifier)
    results = predict_recordings(predictor, by_recording(x, source.lengths), classification)
    outputs = np.vstack([o for o, _ in results])
    predicted = np.concatenate([p for _, p in results])
    model = getattr(predictor, "model", predictor)
    classes = getattr(model, "classes_", None)
    windows = source.windows if isinstance(source, FeatureSet) else source
    return {"output": Predictions(outputs, predicted, classification,
                                  [] if classes is None else np.asarray(classes).tolist(),
                                  windows, predictor)}


def _metrics(pipeline, node_id, params, spec):
    from libemg.offline_metrics import OfflineMetrics
    from libemg._gui._pipeline.fitting import FitError, check_aligned
    predictions = pipeline.input(node_id, "input")
    metadata = pipeline.input(node_id, "labels")
    if predictions is None or metadata is None:
        return {}
    truth = _field(metadata, params.get("label_field") or "classes", spec, node_id)
    try:
        check_aligned(predictions, metadata, "predictions", "labels")
    except FitError as error:
        raise CompileError(str(error)) from error
    predicted = np.asarray(predictions.predicted)
    if predictions.classification:
        # Classification metrics compare indices, and an index that came back as
        # a float would make every comparison and every confusion matrix wrong.
        predicted, truth = predicted.ravel().astype(int), np.asarray(truth).astype(int)
    else:
        predicted = predicted.reshape(len(predicted), -1)
        truth = np.asarray(truth, dtype=float).reshape(len(truth), -1)
    null_label = params.get("null_label", -1)
    results = OfflineMetrics().extract_offline_metrics(
        list(params.get("metrics") or []), truth, predicted,
        null_label=None if null_label in (-1, None) else int(null_label))
    pipeline.results_by_node[node_id] = results
    pipeline.results = results
    return {"output": results}


def _feature_viewer(pipeline, node_id, params, spec):
    from libemg._gui._pipeline.feature_viewer import model_spaces
    from libemg._gui._pipeline.fitting import FitError, check_aligned
    features = pipeline.input(node_id, "features")
    if features is None:
        return {}
    metadata = pipeline.input(node_id, "labels")
    if metadata is not None:
        try:
            check_aligned(features, metadata, "features", "labels")
        except FitError as error:
            raise CompileError(str(error)) from error
    field = params.get("label_field") or "classes"
    fields = dict(metadata.fields) if metadata is not None else {}
    # A filename field is stored as a position in its filter's values; the
    # legend says what the filename said, as 'sets 3', not the position.
    values = metadata.recordings.field_values if metadata is not None else {}
    field_names = {name: {i: f"{name} {v}" for i, v in enumerate(values[name])}
                   for name in fields if name in values}
    view = {"features": features.matrix, "names": list(features.names),
            "labels": fields.get(field) if fields else None, "label_field": field,
            "fields": fields, "field_names": field_names, "params": dict(params),
            "spaces": [], "model_note": ""}
    if metadata is not None and field not in fields:
        view["model_note"] = (f"There is no field called '{field}' to colour by; the "
                              f"recordings carry {', '.join(fields) or 'none'}.")
    links = pipeline.document.links_into(node_id, "model")
    if links:
        # A connected model adds the spaces it works in: its latent space, its
        # decision scores, its outputs. Drawing them needs the model itself.
        predictor = pipeline.predictor(links[0].from_node)
        if predictor is None:
            view["model_note"] = ("The connected model has not been trained yet, so only the "
                                  "features are shown. Train it, then Start again.")
        else:
            view["spaces"] = model_spaces(predictor, features.matrix)
    pipeline.feature_view = view
    return {}


def _prediction_viewer(pipeline, node_id, params, spec):
    from libemg._gui._pipeline.fitting import FitError, STATISTICS, check_aligned
    predictions = pipeline.input(node_id, "model")
    if predictions is None:
        return {}
    metadata = pipeline.input(node_id, "labels")
    if metadata is not None:
        try:
            check_aligned(predictions, metadata, "predictions", "labels")
        except FitError as error:
            raise CompileError(str(error)) from error
    windows = predictions.windows
    field = params.get("label_field") or "classes"
    truth = None
    if metadata is not None and field in metadata.fields:
        truth = np.asarray(metadata.fields[field])
        if predictions.classification:
            truth = truth.reshape(len(truth), -1)[:, 0].astype(int)
    paths = windows.file_paths()
    metas = [windows.metadata.file_meta(p) for p in range(len(windows.lengths))]
    splits, note = _splits(params, predictions.predictor, paths, metas)
    recordings = []
    edges = np.cumsum([0] + list(windows.lengths))
    for position, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        recordings.append({
            "index": position, "meta": metas[position], "file": paths[position],
            "signal": np.asarray(windows.recordings.data[windows.files[position]]),
            "truth": None if truth is None else truth[a:b],
            "outputs": predictions.outputs[a:b], "predicted": predictions.predicted[a:b],
            "split": splits[position]})
    pipeline.prediction_view = {
        "task": "classification" if predictions.classification else "regression",
        "classes": list(predictions.classes),
        "sampling_rate": windows.sampling_rate, "window_size": windows.window_size,
        "window_increment": windows.window_increment, "params": dict(params),
        "note": note, "recordings": recordings}
    return {}


def _splits(params, predictor, paths, metas):
    """Which recordings a model trained on and which it never saw, and how that is known."""
    from libemg._gui._pipeline.fitting import STATISTICS
    count = len(paths)
    field = (params.get("held_out_field") or "").strip()
    values = {v.strip() for v in str(params.get("held_out_values") or "").split(",")
              if v.strip()}
    if field and values:
        return (["held out" if str(m.get(field)) in values else "training" for m in metas],
                f"Held out: {field} {', '.join(sorted(values))}, as the block sets it.")
    statistics = getattr(predictor, STATISTICS, None)
    if statistics and statistics.get("training_files"):
        trained = set(statistics["training_files"])
        return (["training" if p in trained else "held out" for p in paths],
                "Training: the recordings the model was fitted on. Held out: every other one, "
                "including those it was validated on.")
    trained_on = getattr(getattr(predictor, "model", predictor), "trained_on", None) or {}
    held = trained_on.get("validation_indices") or []
    if held and trained_on.get("recordings") == count:
        return (["held out" if i in set(held) else "training" for i in range(count)],
                "Held out: the recordings the model was validated on while it trained.")
    return [""] * count, ""


def _field(metadata, name, spec, node_id):
    try:
        return metadata.field(name)
    except KeyError as error:
        raise CompileError(f"'{spec.title}' ({node_id}): {error.args[0]} Set its Label Field "
                           "to one of them.") from error


_OFFLINE_BLOCKS = {
    "source.offline": _stored_data,
    "transform.filter": _filter_block,
    "transform.channel_mask": _channel_mask,
    "transform.split": _split,
    WINDOW: _window,
    FEATURES: _features,
    MODEL: _model,
    "sink.metrics": _metrics,
    "sink.feature_viewer": _feature_viewer,
    "sink.prediction_viewer": _prediction_viewer,
}


class _NoSpec:
    category = None


_NO_SPEC = _NoSpec()


# ======================================================================
# Compilation
# ======================================================================
def compile_pipeline(document, log=None, probe_rows=400, buffer_rows=2000, models=None):
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
    models: dict or None (optional), default=None
        For a stored-data pipeline: fitted predictors by model block, used
        before any Fitted Model file. What the editor's Train and Load buttons
        produce.

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
        return _compile_offline(document, models)
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


def _compile_offline(document, models=None):
    """Check what can be checked without reading anything, then hand the document over.

    A stored-data pipeline is evaluated block by block (see
    :class:`OfflinePipeline`), so there is no plan to flatten it into. What is
    checked here is what would otherwise only fail once a folder had been
    walked: a Stored Data block with nowhere to read or no way to tell its
    files apart.
    """
    for node_id in document.order():
        node = document.nodes[node_id]
        if node.spec_id == "source.offline":
            _check_stored_data(node.params)
    return OfflinePipeline(document, models)


def compile_for_training(document, models=None):
    """A stored-data pipeline to train model blocks in, before the rest is finished.

    Training a block needs only the blocks upstream of its training inputs, so
    the rest of the document -- a half-wired test branch, a sink not yet
    connected -- is not held against it. What has to hold is that the data is
    stored, and that every Stored Data block can be read.
    """
    unresolved = [n.id for n in document.nodes.values() if n.unresolved]
    if unresolved:
        raise CompileError(f"These blocks are not recognised by this version of LibEMG: "
                           f"{', '.join(sorted(unresolved))}. Remove them to train.")
    if document.mode() != "offline":
        raise CompileError("Training reads stored data. Add a Stored Data source, or open "
                           "the pipeline you collected your training data with.")
    for node in document.nodes.values():
        if node.spec_id == "source.offline":
            _check_stored_data(node.params)
    return OfflinePipeline(document, models)


def _check_stored_data(params):
    if not params.get("folder"):
        raise CompileError("The Stored Data block needs a folder to read from.")
    if not _parse_regex_filters(params.get("regex_filters", "")):
        raise CompileError(
            "The Stored Data block needs at least one regex filter, so it "
            "knows which files belong to what.")
    _parse_regex_filters(params.get("label_files", ""))


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
