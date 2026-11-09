"""A deep model as LibEMG uses one: trained, run, streamed, adapted and saved alike.

:class:`DeepModel` is the one object every part of LibEMG can hold:

    training        ``fit(training_set)`` with a live report per epoch
    offline use     ``predict`` / ``predict_proba`` over a recording in order
    live use        the same calls a window at a time, with ``streaming = True``
                    carrying a recurrent or transformer model's memory between
                    them
    inspection      ``latent(x)``: the embedding its head reads, which the
                    pipeline's Feature Viewer draws as the model's own space
    adaptation      ``adapt(memory)`` and ``save(path)``, the two calls the
                    adaptation manager and AdaptationHook make
    persistence     pickles like any LibEMG model, weights included

Wrapped in an :class:`~libemg.emg_predictor.EMGClassifier` or
:class:`~libemg.emg_predictor.EMGRegressor` it drops into everything that takes
one -- the pipeline's model blocks, OnlineEMGClassifier, offline metrics.

Examples
--------
A GRU classifying a held motion at every window::

    from libemg.deep import DeepModel, TrainingSet
    training = TrainingSet.from_arrays(features, labels, lengths=windows_per_file)
    model = DeepModel("gru", sequence_length=64)
    model.fit(training, on_epoch=lambda e: print(e.describe()))

A CNN on raw windows recognising a discrete gesture once it is over::

    training = TrainingSet.from_arrays(windows, labels, lengths=windows_per_gesture,
                                       timing="discrete")
    model = DeepModel("cnn", timing="discrete", pooling="max").fit(training)

Live, a window at a time::

    model.streaming = True
    model.reset()
    for window_features in stream:
        probabilities = model.predict_proba(window_features[None])
"""

import io
import pickle

import numpy as np

from libemg.deep.data import (CLASSIFICATION, CONTINUOUS, DISCRETE, REGRESSION,
                              TrainingSet)
from libemg.deep.training import Trainer

#: How many windows are run through the network at once outside training. A
#: whole session at once would hold every transformer context in memory.
_CHUNK = 2048


class DeepModel:
    """A deep network with the calls LibEMG expects of a model.

    Parameters
    ----------
    architecture: str
        'mlp' (features), 'cnn' (raw windows), 'transformer' (a horizon of
        windows), 'gru' or 'lstm' (a memory of every window before).
    task: str
        'classification' or 'regression'.
    timing: str
        'continuous' (an answer every window) or 'discrete' (a gesture, labelled
        once). See :mod:`libemg.deep.data`.
    pooling: str
        For a discrete task: how a gesture's steps become one answer. See
        :class:`~libemg.deep.training.Trainer`.
    hidden_size, layers, embedding_size, dropout: network size
    horizon, heads: for a transformer, the windows it attends over and its heads
    epochs, learning_rate, weight_decay, batch_size, sequence_length, patience,
    warmup, seed: training, as :class:`~libemg.deep.training.Trainer` takes them
    adapt_epochs: int
        Epochs per call to :meth:`adapt`. Few: adaptation runs between trials.
    adapt_learning_rate: float or None
        Defaults to a tenth of ``learning_rate``, so adapting nudges the model
        rather than retraining it on one trial's worth of data.
    """

    def __init__(self, architecture="mlp", task=CLASSIFICATION, timing=CONTINUOUS,
                 pooling="max", hidden_size=64, layers=2, embedding_size=32, dropout=0.1,
                 horizon=10, heads=2, epochs=50, learning_rate=1e-3, weight_decay=1e-4,
                 batch_size=32, sequence_length=32, patience=10, warmup=0, seed=0,
                 adapt_epochs=3, adapt_learning_rate=None):
        self.architecture = architecture.lower()
        self.task, self.timing, self.pooling = task, timing, pooling
        self.network_options = dict(hidden_size=hidden_size, layers=layers,
                                    embedding_size=embedding_size, dropout=dropout,
                                    horizon=horizon, heads=heads)
        self.training_options = dict(epochs=epochs, learning_rate=learning_rate,
                                     weight_decay=weight_decay, batch_size=batch_size,
                                     sequence_length=sequence_length, patience=patience,
                                     pooling=pooling, warmup=warmup, seed=seed)
        self.adapt_epochs = adapt_epochs
        self.adapt_learning_rate = adapt_learning_rate
        #: Feed predict() one window at a time and carry memory between calls.
        self.streaming = False
        #: Called with each EpochResult, during fit and adapt. Not saved.
        self.on_epoch = None
        self.network = None
        self.history = []
        #: What fit() trained on, and whether it was stopped: kept for outcome().
        self.trained_on = {}
        self.stopped = False
        self.classes_ = None
        self.input_shape = None
        self._mean = self._scale = None
        self._dofs = 0
        self._state = None

    # -- training ------------------------------------------------------------
    @property
    def fitted(self):
        return self.network is not None

    def fit(self, training, validation=0.2, on_epoch=None, should_stop=None):
        """Train from scratch.

        Parameters
        ----------
        training: TrainingSet
        validation: float, TrainingSet or None
            A fraction of the recordings to hold out (whole recordings, never
            windows), a set of its own, or None to watch the training loss.
        on_epoch: callable or None
            Called with each :class:`EpochResult` as it completes.
        should_stop: callable or None
            Checked between batches. Stopping keeps the best epoch so far.

        Returns
        ----------
        DeepModel
            Itself, so ``DeepModel(...).fit(...)`` reads naturally.
        """
        from libemg.deep.networks import build_network

        self._check(training)
        split_here = isinstance(validation, (int, float)) and not isinstance(validation, bool)
        if split_here:
            training, validation = training.split(float(validation),
                                                  self.training_options["seed"])
        self.input_shape = training.input_shape
        self._fit_scaler(training)
        if self.task == CLASSIFICATION:
            self.classes_ = np.asarray(training.classes)
            outputs = len(self.classes_)
            if outputs < 2:
                raise ValueError("Classification needs at least two classes in the data.")
        else:
            self.classes_ = None
            outputs = training.dofs
        self.network = build_network(self.architecture, self.input_shape, outputs,
                                     **self.network_options)
        self.trained_on = {
            "recordings": len(training.recordings) + (len(validation.recordings)
                                                      if validation is not None else 0),
            "validation_recordings": len(validation.recordings) if validation is not None else 0,
            "windows": training.windows + (validation.windows if validation is not None else 0),
            "input_shape": list(self.input_shape),
            "classes": [int(c) for c in self.classes_] if self.classes_ is not None else None,
            "dofs": training.dofs if self.task == REGRESSION else None,
            # Positions, among the recordings fit() was given, of those held out.
            # Only known when fit() did the split; a validation set passed in is
            # numbered on its own.
            "validation_indices": list(getattr(validation, "source_indices", []))
            if split_here and validation is not None else [],
        }
        trainer = Trainer(**self.training_options)
        self.history = trainer.fit(self.network, training, validation, self.task, self.timing,
                                   prepare=self._prepare,
                                   on_epoch=self._reporter(on_epoch),
                                   should_stop=should_stop)
        self.stopped = bool(should_stop is not None and should_stop())
        self.reset()
        return self

    def outcome(self, model_path="", name="", data=None):
        """This model's training as a :class:`~libemg.deep.training.TrainingOutcome`.

        Parameters
        ----------
        model_path: str
            Where the model was saved, to record alongside.
        name: str
            A label for comparing runs. Defaults to the architecture and time.
        data: dict or None
            More to record about the data, such as the folder it came from.
        """
        from libemg.deep.training import TrainingOutcome
        return TrainingOutcome(
            self.architecture, self.task, self.timing,
            options=dict(self.network_options, **self.training_options),
            data=dict(getattr(self, "trained_on", {}) or {}, **(data or {})),
            epochs=list(self.history), stopped=bool(getattr(self, "stopped", False)),
            model_path=str(model_path), name=name)

    def adapt(self, memory, epochs=None, on_epoch=None, should_stop=None):
        """Fold new data into the model: a few epochs at a low learning rate.

        This is the call the adaptation manager and AdaptationHook make each
        time a slice of memory arrives, followed by :meth:`save`. ``memory``
        is a :class:`~libemg.deep.memory.SequenceMemory`, anything with a
        ``training_set(task, timing)`` method, or a TrainingSet.

        Returns
        ----------
        list of float
            The training loss of each epoch, which the adaptation manager
            writes to its losses file.
        """
        training = memory if isinstance(memory, TrainingSet) else \
            memory.training_set(self.task, self.timing)
        if not self.fitted:
            self.fit(training, validation=None, on_epoch=on_epoch, should_stop=should_stop)
            return [r.train_loss for r in self.history]
        self._check(training)
        if self.task == CLASSIFICATION:
            training = self._known_classes(training)
            if training is None:
                return []
        options = dict(self.training_options, epochs=epochs or self.adapt_epochs, patience=None,
                       learning_rate=self.adapt_learning_rate
                       or self.training_options["learning_rate"] / 10)
        history = Trainer(**options).fit(self.network, training, None, self.task, self.timing,
                                         prepare=self._prepare,
                                         on_epoch=self._reporter(on_epoch),
                                         should_stop=should_stop, phase="adapt",
                                         restore_best=False)
        self.history = list(self.history) + history
        self.reset()
        return [r.train_loss for r in history]

    def evaluate(self, data):
        """Loss and score on data the model has not trained on.

        Returns
        ----------
        (loss, score, score name)
            Accuracy for classification -- per window for a continuous task,
            per gesture for a discrete one -- or RMSE for regression.
        """
        if not self.fitted:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")
        self._check(data)
        if self.task == CLASSIFICATION:
            data = self._known_classes(data)
            if data is None:
                raise ValueError("None of these classes are ones the model was trained on.")
        loss, score, name = Trainer(**self.training_options).evaluate(
            self.network, data, self.task, self.timing, self._prepare)
        return loss, score, name

    # -- using it ------------------------------------------------------------
    def predict_proba(self, x):
        """Class probabilities for each window of ``x``, run as one stream in order."""
        if self.task != CLASSIFICATION:
            raise AttributeError("A regression model has no class probabilities.")
        import torch
        outputs, _ = self._run(x)
        return torch.softmax(torch.as_tensor(outputs), dim=-1).numpy()

    def predict(self, x):
        """A class per window for classification; values per window for regression."""
        outputs, _ = self._run(x)
        if self.task == CLASSIFICATION:
            return self.classes_[np.argmax(outputs, axis=1)]
        return outputs

    def latent(self, x):
        """The embedding the output head reads, one row per window: the model's own space."""
        return self._run(x)[1]

    def reset(self):
        """Forget what a stream has carried so far, as at the start of a new one."""
        self._state = None

    # -- persistence -----------------------------------------------------------
    def save(self, path):
        """Pickle the model, weights included. The call the adaptation manager makes."""
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["on_epoch"] = None        # a window's callback is no part of the model
        state["_state"] = None
        if self.network is not None:
            buffer = io.BytesIO()
            import torch
            torch.save(self.network.state_dict(), buffer)
            state["network"] = buffer.getvalue()
        return state

    def __setstate__(self, state):
        weights = state.get("network")
        self.__dict__.update(state)
        if isinstance(weights, (bytes, bytearray)):
            import torch
            from libemg.deep.networks import build_network
            outputs = len(self.classes_) if self.task == CLASSIFICATION else self._dofs
            self.network = build_network(self.architecture, self.input_shape, outputs,
                                         **self.network_options)
            self.network.load_state_dict(torch.load(io.BytesIO(weights), weights_only=True))
            self.network.eval()

    # -- internals -----------------------------------------------------------
    def _check(self, training):
        if training.task != self.task or training.timing != self.timing:
            raise ValueError(f"This is a {self.timing} {self.task} model, but the data is "
                             f"{training.timing} {training.task}.")
        if self.input_shape is not None and training.input_shape != self.input_shape:
            raise ValueError(f"The model reads windows of shape {self.input_shape}; this data "
                             f"has {training.input_shape}.")

    def _fit_scaler(self, training):
        """Standardise inputs: per feature, or per channel for raw windows."""
        stacked = np.concatenate([r.inputs for r in training.recordings]).astype(np.float64)
        axes = (0,) if stacked.ndim == 2 else (0, 2)
        self._mean = stacked.mean(axis=axes, keepdims=True)[0]
        scale = stacked.std(axis=axes, keepdims=True)[0]
        self._scale = np.where(scale > 0, scale, 1.0)
        if self.task == REGRESSION:
            self._dofs = training.dofs

    def _known_classes(self, training):
        """Drop what the model has no output for: adapting cannot add a class."""
        from libemg.deep.data import Recording
        known = set(int(c) for c in self.classes_)
        kept = []
        for r in training.recordings:
            if self.timing == DISCRETE:
                if int(r.targets) in known:
                    kept.append(r)
            else:
                keep = np.isin(np.asarray(r.targets).astype(int), list(known))
                if keep.any():
                    kept.append(Recording(r.inputs[keep], np.asarray(r.targets)[keep]))
        return TrainingSet(kept, self.task, self.timing) if kept else None

    def _prepare(self, inputs, targets, mask):
        import torch
        x = torch.as_tensor(((inputs - self._mean) / self._scale).astype(np.float32))
        if self.task == CLASSIFICATION:
            lookup = {int(c): i for i, c in enumerate(self.classes_)}
            indices = np.vectorize(lambda v: lookup.get(int(v), 0))(np.asarray(targets))
            y = torch.as_tensor(indices.astype(np.int64))
        else:
            y = torch.as_tensor(np.asarray(targets, dtype=np.float32))
        return x, y, torch.as_tensor(mask)

    def _run(self, x):
        """Outputs and embedding for ``x``, one row per window, carrying state if streaming."""
        if not self.fitted:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")
        import torch
        x = np.asarray(x, dtype=np.float64)
        if x.shape[1:] != tuple(self.input_shape):
            x = x.reshape((len(x),) + tuple(self.input_shape))
        x = ((x - self._mean) / self._scale).astype(np.float32)
        state = self._state if self.streaming else None
        outputs, embeddings = [], []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(x), _CHUNK):
                piece = torch.as_tensor(x[start:start + _CHUNK])[None]
                out, embedding, state = self.network(piece, state)
                outputs.append(out[0].numpy())
                embeddings.append(embedding[0].numpy())
        if self.streaming:
            self._state = state
        if not outputs:
            width = self.network.head.linear.out_features
            return np.empty((0, width)), np.empty((0, 0))
        return np.concatenate(outputs), np.concatenate(embeddings)

    def _reporter(self, on_epoch):
        targets = [f for f in (on_epoch, self.on_epoch) if f is not None]
        if not targets:
            return None

        def report(result):
            for target in targets:
                target(result)
        return report
