"""Fitting a model block on what its training inputs carry, and how well it did.

A model block is trained from two inputs: what it reads -- features, or for a
network on the raw signal, windows -- and the windowed metadata that says what
each window was. One field of that metadata is the label it learns; the block's
Label Field names which.

Every supervised model is judged the same way. Whole recordings are held out
for validation (never windows: neighbouring windows overlap, and a window-level
split scores a model on data it has effectively seen), the model is fitted on
the rest, and both sides are scored: accuracy, per class and as a confusion
matrix, for a classifier; RMSE, MAE and R² per degree of freedom for a
regressor. The statistics travel with the fitted model as
``predictor.training_statistics``, so a model saved now and loaded in another
session still says how it was trained and how well it did.

This module draws nothing. The editor's Train button calls it, and so does
``OfflinePipeline.train`` from a script.
"""

from dataclasses import dataclass, field

import numpy as np

from libemg._gui._pipeline.offline_data import (FeatureSet, by_recording, hold_out,
                                                model_input, predict_recordings)

#: The attribute a fitted predictor carries its statistics in.
STATISTICS = "training_statistics"


class FitError(ValueError):
    """A model block cannot be trained on what it was given, and why."""


@dataclass
class FitResult:
    """A trained model block.

    Attributes
    ----------
    predictor: EMGClassifier or EMGRegressor
        Ready to save, run, or hand to a live pipeline.
    statistics: dict
        As :func:`statistics_text` describes them; also on the predictor.
    epochs: list
        A deep model's EpochResults, one per epoch. Empty otherwise.
    outcome: TrainingOutcome or None
        A deep model's training record, for the Training History window.
    stopped: bool
        Training was asked to stop early; the best epoch so far was kept.
    """

    predictor: object
    statistics: dict
    epochs: list = field(default_factory=list)
    outcome: object = None
    stopped: bool = False


def fit_block(spec_id, params, inputs, metadata, label_field="classes", on_epoch=None,
              should_stop=None):
    """Train a model block on its training inputs.

    Parameters
    ----------
    spec_id: str
        ``'model.classifier'``, ``'model.regressor'`` or ``'model.deep'``.
    params: dict
        The block's settings.
    inputs: FeatureSet or WindowSet
        What the model reads.
    metadata: WindowMetadata
        The windowed metadata, one row per window of ``inputs``.
    label_field: str
        Which field of ``metadata`` is the label.
    on_epoch, should_stop: callable or None
        For a deep model: called with each EpochResult, and checked between
        batches.

    Raises
    ----------
    FitError
        If the labels do not belong to these windows, if the field is
        missing or the wrong shape for the model, or if training was stopped
        before anything was learned.
    """
    from libemg.emg_predictor import (CLASSIFIER_MODELS, EMGClassifier, EMGRegressor,
                                      REGRESSOR_MODELS)

    check_aligned(inputs, metadata, "training features", "training labels")
    try:
        labels = np.asarray(metadata.field(label_field))
    except KeyError as error:
        raise FitError(str(error.args[0])) from error
    x = model_input(inputs)
    lengths = list(inputs.lengths)
    if not len(x):
        raise FitError("There are no windows to train on. Check the folder, the filters, "
                       "and that a window is not longer than the recordings.")

    if spec_id == "model.deep":
        task = str(params.get("task", "Classification")).lower()
        title = f"{params.get('architecture', 'GRU')} ({task})"
    else:
        name = params.get("model", "")
        task = "classification" if spec_id == "model.classifier" else "regression"
        known = CLASSIFIER_MODELS if task == "classification" else REGRESSOR_MODELS
        if name not in known:
            raise FitError(f"'{name}' is not a model this version can fit. Choose one of "
                           f"{sorted(known)}.")
        title = name
    classification = task == "classification"
    if classification:
        if labels.ndim > 1:
            raise FitError(f"A classifier learns one class per window, but '{label_field}' "
                           f"has {labels.shape[1]} columns. Pick a field with one value per "
                           "window, such as classes.")
        if labels.dtype.kind == "f":
            if not np.all(np.isfinite(labels)) or np.any(labels != np.round(labels)):
                raise FitError(f"'{label_field}' holds values that are not whole numbers, so "
                               "it cannot be a class. Train a regressor on it instead.")
            labels = labels.astype(int)
    else:
        labels = labels.astype(float)

    held = hold_out(lengths, labels if classification else None,
                    float(params.get("validation", 0.2)), int(params.get("seed", 0)))
    kept = [i for i in range(len(lengths)) if i not in held]
    pieces_x, pieces_y = by_recording(x, lengths), by_recording(labels, lengths)

    epochs, outcome, stopped = [], None, False
    if spec_id == "model.deep":
        predictor, epochs, stopped = _fit_deep(params, task, x, labels, lengths, held, kept,
                                               on_epoch, should_stop)
    else:
        train_x = np.concatenate([pieces_x[i] for i in kept])
        train_y = np.concatenate([pieces_y[i] for i in kept])
        if train_x.ndim > 2:
            train_x = train_x.reshape(len(train_x), -1)
        predictor = EMGClassifier(name) if classification else EMGRegressor(name)
        if not classification and train_y.ndim == 1:
            # A regressor predicts a column per degree of freedom; a field of
            # one column is one.
            train_y = train_y.reshape(-1, 1)
        predictor.fit({"training_features": train_x, "training_labels": train_y})

    flat = not isinstance(inputs, FeatureSet) and spec_id != "model.deep"
    files = metadata.file_paths()
    statistics = {
        "model": title, "block": spec_id, "task": task, "label_field": label_field,
        "recordings": len(lengths), "windows": int(len(x)),
        "training_files": [files[i] for i in kept],
        "validation_files": [files[i] for i in held],
        "classes": _classes(predictor) if classification else [],
        "dofs": None if classification else (1 if labels.ndim == 1 else int(labels.shape[1])),
        "training": score(predictor, [pieces_x[i] for i in kept], [pieces_y[i] for i in kept],
                          classification, flat),
        "validation": score(predictor, [pieces_x[i] for i in held],
                            [pieces_y[i] for i in held], classification, flat) if held else None,
        "stopped": stopped, "epochs": len(epochs),
    }
    setattr(predictor, STATISTICS, statistics)
    if spec_id == "model.deep":
        outcome = predictor.model.outcome(model_path="", data={"label_field": label_field,
                                                               "statistics": statistics})
    return FitResult(predictor, statistics, epochs, outcome, stopped)


def _fit_deep(params, task, x, labels, lengths, held, kept, on_epoch, should_stop):
    from libemg.deep import DeepModel, TrainingSet
    from libemg.emg_predictor import EMGClassifier, EMGRegressor

    timing = str(params.get("timing", "Continuous")).lower()
    try:
        everything = TrainingSet.from_arrays(x, labels, lengths=lengths, task=task,
                                             timing=timing)
        training = TrainingSet([everything.recordings[i] for i in kept], task, timing)
        validation = TrainingSet([everything.recordings[i] for i in held], task, timing) \
            if held else None
    except ValueError as error:
        raise FitError(str(error)) from error
    patience = int(params.get("patience", 10))
    model = DeepModel(
        str(params.get("architecture", "GRU")).lower(), task=task, timing=timing,
        pooling=params.get("pooling", "max"), hidden_size=int(params.get("hidden_size", 64)),
        layers=int(params.get("layers", 2)), embedding_size=int(params.get("embedding_size", 32)),
        dropout=float(params.get("dropout", 0.1)), horizon=int(params.get("horizon", 10)),
        epochs=int(params.get("epochs", 50)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        batch_size=int(params.get("batch_size", 32)),
        sequence_length=int(params.get("sequence_length", 32)),
        patience=patience or None, seed=int(params.get("seed", 0)))
    epochs = []

    def record(result):
        epochs.append(result)
        if on_epoch is not None:
            on_epoch(result)

    try:
        model.fit(training, validation=validation, on_epoch=record, should_stop=should_stop)
    except ValueError as error:
        raise FitError(str(error)) from error
    if not model.history:
        raise FitError("Training was stopped before its first epoch finished, so there is "
                       "nothing to keep.")
    # Which recordings were held out, by position among those it was given:
    # what lets the Prediction Viewer show training and held-out data apart.
    model.trained_on["validation_indices"] = list(held)
    model.trained_on["recordings"] = len(lengths)
    predictor = EMGClassifier(model) if task == "classification" else EMGRegressor(model)
    stopped = bool(should_stop()) if should_stop is not None else False
    return predictor, epochs, stopped


# ======================================================================
# scoring
# ======================================================================
def score(predictor, inputs, truths, classification, flatten=False):
    """How well a predictor does on some recordings.

    Returns
    ----------
    dict or None
        For a classifier: ``windows``, ``recordings``, ``accuracy``,
        ``per_class`` (class to accuracy), ``classes`` and ``confusion`` (a
        row per true class, a column per predicted class). For a regressor:
        ``windows``, ``recordings``, ``rmse``, ``mae`` and ``r2``, each a list
        with one value per degree of freedom. None with no recordings.
    """
    if not inputs:
        return None
    if flatten:
        inputs = [x.reshape(len(x), -1) for x in inputs]
    results = predict_recordings(predictor, inputs, classification)
    truth = np.concatenate([np.asarray(t) for t in truths])
    predicted = np.concatenate([p for _, p in results])
    out = {"windows": int(len(truth)), "recordings": len(inputs)}
    if classification:
        truth = truth.ravel()
        predicted = np.asarray(predicted).ravel()
        classes = sorted(set(np.unique(truth).tolist()) | set(np.unique(predicted).tolist()))
        index = {c: i for i, c in enumerate(classes)}
        confusion = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(truth.tolist(), predicted.tolist()):
            confusion[index[t], index[p]] += 1
        per_class = {}
        for c in np.unique(truth).tolist():
            mask = truth == c
            per_class[c] = float(np.mean(predicted[mask] == c))
        out.update(accuracy=float(np.mean(predicted == truth)), per_class=per_class,
                   classes=classes, confusion=confusion.tolist())
    else:
        truth = truth.reshape(len(truth), -1)
        predicted = np.asarray(predicted, dtype=float).reshape(len(truth), -1)
        error = predicted - truth
        spread = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
        r2 = [float(1 - (error[:, d] ** 2).sum() / spread[d]) if spread[d] > 0 else None
              for d in range(truth.shape[1])]
        out.update(rmse=[float(v) for v in np.sqrt((error ** 2).mean(axis=0))],
                   mae=[float(v) for v in np.abs(error).mean(axis=0)], r2=r2)
    return out


def check_aligned(values, metadata, what="features", other="labels"):
    """Refuse to pair two things that are not one row per the same window.

    Labels and features line up when they come from the same Window block. Two
    Window blocks with the same settings over the same recordings line up
    too, and are let through by checking the windows per recording match.
    """
    if values is None or metadata is None:
        return
    if getattr(values, "origin", None) is getattr(metadata, "origin", 0):
        return
    if list(values.lengths) != list(metadata.lengths):
        raise FitError(
            f"The {what} and the {other} are not the same windows: {len(values)} windows "
            f"against {len(metadata)}. Take both from one Window block -- its Output for the "
            "signal, its Metadata for the labels -- so each label is the label of its window.")


def statistics_text(statistics):
    """A sentence on how a model did: what the editor shows on its block."""
    if not statistics:
        return ""
    parts = []
    for side in ("training", "validation"):
        result = statistics.get(side)
        if not result:
            continue
        if "accuracy" in result:
            parts.append(f"{side} {result['accuracy']:.1%}")
        else:
            parts.append(f"{side} RMSE {np.mean(result['rmse']):.3g}")
    return ", ".join(parts)


def _classes(predictor):
    model = getattr(predictor, "model", predictor)
    classes = getattr(model, "classes_", None)
    return [] if classes is None else np.asarray(classes).tolist()
