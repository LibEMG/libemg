"""What flows along the links of a stored-data pipeline.

A live pipeline moves rows of samples between processes. A stored-data pipeline
moves whole collections, and every collection keeps track of where each of its
rows came from, so that whatever is computed downstream can still be lined up
with the recording, the file and the label it belongs to.

    Recordings      what a Stored Data block reads: an OfflineDataHandler that
                    also knows each file's path and what its filename said.
                    Filters, channel masks and Split blocks pass it on.
    WindowSet       what a Window block cuts the recordings into.
    WindowMetadata  every metadata field, windowed exactly as the signal was:
                    one value per window, in the same order. What labels a
                    model, colours a plot and scores a prediction.
    FeatureSet      what a Features block extracts from a WindowSet.
    Predictions     what a model made of a WindowSet or a FeatureSet.

Each WindowSet, and everything computed from it, carries the number of windows
in every file it came from. That is what keeps a model with memory from being
trained or run across the join between two recordings, and what lets labels
and predictions be checked to belong to the same windows before they are
compared.
"""

import copy
import os
from glob import glob
from pathlib import Path

import numpy as np

from libemg.data_handler import OfflineDataHandler

#: How a window's value of a metadata field is taken from the samples it spans.
METADATA_RULES = ("last sample", "most common", "first sample", "mean")


# ======================================================================
# recordings
# ======================================================================
class Recordings(OfflineDataHandler):
    """An OfflineDataHandler that remembers each file and what its name said.

    Everything an OfflineDataHandler offers works on it -- ``data``, a list
    per metadata field, ``parse_windows``, ``isolate_data``. It adds:

    files: list of str
        The path of each recording, in the order of ``data``.
    field_values: dict
        For each field read from filenames, the values its filter matched.
        The handler stores a filename field as a position in that list; this
        is what turns it back into what the filename said.
    sampling_rate: int
        Samples per second.
    """

    def __init__(self):
        super().__init__()
        self.extra_attributes = []
        self.files = []
        self.field_values = {}
        self.sampling_rate = 1000

    @classmethod
    def read(cls, folder, regex_filters, metadata_fetchers=None, delimiter=",",
             sampling_rate=1000):
        """Read the recordings in ``folder`` that every filter matches.

        Parameters
        ----------
        regex_filters: list of dict
            RegexFilter keyword arguments: left_bound, right_bound, values,
            description.
        """
        from libemg.data_handler import RegexFilter
        filters = [RegexFilter(**spec) for spec in regex_filters]
        recordings = cls()
        recordings.get_data(folder_location=folder, regex_filters=filters,
                            metadata_fetchers=metadata_fetchers, delimiter=delimiter)
        recordings.files = _matching_files(folder, filters)
        recordings.field_values = {spec["description"]: list(spec["values"])
                                   for spec in regex_filters if spec.get("description")}
        recordings.sampling_rate = int(sampling_rate)
        return recordings

    # -- what it holds -----------------------------------------------------
    def __len__(self):
        return len(self.data)

    def fields(self):
        """The metadata fields every recording carries."""
        return list(self.extra_attributes)

    def value(self, field, index):
        """What a field says about one recording: its filename value, or its first sample."""
        column = getattr(self, field)[index]
        first = np.ravel(column)[0] if np.size(column) else None
        values = self.field_values.get(field)
        if values is not None and first is not None:
            position = int(first)
            return values[position] if 0 <= position < len(values) else position
        return first.item() if hasattr(first, "item") else first

    def file_meta(self, index):
        """Every filename field of one recording, as the filename wrote it."""
        return {field: self.value(field, index) for field in self.field_values
                if hasattr(self, field)}

    def values_of(self, field):
        """Every value a field takes across the recordings, in the order first seen."""
        seen = []
        for index in range(len(self)):
            value = self.value(field, index)
            if value not in seen:
                seen.append(value)
        return seen

    # -- making new ones -----------------------------------------------------
    def subset(self, indices):
        """The recordings at ``indices``, in that order."""
        out = Recordings()
        out.extra_attributes = list(self.extra_attributes)
        out.data = [self.data[i] for i in indices]
        for field in self.extra_attributes:
            column = getattr(self, field)
            setattr(out, field, [column[i] for i in indices])
        out.files = [self.files[i] for i in indices] if self.files else []
        out.field_values = copy.deepcopy(self.field_values)
        out.sampling_rate = self.sampling_rate
        return out

    def with_data(self, data):
        """The same recordings, carrying ``data`` -- a filtered signal -- instead."""
        out = self.subset(range(len(self)))
        out.data = list(data)
        return out

    def split(self, field, wanted):
        """The recordings whose ``field`` is one of ``wanted``, and the rest.

        A recording goes one way or the other as a whole: its value is the one
        its filename gave, or for a field read from a file, its first sample.

        Parameters
        ----------
        wanted: iterable of str or numbers
            Values as the filename writes them. ``'1'`` matches ``1`` and
            ``1.0``.

        Returns
        ----------
        (Recordings, Recordings)
        """
        if field not in self.extra_attributes:
            raise KeyError(field)
        wanted = list(wanted)
        matching, rest = [], []
        for index in range(len(self)):
            (matching if _same_value(self.value(field, index), wanted) else rest).append(index)
        return self.subset(matching), self.subset(rest)


def parse_values(text):
    """The values a Split block's text names: '0,1,2', '0-5' or a mixture.

    A range is inclusive at both ends, as a person counting repetitions means
    it. Anything that is not a whole-number range is taken as written.
    """
    import re
    values = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        span = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
        if span:
            low, high = int(span.group(1)), int(span.group(2))
            step = 1 if high >= low else -1
            values.extend(str(v) for v in range(low, high + step, step))
        else:
            values.append(part)
    return values


# ======================================================================
# windows, and the metadata that goes with them
# ======================================================================
class WindowMetadata:
    """Every metadata field, one value per window, in the order of the windows.

    Attributes
    ----------
    fields: dict
        Field name to an array of one value per window: (windows,) for a
        field of one column, (windows, columns) for one with several, such as
        a regression target over several degrees of freedom.
    lengths: list of int
        Windows in each recording that produced any.
    recordings: Recordings
        Where they came from.
    files: list of int
        For each entry of ``lengths``, the recording's position in
        ``recordings``.
    rule: str
        How each window's value was taken from its samples.
    """

    def __init__(self, fields, lengths, recordings, files, rule, origin):
        self.fields, self.lengths = fields, list(lengths)
        self.recordings, self.files, self.rule = recordings, list(files), rule
        #: Shared with the WindowSet these were windowed alongside.
        self.origin = origin

    def __len__(self):
        return int(sum(self.lengths))

    def names(self):
        return list(self.fields)

    def field(self, name):
        """One field's value per window.

        Raises
        ----------
        KeyError
            Naming the fields there are, if there is no such one.
        """
        if name not in self.fields:
            raise KeyError(f"There is no metadata field called '{name}'. The recordings "
                           f"carry: {', '.join(self.fields) or 'none'}.")
        return self.fields[name]

    def file_meta(self, position):
        """The filename fields of the ``position``-th recording that has windows."""
        return self.recordings.file_meta(self.files[position])

    def file_paths(self):
        return [self.recordings.files[i] if self.recordings.files else str(i)
                for i in self.files]


class WindowSet:
    """Recordings cut into windows, and every metadata field windowed alongside."""

    def __init__(self, windows, lengths, recordings, files, window_size, window_increment,
                 metadata):
        self.windows = windows
        self.lengths = list(lengths)
        self.recordings, self.files = recordings, list(files)
        self.window_size, self.window_increment = int(window_size), int(window_increment)
        self.metadata = metadata
        self.origin = metadata.origin

    def __len__(self):
        return len(self.windows)

    @property
    def sampling_rate(self):
        return self.recordings.sampling_rate

    def signals(self):
        """The signal of every recording that has windows, in order."""
        return [self.recordings.data[i] for i in self.files]

    def file_paths(self):
        return self.metadata.file_paths()


def window_recordings(recordings, window_size, window_increment, rule="last sample"):
    """Cut every recording into windows, and every metadata field with it.

    Recordings shorter than a window produce none and are left out of both.

    Returns
    ----------
    WindowSet
        With its :class:`WindowMetadata` as ``metadata``.
    """
    from libemg.utils import get_windows
    if rule not in METADATA_RULES:
        raise ValueError(f"rule must be one of {METADATA_RULES}, not {rule!r}.")
    windows, lengths, files = [], [], []
    fields = {name: [] for name in recordings.extra_attributes}
    for index, signal in enumerate(recordings.data):
        enframed = get_windows(np.asarray(signal), window_size, window_increment)
        if not enframed.shape[0]:
            continue
        windows.append(enframed)
        lengths.append(int(enframed.shape[0]))
        files.append(index)
        for name in fields:
            fields[name].append(_window_field(getattr(recordings, name)[index],
                                              window_size, window_increment, rule,
                                              enframed.shape[0]))
    origin = object()
    metadata = WindowMetadata({name: np.concatenate(parts) if parts else np.empty(0)
                               for name, parts in fields.items()},
                              lengths, recordings, files, rule, origin)
    stacked = np.vstack(windows) if windows else np.empty((0, 0, window_size))
    return WindowSet(stacked, lengths, recordings, files, window_size, window_increment,
                     metadata)


def _window_field(column, window_size, window_increment, rule, count):
    """One recording's field, a value per window."""
    from libemg.utils import get_windows
    column = np.asarray(column)
    if column.ndim == 0:
        return np.full(count, column.item())
    if column.ndim == 1:
        column = column.reshape(-1, 1)
    enframed = get_windows(column, window_size, window_increment)     # (windows, columns, samples)
    if rule == "last sample":
        values = enframed[:, :, -1]
    elif rule == "first sample":
        values = enframed[:, :, 0]
    elif rule == "mean":
        values = enframed.astype(float).mean(axis=2)
    else:
        distinct, ranks = np.unique(enframed, return_inverse=True)
        ranks = ranks.reshape(enframed.shape)
        values = distinct[np.apply_along_axis(lambda x: np.bincount(x).argmax(), 2, ranks)]
    return values[:, 0] if values.shape[1] == 1 else values


class FeatureSet:
    """Features extracted from a WindowSet, a row per window, a name per column."""

    def __init__(self, matrix, names, windows):
        self.matrix, self.names, self.windows = matrix, list(names), windows
        self.lengths = windows.lengths
        self.origin = windows.origin
        self.metadata = windows.metadata

    def __len__(self):
        return len(self.matrix)


class Predictions:
    """What a model made of every window, recording by recording.

    Attributes
    ----------
    outputs: array
        Class probabilities (windows, classes) for a classifier, or values
        (windows, dofs) for a regressor.
    predicted: array
        The class of each window, or the values again.
    classification: bool
    classes: list
        The classes the model knows, in the order of ``outputs``' columns.
    windows: WindowSet
        The windows they are predictions of.
    predictor: object
        The model that made them.
    """

    def __init__(self, outputs, predicted, classification, classes, windows, predictor):
        self.outputs, self.predicted = outputs, predicted
        self.classification, self.classes = classification, list(classes)
        self.windows, self.predictor = windows, predictor
        self.lengths, self.origin = windows.lengths, windows.origin

    def __len__(self):
        return len(self.predicted)


def model_input(value):
    """A model's input as an array: features as they are, windows flattened or not."""
    if isinstance(value, FeatureSet):
        return value.matrix
    return value.windows


def by_recording(array, lengths):
    """Cut an array of windows back into its recordings."""
    edges = np.cumsum([0] + list(lengths))
    return [array[a:b] for a, b in zip(edges[:-1], edges[1:])]


def predict_recordings(predictor, inputs, classification=True):
    """A model's outputs for each recording on its own.

    Parameters
    ----------
    predictor: EMGClassifier, EMGRegressor or a model
    inputs: list of arrays
        One per recording, one row per window.

    Returns
    ----------
    list of (outputs, predicted)
        ``outputs`` is class probabilities (windows, classes) for a
        classifier, or values (windows, dofs) for a regressor; ``predicted``
        is the class per window, or the values again.
    """
    model = getattr(predictor, "model", predictor)
    results = []
    for x in inputs:
        if hasattr(model, "reset"):
            model.reset()           # a fresh stream: no memory of the last recording
        if classification and hasattr(predictor, "_predict_proba"):
            probabilities = np.atleast_2d(np.asarray(predictor._predict_proba(x), dtype=float))
            classes = getattr(model, "classes_", None)
            index = probabilities.argmax(axis=1)
            predicted = np.asarray(classes)[index] if classes is not None else index
            results.append((probabilities, predicted))
        else:
            values = np.asarray(predictor._predict(x), dtype=float)
            values = values.reshape(len(x), -1)
            results.append((values, values))
    if hasattr(model, "reset"):
        model.reset()
    return results


def hold_out(lengths, labels=None, share=0.2, seed=0):
    """Recordings to hold out for validation: whole recordings, never windows.

    Neighbouring windows overlap and are nearly the same, so holding out
    windows would score a model on data it has effectively seen. With labels,
    each class gives up its share, so no class is missing from either side by
    chance.

    Parameters
    ----------
    lengths: list of int
        Windows in each recording.
    labels: array or None
        A label per window; a recording's class is its most common one.

    Returns
    ----------
    list of int
        Positions of the recordings held out, sorted. Empty if the share is
        zero or there are too few recordings to spare one.
    """
    count = len(lengths)
    if share <= 0 or count < 2:
        return []
    order = list(np.random.default_rng(seed).permutation(count))
    if labels is not None and np.ndim(labels) == 1:
        groups = {}
        for index, piece in zip(range(count), by_recording(np.asarray(labels), lengths)):
            values, counts = np.unique(piece, return_counts=True)
            groups.setdefault(values[np.argmax(counts)].item(), []).append(index)
        held = []
        for members in groups.values():
            members = [i for i in order if i in members]
            if len(members) > 1:
                held += members[:max(1, int(round(share * len(members))))]
        if held:
            return sorted(int(i) for i in held)
    take = min(max(1, int(round(share * count))), count - 1)
    return sorted(int(i) for i in order[:take])


# ----------------------------------------------------------------------
def _matching_files(folder, filters):
    """The files OfflineDataHandler.get_data reads, in the order it reads them."""
    found = []
    for pattern in ["*.csv", "*.txt", "*.hea"]:
        found.extend([y for x in os.walk(folder) for y in glob(os.path.join(x[0], pattern))])
    found = [Path(f).as_posix() for f in found]
    for regex_filter in filters:
        found = regex_filter.get_matching_files(found)
    return found


def _same_value(value, wanted):
    """Whether a recording's value is one of the wanted ones, '1' matching 1.0."""
    for candidate in wanted:
        if str(value) == str(candidate):
            return True
        try:
            if float(value) == float(candidate):
                return True
        except (TypeError, ValueError):
            continue
    return False
