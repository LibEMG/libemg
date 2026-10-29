"""What a deep model is trained on: recordings of windows, and what they should produce.

A deep model in LibEMG reads a *sequence* of windows, not a bag of them. Even a
model that looks at one window at a time is trained on sequences, so that every
architecture -- a network on features, a CNN on the raw signal, a transformer
over a horizon of windows, a recurrent network carrying memory forward -- is
fed the same way and differs only in what it does with the time axis.

The unit is a :class:`Recording`: one uninterrupted stretch of windows, in the
order they were taken. Windows from two different files are never stitched into
one sequence, because the join between them is not something the signal ever
did, and a temporal model would learn it.

What a recording should produce depends on the task:

    continuous   every window has a target -- the motion being held, or the
                 regression values for that moment. The model answers at every
                 step.
    discrete     the recording is one gesture with one label, and nothing says
                 which windows the gesture is in. The model answers once the
                 gesture is over (see ``pooling`` in
                 :class:`~libemg.deep.training.Trainer`).

and on what it predicts: a class (``classification``) or one value per degree
of freedom (``regression``).
"""

from dataclasses import dataclass

import numpy as np

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASKS = (CLASSIFICATION, REGRESSION)

CONTINUOUS = "continuous"
DISCRETE = "discrete"
TIMINGS = (CONTINUOUS, DISCRETE)


@dataclass
class Recording:
    """One uninterrupted stretch of windows, and what it should produce.

    Attributes
    ----------
    inputs: array
        One row per window, in time order: ``(windows, features)`` for
        features, or ``(windows, channels, samples)`` for the raw signal.
    targets: array
        For a continuous task, one per window: ``(windows,)`` class indices or
        ``(windows, dofs)`` values. For a discrete task, the gesture's single
        class index.
    """

    inputs: np.ndarray
    targets: np.ndarray

    def __len__(self):
        return len(self.inputs)


class TrainingSet:
    """Recordings to train on, and the task they describe.

    Parameters
    ----------
    recordings: list of Recording
    task: str
        ``'classification'`` or ``'regression'``.
    timing: str
        ``'continuous'`` (a target per window) or ``'discrete'`` (one label
        per recording, which is one gesture).

    Examples
    --------
    From the arrays an offline run produces, split back into its files::

        training = TrainingSet.from_arrays(features, labels, lengths=windows_per_file)
        model = DeepModel("gru").fit(training)
    """

    def __init__(self, recordings, task=CLASSIFICATION, timing=CONTINUOUS):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, not {task!r}.")
        if timing not in TIMINGS:
            raise ValueError(f"timing must be one of {TIMINGS}, not {timing!r}.")
        if timing == DISCRETE and task == REGRESSION:
            raise ValueError("A discrete task is a gesture with one class; regression is "
                             "continuous by nature. Use timing='continuous'.")
        self.recordings = [r for r in recordings if len(r)]
        #: Positions of these recordings in the set they were split from, if any.
        self.source_indices = list(range(len(self.recordings)))
        if not self.recordings:
            raise ValueError("There are no windows to train on.")
        self.task, self.timing = task, timing
        shapes = {r.inputs.shape[1:] for r in self.recordings}
        if len(shapes) != 1:
            raise ValueError(f"Every window has to have the same shape; found {sorted(shapes)}.")
        self.input_shape = shapes.pop()
        for r in self.recordings:
            if timing == CONTINUOUS and len(r.targets) != len(r.inputs):
                raise ValueError("A continuous task needs a target for every window: "
                                 f"got {len(r.targets)} targets for {len(r.inputs)} windows.")

    # -- what it holds -----------------------------------------------------
    @property
    def windows(self):
        return sum(len(r) for r in self.recordings)

    @property
    def classes(self):
        """The class indices present, sorted. Empty for regression."""
        if self.task != CLASSIFICATION:
            return []
        values = np.concatenate([np.ravel(r.targets) for r in self.recordings])
        return sorted(int(v) for v in np.unique(values))

    @property
    def dofs(self):
        """Values per target, for regression. 0 for classification."""
        if self.task != REGRESSION:
            return 0
        targets = self.recordings[0].targets
        return 1 if np.ndim(targets) == 1 else int(np.shape(targets)[1])

    def describe(self):
        what = (f"{len(self.classes)} classes" if self.task == CLASSIFICATION
                else f"{self.dofs} degree(s) of freedom")
        return (f"{len(self.recordings)} recordings, {self.windows} windows of shape "
                f"{self.input_shape}, {self.timing} {self.task} over {what}")

    # -- building one ------------------------------------------------------
    @classmethod
    def from_arrays(cls, inputs, targets, lengths=None, task=CLASSIFICATION,
                    timing=CONTINUOUS):
        """Split arrays laid end to end back into the recordings they came from.

        Parameters
        ----------
        inputs: array (windows, ...)
        targets: array
            One per window, laid out like ``inputs``. For a discrete task the
            recording's label is the most common one among its windows.
        lengths: list of int or None
            Windows in each recording, in order. None treats everything as one
            recording -- right only when it really was recorded in one piece.
        """
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        if task == REGRESSION:
            targets = targets.astype(float)
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)
        else:
            targets = targets.reshape(len(targets), -1)[:, 0].astype(int)
        lengths = [len(inputs)] if lengths is None else [int(n) for n in lengths]
        if sum(lengths) != len(inputs):
            raise ValueError(f"The lengths add up to {sum(lengths)} windows, but there are "
                             f"{len(inputs)}.")
        recordings, start = [], 0
        for n in lengths:
            piece = targets[start:start + n]
            if timing == DISCRETE and n:
                values, counts = np.unique(piece, return_counts=True)
                piece = np.asarray(values[np.argmax(counts)])
            recordings.append(Recording(inputs[start:start + n], piece))
            start += n
        return cls(recordings, task=task, timing=timing)

    def split(self, validation_fraction=0.2, seed=0):
        """Hold whole recordings out for validation.

        Never a random subset of windows: neighbouring windows overlap and are
        nearly the same, so a window-level split would score the model on data
        it has effectively already seen. Returns ``(training, validation)``;
        validation is None when there are too few recordings to spare one.
        """
        if validation_fraction <= 0 or len(self.recordings) < 2:
            return self, None
        order = np.random.default_rng(seed).permutation(len(self.recordings))
        held = max(1, int(round(validation_fraction * len(self.recordings))))
        held = min(held, len(self.recordings) - 1)
        if self.task == CLASSIFICATION and self.timing == DISCRETE:
            # Stratified, so a class is not missing from validation by chance.
            by_class = {}
            for i in order:
                by_class.setdefault(int(self.recordings[i].targets), []).append(i)
            chosen = []
            for indices in by_class.values():
                take = max(1, int(round(validation_fraction * len(indices)))) \
                    if len(indices) > 1 else 0
                chosen += indices[:take]
            held_out = set(chosen) if chosen else set(order[:held])
        else:
            held_out = set(order[:held].tolist())
        train = [r for i, r in enumerate(self.recordings) if i not in held_out]
        valid = [r for i, r in enumerate(self.recordings) if i in held_out]
        training = TrainingSet(train, self.task, self.timing)
        validation = TrainingSet(valid, self.task, self.timing)
        # Which recordings went where, by position in this set: what lets a
        # model's predictions be shown later as on training or held-out data.
        validation.source_indices = sorted(int(i) for i in held_out)
        training.source_indices = [i for i in range(len(self.recordings)) if i not in held_out]
        return training, validation

    # -- feeding a network ---------------------------------------------------
    def batches(self, sequence_length, batch_size, rng=None, shuffle=True):
        """Batches of sequences, covering every window about once.

        Continuous: crops of ``sequence_length`` windows from random places in
        random recordings. Discrete: whole recordings, since the label belongs
        to the whole gesture. Shorter sequences are padded at the end and the
        mask says which steps are real.

        Yields
        ------
        inputs: float32 array (batch, steps, ...)
        targets: array -- (batch, steps[, dofs]) continuous, (batch,) discrete
        mask: bool array (batch, steps)
        """
        rng = rng or np.random.default_rng(0)
        if self.timing == DISCRETE:
            order = rng.permutation(len(self.recordings)) if shuffle \
                else np.arange(len(self.recordings))
            for start in range(0, len(order), batch_size):
                chosen = [self.recordings[i] for i in order[start:start + batch_size]]
                steps = max(len(r) for r in chosen)
                inputs, mask = self._padded([r.inputs for r in chosen], steps)
                yield inputs, np.array([int(r.targets) for r in chosen]), mask
            return

        crops = []
        for index, recording in enumerate(self.recordings):
            length = min(sequence_length, len(recording))
            count = max(1, int(np.ceil(len(recording) / max(1, sequence_length))))
            if shuffle:
                starts = rng.integers(0, len(recording) - length + 1, size=count)
            else:
                starts = np.arange(0, len(recording), length)[:count]
            crops += [(index, int(s), length) for s in starts]
        if shuffle:
            crops = [crops[i] for i in rng.permutation(len(crops))]
        for start in range(0, len(crops), batch_size):
            chosen = crops[start:start + batch_size]
            steps = max(length for _, _, length in chosen)
            pieces = [self.recordings[i].inputs[s:s + n] for i, s, n in chosen]
            targets = [self.recordings[i].targets[s:s + n] for i, s, n in chosen]
            inputs, mask = self._padded(pieces, steps)
            padded_targets, _ = self._padded(targets, steps, dtype=targets[0].dtype)
            yield inputs, padded_targets, mask

    @staticmethod
    def _padded(pieces, steps, dtype=np.float32):
        shape = (len(pieces), steps) + tuple(np.shape(pieces[0])[1:])
        out = np.zeros(shape, dtype=dtype)
        mask = np.zeros((len(pieces), steps), dtype=bool)
        for i, piece in enumerate(pieces):
            out[i, :len(piece)] = piece
            mask[i, :len(piece)] = True
        return out, mask
