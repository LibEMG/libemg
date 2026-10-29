"""What live adaptation remembers, kept in the shape a deep model trains on.

The adaptation suite pairs what the model read with what the environment says
it should have produced: a CIIL pseudo-label, a class, a regression target. The
:class:`~libemg.adaptation.managers.MemoryManager` (or
:class:`~libemg.adaptation.hooks.MemoryHook`) appends each pair as it arrives
and saves a slice when a trial ends; the adaptation side adds the slices up and
hands the total to ``model.adapt(memory)``.

:class:`SequenceMemory` keeps every trial as its own recording, in the order
its windows arrived, so a model with memory -- a GRU, a transformer over a
horizon -- adapts on sequences as they happened, exactly as it was trained.

Examples
--------
::

    from libemg.adaptation.managers import AdaptationManager, MemoryManager
    from libemg.deep import DeepModel, SequenceMemory

    model = DeepModel.load('gru.pkl')            # trained offline first
    memory_manager = MemoryManager(SequenceMemory(), memory_smi, memory_ow, 'memory/')
    adaptation_manager = AdaptationManager(model, adapt_smi, adapt_ow, None,
                                           'memory/', 'models/')
"""

import pickle

import numpy as np

from libemg.adaptation.memory import Memory
from libemg.deep.data import CLASSIFICATION, DISCRETE, Recording, TrainingSet


class SequenceMemory(Memory):
    """Model inputs and their feedback, one sequence per trial."""

    def __init__(self):
        self._trials = []           # [(trial, [input rows], [feedback rows])], in arrival order

    # -- the memory manager's calls -------------------------------------------
    def append(self, feedback, input, trial=0):
        """Remember one window: what the model read, and what it should have produced.

        ``input`` may hold no rows when the model's input for that moment had
        already left shared memory; there is then nothing to learn from, and
        nothing is kept.
        """
        rows = np.atleast_2d(np.asarray(input, dtype=float))
        if rows.size == 0:
            return
        target = np.ravel(np.asarray(feedback, dtype=float))
        if not self._trials or self._trials[-1][0] != trial:
            self._trials.append((trial, [], []))
        for row in rows:
            self._trials[-1][1].append(row)
            self._trials[-1][2].append(target)

    def reset(self):
        self._trials = []

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            loaded = pickle.load(handle)
        self._trials = loaded._trials
        return self

    def __add__(self, other):
        combined = SequenceMemory()
        combined._trials = list(self._trials) + list(getattr(other, "_trials", []))
        return combined

    def __len__(self):
        return sum(len(inputs) for _, inputs, _ in self._trials)

    @property
    def processed_data(self):
        """Everything as two arrays, inputs and feedback, as the adaptation manager logs it."""
        if not self._trials:
            return [np.empty((0, 0)), np.empty((0, 0))]
        return [np.vstack([np.vstack(i) for _, i, _ in self._trials]),
                np.vstack([np.vstack(f) for _, _, f in self._trials])]

    # -- what a model adapts on --------------------------------------------------
    def training_set(self, task=CLASSIFICATION, timing="continuous"):
        """Every trial as a recording, its feedback as the targets.

        For classification the feedback's first value is the class; for
        regression the feedback is the target vector. A discrete task takes the
        trial's most common class as the gesture's label.
        """
        recordings = []
        for _, inputs, feedback in self._trials:
            if not inputs:
                continue
            inputs, feedback = np.vstack(inputs), np.vstack(feedback)
            if task == CLASSIFICATION:
                classes = np.round(feedback[:, 0]).astype(int)
                if timing == DISCRETE:
                    values, counts = np.unique(classes, return_counts=True)
                    classes = np.asarray(values[np.argmax(counts)])
                recordings.append(Recording(inputs, classes))
            else:
                recordings.append(Recording(inputs, feedback))
        return TrainingSet(recordings, task=task, timing=timing)
