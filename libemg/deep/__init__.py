"""Deep models in LibEMG: one way to train, run, stream and adapt them.

    from libemg.deep import DeepModel, TrainingSet

    training = TrainingSet.from_arrays(features, labels, lengths=windows_per_file)
    model = DeepModel("transformer", horizon=10).fit(
        training, on_epoch=lambda epoch: print(epoch.describe()))

=====================  ==================================================
:class:`TrainingSet`   recordings of windows and what they should produce
                       (:mod:`libemg.deep.data`)
:class:`DeepModel`     the model: fit, predict, stream, latent, adapt, save
                       (:mod:`libemg.deep.model`)
:class:`Trainer`       the training loop and its per-epoch reports
                       (:mod:`libemg.deep.training`)
:class:`SequenceMemory`  what live adaptation remembers
                       (:mod:`libemg.deep.memory`)
:class:`TrainingOutcome`  a training run kept as JSON: settings, data, every
                       epoch -- to export, reload and compare later
=====================  ==================================================

Architectures: 'mlp' on features, 'cnn' on the raw signal, 'transformer' over
a horizon of windows, 'gru' and 'lstm' with a memory of every window before.
Tasks: 'classification' or 'regression'. Timings: 'continuous' (an answer at
every window) or 'discrete' (one label per gesture).

PyTorch is needed only to train or run a model, and is imported only then.
"""

from libemg.deep.data import (CLASSIFICATION, CONTINUOUS, DISCRETE, REGRESSION, TASKS,
                              TIMINGS, Recording, TrainingSet)
from libemg.deep.memory import SequenceMemory
from libemg.deep.model import DeepModel
from libemg.deep.training import OUTCOME_SUFFIX, POOLINGS, EpochResult, Trainer, TrainingOutcome

#: The networks a DeepModel can be. Kept here as well as in
#: :mod:`libemg.deep.networks` so naming one does not import PyTorch.
ARCHITECTURES = ("mlp", "cnn", "transformer", "gru", "lstm")

__all__ = ["ARCHITECTURES", "CLASSIFICATION", "CONTINUOUS", "DISCRETE", "DeepModel",
           "EpochResult", "POOLINGS", "Recording", "REGRESSION", "SequenceMemory", "TASKS",
           "TIMINGS", "Trainer", "TrainingOutcome", "TrainingSet", "OUTCOME_SUFFIX"]
