"""The training loop every deep model shares, and what it reports as it goes.

One loop for every architecture and every task, because what differs between
them is only the loss:

    continuous classification   cross entropy at every real step
    continuous regression       mean squared error at every real step
    discrete classification     the gesture's steps pooled into one answer,
                                then cross entropy against its one label

Pooling is the discrete case's whole difficulty: the label says a gesture
happened in this recording, not where. ``max`` takes each class's strongest
step (multiple-instance learning: the gesture is evidenced somewhere),
``last`` takes the final step (the model answers once the gesture is over,
which suits a recurrent or transformer model that has seen all of it), and
``mean`` averages the steps.

Every epoch ends with an :class:`EpochResult` passed to ``on_epoch``. That is
what a window draws a live loss curve from, what a script prints, and what
:meth:`~libemg.deep.model.DeepModel.adapt` reports while adapting live. The
loop checks ``should_stop`` between batches, so a Stop button is answered
within one batch, and the best epoch's weights are what is kept either way.
"""

import copy
import time
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from libemg.deep.data import CLASSIFICATION, CONTINUOUS, DISCRETE

POOLINGS = ("max", "last", "mean")

#: The suffix a pipeline writes a training record under, beside its model.
OUTCOME_SUFFIX = ".training.json"


@dataclass
class EpochResult:
    """How one epoch went.

    Attributes
    ----------
    epoch: int
        Counting from 1.
    train_loss: float
    validation_loss: float or None
        None when nothing was held out.
    score: float or None
        On the validation set (or the training set when nothing was held
        out): accuracy for classification, RMSE for regression.
    score_name: str
        ``'accuracy'`` or ``'rmse'``.
    seconds: float
        How long the epoch took.
    best: bool
        Whether this is the best epoch so far, the one whose weights are kept.
    phase: str
        ``'train'``, or ``'adapt'`` for an epoch of live adaptation.
    """

    epoch: int
    train_loss: float
    validation_loss: float = None
    score: float = None
    score_name: str = ""
    seconds: float = 0.0
    best: bool = False
    phase: str = "train"

    def describe(self):
        parts = [f"epoch {self.epoch}", f"train loss {self.train_loss:.4f}"]
        if self.validation_loss is not None:
            parts.append(f"validation loss {self.validation_loss:.4f}")
        if self.score is not None:
            parts.append(f"{self.score_name} {self.score:.3f}" if self.score_name == "rmse"
                         else f"{self.score_name} {self.score:.1%}")
        return ", ".join(parts) + (" (best)" if self.best else "")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class TrainingOutcome:
    """A training run, kept: what was trained, on what, and how every epoch went.

    Saved as JSON, so it can be read back -- and drawn, compared, archived with
    a paper's results -- without PyTorch and without the model. A pipeline
    writes one beside every deep model it trains (``<model>.training.json``),
    and :meth:`load` also reads one out of a saved model, which carries its
    own history.

    Attributes
    ----------
    architecture, task, timing: str
    options: dict
        The model's network and training settings.
    data: dict
        What it was trained on: recordings, windows, input shape, classes or
        degrees of freedom, and the source folder when known.
    epochs: list of EpochResult
    stopped: bool
        Whether training was stopped before it would have ended by itself.
    model_path: str
        Where the trained model was saved, if it was.
    created: str
        When training finished, ISO 8601.
    name: str
        A label for comparing runs; defaults to the architecture and the time.
    """

    architecture: str
    task: str
    timing: str
    options: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    epochs: list = field(default_factory=list)
    stopped: bool = False
    model_path: str = ""
    created: str = ""
    name: str = ""

    def __post_init__(self):
        self.epochs = [e if isinstance(e, EpochResult) else EpochResult.from_dict(e)
                       for e in self.epochs]
        if not self.created:
            self.created = datetime.now().isoformat(timespec="seconds")
        if not self.name:
            self.name = f"{self.architecture.upper()} {self.created.replace('T', ' ')}"

    @property
    def best(self):
        """The epoch whose weights were kept, or None before the first epoch."""
        if not self.epochs:
            return None
        return min(self.epochs, key=lambda e: e.validation_loss
                   if e.validation_loss is not None else e.train_loss)

    def describe(self):
        best = self.best
        tail = f"; best epoch {best.epoch}: {best.describe()}" if best else ""
        return (f"{self.name}: {self.timing} {self.task} {self.architecture}, "
                f"{len(self.epochs)} epochs{' (stopped)' if self.stopped else ''}{tail}")

    def to_dict(self):
        raw = asdict(self)
        raw["epochs"] = [e.to_dict() for e in self.epochs]
        return raw

    @classmethod
    def from_dict(cls, raw):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path):
        """Write the record as JSON. Returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_jsonable))
        return path

    @classmethod
    def load(cls, path):
        """Read a record: a ``.json`` written by :meth:`save`, or a saved model.

        A model file -- a DeepModel, or an EMGClassifier/EMGRegressor wrapping
        one -- is unpickled and its own history read, which needs PyTorch. A
        JSON record does not.
        """
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_dict(json.loads(path.read_text()))
        import pickle
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        model = getattr(model, "model", model)
        if not hasattr(model, "outcome"):
            raise ValueError(f"{path.name} is not a deep model, so it has no training record.")
        return model.outcome(model_path=str(path))


def _jsonable(value):
    """What JSON cannot hold, as something it can: arrays as lists, the rest as text."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


@dataclass
class Trainer:
    """How to train: the optimizer, the schedule and when to stop.

    Parameters
    ----------
    epochs: int
        The most epochs to run.
    learning_rate, weight_decay: float
        For Adam.
    batch_size: int
        Sequences per batch.
    sequence_length: int
        Windows per training sequence, for a continuous task. A memoryless
        network gains nothing from more than 1; a transformer or recurrent one
        learns its temporal behaviour from them, so give it several times its
        horizon.
    patience: int or None
        Stop after this many epochs without a better validation loss. None
        runs every epoch.
    pooling: str
        How a discrete gesture's steps become one answer: 'max', 'last' or
        'mean'.
    warmup: int
        Steps at the start of each continuous sequence left out of the loss,
        so a model with memory is not scored before it has any.
    seed: int
    """

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    sequence_length: int = 32
    patience: int = 10
    pooling: str = "max"
    warmup: int = 0
    seed: int = 0
    history: list = field(default_factory=list)

    def fit(self, network, training, validation=None, task=CLASSIFICATION, timing=CONTINUOUS,
            prepare=None, on_epoch=None, should_stop=None, phase="train", restore_best=True):
        """Train ``network`` and keep its best weights.

        Parameters
        ----------
        network: torch.nn.Module
            One of :mod:`libemg.deep.networks`, or anything with the same call.
        training, validation: TrainingSet or None
        prepare: callable or None
            Turns a batch's inputs and targets into tensors: standardising,
            mapping class values to output indices. Given by DeepModel.
        on_epoch: callable or None
            Called with each :class:`EpochResult`.
        should_stop: callable or None
            Checked between batches; True ends training after that batch.

        Returns
        ----------
        list of EpochResult
        """
        import torch

        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, not {self.pooling!r}.")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        prepare = prepare or _as_tensors
        should_stop = should_stop or (lambda: False)
        best_loss, best_state, waited = np.inf, None, 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            started = time.perf_counter()
            network.train()
            losses, stopped = [], False
            for inputs, targets, mask in training.batches(self.sequence_length,
                                                          self.batch_size, rng):
                if should_stop():
                    stopped = True
                    break
                x, y, m = prepare(inputs, targets, mask)
                outputs, _, _ = network(x)
                loss = self._loss(outputs, y, m, task, timing)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if not losses:
                break
            judged = validation if validation is not None else training
            validation_loss, score, score_name = self.evaluate(network, judged, task, timing,
                                                               prepare)
            watched = validation_loss if validation is not None else float(np.mean(losses))
            best = watched < best_loss
            if best:
                best_loss, waited = watched, 0
                best_state = copy.deepcopy(network.state_dict())
            else:
                waited += 1
            result = EpochResult(epoch, float(np.mean(losses)),
                                 validation_loss if validation is not None else None,
                                 score, score_name, time.perf_counter() - started, best, phase)
            self.history.append(result)
            if on_epoch is not None:
                on_epoch(result)
            if stopped or should_stop():
                break
            if self.patience is not None and waited >= self.patience:
                break

        if restore_best and best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        return self.history

    def evaluate(self, network, data, task, timing, prepare=None):
        """Loss and score on ``data``, without training: ``(loss, score, score name)``."""
        import torch
        prepare = prepare or _as_tensors
        network.eval()
        losses, weights, right, total, squared = [], [], 0, 0, 0.0
        with torch.no_grad():
            for inputs, targets, mask in data.batches(self.sequence_length, self.batch_size,
                                                      shuffle=False):
                x, y, m = prepare(inputs, targets, mask)
                outputs, _, _ = network(x)
                losses.append(float(self._loss(outputs, y, m, task, timing)))
                weights.append(int(m.sum()))
                if task == CLASSIFICATION:
                    if timing == DISCRETE:
                        guess = self._pool(outputs, m).argmax(dim=-1)
                        right += int((guess == y).sum())
                        total += len(y)
                    else:
                        valid = self._scored(m)
                        right += int((outputs.argmax(dim=-1)[valid] == y[valid]).sum())
                        total += int(valid.sum())
                else:
                    valid = self._scored(m)
                    squared += float(((outputs[valid] - y[valid]) ** 2).sum())
                    total += int(valid.sum()) * outputs.shape[-1]
        loss = float(np.average(losses, weights=weights)) if losses else float("nan")
        if task == CLASSIFICATION:
            return loss, right / max(1, total), "accuracy"
        return loss, float(np.sqrt(squared / max(1, total))), "rmse"

    # -- losses --------------------------------------------------------------
    def _scored(self, mask):
        """Which steps count: the real ones, after the warm-up."""
        if self.warmup <= 0:
            return mask
        scored = mask.clone()
        scored[:, :self.warmup] = False
        return scored if scored.any() else mask

    def _pool(self, outputs, mask):
        import torch
        if self.pooling == "last":
            last = mask.long().sum(dim=1).clamp(min=1) - 1
            return outputs[torch.arange(len(outputs)), last]
        if self.pooling == "mean":
            weights = mask.unsqueeze(-1).float()
            return (outputs * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        hidden = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return hidden.max(dim=1).values

    def _loss(self, outputs, targets, mask, task, timing):
        import torch.nn.functional as F
        if timing == DISCRETE:
            return F.cross_entropy(self._pool(outputs, mask), targets)
        valid = self._scored(mask)
        if task == CLASSIFICATION:
            return F.cross_entropy(outputs[valid], targets[valid])
        return F.mse_loss(outputs[valid], targets[valid])


def _as_tensors(inputs, targets, mask):
    import torch
    targets = torch.as_tensor(targets)
    if targets.dtype in (torch.float64, torch.float16):
        targets = targets.float()
    return torch.as_tensor(inputs, dtype=torch.float32), targets, torch.as_tensor(mask)
