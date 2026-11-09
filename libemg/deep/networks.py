"""The networks a :class:`~libemg.deep.model.DeepModel` can be, behind one interface.

Every network reads a batch of sequences of windows and answers at every step::

    outputs, embedding, state = network(inputs, state=None)

    inputs      (batch, steps, features) or (batch, steps, channels, samples)
    outputs     (batch, steps, outputs): logits per class, or values per DOF
    embedding   (batch, steps, embedding): the latent space the head reads,
                which is what the Feature Viewer draws as the model's own space
    state       what a stream carries from one call to the next: None for a
                network with no memory, the recurrent state for an RNN, the
                last ``horizon - 1`` windows for a transformer

Answering at every step is what lets one training loop serve every timing: a
continuous task scores every step, a discrete one pools the steps of a gesture
(see :class:`~libemg.deep.training.Trainer`). And carrying ``state`` is what
lets a stream be fed a window at a time, live, and get exactly the answers it
would have got fed all at once.

Torch is imported here and nowhere else in LibEMG's core, so it is only needed
by someone who trains a deep model.
"""

import math

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised only without torch
    raise ImportError("Deep models need PyTorch. Install it with `pip install torch`, "
                      "or install LibEMG with the deep extra: `pip install libemg[deep]`."
                      ) from error

MLP, CNN, TRANSFORMER, GRU, LSTM = "mlp", "cnn", "transformer", "gru", "lstm"
ARCHITECTURES = (MLP, CNN, TRANSFORMER, GRU, LSTM)


class _Head(nn.Module):
    """Embedding to outputs: one linear layer, so the embedding is the last thing learned."""

    def __init__(self, embedding_size, outputs):
        super().__init__()
        self.linear = nn.Linear(embedding_size, outputs)

    def forward(self, embedding):
        return self.linear(embedding)


def _flat(inputs):
    """(batch, steps, ...) -> (batch, steps, everything else), for a network that reads vectors."""
    return inputs.reshape(inputs.shape[0], inputs.shape[1], -1)


class MLPNetwork(nn.Module):
    """Linear layers on each window's features, with no memory of the windows before."""

    def __init__(self, input_size, outputs, hidden_size=64, layers=2, embedding_size=32,
                 dropout=0.1):
        super().__init__()
        blocks, width = [], input_size
        for _ in range(max(0, layers - 1)):
            blocks += [nn.Linear(width, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden_size
        blocks += [nn.Linear(width, embedding_size), nn.ReLU()]
        self.body = nn.Sequential(*blocks)
        self.head = _Head(embedding_size, outputs)

    def forward(self, inputs, state=None):
        embedding = self.body(_flat(inputs))
        return self.head(embedding), embedding, None


class CNNNetwork(nn.Module):
    """1D convolutions along each window's samples, every channel an input plane.

    Reads the signal itself rather than features computed from it, so it
    needs raw windows: ``(batch, steps, channels, samples)``.
    """

    def __init__(self, channels, samples, outputs, hidden_size=32, layers=3,
                 embedding_size=32, dropout=0.1, kernel=7):
        super().__init__()
        blocks, width = [], channels
        for _ in range(max(1, layers)):
            blocks += [nn.Conv1d(width, hidden_size, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(hidden_size), nn.ReLU(),
                       nn.MaxPool1d(2) if samples >= 8 else nn.Identity()]
            width = hidden_size
            samples //= 2
        self.convolutions = nn.Sequential(*blocks)
        self.project = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, embedding_size),
                                     nn.ReLU())
        self.head = _Head(embedding_size, outputs)

    def forward(self, inputs, state=None):
        if inputs.dim() != 4:
            raise ValueError("A CNN reads raw windows, shaped (batch, steps, channels, "
                             f"samples); got {tuple(inputs.shape)}. Feed it windows, not "
                             "features.")
        batch, steps = inputs.shape[:2]
        planes = self.convolutions(inputs.reshape(batch * steps, *inputs.shape[2:]))
        pooled = planes.mean(dim=-1)                    # over what is left of the samples
        embedding = self.project(pooled).reshape(batch, steps, -1)
        return self.head(embedding), embedding, None


class TransformerNetwork(nn.Module):
    """Attention over the last ``horizon`` windows, answering for the newest.

    Each step sees itself and the ``horizon - 1`` windows before it, never
    after: what is causal live has to be causal in training. Windows before the
    start of a stream are zeros, which is also what ``state`` starts as.
    """

    def __init__(self, input_size, outputs, embedding_size=32, horizon=10, heads=2,
                 layers=2, feedforward=64, dropout=0.1):
        super().__init__()
        self.horizon = int(horizon)
        self.input_size = input_size
        self.tokens = nn.Linear(input_size, embedding_size)
        position = torch.arange(self.horizon).unsqueeze(1)
        scale = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        encoding = torch.zeros(self.horizon, embedding_size)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)[:, :embedding_size // 2]
        self.register_buffer("position", encoding)
        layer = nn.TransformerEncoderLayer(embedding_size, heads, feedforward, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, max(1, layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_size)
        self.head = _Head(embedding_size, outputs)

    def forward(self, inputs, state=None):
        windows = _flat(inputs)
        batch, steps, width = windows.shape
        history = state if state is not None else windows.new_zeros(batch, self.horizon - 1, width)
        stream = torch.cat([history, windows], dim=1)            # (batch, H-1+steps, F)
        contexts = stream.unfold(1, self.horizon, 1)              # (batch, steps, F, H)
        contexts = contexts.permute(0, 1, 3, 2).reshape(batch * steps, self.horizon, width)
        encoded = self.encoder(self.tokens(contexts) + self.position)
        embedding = self.norm(encoded[:, -1]).reshape(batch, steps, -1)
        new_state = stream[:, -(self.horizon - 1):] if self.horizon > 1 else None
        return self.head(embedding), embedding, new_state


class RecurrentNetwork(nn.Module):
    """A GRU or LSTM: one window at a time, with a memory of every window before."""

    def __init__(self, input_size, outputs, hidden_size=64, layers=2, embedding_size=32,
                 dropout=0.1, cell=GRU):
        super().__init__()
        recurrent = {GRU: nn.GRU, LSTM: nn.LSTM}[cell]
        self.rnn = recurrent(input_size, hidden_size, max(1, layers), batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.project = nn.Sequential(nn.Linear(hidden_size, embedding_size), nn.ReLU())
        self.head = _Head(embedding_size, outputs)

    def forward(self, inputs, state=None):
        hidden, new_state = self.rnn(_flat(inputs), state)
        embedding = self.project(hidden)
        return self.head(embedding), embedding, new_state


def build_network(architecture, input_shape, outputs, hidden_size=64, layers=2,
                  embedding_size=32, dropout=0.1, horizon=10, heads=2):
    """The network ``architecture`` names, sized for windows of ``input_shape``.

    Parameters
    ----------
    architecture: str
        One of :data:`ARCHITECTURES`.
    input_shape: tuple
        One window: ``(features,)`` or ``(channels, samples)``. Anything but a
        CNN flattens a raw window into a vector.
    outputs: int
        Classes, or degrees of freedom.
    """
    architecture = architecture.lower()
    vector = int(math.prod(input_shape))
    if architecture == MLP:
        return MLPNetwork(vector, outputs, hidden_size, layers, embedding_size, dropout)
    if architecture == CNN:
        if len(input_shape) != 2:
            raise ValueError("A CNN reads the signal itself, so it needs raw windows "
                             "(channels, samples), not features. Connect it to a Window "
                             "block rather than a Features block.")
        return CNNNetwork(input_shape[0], input_shape[1], outputs, hidden_size, layers,
                          embedding_size, dropout)
    if architecture == TRANSFORMER:
        if embedding_size % heads:
            raise ValueError(f"The embedding size ({embedding_size}) has to divide evenly "
                             f"among the attention heads ({heads}).")
        return TransformerNetwork(vector, outputs, embedding_size, horizon, heads, layers,
                                  feedforward=2 * max(hidden_size, embedding_size),
                                  dropout=dropout)
    if architecture in (GRU, LSTM):
        return RecurrentNetwork(vector, outputs, hidden_size, layers, embedding_size, dropout,
                                cell=architecture)
    raise ValueError(f"Unknown architecture {architecture!r}. Choose one of {ARCHITECTURES}.")
