`libemg.deep` is one way to train, run, stream and adapt neural networks in LibEMG. The same model object is trained offline with a live report per epoch, runs over a stored recording, runs live one window at a time, shows its latent space in the pipeline's Feature Viewer, and adapts during a session through the adaptation managers.

It needs PyTorch, which is imported only when a model is trained or run.

```
pip install libemg[deep]
```

# A first model

```Python
from libemg.deep import DeepModel, TrainingSet

# features: (windows, features); labels: one per window; lengths: windows per file
training = TrainingSet.from_arrays(features, labels, lengths=windows_per_file)

model = DeepModel("gru", sequence_length=64)
model.fit(training, on_epoch=lambda epoch: print(epoch.describe()))
# epoch 1, train loss 1.0921, validation loss 1.0544, accuracy 61.2% (best)
# ...

model.predict(test_features)       # a class per window
model.latent(test_features)        # the embedding the output reads
model.save("gru.pkl")
```

# Recordings, not bags of windows

A deep model is trained on *sequences* of windows, so every architecture is fed the same way, whether it uses the time axis or not. The unit is a `Recording`: one uninterrupted stretch of windows in the order they were taken, with what it should produce. `TrainingSet.from_arrays` splits arrays laid end to end back into their recordings using `lengths`. Windows from two files are never stitched into one sequence: the join was never in the signal, and a temporal model would learn it.

Validation holds out whole recordings, never windows. Neighbouring windows overlap and are nearly identical, so a window-level split scores a model on data it has effectively seen.

# Architectures

| Architecture | Reads | Uses the time axis by |
| --- | --- | --- |
| `mlp` | Features | Not at all: each window on its own |
| `cnn` | Raw windows, `(channels, samples)` | Not at all: convolutions along each window's samples |
| `transformer` | Features | Attending over the last `horizon` windows, the newest included |
| `gru`, `lstm` | Features | A memory of every window before, carried step to step |

Every network answers at every step and exposes the embedding its output layer reads. That embedding is `DeepModel.latent`, and it is what the Feature Viewer draws as the model's own space.

# Tasks and timing

| `task` | `timing` | Target | Loss |
| --- | --- | --- | --- |
| `classification` | `continuous` | A class per window: the motion being held | Cross entropy at every window |
| `regression` | `continuous` | Values per window, one per degree of freedom | Mean squared error at every window |
| `classification` | `discrete` | One class per recording, which is one gesture | The gesture's windows pooled into one answer, then cross entropy |

A discrete label says a gesture happened in this recording, not where. `pooling` says how the gesture's windows become one answer.

| `pooling` | Meaning | Suits |
| --- | --- | --- |
| `max` | Each class's strongest window: the gesture is evidenced somewhere (multiple-instance learning) | Any architecture |
| `last` | The final window: the model answers once the gesture is over | A GRU, LSTM or transformer, which has seen all of it by then |
| `mean` | The average over the gesture | Gestures that last their whole recording |

```Python
training = TrainingSet.from_arrays(windows, labels, lengths=windows_per_gesture,
                                   timing="discrete")
model = DeepModel("cnn", timing="discrete", pooling="max").fit(training)
```

# Training, and watching it

`fit` reports each epoch as an `EpochResult`: the training loss, the validation loss, the validation score (accuracy, or RMSE for regression), the time it took, and whether it is the best so far. `should_stop` is checked between batches. Whether training ends by stopping, patience or the last epoch, the best epoch's weights are the ones kept.

| `DeepModel` option | Default | What it does |
| --- | --- | --- |
| `epochs` | 50 | The most epochs to run |
| `learning_rate`, `weight_decay` | 1e-3, 1e-4 | For Adam |
| `batch_size` | 32 | Sequences per batch |
| `sequence_length` | 32 | Windows per training sequence. A model with memory learns its timing from these: give it several times its horizon |
| `patience` | 10 | Epochs without a better validation loss before stopping. None runs every epoch |
| `warmup` | 0 | Steps at the start of each sequence left out of the loss |
| `hidden_size`, `layers`, `embedding_size`, `dropout` | 64, 2, 32, 0.1 | Network size |
| `horizon`, `heads` | 10, 2 | A transformer's context and attention heads |

`evaluate(training_set)` scores a trained model on data it has not seen: per window for a continuous task, per gesture for a discrete one.

When `fit` splits off the validation recordings itself (`validation=0.2`, say), it records which ones in `model.trained_on["validation_indices"]`: their positions among the recordings it was given. That is what lets a Prediction Viewer block show the model's predictions on its training data and on the data it was validated on, separately.

# Running live

Live, a model is fed one window at a time. A model with memory has to carry it from one call to the next.

```Python
model.streaming = True
model.reset()                           # a new stream
for features in live_windows:
    probabilities = model.predict_proba(features[None])
```

A stream fed one window at a time gets exactly what it would have got fed all at once, for every architecture. Wrapped in an `EMGClassifier` or `EMGRegressor`, a `DeepModel` works anywhere those do. The pipeline's live model hook switches streaming on when it loads one.

# Adapting live

The adaptation managers make two calls on a model: `adapt(memory)`, then `save(path)`. `DeepModel` answers both. `adapt` runs `adapt_epochs` epochs (3 by default) at a tenth of the training learning rate, so it nudges the model rather than retraining it on one trial. It reports each epoch through `on_epoch`, with `phase='adapt'`, and returns the losses the manager logs.

`SequenceMemory` is the memory to give the memory manager. It keeps each trial as its own recording, in the order its windows arrived, so a model with memory adapts on sequences as they happened.

```Python
from libemg.adaptation.managers import AdaptationManager, MemoryManager
from libemg.deep import DeepModel, SequenceMemory

model = DeepModel.load("gru.pkl")      # trained offline first
memory_manager = MemoryManager(SequenceMemory(), memory_smi, memory_ow, "memory/")
adaptation_manager = AdaptationManager(model, adapt_smi, adapt_ow, None,
                                       "memory/", "models/")
```

For classification the feedback's first value is the class. For regression the feedback is the target, which is where a CIIL pseudo-label goes. Feedback for a class the model has no output for is left out: adapting cannot add a class.

# In the pipeline editor

The **Deep Model** block trains any of these on stored data. Connect Train Features for an MLP, transformer, GRU or LSTM, or a Window block's Output to Train Windows for a CNN, and that Window's Metadata to Train Labels; then press **Train** on the block. A window draws the training and validation loss as each epoch ends, with the best epoch marked and the validation score below. **Stop** ends training after the current batch and keeps the best epoch so far. **Statistics** shows training and validation accuracy, per class, and the loss of every epoch; **Save** writes the model to **Fitted Model**, with its training record beside it. Connect Features (or Windows) as well, from another branch of a Split block or the same one, and the same document runs the model, offline or live (see [Training a model in the editor](../pipeline/pipeline_doc.md#training-a-model-in-the-editor)). Add a **Prediction Viewer** block on the model's output to see what it predicted over every recording, training and held out alike (see [Looking at the predictions](../pipeline/pipeline_doc.md#looking-at-the-predictions)).
