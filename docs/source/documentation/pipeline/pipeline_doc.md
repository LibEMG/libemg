A myoelectric control pipeline is a chain of stages. Samples are filtered, cut into windows, reduced to features, handed to a model, and the model's output is sent somewhere useful. Writing that chain out in code means knowing which class each stage belongs to, and which argument carries the window size, before anything can be run at all.

The pipeline editor is a node editor for that chain. Open the LibEMG GUI, choose **Pipeline**, then **Pipeline Editor**. Blocks are placed from a palette on the left and connected by dragging from one port to another. The editor creates its own sources, so it opens with no `OnlineDataHandler` and no hardware attached.

```Python
from libemg.gui import GUI

if __name__ == "__main__":
    # No handler is passed. The pipeline editor supplies its own sources.
    GUI().start_gui()
```

# Three layers, and why the editor is the thin one

A pipeline is built from three modules, and only the last one needs a display.

| Layer | Module | What it is |
| --- | --- | --- |
| Registry | `libemg._gui._pipeline.registry` | A description of every block that can be placed. |
| Document | `libemg._gui._pipeline.document` | The pipeline itself, as plain data. |
| Compiler | `libemg._gui._pipeline.compile` | Turns a document into something that runs. |

The registry is generated from the library rather than hand-listed. Feature names come from the feature extractor, metric names from the offline metrics, model names from the predictors, and each device's parameters from its own signature. A block added to LibEMG appears in the palette without the registry being edited.

The document holds the nodes, the links, the probes and the canvas positions. It imports neither DearPyGui nor the LibEMG runtime.

The consequence matters more than the structure. A pipeline is plain data, so it can be built, saved, validated and run with no display at all. The editor is a view onto a document, not the thing itself.

# Building a pipeline without the editor

The headless route is the same document the editor edits. This builds a live pipeline, compiles it and runs it.

```Python
from libemg._gui._pipeline import PipelineDocument
from libemg._gui._pipeline.compile import compile_pipeline

doc = PipelineDocument()
src = doc.add_node('source.synthetic_streamer',
                   params={'num_channels': 4, 'sampling_rate': 1000,
                           'pattern': 'bursts'})
flt = doc.add_node('transform.filter',
                   params={'name': 'bandpass', 'cutoff': '20,450',
                           'order': 4, 'sampling_rate': 1000})
win = doc.add_node('window.enframe',
                   params={'window_size': 200, 'window_increment': 50})
fea = doc.add_node('features.extract', params={'features': ['MAV', 'RMS']})
mdl = doc.add_node('model.classifier', params={'model_path': 'mdl.pkl'})
snk = doc.add_node('sink.console')

doc.connect(src, 'emg', flt, 'input')
doc.connect(flt, 'output', win, 'input')
doc.connect(win, 'output', fea, 'input')
doc.connect(fea, 'output', mdl, 'input')
doc.connect(mdl, 'output', snk, 'input')

print(doc.mode(), doc.validate())
# online []

doc.save('pipeline.json')
pipeline = compile_pipeline(doc)
pipeline.start()
```

`validate()` returns every reason a pipeline could not run, as sentences. An empty list means it is ready. `check()` raises `ValidationError` instead, for a script that should stop. This is what makes a pipeline testable without a person watching it.

# The blocks

Blocks are grouped in the palette by category.

| Category | Block | Consumes | Produces | Notes |
| --- | --- | --- | --- | --- |
| Source | Device streamers | | samples | One block per entry in `libemg.streamers`. Live only. |
| Source | Synthetic Source | | samples | Generated samples, no hardware. Live only. |
| Source | Stored Data | | samples, labels | Replays recordings from disk, with labels taken from the filenames or from a file packaged beside them. Stored only. |
| Transform | Filter | samples | samples | Conditions the signal once, for every consumer downstream. |
| Transform | Channel Mask | samples | samples | Narrows the stream to a subset of channels. |
| Window | Window | samples | windows | Sets the window size and increment. Never a stage of its own. |
| Features | Features | windows | features | Extracts features once, so several models can share them. |
| Model | Classifier | features or windows | prediction | Runs a fitted classifier. |
| Model | Regressor | features or windows | continuous | Runs a fitted regressor. |
| Model | Deep Model | features or windows | prediction | Trains a neural network (MLP, CNN, transformer, GRU, LSTM) with its loss drawn live, and runs it, streaming, live. See Deep Models. |
| Sink | Socket Output | prediction | | Sends each output over UDP or TCP. Live only. |
| Sink | File Output | prediction | | Appends each output to a file. |
| Sink | Console Output | prediction | | Prints each output, while a pipeline is being built up. |
| Sink | Offline Metrics | prediction, labels | metrics | Scores predictions against ground truth. Stored only. |
| Sink | Feature Viewer | features, labels and a model (both optional) | | Draws the features, and a connected model's latent space, once the recording has been read. Stored only. See [Looking at the features](#looking-at-the-features). |
| Sink | Prediction Viewer | a model's output, labels (optional) | held-out field and values | Draws what the model predicted over the signal, recording by recording or as a stream, on training and held-out data. Stored only. See [Looking at the predictions](#looking-at-the-predictions). |

The device source blocks in this release of LibEMG are:

| Block id |
| --- |
| `source.delsys_streamer` |
| `source.delsys_api_streamer` |
| `source.emager_streamer` |
| `source.leap_streamer` |
| `source.myo_streamer` |
| `source.oymotion_streamer` |
| `source.sifi_bioarmband_streamer` |
| `source.sifi_biopoint_streamer` |

Each one carries the keyword arguments of the streamer function it was generated from. Those appear under an **Advanced** disclosure on the block, because they have sensible defaults and rarely need touching.

# Port types, and why a wrong connection cannot be made

Every port carries one of seven types.

| Port type | Carries |
| --- | --- |
| `samples` | An N by C continuous stream. |
| `windows` | Enframed windows. |
| `features` | An N by F feature matrix. |
| `prediction` | A class index with its confidence. |
| `continuous` | A regression output vector. |
| `labels` | Ground truth, stored data only. |
| `metrics` | A summary result table. |

The port type becomes the `category` on the underlying DearPyGui node attribute, and DearPyGui refuses to link two attributes whose categories differ. A features input simply will not accept a samples output as the link is dragged. Typing is enforced by the toolkit while the user drags, not by validation that runs afterwards and has to explain itself.

Three rules cannot be expressed as a category. Those are checked when the link is released, and reported in words.

| Situation | What the editor says |
| --- | --- |
| A block linked to itself | `A block cannot feed itself.` |
| An input that already has a link | `'Input' already has a connection, and two sources into one input would interleave with no defined order.` |
| A link that would close a loop | `That would make a loop, and each stage would wait for the other.` |

A type mismatch is still explained if it reaches the document, for instance when the document is built in Python rather than dragged:

```Python
doc.why_not_connect(src, 'emg', fea, 'input')
# "'EMG' carries samples and 'Input' expects windows."
```

`why_not_connect` returns `None` when a link would be accepted. `connect` raises `ValueError` with the same sentence.

# The window block is fused, not a stage

This is the central design point of the compiler. A window block does not become a stage. What it configures is how the next stage observes the one before it.

| Window parameter | What it becomes |
| --- | --- |
| `window_increment` | The criterion on the next stage, as `OnSamples(increment)`. |
| `window_size` | The number of rows that stage is handed, in window mode. |

So a Window between a Filter and a Features block compiles to the Features block observing the Filter directly. Nothing copies an enframed array into shared memory just to hand it along. The criteria and modes referred to here are the ones described in the [reactive pipelines guide](../reactive/reactive_doc.md).

The document above compiles to this graph. It is the compiler's own description, printed from a real run:

```
executor 'filter_2':
  filter_2: observes pipe_synthetic_streamer_1_emg [OnSamples(1), window(64)] -> writes pipe_filter_2_output
executor 'extract_4':
  extract_4: observes pipe_filter_2_output [OnSamples(50), window(200)] -> writes pipe_extract_4_output
executor 'classifier_5':
  classifier_5: observes pipe_extract_4_output [OnSamples(1), latest] -> writes pipe_classifier_5_output
executor 'console_6':
  console_6: observes pipe_classifier_5_output [OnCommit(), latest] -> writes -
executor 'probe_filter_2':
  probe_filter_2_output: observes pipe_filter_2_output [Periodic(20), window(400)] -> writes probe_filter_2_output
executor 'probe_classifier_5':
  probe_classifier_5_output: observes pipe_classifier_5_output [Periodic(20), latest] -> writes probe_classifier_5_output
```

There are six blocks in the document and no stage named after the window. Its two numbers reappear as `OnSamples(50), window(200)` on the features stage. That is the whole of its effect.

Because a window is only ever a pair of numbers on the stage it feeds, it has no online-or-offline mode to set. Live, those numbers trigger the next stage when an increment has accrued. Over stored data, the same numbers enframe a recording in batches.

Running that pipeline for six seconds on a synthetic source produced the following, and the counts are what the fusion predicts:

| Item | Rows committed |
| --- | --- |
| `pipe_synthetic_streamer_1_emg` | 6287 |
| `pipe_filter_2_output` | 6286 |
| `pipe_extract_4_output` | 125 |
| `pipe_classifier_5_output` | 125 |

6287 samples at an increment of 50 gives 125 windows. Every window produced one feature row, and every feature row produced one prediction.

# A model reads features, or raw windows

A model block has two inputs and takes exactly one of them. Connect features for a statistical model. Connect a window directly for a model that works on raw windows, which is the usual shape for a deep model.

| Connected to | What the model observes |
| --- | --- |
| Features | One feature row at a time |
| Window | The window itself, on the window's increment |

Connecting both is refused, because a model reads one or the other. Connecting neither is refused too.

```Python
# a deep model, straight off the window
doc.connect(window, 'output', model, 'windows')
```

# Standardize is a stored-data filter

Standardizing subtracts a mean and divides by a deviation, and both have to be measured from data. A recording supplies them, so the filter works offline. A live stream has nothing to measure before it starts, so choosing Standardize on a live pipeline is refused while compiling rather than failing later inside a running stage.

| Pipeline | Standardize |
| --- | --- |
| Stored data | Works. The loaded recording supplies the statistics. |
| Live stream | Refused at compile time, with the reason and the alternatives. |

# Online or offline is inferred, never configured

A pipeline's mode comes from its sources.

| Sources | `doc.mode()` |
| --- | --- |
| No source at all | `empty` |
| Live devices or the synthetic source | `online` |
| Stored Data only | `offline` |
| Both kinds at once | `mixed` |

There is no mode setting anywhere in the editor. A user who had to declare the mode could declare it wrongly, and a pipeline built for a live stream but told to read a recording would start cleanly and then silently do nothing. Inferring it makes that mistake unrepresentable.

Mixing the two is an error, and validation names the blocks on each side:

```
This mixes live sources (synthetic_streamer_1) with stored data (offline_2).
A run is either one or the other.
```

Blocks that only make sense in one mode are checked against the inferred mode too. A Socket Output in a stored-data pipeline is reported as only working on a live stream.

# Probes

A probe watches one output port while the pipeline runs. It is a toggle on the port, not a block to place. Making it a block would clutter the canvas and force the user to wire up something they only want to look at.

How a probe draws is derived from the type of the port it watches, so probing is one click and there is nothing to configure.

| Port type | Rendering |
| --- | --- |
| `samples` | Time series |
| `continuous` | Time series |
| `windows` | Window overlay |
| `features` | Bars |
| `prediction` | Probabilities |
| `labels` | Time series |
| `metrics` | Table |

Probes open in a **Scope** window alongside the editor when a live pipeline starts. Each one is rate limited, by default to 30 updates per second, and publishes into its own shared-memory item rather than calling back into the pipeline. A probe that cannot keep up with its source is still a probe. It shows less, and it can never stall what it watches.

Two probes were set at 20 Hz on the six-second run above. Neither held its stage back.

| Stage or probe | Times it ran |
| --- | --- |
| `filter_2`, the stage being watched | 6291 |
| `probe_filter_2_output` | 120 |
| `probe_classifier_5_output` | 75 |

```Python
doc.add_probe(flt, 'output', hz=20)
doc.add_probe(mdl, 'output', hz=20)
doc.probe_render(doc.probes[0])
# 'timeseries'
```

A window cannot be probed. It is folded into the block it feeds rather than becoming a stage, so it publishes nothing to watch. Asking for one is refused straight away, with a suggestion:

```
A window cannot be probed. It is folded into the block it feeds rather than
producing anything of its own, so there is nothing to watch. Probe the block
before it to see the samples going in, or the block after it to see what
comes out.
```

Refusing at the point it is asked for matters. Accepting the probe and dropping it when the pipeline compiled would leave an empty plot with no explanation, and would also let it count as a reader of a branch that nothing actually reads.

# Running

Both modes get a **Start** and a **Stop** button, and both report what they are doing under the canvas. What they can honestly report differs.

| | Live pipeline | Stored-data pipeline |
| --- | --- | --- |
| Started by | Start | Start |
| Stopped by | Stop | Stop |
| Progress shown as | A status line of rows committed per stage | A progress bar |
| Probes | A Scope window | Not applicable |
| Finishes into | Nothing. It runs until stopped | A Results window |

A live run has no total, because a stream has no end. The status line reports rows committed per stage instead, which is the honest thing a live run knows. A recording knows its own length, so the bar is a real fraction rather than a spinner.

The stored-data run reports its progress as it works through the files. Running the pipeline above over a two-file recording gave:

```Python
results = pipeline.run(on_progress=print)
# 0.5
# 1.0
# 1.0
```

A cancelled run says so rather than claiming to have finished. Stopping part way leaves the fraction where it stopped, and `stopped` records that it was cut short.

| After a run | `stopped` | `progress` | `results` |
| --- | --- | --- | --- |
| Finished | False | 1.0 | Over every recording |
| Stopped part way | True | Where it stopped | Over the recordings it read |

The metrics from a cancelled run are kept, because throwing away work already done is worse than reporting less. They are computed over part of the recording, which is exactly why the run has to be distinguishable from a complete one.

The fraction is reported once per recording read, and once more when the run completes.

When it finishes, the Results window opens with a metrics table and a heat map of any confusion matrix. **Copy as CSV** puts the whole table on the clipboard. The same run, scored on the data it was fitted on:

| Metric | Value |
| --- | --- |
| CA | 1.0 |
| RECALL | 1.0 |

A run in progress can be asked to finish early with **Stop**, or with `request_stop` from a script.

Compilation happens when Start is pressed, and anything wrong is reported before a single process is spawned. The messages name the block and say what to do:

```
This pipeline cannot run yet:
  - 'Offline Metrics' (metrics_3) has nothing connected to its Input input.
  - 'Offline Metrics' (metrics_3) has nothing connected to its Labels input.
  - 'Window' (enframe_2) produces something that nothing reads and no probe watches.
```

# Looking at the features

Before fitting a model it is worth knowing whether the classes come apart in the feature space at all, and which features pull them apart. The **Feature Viewer** block answers that. Connect a Features block's output to its Features input, and the Stored Data block's Labels to its Labels input, then press **Start**. Once the recording has been read, a window opens on the feature matrix: one point per window, coloured by its label.

```
Stored Data --EMG--> Window --> Features --> Feature Viewer
            --Labels-------------------------------^
```

The view is changed in that window, not on the block, so flipping between views does not read the recording again. The block's own settings are only where the window starts.

| Method | What is drawn |
| --- | --- |
| Columns | Feature columns as they are. 1D: a histogram per class of the column chosen as X. 2D and 3D: the X, Y and Z columns against each other. |
| PCA | The directions of greatest variance. Each axis says how much of the variance it holds. |
| ICA | Statistically independent directions (FastICA). |
| Isomap | A nonlinear unfolding that keeps neighbouring windows close. |

With labels connected, **Most telling columns** ranks every column on its own by how well it separates them: the ANOVA F score for classes, the absolute correlation for a regression target (the chosen degree of freedom, for several). Clicking one plots it, and the one plotted before moves to Y, so clicking the top two shows the best pair at once. With a feature set such as WENG, 72 columns over 8 channels, this is the quick way to the few worth looking at.

Every column is named for its feature and channel, as `MAV ch1`, `WL ch3`, in the order the matrix holds them. A feature with more than one column per channel, such as `AR`, is named by position: `AR [1]`, `AR [2]`. **Standardize** scales each column to unit variance before projecting. Without it a projection is mostly a measure of which feature has the largest numbers.

| Colour by | Used for |
| --- | --- |
| Class | One colour per class, with a legend. |
| Label value | A heat scale for a regression target, with a colour bar. A target with several degrees of freedom gets a Label column choice. In 1D, the feature is drawn against the label value instead of as histograms. |
| Auto | Class when the labels are a handful of whole numbers, Label value otherwise. |
| None | One colour. Also what a viewer with no Labels link draws. |

DearPyGui has no 3D plot, so a 3D view draws the points turned by an azimuth and an elevation, with the three axes drawn from the centre and labelled. **Spin** turns it continuously.

## A model's own spaces

Connect a Classifier or Regressor block's output to the viewer's **Model** input as well, and the window offers the spaces that model works in beside the features, under **Space**. Every view above works in each of them. The features show what a model *could* separate; these show what it did.

| Model | Spaces offered |
| --- | --- |
| LDA | Discriminant axes (`LD 1` to one fewer than the classes), class probabilities |
| MLP | Last hidden layer (one column per unit), class probabilities |
| SVM | Decision scores, class probabilities |
| Any other classifier | Class probabilities |
| Regressor | Its outputs, one column per degree of freedom |
| A model of your own | Its latent space, from a `latent(features)` method, before any of the above |

The last row is how a model built elsewhere, such as a recurrent network, shows its embedding here. Give the object the Classifier block loads a `latent` method that takes the features and returns one row per window; a torch tensor is accepted as it is.

```Python
class MyModel:
    def predict(self, x): ...
    def predict_proba(self, x): ...
    def latent(self, x):
        return self.encoder(x)      # (windows, embedding size)
```

The model has to be trained first: press **Train**, then **Start**. If it has not been, the viewer still opens on the features, and the status line says why the model's spaces are missing.

Projections are fitted to at most **Points** windows, chosen at random and the same each time. Isomap's cost grows with the square of that number, so it runs off the render thread and the window says when it is computing. Each view is computed once and kept, so going back to one is immediate.

# Looking at the predictions

A single accuracy says how often a model is right, not *when* it is wrong: at the onset of a motion, at its end, on one class, on one repetition. The **Prediction Viewer** block shows that. Connect a model block's output to its Model input and the Stored Data block's Labels to its Labels input, train the model, then press **Start**.

```
Stored Data --EMG--> Window --> Features --> Classifier --> Prediction Viewer
     \--Labels-----------------------------------------------/
```

Once the recording has been read, a window opens with three plots that share a time axis:

- the EMG, one trace per channel, each scaled to its own spread, tinted by the class that was performed;
- the model's probability for every class over time (for a regressor, its outputs beside their targets, per degree of freedom);
- the class predicted, under the class performed, as two bands of colour.

**Show** picks how the recordings are laid out:

| View | What it shows |
| --- | --- |
| Recording | One file: one repetition of one motion, as a collection writes them. **Previous** and **Next** step through them. |
| Stream | Every recording that shares a value of a filename field, such as a set or repetition number, end to end in time with a short gap between them. Pick the field under **Stream by** and the value under **Value**. |

Each recording is predicted on its own, and a model with memory -- a recurrent network, a transformer -- starts each one fresh, as it was trained to, rather than carrying a state across the join between two files.

## Training and held-out data

**Data** narrows the recordings to the ones the model trained on or the ones it never saw, and the accuracy is reported for each. Which is which comes from one of two places:

- The block's **Held Out Field** and **Held Out Values**. With `sets` and `3`, every recording whose filename says set 3 is held out and every other one is training data. Use this when the model was trained on a different selection of files, or by a script.
- A Deep Model block trained from this pipeline. It holds whole recordings out for validation while it trains and remembers which; left empty, the block uses that.

With neither, every recording is shown and **Data** offers only *All*.

## Saving the session as a figure

**Save figure...** writes every recording the Data filter lets through to one PNG: a row per class, a column per value of the first filename field (a set or repetition), and in each panel the recording's EMG above a strip whose background is the class performed and whose line is coloured by the class predicted, at the model's probability for it. Held-out recordings are framed. The same figure is available from code:

```Python
from libemg.pipeline import compile_pipeline
from libemg._gui._pipeline.prediction_viewer import save_grid_figure

pipeline = compile_pipeline(document)
pipeline.run()
save_grid_figure(pipeline.prediction_view, "predictions.png")
```

`pipeline.prediction_view["recordings"]` holds, for each file, its `signal`, its filename `meta`, the window labels as `truth`, the model's `outputs` and `predicted` class, and its `split`: everything needed for a figure of your own.

# Saving and loading

**Save** writes a JSON file holding the schema version, the nodes with their parameters and canvas positions, the links and the probes. Canvas positions are read back from the editor before writing, so layout survives a save. **Open** restores all of it, including the probe toggles and the links on the canvas.

The registry is generated from the library, so it legitimately differs between LibEMG versions. A file written against a newer version will name blocks this one has never heard of. Refusing to open that file loses the user's work, and opening it while quietly dropping what was not recognised loses it more insidiously.

An unknown block therefore loads as an **unresolved** node.

| Behaviour | Detail |
| --- | --- |
| Kept | Its block id, its parameters and its canvas position. |
| Drawn | As a node marked `Not recognised by this LibEMG.` |
| Reported | By `validate`, naming the node. |
| Blocks the run | Yes. Compilation refuses until it is dealt with. |
| Written back | Byte for byte as it came in. |

Loading a file containing `transform.timewarp`, a block that does not exist here, gives:

```
These blocks are not recognised by this version of LibEMG: future_1.
They were kept so nothing is lost, but the pipeline cannot run until they
are removed.
```

Saving that document again writes the unresolved node back unchanged, parameters and position included. Nothing is destroyed by looking at the file with the wrong version installed.

A parameter the current registry no longer declares is kept the same way, so downgrading LibEMG and upgrading again does not lose it.

# Building without hardware

The Synthetic Source writes into shared memory exactly as a device streamer does. A pipeline built on it is the same pipeline, and swapping in the real device changes one block.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `sampling_rate` | 1000 | Samples per second. |
| `num_channels` | 8 | Channels produced. |
| `pattern` | `bursts` | `noise`, `sine` or `bursts`. |
| `amplitude` | 1.0 | Scale of the generated signal. |

The `bursts` pattern alternates four seconds quiet with four seconds active. That gives a classifier something to separate and a probe something visible to draw. Every figure quoted on this page was produced by a synthetic source on a machine with no electrodes attached.

It works outside the editor as well, anywhere a streamer is accepted:

```Python
from libemg._gui._pipeline.synthetic import synthetic_streamer
from libemg.data_handler import OnlineDataHandler

streamer, shared_memory = synthetic_streamer(pattern='bursts', num_channels=8)
odh = OnlineDataHandler(shared_memory)
```

# Training a model from the pipeline that describes it

**Train** fits the model the pipeline names, on the recording the pipeline reads. It is only for a stored-data pipeline, because a fit needs labelled data that has already been collected.

Everything a fit needs is already on the canvas. The Stored Data block says where the recordings are, which regex filters parse their names and which metadata field carries the labels. The filter and window blocks say how the signal is conditioned and enframed. The features block says what to extract. The model block says which model to fit and where to put it. Training walks the recording exactly as a scoring run does, so the model is fitted on precisely the data it will then be scored on.

Build the pipeline, press **Train**, then press **Start**. The second run scores what the first one fitted.

```
Trained on 71972 windows of 27 features over classes [0, 1, 2].
Saved to C:\work\models\lda.pkl. Point a live pipeline at that file to run it.
```

Which kind of predictor it builds follows from the block. A **Classifier** block fits an `EMGClassifier` and rounds its labels to class indices. A **Regressor** block fits an `EMGRegressor` and treats the label field as one degree of freedom per column, so a field holding a single value per sample trains as one DOF.

The same document then runs live. Point its source at a device instead of a folder and the model block already holds the file that was just written.

## Reproducing an offline analysis from the editor

The two offline examples in the documentation — a classifier over the `OneSubjectMyoDataset` and a regressor over the `OneSubjectEMaGerDataset` — are scripts. The same analyses can be built on the canvas, and the numbers `OfflineMetrics` reports come out identical, not approximately identical: both paths run the same estimator over the same windows, so any difference at all would mean one of them is reading the data differently. `tests/test_pipeline_offline_parity.py` holds that.

A train/test split is expressed as two runs of the same graph with different filter values, because a pipeline reads one folder. Build the graph with the training values, press **Train**, change the split values, press **Start**. The second run scores what the first fitted.

### Classification

| Block | Parameter | Value |
| --- | --- | --- |
| Stored Data | Folder | `OneSubjectMyoDataset/` |
| | Regex Filters | `/trial_\|/\|1,2,3\|sets`<br>`C_\|.csv\|0,1,2,3,4\|classes`<br>`R_\|_\|0,1\|reps` |
| | Label Key | `classes` |
| | Sampling Rate | `200` |
| Window | Window Size / Increment | `20` / `10` |
| Features | Feature Group | `HTD` |
| Classifier | Model / Fitted Model | `LDA` / wherever you want it |
| Metrics | Metrics / Null Label | `CA, AER, INS` / `2` |

Train on `1,2,3`, change the sets filter to `4,5,6`, and run. The script and the editor both report:

```
CA  0.9853107344632769
AER 0.004297994269340966
INS 0.015819209039548022
```

### Regression, and labels that live in their own file

A filename filter can only say what a whole recording was. A regression target is a value per sample, so it lives in a file beside the recording — which is what **Label Files** is for. It takes the same `left|right|values|description` lines as the regex filters, but they match *metadata* files, and **Package By** says how each one is paired with the recording it belongs to (`same folder` is the LibEMG convention, `same stem` pairs a recording with the file named after it).

| Block | Parameter | Value |
| --- | --- | --- |
| Stored Data | Folder | `OneSubjectEMaGerDataset/` |
| | Regex Filters | `/\|/\|open-close,pro-sup\|movements`<br>`_R_\|_emg.csv\|0,1,2,3\|reps` |
| | Label Files | `/\|.txt\|labels\|labels` |
| | Label Key | `labels` |
| | Sampling Rate | `1010` |
| Window | Window Size / Increment | `150` / `40` |
| Features | Feature Group | `HTD` |
| Regressor | Model | `LR` |
| Metrics | Metrics | `R2, NRMSE, MAE` |

Train on reps `0,1,2,3`, change the reps filter to `4`, and run. Both paths report one value per degree of freedom:

```
R2    [0.9394276836613275, 0.9576579094961605]
NRMSE [0.06087835488847762, 0.050867671453071764]
MAE   [0.09103975298336556, 0.07208576409479191]
```

The label of a window is the label at its *end*, which is the same convention as the `metadata_operations={'labels': lambda x: x[-1]}` the script passes to `parse_windows`. A label field with several columns stays several columns, so a two-DOF target trains and scores as two DOFs rather than silently becoming one.


## What Train refuses, and why

| Message | What to do |
| --- | --- |
| The model block has no Fitted Model path | Set that parameter. It is both where training saves and where a live run loads. |
| That recording produced no windows | Check the folder, the regex filters, and that the window is not longer than the recordings. |
| That recording carries no labels | Set the Stored Data block's **Label Key** to a metadata field its regex filters produce. |
| Training was stopped part way | Nothing was saved. A model fitted on half a recording would be indistinguishable from one fitted on all of it, so the existing file is left alone. |

A live model still will not compile without a file, and says so:

```
'Classifier' (classifier_5) needs a fitted model to run live.
Set its Fitted Model parameter to a saved predictor.
```

An **offline** pipeline compiles without one, which is what makes the Train button reachable: the only way to get the file is to run the training that demanding it would refuse.
