A myoelectric control session has always been four jobs. A device has to be brought up, training data has to be collected, a pipeline has to be built and run, and something has to be controlled with it. Each of those used to be a script, and the first one had to be a script before any of the others could be tried at all.

The LibEMG window now does all four. Open it with no arguments and no handler.

```Python
from libemg.gui import GUI

if __name__ == "__main__":
    GUI().start_gui()
```

Nothing is passed in. The window starts the device itself, and every other panel works off the one it started.

# The whole workflow, in order

| Step | Menu | Item | What it is for |
| --- | --- | --- | --- |
| 1 | Device | Streamer | Brings a device up and hands it to the rest of the window. |
| 2 | Data | Collect Data | Records labelled training data with screen guided prompts. |
| 3 | Pipeline | Pipeline Editor | Builds a pipeline as a graph, then runs it. |
| 4 | Environments | Launch Environment | Plays a task driven by the pipeline's output. |

There is a fifth item that is useful at any point. **Visualize** then **Live Signal** plots whatever the started device is delivering, which is the quickest way to see that electrodes are on properly before anything is recorded.

The order matters in one place only. Step 1 comes first. The other three can be revisited in any order once a device is running.

# The streamer panel

Choose **Device** then **Streamer**. The panel has a device drop-down, a **Start** button, a **Stop** button, a block of options for the chosen device, and a table of what is arriving.

The device list is generated rather than written out. It is the streamer functions in `libemg.streamers`, filtered to the ones that accept `shared_memory_items`, with the synthetic device added at the top. A streamer that talks over a socket is left out, because it publishes nothing the rest of the window could attach to. That is why `mock_emg_stream` does not appear. These are the devices this release offers.

| Device |
| --- |
| Synthetic (no hardware) |
| `delsys_api_streamer` |
| `delsys_streamer` |
| `emager_streamer` |
| `leap_streamer` |
| `myo_streamer` |
| `oymotion_streamer` |
| `sifi_bioarmband_streamer` |
| `sifi_biopoint_streamer` |

Each device's options are generated too. They are the keyword arguments of that device's own function, read from its signature, with the control chosen from the type of each default. A device added to `libemg.streamers` appears here with its options intact and nothing in the panel changes. Three devices, counted from a real run.

| Device | Options drawn |
| --- | --- |
| `delsys_streamer` | 8 |
| `sifi_biopoint_streamer` | 23 |
| Synthetic (no hardware) | 4 |

Press **Start** and the panel launches that device, attaches an `OnlineDataHandler` to it, and begins reporting. The table has one row per modality. Which modalities appear depends on what the device was asked for. A Delsys gives `emg`, and `imu` as well when its Imu option is turned on. A SiFi BioPoint gives whichever of `ecg`, `emg`, `eda`, `imu`, `ppg` and `temperature` are enabled. The synthetic device gives `emg`.

| Column | What it counts |
| --- | --- |
| Modality | The shared memory item the device writes to. |
| Samples | Every sample committed since this run started. |
| Rate (Hz) | Samples per second, smoothed over the last few refreshes. |
| Writes | Commits made by the device, which is how often it delivered. |

The synthetic device at 1000 Hz on 6 channels, four seconds after Start, read as follows.

| Modality | Samples | Rate (Hz) | Writes |
| --- | --- | --- | --- |
| `emg` | 4006 | 1000 | 4006 |

The header above the table said this.

```
Incoming data    1000 samples per second across 1 modalities
```

The rate costs nothing to show. The panel reads each modality's state block, which is a handful of integers, rather than its data. Reading the samples themselves every frame would compete with the device for the same memory.

Press **Stop** and the device is shut down. A streamer that has a stop signal is asked to stop and given three seconds to finish. A streamer that has none is terminated.

# Why starting a device matters to everything else

The panel does not keep the device to itself. When a device starts, the panel hands the GUI the `OnlineDataHandler` it built and the shared memory items behind it. Every other panel reads that handler when it opens.

| Panel | What it does with the started device |
| --- | --- |
| Collect Data | Records the live stream against the prompts it shows. |
| Live Signal | Plots each modality as it arrives. |
| Pipeline Editor | Offers the running device as a source block. |
| Launch Environment | Drives a task from a model reading the same stream. |

This is the reason step 1 comes first. Before a device is started, nothing in the window can see data. Opening Collect Data with no device running gives a panel with nothing to record.

The handler is a normal `OnlineDataHandler`, so anything that accepts one accepts this. The one published by a synthetic device asked for 6 channels answered `get_data(N=200)` with the following.

| Modality | Rows returned | Channels returned |
| --- | --- | --- |
| `emg` | 200 | 6 |

**Stop** withdraws the handler as well as stopping the device. The other panels are told the device has gone rather than being left holding a handler attached to nothing.

# The synthetic device

The synthetic device is offered alongside the real ones. It commits generated samples into shared memory at a fixed rate, exactly as a device streamer does. Nothing downstream can tell the difference, so the entire window can be rehearsed with no hardware on the desk. A pipeline built against it is the same pipeline, and swapping in the real device later changes one choice in one drop-down.

| Option | Default | What it sets |
| --- | --- | --- |
| Sampling Rate | 1000 | Samples committed per second. |
| Num Channels | 8 | Columns in each sample. |
| Pattern | bursts | The shape of the generated signal. |
| Amplitude | 1.0 | Scale of the generated signal. |

There are three patterns.

| Pattern | What it produces | Good for |
| --- | --- | --- |
| `noise` | Gaussian noise on every channel. | Checking shapes, rates and connections. |
| `sine` | A sine per channel, each at a different frequency. | Seeing a filter do something visible. |
| `bursts` | Four seconds quiet, then four seconds active. | Giving a classifier two states to separate. |

Bursts is the default because it is the only one of the three a classifier can learn anything from. The quiet and active halves are easy to recognise in a probe and easy to label.

# Two behaviours worth knowing

**Starting resets the counters.** Shared memory outlives the process that made it, and every device writes to the same modality names. A device started now attaches to whatever the last one left behind. Without a reset, counts from a previous session would read as live data, and a device that is not connected would look like it was working. So the panel zeroes the counters on every Start. A second run begins near zero rather than continuing from the first.

**A device that never delivers is reported.** A streamer spawns a process that goes and finds the device on its own. A device that is unplugged, asleep or already paired to something else starts perfectly well and simply never produces a sample. So the panel does not claim success at Start. It says it is waiting.

```
Synthetic (no hardware) started. Waiting for the first samples.
```

Only once samples actually arrive does the message change.

```
Synthetic (no hardware) is streaming. Collect Data, Live Signal, the pipeline
editor and the environments can all use it now.
```

A device that has delivered nothing after five seconds is said so plainly. This is a real Myo with no dongle attached.

```
myo_streamer started, but no samples have arrived in five seconds. Check that
the device is on, paired and not in use by another program. It is left running
in case it is still connecting.
```

It is left running rather than shut down, because a device that is slow to connect and a device that is not there look identical for the first few seconds. A device that fails to launch at all is a different case and reports its own reason instead.

Switching device while one is running is refused. The drop-down is put back to the running device and the panel says to stop it first.

**Clicking a menu item twice brings the panel forward.** It does not build a second one. This matters most for the streamer: a second panel would take over the widget tags the first one owns, leaving the running device held by a panel with no window able to stop it. The same applies to the pipeline editor and the environments panel, so a running pipeline is never orphaned by a stray click.

# Fitting the model

The fourth job is in the window too. Build a pipeline whose source is **Stored Data**, pointed at the folder the collection panel wrote, and press **Train**. The pipeline already says which regex filters parse the file names, which metadata field carries the labels, how the signal is filtered and windowed, which features to take and which model to fit, so nothing has to be said twice.

```
Trained on 71972 windows of 27 features over classes [0, 1, 2].
Saved to C:\work\models\lda.pkl. Point a live pipeline at that file to run it.
```

The file it writes is the one the model block's **Fitted Model** parameter already names, so the same document scores itself on the next **Start**, and a live pipeline built on the same model block picks it up with nothing else to set.

A live pipeline still will not compile without that file, and says so rather than failing later.

```
'Classifier' (classifier_5) needs a fitted model to run live.
Set its Fitted Model parameter to a saved predictor.
```

# The whole loop, in the window

1. **Device ▸ Streamer**, pick the hardware, press Start. The handler it publishes reaches every other panel.
2. **Data ▸ Collect Data**, record the prompted gestures.
3. **Pipeline ▸ Pipeline Editor**, build a stored-data pipeline over that folder, press **Train**, then **Start** to see its offline metrics.
4. Rebuild the same pipeline against the live device, or open the one you saved, and press **Start**.
5. **Environments**, set up a task and launch it. It reads the predictions the pipeline is publishing.

No step in that list needs Python.
