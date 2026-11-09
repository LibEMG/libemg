<style>
    table {
        width: 100%;
    }
</style>

Although not directly part of the EMG pipeline, the `Screen Guided Training` module provides developers a convenient means to collect real-time screen-guided training (SGT) data. This SGT data can be used to train a system (i.e., machine learning model) or for future offline analyses. Note that the SGT tool can handle image types of `png`, `jpg`, and `jpeg`. Additionally, it can handle `.gifs` for dynamic movements/contractions.

| <center>Menu</center>  | <center>SGT</center> |
| ------------- | ------------- |
| ![](menu.PNG) | ![](training.PNG) |
<center> <p> Table 1: Screen Guided Training UI</p> </center>

# Collection Modes
**Data > Collect Data** offers two modes, chosen at the top of its window. The mode cannot be changed while a collection is running, and every session records which mode it was collected in (`metadata.collection_mode` in `_session_annotations.json`, and `collection_mode` in `collection_details.json`).

| Mode | What the participant sees | Labels |
| --- | --- | --- |
| **3D Hand Choreography** (default) | An animated 3D right hand that ramps into each gesture, holds it, ramps out and rests. | Every phase: warm-up, ramp-in, hold, ramp-out, rest, cycle pause, paused. |
| **Image Folder / Optional Protocol** (backup) | One image or video per class from a media folder, optionally ordered by a collection protocol. | `prompt` / `rest` / `pause` stages, as before. |

The image-folder mode is unchanged and is described in the rest of this page. A protocol built under **Data > Collection Protocol** opens Collect Data in that mode.

## 3D Hand Choreography
Tick the gestures to collect (16 are available; *rest* is the relaxed hand and is never a selectable class), put them in order, and set the timing: warm-up, ramp-in, hold, ramp-out, rest between gestures, repetitions per gesture, and an optional pause between cycles. Each gesture can override its own repetition count and any of its ramp-in, hold, ramp-out and following rest. Order is either fixed or a seeded shuffle; the seed is always shown and saved. Settings can be saved and loaded as JSON presets. The window shows the estimated duration and the number of repetitions before anything starts.

**Preview** animates the run without recording or writing anything, and needs no device. **Start** (with a device streaming) checks the signal as the image-folder mode does, then opens the hand window in recording mode; the recording begins only when **Start** is pressed in that window. The window has **Start**, **Pause**, **Resume**, **Stop** and **Reset**, and shows the current motion, phase, a countdown, the cycle, and that gesture's repetition count.

How a run is laid out:

* One repetition is rest → ramp-in → hold → ramp-out → rest, with each rest shared with the neighbouring repetitions.
* A cycle holds one repetition of every selected gesture that still has repetitions left. Three gestures with three repetitions each make three cycles; a gesture with fewer repetitions drops out of the later cycles.
* The run opens with the warm-up (relaxed hand). Gestures within a cycle are separated by the rest (the rest after a gesture uses that gesture's rest override, if any). Between cycles the cycle pause is used *instead of* the rest when it is enabled. The run ends on the last ramp-out, with no trailing rest. Zero-length phases are left out. Ramp-in and ramp-out must be longer than zero.
* **Rest between gestures** can be unticked to collect direct *transitions*: each ramp-out then runs straight into the next gesture's ramp-in, so the hand returns through rest without stopping there (between cycles too, unless the cycle pause is on). Rest is still collected deliberately in the warm-up and the cycle pauses, and one of them has to be kept. The rest length and the per-gesture rest overrides are not used. Each such ramp-in records the gesture it came from in `transition_from`, and the window warns of the next gesture during the hold and ramp-out ("Then straight into: Wrist Flexion").
* The phases are mapped onto the model's clips, not played at a different speed: ramp-in plays clip time 0.3 → 1.0 s, hold 1.0 → 1.5 s, ramp-out 1.5 → 2.2 s, and every resting phase shows the rest frame (0.0 s). Lengthening the hold never changes how the ramps look.
* The hand moves **linearly in time** over each ramp. The clips as authored are eased (slow at both ends), so each clip's ramps are re-timed when the hand is loaded: the skinned hand's travel from rest toward the gesture is measured and inverted, so at any instant the fraction of the way it has moved equals the `position` label (to within 0.3 %).
* **Pause** freezes the hand and the schedule; **Resume** continues from the same point. **Stop** cancels the run and returns the hand to rest. A run ends **completed**, **cancelled** or **failed**.

**Intensity (optional).** Tick *Intensity targets* and give levels as percents of maximum effort (for example `30, 60, 90`). Each gesture's repetitions are dealt these levels, in turn or in a seeded shuffle, as evenly as its repetition count allows. A bar beside the hand shows the target: it fills with the hand over the ramp-in, holds at the level (and turns green) through the hold, and empties over the ramp-out. During a rest it previews the next repetition's level. The window also names the target ("Target effort: 60% of maximum").

The run pauses itself, and records why, when the window stops updating (a dragged window, say), when the renderer falls behind the schedule, or (when recording) when the device stops delivering data. Resume is refused until data arrives again. Stretches the participant could not see, or that have no data, are marked and left out of training. Closing the application mid-run cancels it with reason `application_shutdown` and saves what was recorded.

### What a choreography recording writes
| File | Contents |
| --- | --- |
| `_session_<modality>.csv`, `_session_annotations.json` | The continuous recording and its stages, written by the same code as the image-folder mode. Holds are `prompt` stages, so `C_<class>_R_<rep>_<modality>.csv` are steady holds. Ramps are `ramp_in`/`ramp_out` stages; warm-up, rests and cycle pauses are `rest` stages with a `phase` field. |
| `_session_choreography.json` | The run record: configuration snapshot, asset SHA-256, versioned class map, the full schedule (cycles, seed, per-phase timing), the clock mapping, every event and every annotation interval. |
| `_session_choreography_events.jsonl` | An append-only journal written while the run goes. `python -m libemg.choreography recover <folder>` rebuilds the run record from it after a crash. |
| `_choreography_targets_<modality>.csv` | One row per row of `_session_<modality>.csv`: per-sample regression targets (below). |
| `collection_details.json`, `subject_details.yaml` | As in the image-folder mode, with `collection_mode: "choreography"`, the intensity settings and the target-file description. |

Class IDs come from a versioned map (`right_hand_choreography.v1`: 0 rest, 1 hand_closed, 2 hand_open, … 16 thumb_extend), never from which gestures were ticked or their order.

Every interval in `_session_choreography.json` carries `run_id`, `collection_session_id`, `interval_id`, `parent_repetition_id`, `motion_class` (the stable ID), `phase`, `derived_label` (`hand_closed/ramp_in`, `hand_closed/hold`, `rest/warmup`, `rest/cycle_pause`, or plain `rest`), `cycle_index`, `gesture_repetition_index`, `start_collection_time`/`end_collection_time`, `start_sample_index`/`end_sample_index`, `completion_status` and `training_eligible`. Intervals are **half-open, [start, end)**: a sample produced exactly on a boundary belongs to the later interval, and adjacent phases share a boundary without sharing a sample. Phase intervals are on the `phase` track and never overlap; each repetition also has a parent interval (ramp-in start to ramp-out end) on the `repetition` track. Collection times are in the logger's `time.time()` clock, mapped from the scheduler's monotonic clock; sample indices come from the session's fitted sample clock. Labels describe the instruction that was shown, not verified participant movement.

```Python
import json
record = json.load(open("data/_session_choreography.json"))
holds = [iv for iv in record["intervals"]
         if iv["phase"] == "hold" and iv["training_eligible"]]
```

### Regression targets: position and intensity
Besides the class, every instant of a choreography recording has a **position**: where the displayed hand is between rest (0) and the gesture (1). It rises linearly from 0 to 1 over the ramp-in, is 1 through the hold, falls linearly from 1 to 0 over the ramp-out, and is 0 while resting; while paused it holds the frozen value. With intensity on, each repetition has an `intensity_level`, and the **intensity** target is `position × intensity_level` (0 while resting). Through a direct transition the position falls to 0 at the end of one gesture's ramp-out and rises from 0 at the start of the next ramp-in, and the class changes at that instant. Each interval records both at its ends (`position_start`/`position_end`, `intensity_start`/`intensity_end`, and `intensity_level`; rests carry `next_intensity_level`). Within an interval the values are a straight line between the two ends. Hold stages in `_session_annotations.json` carry the level as a factor, `{"intensity": 0.6}`.

`_choreography_targets_<modality>.csv` writes these out per sample. Its columns are named in a `#` header line:

| Column | Meaning |
| --- | --- |
| `sample_index` | The row of `_session_<modality>.csv`. |
| `collection_time` | When the sample was produced (the session's fitted clock). |
| `motion_class_id` | The versioned class ID; 0 (rest) while resting. |
| `phase_code` | 0 warm-up, 1 ramp-in, 2 hold, 3 ramp-out, 4 rest, 5 cycle pause, 6 paused, -1 outside the run. |
| `position` | 0 rest … 1 gesture, as above. |
| `intensity_level`, `intensity` | Only when intensity was on. |
| `repetition_index` | The repetition's place in the run (0-based); -1 while resting. |
| `training_eligible` | 1 where the interval was shown cleanly (not paused, interrupted, or overlapping a display or data gap). |

```Python
from libemg.choreography import load_targets

t = load_targets("data/", "emg")          # samples and targets, row for row
t = t.select(t.training_eligible == 1)    # drop paused or interrupted stretches
X, y_class, y_position = t.samples, t.motion_class_id, t.position
y_intensity = t.intensity                 # None if the run did not ask for intensity
```

For loaders that read one file per repetition, `python -m libemg.choreography cut-repetitions data/ --out data_repetitions/` writes each completed repetition (ramp-in start to ramp-out end) as `C_<class>_R_<rep>_<modality>.csv` with a matching `C_<class>_R_<rep>_<modality>_targets.csv` of the same length. `--lead-in` and `--lead-out` add rest either side. The output folder has to be outside the recording folder: the offline loaders search folders recursively and would otherwise read these repetitions a second time alongside the hold files.

Like every label here, position and intensity describe the instruction shown, not measured movement or force.


# Gesture Library
One of the tedious aspects (and annoyances) of performing data collection is the need to accumulate photos (or videos) of each contraction. This process is not only time-consuming but also leads to inconsistencies among studies. To streamline this process, LibEMG includes a database of [~35 gestures](https://github.com/libemg/LibEMGGestures) (both static and dynamic) that can be interfaced and downloaded through the API.

![](gesture_example.png)
<center> <p> Figure 1: A screenshot of the first four gestures of the <a href="https://github.com/libemg/LibEMGGestures"> gesture library </a>. Each gesture includes an index, a static image, and a dynamic gif. </p> </center>

Each gesture in the library is associated with a unique id that is referenced when specifying the gestures to download. A simple pipeline for setting up the SGT module is as follows: 

```Python
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler
from libemg.streamers import myo_streamer

# Create data handler and streamer 
_, sm = myo_streamer()
odh = OnlineDataHandler(sm)

training_ui = GUI(odh, width=700, height=700, gesture_height=300, gesture_width=300)
training_ui.download_gestures([1,2,3,4,5], "images/")
training_ui.start_gui()
```

# Types of Training 
In general, the accumulation of training data can be split into one of two types: <b>static reps</b> and <b>ramp reps</b>. While ramp reps often lead to better performance, both training methods are viable. Static contractions involve eliciting the contraction for the entire rep duration. In this case, there is little variability other than that occurring throughout the contraction's duration. Ramp contractions start at rest and involve a transition to the contraction within the rep duration. This captures a range of variability for each contraction, which is crucial for velocity-based control systems. Table 2 highlights the differences between the two strategies. 

**Note, that when using velocity control, ramp contractions should be used.**

| <center>Type</center>  | <center>EMG</center> |
| ------------- | ------------- |
|  <center><b>Static</b></center> | ![](static.png) |
|  <center><b>Ramp</b></center> | ![](ramp.png) |
<center> <p> Table 2: Static training repetition (top) compared to a ramp training repetition (bottom). </p> </center>

# Subject Details
The configuration window has a **Subject Details** section: subject number, subject name, session number, day number and condition number. Whatever it is showing when Start is pressed is written to `subject_details.yaml` in the output folder, alongside the data:

```yaml
subject_number: 7
subject_name: Ada
session_number: 2
day_number: 3
condition_number: 4
collected: '2026-09-18T14:02:11.418293'
```

This is what a dataset has to be grouped by later, and it is the one thing in the folder that cannot be recovered from the recording itself -- so it is written before the first repetition rather than left to a lab notebook. It is kept out of `collection_details.json` because the two go stale at different times: that file describes the prompts a run happened to use, this one describes the person and the condition. YAML because it is the file most likely to be opened and corrected by hand.

The fields can be seeded from the GUI's `args` so a rig set up for one participant opens on the right numbers:

```Python
training_ui = GUI(odh, args={'media_folder': 'images/', 'data_folder': 'data/S07/',
                             'subject_number': 7, 'subject_name': 'Ada',
                             'session_number': 2, 'day_number': 3, 'condition_number': 4})
```

A numbered field that is not a number is written through as typed, so a study labelling its subjects `P01` works and a blank field does not abort the collection at the moment Start is pressed.


# Designing the Protocol

Most of what decides whether a collection produces usable data is fixed before anyone contracts a muscle: what is prompted, in what order, for how long, with what rests, under what conditions, and against what acquisition settings. Those decisions used to live in whichever script launched the GUI, in the alphabetical order of a media folder, and in a lab notebook.

**Data > Collection Protocol** makes them a document. The panel builds a `CollectionProtocol`, renders its prompts with the `Animator`, and hands both to screen guided training. The protocol is saved into the data folder, so a recording carries the description of the experiment that produced it.

```Python
from libemg.protocol import CollectionProtocol, within_and_combined_dof

protocol = CollectionProtocol(name='wrist-2dof', kind='regression',
                              dof_labels=['Flexion/Extension', 'Pronation/Supination'],
                              factors={'limb_position': ['neutral', 'elevated', 'across body']})
protocol.trajectory = within_and_combined_dof(n_dof=2, hold_seconds=3, ramp_seconds=2)
protocol.save('wrist-2dof.yaml')
protocol.build_media('animation/')     # writes the video and its per-frame labels
```

Or from a terminal:

```
python -m libemg.protocol template study.yaml --dofs Flex Extend
python -m libemg.protocol show study.yaml
python -m libemg.protocol build study.yaml --media-folder animation/
```

## The trajectory

A prompt trajectory is a list of segments — `rest`, `ramp`, `hold`, `sinusoid`, `random` — rendered end to end. Each segment starts where the last one finished, so there is never a jump the subject cannot follow, and a `rest` releases over a quarter of a second rather than snapping to zero. The animation's own coordinates are written beside the video as a `.txt`, which screen guided training already picks up as per-frame labels, so a regression collection comes out with its ground truth aligned to its prompts.

Three templates are built in, each matching a standard prompt in the literature:

| Template | What it prompts | What it is for |
| --- | --- | --- |
| Within and combined DOF | each DOF ramped to both extremes alone, then the pairs driven together | proportional control, which exists for the simultaneous case a within-DOF-only training set never shows |
| Ramp and hold | one DOF ramped to a series of intensity levels | pacing the contraction, and covering more than the single effort level a classifier would otherwise be trained at |
| Random exploration | randomly placed targets inside the unit ball | the space between the axes, which is most of where a user actually operates |

An animated prompt matters even for classification. A static image says what to do but not how hard or how fast, so the contraction profile is whatever the subject felt like; a prompt that ramps to a level and holds paces them. This is the same recommendation as the ramp contractions described above, generated rather than described.

## Order, rests and design factors

| Setting | Default | Why |
| --- | --- | --- |
| Order | `random` | Collecting class 1 first and class 8 last in every rep confounds class with everything that drifts over a session: fatigue, gel drying, skin impedance, the subject settling in. `counterbalanced` rotates instead, so every class sits in every position. |
| Seed | recorded | An order that cannot be reproduced cannot be reported. The seed is saved with the protocol. |
| Rest jitter | 1.0 s | A rest of exactly three seconds teaches the subject when the prompt is coming, and they pre-empt it — the onset then sits before the cue and the start of the rep is not what it is labelled. |
| Design factors | none | Accuracy falls when the limb is in a position, or the contraction at an intensity, that was not in the training set. Levels are declared here, varied between reps (a subject cannot change arm position mid-rep), shown in the prompt, and written into every stage of `_session_annotations.json`. |
| Reaction time | 0.2 s | Recorded, not applied. The recording is continuous and the annotations mark the cue, so trimming the reaction delay is a decision for analysis rather than one baked into the files. |

## Acquisition details

The **Acquisition (ISEK reporting)** section collects what the ISEK standards for reporting EMG ask for: electrode type, size, inter-electrode distance, placement, skin preparation, ground location, sampling rate, amplifier bandwidth, notch, gain and A/D resolution. None of it can be recovered from the recording afterwards, and all of it decides whether the data can be compared with anyone else's. `protocol.describe()` says how many fields are still blank.

# Checking the Signal First

Most unusable EMG datasets were unusable from the first repetition, and nothing in the recording says so. A disconnected channel writes a flat line, a loose electrode writes artifact, a badly grounded rig writes mains hum, and an over-gained amplifier writes a clipped square wave — all four look like data, and all four are found weeks later in analysis.

The configuration window has a **Check Signal** button, and the same check runs when Start is pressed. It reports dead channels, clipping, mains pickup at 50 or 60 Hz, and channels far out of balance with the rest, and it measures the rate the device is *actually* delivering rather than the one it was configured for. The result is written to `signal_quality.yaml` in the data folder, so the state of the electrodes at the start of the session is part of the record.

```Python
from libemg.signal_quality import check_signal

report = check_signal(odh.get_data()[0]['emg'], sampling_rate=1000)
print(report.summary())
```

```
emg: 8 channels, 2000 samples (2.0 s at 1000 Hz)
  ! Channel(s) 3 are effectively flat next to the others. Check those leads are seated before collecting.
  ! Channel(s) 6 carry heavy 60 Hz mains pickup (62% of in-band power). Check the ground electrode ...
```

Everything is measured relative to the signal itself rather than in microvolts, since a LibEMG stream carries whatever units its device chose. Pass `signal_check='block'` to refuse to start on a bad signal, or `'off'` to skip it; the default warns and collects anyway, because a heuristic should not be able to cancel a booked session hour.

At the end of a session the median frequency of the first repetition is compared against the last. A drop is the shift a fatiguing muscle makes, and it means the later repetitions are not quite the same task as the earlier ones — worth knowing before the data is treated as one homogeneous set.


# Continuous Collection
A collection is recorded in one piece. Every sample the device delivers between pressing Start and the final repetition ending goes into `_session_<modality>.csv` -- the prompts, the grey rest screens in front of them, and the pauses where the collection waits for someone to press Continue. Nothing is dropped for falling outside a repetition.

This matters for two reasons. Starting and stopping a log around each repetition cuts the stream on the time samples *arrived*, and a device does not deliver on a schedule: the tail of a repetition is still in flight when its log stops, while a burst landing early in a window carries samples produced during the preceding rest. Neither is lost data -- both are samples credited to the wrong repetition, and both disappear when the stream is never cut while it is running. And a temporal model that wants to read a longer stream than one repetition, through the transition into and out of a contraction, needs the stretch between repetitions to exist.

Alongside the recording, the collection writes `_session_annotations.json`: one entry per stage, with the wall-clock time it ran and the sample indices it occupies in each modality's file. It is written after every stage, so a collection that is interrupted still leaves a file describing what reached the disk. The standard `C_<class>_R_<rep>_<modality>.csv` files are then cut from the recording at those indices -- so the same cut can be made again later, differently, without collecting again.

```Python
from libemg.session import SessionAnnotations

session = SessionAnnotations.load('data/_session_annotations.json')

# Cut it again: keep the rest periods this time, with a quarter second of
# context either side of every prompt.
session.cut('data/recut/', kinds=('prompt', 'rest'), lead_in=0.25, lead_out=0.25)

# Or read a modality end to end and ask what each part of it was. The spans
# tile the recording with no gaps between them.
arrivals, emg = session.read('emg')
for span in session.timeline('emg'):
    print(span.kind, span.label, span.start, span.end)
```

The same thing from a terminal:

```
python -m libemg.session show data/_session_annotations.json
python -m libemg.session cut data/_session_annotations.json --kinds prompt rest --lead-in 0.25
```

A stage is one of `prompt` (a repetition), `rest` (the grey screen before one) or `pause` (the hold on Redo/Continue). Only prompts are cut by default, and only prompts are named `C_<class>_R_<rep>_`; a rest or a pause is written under a name carrying no `C_`/`R_` tokens, so a regex parser pointed at the folder cannot mistake one for a repetition. A repetition that was redone stays in the annotations marked `superseded`, since those samples are still in the recording, but it is not cut unless asked for.

Passing `'continuous_log': False` in the GUI's `args` restores the old behaviour of a separate log per repetition, with no session file and no annotations.
