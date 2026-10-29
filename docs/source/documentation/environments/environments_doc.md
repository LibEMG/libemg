LibEMG's environments are its real-time tasks. Each one is a pygame game with its own loop, its own drawing and its own results log. Until now each one opened a window of its own, and reaching it meant writing a script first.

The Environments window runs them inside the LibEMG window instead. Open the LibEMG GUI, choose **Environments**, then **Launch Environment**. A task is picked from a list at the top, set up on the screen that appears, and launched into a panel beside its settings. The window builds its own environment, so it opens with no `OnlineDataHandler` and no hardware attached. A handler is needed only to save the EMG of a run.

```Python
from libemg.gui import GUI

if __name__ == "__main__":
    # No handler is passed. The Environments window does not need one.
    GUI().start_gui()
```

Four tasks are offered in this release.

| Task | What it is | Frame it opens at |
| --- | --- | --- |
| Fitts' Law | A cursor and a single target. The classic test of how quickly a control scheme acquires a target. | 1250 by 750 |
| ISO Fitts' Law | Targets in a ring, acquired in the standard ISO 9241-9 order. The usual way to report throughput. | 1250 by 750 |
| Curricular Fitts | A Fitts task whose difficulty adapts as the user improves. The task used for user-in-the-loop adaptation. | 1000 by 1080 |
| EMG Hero | Notes fall down the screen and are hit with the matching gesture. A rhythm game for discrete control. | 1500 by 750 |

The frame size is whatever the task's own width and height settings say. The numbers above are the defaults the setup screen starts with.

# How a pygame game ends up in a DearPyGui window

This is the part worth understanding, because it explains why nothing about the games had to change.

SDL has a video driver called `dummy`. Under it `pygame.display.set_mode` still returns a real surface, and everything still draws onto that surface. What is missing is the window. The environment never learns this. Its `game_setup` and its `_run_loop` run exactly as they always did.

The frame then has to reach the interface. It does so by both sides pointing at the same memory.

| Side | What it does with the memory |
| --- | --- |
| The GUI | Hands the array to `add_raw_texture` as the texture's backing store. |
| The environment | Writes the surface it just drew into that same array. |

DearPyGui draws from the array it was given rather than from a copy taken when the texture was made. So there is no upload per frame and nothing to copy on the GUI side. The environment paints, and the window shows what it painted.

The consequence matters more than the mechanism. An environment needs no special support to be embeddable. A new task written against the same base class runs in the window unchanged, and what it needs is a registry entry and a factory rather than a new drawing path.

| Piece | Module | What it is |
| --- | --- | --- |
| Bridge | `libemg._gui._environments.frame_bridge` | The shared memory carrying one frame, plus control and input. |
| Runner | `libemg._gui._environments.embedded` | The process that runs the game offscreen. |
| Registry | `libemg._gui._environments.registry` | What each environment can be set up with. |
| Factories | `libemg._gui._environments.factories` | What crosses into the child process to be built there. |

Nothing about the game is constructed in the GUI. A pygame object cannot cross a process boundary, and neither can an open socket. What crosses is a small description of what to build, and the building happens on the other side.

# Why the environment stays in its own process

A game that stalls must not take the interface with it. A thread would share the interpreter with the render loop, so a game stuck in its own update would freeze the window that was meant to stop it. A separate process cannot do that. The **Stop** button is still answered by the GUI even when the game answers nothing.

The cost of that separation is one write of the frame per drawn frame. It was measured on the machine this page was written on.

| Frame size | Bytes in one frame | Shared segment | Cost of one publish |
| --- | --- | --- | --- |
| 640 by 480 | 1.23 MB | 4.92 MB | 1.9 ms |
| 800 by 600 | 1.92 MB | 7.68 MB | 2.9 ms |
| 1250 by 750 | 3.75 MB | 15.0 MB | 6.5 ms |

The segment is four times the frame because the texture holds float RGBA rather than bytes. Almost all of the cost is the pixels themselves, not the coordination.

| Step in one publish, at 800 by 600 | Cost |
| --- | --- |
| Reading the surface as RGBA bytes | 1.0 ms |
| Scaling those bytes into the texture memory | 1.8 ms |
| Stamping the control block under its lock | 0.001 ms |

The pixels are not locked. There is one writer, one reader, and a frame replaced whole sixty times a second. The worst a race can do is show one frame half new and half old, for one sixtieth of a second. Locking two megabytes sixty times a second would cost more than it saves, and would let a slow reader hold up the game. The small control block beside the pixels is guarded, because a torn integer there would be a real fault.

An embedded ISO Fitts run at 640 by 480, asked for 60 frames per second, gave the following.

| Measure | Value |
| --- | --- |
| Frames published in the first three seconds | 187 |
| Frames published in one measured second | 60 |
| Rate the bridge reports | 61 fps |

The task ran at its requested rate with the frame going through shared memory every frame.

# Input, going the other way

With no window there are no keyboard events. That matters more than it sounds, because a keyboard-driven environment does not read the event queue at all. It calls `pygame.key.get_pressed` and asks what is held right now. SDL has no key state to report when there is no window, so that call would always answer nothing.

Inside the environment's process, `pygame.key.get_pressed` is therefore replaced. The replacement reports the keys the GUI forwarded through the bridge. This replacement lives only in that process. Anywhere else in LibEMG, pygame behaves exactly as it always did.

Posting synthetic key events instead would look correct and do nothing. `get_pressed` reflects SDL's own view of the physical keyboard, which posted events never reach.

These are the keys the GUI forwards, listed in `FORWARDED_KEYS`.

| Key names | Read by |
| --- | --- |
| `left`, `right`, `up`, `down` | The keyboard controller, as the four cursor directions. |
| `1`, `2`, `3`, `4` | The keyboard controller, for a task whose prediction map covers them. |
| `w`, `a`, `s`, `d` | Forwarded and available. No environment in this release reads them. |
| `space`, `escape` | Forwarded and available. No environment in this release reads them. |

The keyboard is read once per rendered frame rather than through key-down handlers. A game wants to know what is held right now, every frame, not to be told once when a key went down.

Click the game panel before typing. The forwarded keys are the ones the LibEMG window has, so the window has to have the focus.

# The setup screens are generated

No setup screen is hand-written. Each is built from the environment's own configuration, the same declaration the API documentation is built from.

| Environment | Where its settings are declared | Settings offered |
| --- | --- | --- |
| Fitts' Law | `FittsConfig`, a dataclass with typed fields | 17 |
| ISO Fitts' Law | `FittsConfig`, plus the two ring settings `ISOFitts` takes | 19 |
| Curricular Fitts | `CurricularFittsConfig`, a dataclass with typed fields | 19 |
| EMG Hero | The constructor arguments of `EMGHero` | 6 |

Every one of those settings gets a control, with one exception. `save_file`, the environment's own pickle of its log, is left off: the setup screen saves through its **Save** section instead (see [Saving a run](#saving-a-run)), and offering both would be two different answers to where the data went. Scripts can still pass `save_file` as before. Hovering over a setting's name shows the author's own docstring sentence for that field, so what a user reads is what the author wrote where the setting is declared. A setting added to an environment appears on its setup screen without the screen being edited.

The screen is a form in two columns. On the left is how the task is driven and what is saved of it, the things that change between sessions. On the right are the task's settings, each with its name on the left and its unit after the value, under headings.

| Heading | Settings under it |
| --- | --- |
| Trials | Trials, dwell time, trial timeout, time limit |
| Game | EMG Hero's duration, note speeds and the gaps between notes |
| Cursor | Cursor speed and radius, proportional control, cursor space |
| Targets | Target radius and speed, the ISO ring, moving targets |
| Window | Width, height, frame rate, Curricular Fitts' info bar |
| Appearance | Colours, folded away |

Field names are written for code: `num_trials`, `default_timeout`, and they carry no units. How each is named on screen, its unit and its heading are set in `PRESENTATION` in `libemg._gui._environments.registry`. A setting not listed there still appears, under **Other**, with a name made from its field.

The control is chosen from the setting's kind, which is read from its type annotation and its default.

| Kind | Control drawn |
| --- | --- |
| `int` | Number box |
| `float` | Number box, two decimals |
| `bool` | Checkbox |
| `enum` | Drop-down of the allowed values |
| `color` | Colour picker |
| `str` | Text box |
| `path` | Text box |

The `mapping` setting on a Fitts task is the one enumeration. A free text box would let somebody type a value that is only rejected once the task starts. Note that bare `polar` is not one of the choices. The environment accepts `polar+` and `polar-`, which say which way up maps, and raises on anything else.

Colour settings are collected under an **Appearance** disclosure. They are about how the task looks rather than what it measures, and they would otherwise crowd out the settings that change the result.

| Environment | Colour settings under Appearance |
| --- | --- |
| Fitts' Law and ISO Fitts' Law | 5 |
| Curricular Fitts | 6 |
| EMG Hero | 0 |

# Two settings that behave specially

Both come from the same place. A control has to be able to say things a plain number cannot.

**A limit that can be switched off has a box that says Off.** A timeout has no number meaning "off". The box beside it reads **Off** until it is ticked, and ticking it shows the number. The number is kept while the box is unticked, so switching it back on restores it.

| Setting | On screen | Value the task receives |
| --- | --- | --- |
| `timeout` | Off | No timeout |
| `timeout` | Ticked, 2.5 s | 2.5 seconds |
| `game_time` | Off | No time limit |

**A required setting starts somewhere usable.** `num_trials` on a Fitts task has no default, because the environment cannot run without being told how many trials to run. The control starts at 1 rather than at 0, because a task asking for zero trials is not runnable.

# Controllers

A task needs something to drive it. The controller is chosen under **Control**, at the top of the left column. Only the fields the chosen controller uses are shown.

| Controller | What drives the task | Fields shown |
| --- | --- | --- |
| Keyboard | The arrow keys, read every frame. | None |
| Classifier | A running classifier's output, over a socket. | Listens on (address and port), message format, classes in the model for the probabilities format, and which class does what |
| Regressor | A running regressor's output, over a socket. | Listens on (address and port) |

Keyboard is for trying a task out and seeing that it behaves. Classifier and Regressor listen on the address given for a model that is already running elsewhere. A task that takes only one controller, as Curricular Fitts takes only a regressor, says so instead of offering a choice.

A classifier can send one of two formats, chosen under **Message format**.

| Format | Each message | Sent by |
| --- | --- | --- |
| `predictions` | `<class> [speed]` | A script that sends the class itself. The time the message was read stands in for its timestamp. |
| `probabilities` | One probability per class, then the speed if velocity is on, then a timestamp | libemg's `OnlineEMGClassifier`. **Classes in the model** has to match the model. |

A Fitts task has proportional control on by default and then reads a speed from every message. A classifier sending no speed needs it switched off.

Choosing Keyboard also supplies a prediction map, without being asked. A Fitts task turns a prediction into a direction through that map, and the default map covers class indices 0 to 4. The keyboard controller does not produce class indices. It produces pygame key codes, and -1 when nothing is held. Launching with the default map would fail on the first frame with a key error. The map supplied instead is this one.

| Key | Direction |
| --- | --- |
| Up | `N` |
| Down | `S` |
| Right | `E` |
| Left | `W` |
| Nothing held | `NM` |

The four arrows steer, and every other forwarded key maps to no motion.

# Which class does what

A classifier sends class indices. A task wants commands: a direction for a cursor, a lane for EMG Hero. Which class means which command belongs to the model, not to the task, so with the Classifier controller chosen the setup screen shows one class per action.

| Task | Actions, and the class each starts at |
| --- | --- |
| Fitts' Law, ISO Fitts' Law | Up 1, Down 0, Left 4, Right 3, No motion 2 |
| EMG Hero | Lane 1 0, Lane 2 1, Lane 3 3, Lane 4 4, Nothing 2 |
| Curricular Fitts | None. It is driven by a regressor's two outputs. |

The starting classes are the environment's own default map, so leaving them alone changes nothing. The screen warns when two actions are given the same class (the first one listed gets it) and, for the probabilities format, when a class is beyond the number the model sends. The cursor defaults are the five classes of libemg's Fitts example: hand close, hand open, no motion, wrist extension, wrist flexion.

Every class without an action of its own takes the idle one: the cursor holds still, the game plays nothing. So a model with more classes than the task has actions can drive it. A 16-class model can steer a Fitts task with its hand open, power grasp and two wrist postures, and its other classes simply do nothing. Before, a class missing from the map ended the task with a key error. A class of -1 leaves an action with no class at all, and the task accepts a map that never produces it.

The same map works in a script through `PredictionMap`.

```Python
from libemg.environments._base import PredictionMap
from libemg.environments.fitts import Fitts, FittsConfig

# action -> class; any other class holds the cursor still
mapping = PredictionMap.from_actions({'N': 8, 'S': 7, 'E': 1, 'W': 2, 'NM': 0}, default='NM')
fitts = Fitts(controller, FittsConfig(num_trials=20), prediction_map=mapping)
```

The map also decides the class a frame is labelled with for adaptation. The CIIL label is a direction, and the class it is saved as is the class mapped to that direction.

# Saving a run

The **Save** section sits under the controller. Type a folder or pick one with **Browse...**, and every run launched from then on is saved, each into a folder of its own named for the task and the time it started, such as `fitts_2026-10-28_14-03-11`. Leave it empty and nothing is saved. Five boxes choose what goes in.

| Box | Columns it adds |
| --- | --- |
| Timestamps | `timestamp`: when each sample was produced, in `time.time()` seconds. |
| EMG | `emg_0` to `emg_<n>`, one row per EMG sample. Without it the files hold one row per frame instead, and there is nothing to train on. |
| Model outputs | `prediction` and `speed` from a classifier, `output_x` and `output_y` from a regressor. |
| Game state | Cursor tasks: `cursor_x`, `cursor_y`, `target_x`, `target_y`, `target_radius`, `in_target`, `velocity_x`, `velocity_y`. EMG Hero: `lane`, and `note_0` to `note_3` for which lanes have a note on the hit line. |
| Adaptation (CIIL) | `optimal_x` and `optimal_y`, the unit vector from cursor to target, zero inside it. `ciil_class` for a classifier, or `ciil_x` and `ciil_y` for a regressor. EMG Hero: `ciil_lane` and `ciil_class`. |

`trial` is always kept, because the files are split by it. Screen positions are in pixels with y growing downwards, as pygame draws. The optimal direction, the outputs and the CIIL labels are in the model's own space, where an up prediction has y positive.

EMG comes from the online data handler, so it can only be saved when one is streaming in the window. Pass one to `GUI(online_data_handler=...)` or start a device from **Device**, **Streamer**. The setup screen says when there is none.

## What is in a run's folder

| File | What it is |
| --- | --- |
| `trial_<n>.csv` | The training file for trial `n`. One row per EMG sample produced during it, carrying the state of the frame on screen when it was produced. Plain numeric CSV with a header row. |
| `_session_emg.csv` | Every EMG sample the handler received: arrival time, then the channels. The same file continuous screen guided training writes. |
| `_session_environment.csv` | One row per frame the environment drew: its time, its trial and every field it recorded. |
| `recording.json` | What was recorded, the settings, which class each action was mapped to, and the column layout of the trial files. |

The two `_session_` files are the raw halves. The environment sees what it showed and what it was sent but not the EMG; the handler sees the EMG but not the game. When the task ends they are joined. The EMG file stamps every sample with the time its batch was read, which is later than it was produced, so the sample clock is recovered first with `libemg.session.fit_production_clock`, the fit continuous screen guided training cuts its repetitions with. Each sample is then given the frame that was on screen when it was produced. The join runs in the background after **Stop**, and the setup screen says when the files are written.

The join reads only those raw files, so it can be run again at any time, for instance on a run whose window was closed while it was saving.

```Python
from libemg.environments.recording import assemble
assemble('data/fitts_2026-10-28_14-03-11')
```

## Reading a run back for training

`load_recording` puts a run into an `OfflineDataHandler`. `data` holds the EMG, one array per trial, and every other column is per-sample metadata under its own name.

```Python
from libemg.environments.recording import load_recording

odh = load_recording('data/fitts_2026-10-28_14-03-11')
odh += load_recording('data/fitts_2026-10-28_14-20-40')     # runs combine with +

windows, meta = odh.parse_windows(80, 16, metadata_operations={'ciil_class': 'last_sample'})
labels = meta['ciil_class']        # what the user should have been doing
```

It is one call to `get_data` and nothing more, so the files read just as well by hand. The column numbers are in `recording.json` under `trial_files`.

```Python
from libemg.data_handler import OfflineDataHandler, RegexFilter, ColumnFetcher

odh = OfflineDataHandler()
odh.get_data('data/fitts_2026-10-28_14-03-11',
             [RegexFilter('trial_', '.csv', [str(t) for t in range(20)], 'trial')],
             metadata_fetchers=[ColumnFetcher('ciil_class', 21)],
             data_column=list(range(1, 9)), skiprows=1)
```

Windowing takes the mode of each metadata field by default, which now works for every field a run records. It used to cast each field to an integer and count, which failed on the -1 EMG Hero uses for no lane and on the empty CIIL label of a frame with no model output yet, and truncated fields such as `optimal_x`. Non-negative whole-number labels get exactly the mode they always did.

## The CIIL label for a classifier

`ciil_class` is the class the user should have been producing at that moment, by the context-informed incremental learning rule of Campbell et al. (2024), in `libemg.adaptation._base.ciil_direction`.

| Situation | Label |
| --- | --- |
| The cursor is in the target | No motion, so it dwells |
| The cursor is moving in a direction that brings it closer | That direction: any output that helps is kept as it was |
| Otherwise | The direction along whichever axis the target is further away on |

For a regressor, `ciil_x` and `ciil_y` are the tolerant label of Morrell et al. (2025) from `produce_tciil_feedback`, the one Curricular Fitts already feeds its adaptation with. `optimal_x` and `optimal_y` are saved alongside, so a different rule can be applied later without running the task again.

## Saving from a script

The setup screen is a convenience. The same recording works with no GUI at all.

```Python
from libemg.environments.fitts import Fitts, FittsConfig
from libemg.environments.recording import RecordingConfig, EnvironmentRecording, load_recording

config = RecordingConfig(folder='data/fitts_run_1')        # every box ticked by default
fitts = Fitts(controller, FittsConfig(num_trials=20), recording=config)

with EnvironmentRecording(config, online_data_handler=odh):  # logs EMG, joins on exit
    fitts.run()

odh_offline = load_recording(config.folder)
```

Every environment takes `recording=`. A new environment records by calling `self.record_frame(trial, model={...}, game={...}, adaptation={...})` once a frame. Fields are not declared anywhere else first: a field becomes a column the first frame it is recorded in.

# When an environment refuses its settings

An environment checks its own settings and raises when they conflict. It does that in its own process, where a traceback reaches nobody who is looking at the GUI.

The clearest example is an ISO Fitts ring that will not fit. The ring radius has to be smaller than half of each screen dimension, or the targets are drawn off the edge. Asking for a radius of 400 on a 640 by 480 frame gives this.

```
ValueError: Radius between ISO Fitts targets is larger than screen size will allow.
Target distance radius must be less than half the screen dimensions.
Please increase screen width and height or reduce target distance radius.
```

The bridge reserves 2048 bytes for exactly this reason. The runner writes the reason there as well as printing the full traceback to the console, so the setup window shows the message rather than leaving a black rectangle and no explanation. The play window closes on its own and the setup window reads `That task could not start:` followed by the reason.

The settings that can refuse a launch are these.

| Setting | Rule |
| --- | --- |
| `target_distance_radius` | Must be less than half the width and less than half the height. |
| `mapping` | Must be `cartesian`, `polar+` or `polar-`. The drop-down offers only these. |

Anything that goes wrong before the process is spawned is reported on the setup screen directly, because it happens where the window can see it.

# Driving an environment without the GUI

The same pieces work in a script. This runs ISO Fitts offscreen and reads its frames, with no window of any kind.

```Python
import time
from libemg._gui._environments import (EmbeddedEnvironment, ControllerSpec,
                                       build_factory, default_registry)

if __name__ == "__main__":
    spec = default_registry()["iso_fitts"]
    values = spec.defaults()
    values.update({'num_trials': 5, 'width': 640, 'height': 480,
                   'target_distance_radius': 180})

    factory = build_factory(spec, ControllerSpec(kind='keyboard'), values)
    env = EmbeddedEnvironment('demo', factory, *spec.frame_size(values)).start()

    time.sleep(3.0)
    print(env.status())
    # {'running': True, 'finished': False, 'frames': 187, 'fps': 60.8,
    #  'generation': 187, 'error': ''}

    env.send_input({'right'})   # the same keys the GUI forwards
    frame = env.pixels().reshape(480, 640, 4)
    env.stop()
```

`pixels` returns the array a raw texture would be backed by. `status` is what the play window's status line is drawn from. `send_input` is what the GUI calls once per rendered frame.

The factory has to be picklable, which is why it is a small class holding settings rather than a closure. A lambda capturing a configuration cannot be sent to another process, and the failure it produces names an anonymous function rather than anything a user could act on.

# What is not built yet

Embedding is an addition, not a replacement. Every environment still runs standalone exactly as it did before, in its own window, from its own script. Nothing on this page changes that.

| Not implemented | What happens instead |
| --- | --- |
| Mouse control of a task | The pointer position and button are forwarded every frame. No environment in this release reads them. |
| Mapping a regressor's outputs | A regressor's first output is x and its second y. Which output drives which axis, or flipping one, is not set on the setup screen. |
| Saving frames as they happen | An environment writes its frames when it stops. A run lost to the whole GUI being killed keeps its EMG but not its frames. |
| Saving a set of settings | Settings last as long as the window. **Reset settings** returns them to the environment's own defaults. |
| Two tasks at once | One environment runs at a time. Launching while one is running says so rather than starting a second. |
