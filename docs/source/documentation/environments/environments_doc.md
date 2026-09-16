LibEMG's environments are its real-time tasks. Each one is a pygame game with its own loop, its own drawing and its own results log. Until now each one opened a window of its own, and reaching it meant writing a script first.

The Environments window runs them inside the LibEMG window instead. Open the LibEMG GUI, choose **Environments**, then **Launch Environment**. A task is picked from a list at the top, set up on the screen that appears, and launched into a panel beside its settings. The window builds its own environment, so it opens with no `OnlineDataHandler` and no hardware attached.

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
| Fitts' Law | `FittsConfig`, a dataclass with typed fields | 18 |
| ISO Fitts' Law | `FittsConfig`, plus the two ring settings `ISOFitts` takes | 20 |
| Curricular Fitts | `CurricularFittsConfig`, a dataclass with typed fields | 19 |
| EMG Hero | The constructor arguments of `EMGHero` | 7 |

Every one of those settings gets a control. The help text beside a control is the author's own docstring sentence for that field, so what a user reads is what the author wrote where the setting is declared. A setting added to an environment appears on its setup screen without the screen being edited.

The control is chosen from the setting's kind, which is read from its type annotation and its default.

| Kind | Control drawn |
| --- | --- |
| `int` | Integer spinner |
| `float` | Float spinner |
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

**Zero means off, for an optional number.** A timeout that can be switched off has no number meaning "off", and a spinner cannot show a blank. Zero is the only value a spinner can offer that is not a real duration. So zero is read as off. Without this a timeout left at zero would fail every trial the instant it began.

| Setting | Value in the control | Value the task receives |
| --- | --- | --- |
| `timeout` | 0 | No timeout |
| `timeout` | 2.5 | 2.5 seconds |
| `game_time` | 0 | No time limit |
| `save_file` | Empty | Nothing is saved |

**A required setting says so and still starts somewhere usable.** `num_trials` on a Fitts task has no default, because the environment cannot run without being told how many trials to run. Its label is marked, and the control starts at 1 rather than at 0. A control showing zero looks like a setting rather than a blank, and a task asking for zero trials is not runnable.

| Setting | Label shown | Starting value |
| --- | --- | --- |
| `num_trials` on Fitts and ISO Fitts | `Num Trials  (required)` | 1 |

# Controllers

A task needs something to drive it. The controller is chosen at the top of the setup screen, above the settings.

| Controller | What drives the task | Fields shown |
| --- | --- | --- |
| Keyboard | The arrow keys, read every frame. | None |
| Classifier | A running classifier's output, over a socket. | Address, port, classes |
| Regressor | A running regressor's output, over a socket. | Address, port |

Keyboard is for trying a task out and seeing that it behaves. Classifier and Regressor listen on the address given for a model that is already running elsewhere.

Choosing Keyboard also supplies a prediction map, without being asked. A Fitts task turns a prediction into a direction through that map, and the default map covers class indices 0 to 4. The keyboard controller does not produce class indices. It produces pygame key codes, and -1 when nothing is held. Launching with the default map would fail on the first frame with a key error. The map supplied instead is this one.

| Key | Direction |
| --- | --- |
| Up | `N` |
| Down | `S` |
| Right | `E` |
| Left | `W` |
| Nothing held | `NM` |

The four arrows steer, and every other forwarded key maps to no motion. A Fitts task looks a prediction up in that map with no fallback, so a key with no entry would end the task with an error rather than being ignored.

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
| Saving a set of settings | Settings last as long as the window. **Reset settings** returns them to the environment's own defaults. |
| Two tasks at once | One environment runs at a time. Launching while one is running says so rather than starting a second. |
