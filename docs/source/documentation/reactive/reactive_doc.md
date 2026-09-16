Online processing in LibEMG used to work by asking. A classifier sat in a loop asking "is there a window yet?", and asking was expensive: each question copied an entire shared-memory buffer and, if a filter was installed, ran that filter over all of it, only to read a single counter and throw the data away. Several consumers meant several such loops, each independently re-deriving the same filtered signal and the same features, with nowhere to put a result another process could use.

The reactive layer replaces asking with being told. A write to a shared-memory item announces itself, and anything hooked into that item is notified and decides for itself whether the change matters.

The change is measurable. Streaming 1000 Hz of 8-channel data through a classifier with a bandpass and notch filter installed, window 200 and increment 50:

| Streaming loop | CPU used by the streaming process |
| --- | --- |
| Polling, the original loop | 93.7% |
| Reactive, hooked | 10.7% |

Same predictions, 8.8 times less CPU. End-to-end latency from the final sample of a window to a prediction on the wire measures 0.57 ms on average.

**Existing code does not need to change.** `OnlineEMGClassifier`, `OnlineEMGRegressor` and `OnlineDataHandler` keep their signatures and their behaviour. The streaming loop switched underneath them.

# Why dirtiness cannot be a flag on the item

The obvious design is to mark an item "dirty" when it changes and "clean" when somebody handles it. That breaks as soon as an item has more than one observer, because whichever observer clears the flag starves the rest.

It fails for a second and more interesting reason: observers legitimately disagree about what counts as a change.

- A **filter** wants to run on a single new sample. One sample in, one filtered sample out.
- A **windowing** stage does not care until a full window increment has accrued. Forty-nine new samples are not a window.
- A **live plot** wants thirty updates a second no matter how fast samples arrive. Anything more is invisible.

Given the same item in the same state, those three answer "has it changed enough to act?" differently. So the item does not hold an opinion. It publishes facts, and each observer pairs those facts with its own criterion and its own memory of what it last consumed.

- **Propagation** is the notification, and it goes to everybody hooked in.
- **Dirtiness** is a per-observer judgement made against shared facts.

# What a stateful item publishes

Every shared-memory item now carries a small block of counters alongside its buffer, in its own segment. Reading it costs a handful of integers rather than a copy of the buffer, which is what makes it affordable to consult often.

| Fact | Meaning |
| --- | --- |
| `generation` | Writes since the item was created. Unchanged means nothing happened. |
| `total_samples` | Rows ever committed. What a volume-based criterion compares against. |
| `epoch` | Bumped on reset, so a reader can tell a reset from a long silence. |
| `closed` | The writer has finished. This is how a finite pipeline knows to stop. |
| `commits`, `dropped` | Diagnostics, including rows overwritten before anyone read them. |
| `t_last_ns` | When the newest write landed, for latency accounting. |

Read it through the data handler:

```Python
state = odh.get_state('emg')
print(state.total_samples, state.generation, state.closed)
```

Because the block lives in shared memory and the wake-up is a synchronisation primitive shared between processes, an observer does not have to live in the process that did the writing.

# Criteria

A criterion is an observer's own definition of a meaningful change.

| Criterion | Dirty when | Suits |
| --- | --- | --- |
| `OnCommit()` | Any write at all | Filters, loggers |
| `OnSamples(n)` | `n` new samples have accrued | Windowing |
| `AfterSamples(n)` | `n` accrued, then skip the backlog | Models that must not fall behind |
| `Periodic(hz)` | At most `hz` times a second, and only if something changed | Plots, probes |
| `WhenClosed()` | The writer declares itself finished | Summarising a finite run |
| `Always()` | Whenever asked | Heartbeats |
| `AllOf(...)`, `AnyOf(...)` | Combinations | Multimodal inputs |
| `Custom(fn, name)` | Whatever you write | Anything else |

`OnSamples` and `AfterSamples` share a threshold and differ in what they do about a backlog. `OnSamples` advances by exactly its increment on each firing, so a burst that delivers three increments at once produces three firings and window boundaries stay evenly spaced. `AfterSamples` jumps to the newest total, so the same burst produces one firing on the latest data. Use the first when every window matters and the second when freshness matters more than completeness.

Writing your own takes one method:

```Python
from libemg.reactive import Criterion

class OnEnergy(Criterion):
    """Dirty once the accumulated signal energy passes a threshold."""
    def __init__(self, threshold):
        self.threshold = threshold

    def is_dirty(self, snapshot, memory):
        return snapshot.total_samples - memory.samples >= self.threshold

    def describe(self):
        return f'OnEnergy({self.threshold})'
```

`is_dirty` reads the item's published facts and the observer's memory and returns a judgement without mutating either. `describe` names the criterion in the event log so a recorded decision can be read back and understood.

# Hooks

A hook declares what it observes and what it writes. The runtime does the attaching, the locking and the committing, so a hook never takes a lock and never commits. That is what makes one testable on its own: build the input dict by hand, call `step`, and check what comes back.

Here is one written from scratch. `RmsHook` is not part of LibEMG; it is an example of the shape your own hooks take, and it is reused further down to show how a hook joins a pipeline.

```Python
from libemg.reactive import Hook, Input, Output, OnSamples, WINDOW
import numpy as np

class RmsHook(Hook):
    def __init__(self):
        super().__init__(
            'rms',
            inputs=[Input('filtered_emg', OnSamples(50), mode=WINDOW, size=200)],
            outputs=[Output('rms', (100, 8), np.double)],
        )

    def step(self, data, snapshots):
        window = data['filtered_emg']
        return {'rms': np.sqrt(np.mean(window ** 2, axis=0, keepdims=True))}
```

An input's `mode` says how the data should arrive: `WINDOW` for the newest `size` rows oldest-first, `DELTA` for everything since this hook last consumed, `LATEST` for one row, `FULL` for the whole buffer newest-first as `get_data` returns it, and `STATE` for no data at all when only the counters matter.

Anything a hook cannot pickle, such as a model or a socket, is built in `setup` rather than in `__init__`, because on Windows the executor is a spawned process and the hook has to reach it.

Built-in hooks cover the common stages: `FilterHook`, `FeatureHook`, `ProbeHook`, and `CallbackHook` for wrapping a plain function.

# Incremental learning

Adapting a model while a person is using it is built on these same primitives. `libemg.adaptation.hooks` adds three more: `MemoryHook` assembles training data from what the environment judged, `AdaptationHook` folds a finished slice into the model, and `ModelSwapHook` reacts to newly adapted weights. They are a worked example of the argument above, since the three stages watch one chain of items and disagree about what counts as a change. The loop they build is documented in the [online adaptation guide](../adaptation/adaptation_doc.md).

# The cascade

A hook's output is itself a stateful item, so committing to it notifies that item's observers in turn. That is the whole mechanism: `emg` to `filtered_emg` to `features` to `predictions` needs no coordinating loop, because each stage wakes the next.

```Python
from libemg.reactive import ReactiveGraph, FilterHook

graph = ReactiveGraph(shared_memory_items)
graph.add(FilterHook('filter', 'emg', 'filtered_emg', fi=fi, shape=(2000, 8)))
graph.add(RmsHook(), executor='features')   # the hook defined in the section above
graph.start()
...
graph.stop()
```

Two consequences are worth being explicit about, because they are what the old design could not express.

**An item may have any number of observers.** Filtering happens once and every consumer reads the result, instead of each consumer filtering the same samples again.

**A model may hook whichever stage it wants.** Because what a hook observes is declared rather than hard-wired, subscribing a model to `features` and subscribing it to raw `emg` are the same code path. A model that takes raw windows and one that takes features differ by one line.

Hooks are grouped into executors, one process each. Hooks in the same executor share a wake-up and run in sequence, which is what you want for stages that are individually cheap. A hook that is expensive, or that must not be delayed by its neighbours, belongs in its own executor.

# Processes and threads

A process executor reaches its process by pickling its hooks, which has two consequences worth knowing before you hit them.

A hook that closes over a lambda or a local function cannot be pickled. Neither can one holding a live handle such as an open socket or a plotting window. Register such a hook with `in_process=True` and it runs as a thread in the process that built the graph instead:

```Python
graph.add(ProbeHook('scope', 'emg', lambda data, snaps: plot(data)),
          executor='scope', in_process=True)
```

This is also the answer for a probe that draws. A GUI toolkit will not accept calls from another process, so anything touching a window has to run where the window lives.

The trade is the interpreter lock: a threaded hook shares this process's, so heavy numeric work belongs in a process executor. Numpy releases the lock for the arithmetic itself, so probes and light transforms are usually fine as threads.

The graph refuses to start rather than misbehave later. Two hooks writing the same item, a hook observing an item nothing declares or produces, a duplicate hook name and a cycle are all rejected before any process spawns. So is a hook that cannot be pickled, with a message naming the hook and the three ways out, because the alternative is a bare pickling error raised from inside multiprocessing about an anonymous function.

# How a graph over stored data finishes

A writer that runs out of data calls `mark_closed`. Observers see `closed` on the item's state block, and an executor whose watched items are all closed with no criterion still dirty flushes and exits. Without that, a pipeline over a recording could only be stopped from outside.

# The debug log

The reactive layer records every decision it makes, which matters because the failure you hit is usually "why did this not fire?" or "why did it fire that often?"

```Python
from libemg.event_log import EventLog

log = EventLog(path='reactive.log')
graph = ReactiveGraph(shared_memory_items, log=log)
```

Events carry a wall-clock timestamp, the emitting process, the item that changed, the observer concerned, the criterion applied and the outcome. Kinds are `commit`, `notify`, `evaluate`, `dirty`, `clean`, `invoke`, `complete`, `drop`, `error` and `lifecycle`.

Recorded lines look like this, and the pair below is the design's central claim in evidence. The same item, at the same instant, judged differently by two observers:

```
1789068702.012910  pid=44620  clean  origin=doubled  observer=windowmean  criterion=OnSamples(25)  generation=1  total_samples=1
1789068702.017528  pid=46664  dirty  origin=emg      observer=doubler     criterion=OnCommit()     generation=3  total_samples=3
```

`summary()` gives the counts that tell you whether a criterion is set sensibly, since a criterion that is always dirty or never dirty is usually a mistake:

```Python
log.stop()
print(log.summary())
# {'kinds': {...},
#  'invocations': {'doubler': 200, 'windowmean': 8},
#  'dirty': {'doubler': 200, 'windowmean': 8},
#  'clean': {'windowmean': 211, 'doubler': 211}}
```

Logging is cheap to leave on. An event is a small tuple pushed onto a queue by the process that observed it, and one drain thread in the owning process does all the formatting and file writing. Pass `kinds=` to filter in the emitting process so excluded events cost nothing beyond the check. With no log installed, a null object absorbs every call.

An online streamer takes a log the same way:

```Python
classifier.install_event_log(log)
```

# What changed inside the library

**Online streamers no longer poll.** `OnlineStreamer._run_helper` was an unthrottled loop asking `window_trigger_function_handle` whether a window was ready, and that question copied and filtered the whole buffer. It now subscribes to the items it consumes and blocks until it is woken, then applies `OnSamples(window_increment)` to decide whether to run. Everything else is untouched: the same startup, window, prediction and postprocessing handles run in the same order in the same separate process.

If you have replaced `window_trigger_function_handle` with your own predicate, the original loop is used automatically, because an arbitrary predicate cannot be restated as a per-item criterion. Setting `classifier.reactive = False` selects it explicitly.

**Writers commit rather than read-modify-write.** Streamers used to prepend samples with `modify_variable` and a whole-buffer `vstack`, then update the counter in a second locked call. `commit` does both under one lock acquisition, advances the state block and wakes subscribers. Taking the lock once also closes a real gap: writing the buffer and its counter separately let a reader observe a count running ahead of the data it had just copied.

**A reset now moves the state too.** `OnlineDataHandler.reset` zeroes the counters and bumps the epoch. An observer that missed a reset would otherwise compare against a total that had gone backwards and conclude nothing had arrived.

**The old read API still works, unchanged.** `get_data`, `get_variable`, `get_variables`, `get_samples_since` and `modify_variable` behave exactly as before, and the newest-first buffer layout is preserved. The state block is additive.

# Notification across processes

Each executor owns one wake-up slot and one blocking wait. A writer signals the slots subscribed to the item it just wrote, which it finds in that item's state block, so a writer that started before an observer existed still reaches it.

Slots come from a `NotifierPool` created up front, because a synchronisation primitive cannot be looked up by name after the fact on every platform LibEMG supports. Everything built in one process shares one pool via `default_notifier_pool()`, which is how a streamer started early in a script can wake a classifier constructed later in the same script.

A writer that never obtains the pool still works correctly. Its commits update the state block, and observers fall back to re-reading that block at their fallback interval, which defaults to 2 ms. That fallback reads a few integers instead of copying and filtering a buffer, so it is still orders of magnitude cheaper than the loop it replaces; holding the pool turns a cheap check into no check at all. The fallback is also the safety net that keeps a lost notification from becoming a hang.

# Hooking a live stream directly

For one or two observers on a live stream, the data handler is enough and you do not need to build a graph:

```Python
from libemg.reactive import ProbeHook

odh.install_hook(ProbeHook('watch', 'emg', print, hz=5))
odh.start_hooks()
...
odh.stop_hooks()
```
