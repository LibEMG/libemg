[View Source Code](https://github.com/ECEEvanCampbell/CoAdaptUnderCurriculum)

The offline performance of a myoelectric model (e.g., $R^2$, mean absolute error on screen-guided training data) does not necessarily reflect how usable it is once a person is actually in the loop. A model that looks excellent on a calibration set can still drift or feel unresponsive online. `LibEMG` provides an **adaptation suite** that keeps improving a model *while the user operates it*, using the ongoing interaction as a source of labels. This is the mechanism behind context-informed incremental learning (see [Context-Informed Incremental Learning Improves Throughput and Reduces Drift in Regression-Based Myoelectric Control, Morrell et al. 2025](https://github.com/ECEEvanCampbell/CoAdaptUnderCurriculum)).

This tutorial builds the minimal pipeline: a **within-subject initialization** from screen-guided training (SGT) data, followed by an **online environment that adapts the model** as the user plays through it.

# Architecture
Adaptation runs as **four cooperating processes** that communicate over shared memory. The `libemg.adaptation._base.get_edil_adaptation_objects` helper pre-builds every shared-memory item and `OutputWriter` needed to wire them together (each modality buffer and its sample counter are paired on a shared lock so readers always see a consistent snapshot), so you never allocate them by hand.

1. **`OnlineEMGRegressor`** — the live model. It windows the incoming EMG, extracts features, predicts, and publishes each feature vector (`model_input`) with a timestamp to shared memory. It also watches an `adapt_flag`; when the adaptation manager raises it, the regressor hot-swaps in the newly adapted model. The flag is consulted through its state block, so the common answer that nothing has changed costs a handful of integers rather than three locked reads of the variable.
2. **Environment** (`CurricularFitts`) — the task the user performs. Every frame it turns the current cursor/target geometry into a *pseudo-label* through a `feedback_handle`, and writes that label (`environment_feedback`) with the same timestamp and trial number.
3. **`MemoryManager`** — joins each `environment_feedback` row to the `model_input` row with the matching timestamp, appends the pair to a `Memory`, and saves one memory slice (`memory_<trial>.pkl`) at the end of every trial. It is woken by the environment's write instead of asking whether one has happened.
4. **`AdaptationManager`** — loads memory slices (seeded by the SGT data for stability), calls `model.adapt(memory)`, saves the updated model (`mdl<N>.pkl`), and — if `notify=True` — sets the `adapt_flag` so the live model reloads it. It is woken by the memory manager's write in the same way.

The loop is therefore: **predict → act → pseudo-label → remember → adapt → reload → predict …**, all without pausing the user.

**Neither manager spins.** Each one waits on a notifier slot and is woken by the write it used to poll for. The memory manager's old loop copied both the feedback buffer and the whole model-input buffer on every pass, just to compare one counter. Their signatures and their behaviour are otherwise unchanged.

By default the adaptation manager still calls `model.adapt` on every pass of its loop, including passes where no new slice has arrived. Set its `wait_for_memory` attribute to adapt only when a slice actually arrives:

```Python
adaptation_manager.wait_for_memory = True
```

**You supply three things.** The suite is model-agnostic; it only assumes:
- a **model** with `predict(features)`, `adapt(memory)`, `save(path)`, and `load(path)` (the reference repository uses a small MLP / Transformer),
- a **memory** subclassing `libemg.adaptation.memory.Memory` (implementing `append`, `reset`, `save`, `load`, and `__add__`),
- a **feedback function** mapping game state to a pseudo-label — `libemg.adaptation._base.produce_tciil_feedback` is a ready-made regression example.

**Note:** The snippets below use the `Myo Armband` and a 2-DOF wrist regressor (flexion/extension and radial/ulnar deviation). Any hardware works by switching the `streamer`, `window_size`, and `window_increment`.

# Building the Same Loop with Hooks

The four processes above pass work between them by writing to shared memory. A write now announces itself, so the memory and adaptation stages can be written as observers rather than as loops of their own. `libemg.adaptation.hooks` provides them. The layer they are built on is described in the [reactive pipelines guide](../reactive/reactive_doc.md).

**The existing API is unchanged.** `MemoryManager` and `AdaptationManager` keep their signatures and their behaviour, and Step 2 below still runs exactly as written. The hooks are an alternative way to assemble the same loop, with the same behaviour, for a pipeline that is already reactive.

Three hooks cover the chain:

- `MemoryHook` observes `environment_feedback`, pairs each judgement with the model input that produced it, and appends the pair to a memory. A judgement carrying a new trial number closes the slice: the memory is saved as `memory_N.pkl` and `memory_flag` is advanced.
- `AdaptationHook` observes `memory_flag`, loads whatever slices have appeared since it last looked, calls `model.adapt`, saves the new weights as `mdlN.pkl`, and advances `adapt_flag`.
- `ModelSwapHook` observes `adapt_flag` and calls a function of your own with the new model number. It is for anything outside the streaming process that wants to know the model changed, such as a plot marking the moment or a counter of how often adaptation reached the user.

**The three stages disagree about what a change is.** That disagreement is the argument the whole reactive layer is built around. An item publishes facts, and each observer pairs those facts with its own criterion.

| Stage | Dirty when | Why |
| --- | --- | --- |
| Memory | Any feedback row arrives, `OnCommit()` | Every judgement the person produced is data worth keeping. |
| Adaptation | A slice is finished, `OnNewSlice()` | Training on a fraction of a trial is worse than waiting for the whole one. |
| Predictor | A new model is published | A swap is rare, and acting on one costs a model load. |

`MemoryHook` reads `model_input` as well, but its arrival is not a reason to run. It is declared with a criterion that is never dirty, so the inputs are delivered on every run without ever firing one. Feedback rows are joined to inputs **by timestamp** rather than by position, because the two are written by different processes at different rates and their indices do not correspond.

**Assembling the graph.** This replaces the `MemoryManager` and `AdaptationManager` block of Step 2. Everything else in Step 2 is unchanged, including the shared-memory items from `get_edil_adaptation_objects`.

```Python
from libemg.adaptation.hooks import MemoryHook, AdaptationHook
from libemg.reactive import ReactiveGraph, default_notifier_pool

# Every item the two hooks touch, with duplicate tags removed.
adaptation_items = list({item[0]: item for item in
                         memory_manager_smi + adaptation_manager_smi + model_smi}.values())

# Sharing the output writers' pool is what lets their writes wake the hooks.
graph = ReactiveGraph(adaptation_items, notifier_pool=default_notifier_pool())

graph.add(
    MemoryHook(
        memory   = MyMemory(...),   # empty memory of the same type as the SGT seed
        save_dir = ADAPT_DIR,       # writes memory_N.pkl
    ),
    executor='memory')

graph.add(
    AdaptationHook(
        model              = model,
        load_dir           = ADAPT_DIR,   # reads memory_N.pkl (== MemoryHook.save_dir)
        save_dir           = ADAPT_DIR,   # writes mdlN.pkl    (== OnlineEMGRegressor.file_path)
        initial_memory_loc = ADAPT_DIR + 'sgt_memory.pkl',   # SGT seed for stability
        stop_after         = NUM_TRIALS - 1,
        notify             = True,        # advance adapt_flag so the live model hot-swaps
    ),
    executor='adapt')

graph.start()

env.run_helper(block=False)   # spawn the game
env.process.join()            # adapt for as long as the user is playing
graph.stop()
```

The two path linkages are the ones Step 2 describes. The adaptation hook's `load_dir` equals the memory hook's `save_dir`, and its `save_dir` equals the online regressor's `file_path`. Each hook runs in its own executor process, so a long training call never delays memory assembly.

**One behavioural difference is deliberate.** `AdaptationManager` calls `model.adapt` on every pass of its loop, including passes where no new memory arrived, so it retrains continuously on unchanged data and republishes a model each time. `AdaptationHook` trains when a slice arrives. Pass `continuous=True` to restore the old behaviour. The same choice is available on the manager through the `wait_for_memory` attribute shown above.

**Diagnostics.** `MemoryHook.unmatched()` counts feedback rows that had no model input to pair with. A non-zero count means the predictor's input buffer is too small for the delay between a prediction and the environment's judgement of it, and that training data is being lost. `MemoryHook.appended()` counts the pairs that reached the memory. `MemoryHook.flush()` closes the slice in progress, which is how the final trial gets saved; a slice is otherwise closed by the arrival of the next trial number.

**Acting where the model lives.** `ModelSwapHook` runs in whichever executor you give it, which is not the process holding the live model. For work that has to happen in the streaming process when a new model is loaded, such as resetting a decision history that the old model's outputs populated or re-fitting a scaler, override `OnlineStreamer.on_model_update(number)`. Override it in a subclass rather than assigning a lambda to it. The streamer is a spawned process, and a lambda cannot be pickled.

**What made the flags observable.** `SharedMemoryManager.apply(tag, fn, count_fn=None)` is the write path for a value that is set rather than appended. `SharedMemoryOutputWriter` now writes through it, which is what turns an adaptation flag or a row of environment feedback into something a hook can be triggered by at all. It takes the variable's lock once, where the old pair of `modify_variable` calls took it twice, so a reader can no longer see a count running ahead of the data it counts.

**A bug fixed on this path.** `OnlineStreamer.load_emg_predictor` used an exact type check, `type(loaded) == EMGPredictor`. `EMGClassifier` and `EMGRegressor` both subclass `EMGPredictor`, and they are exactly what an adaptation run saves, so a saved classifier was misrouted into `.model`. The next prediction then failed on a classifier having no `predict_proba`, after the swap had already been reported as successful. The check now uses `isinstance`.

# Step 1 — Within-Subject Initialization
First, record a short screen-guided training session and fit an initial model to it. This is the same collection flow used elsewhere in `LibEMG`; for regression we prompt the four wrist directions and record continuous labels.

```Python
import libemg
from libemg.streamers import myo_streamer

# Stream from the device and collect calibration (SGT) data.
streamer, sm_items = myo_streamer()
odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=sm_items)

gui = libemg.gui.GUI(odh, args={
    'media_folder': 'media/',        # regression prompts (wrist flexion/extension, radial/ulnar)
    'data_folder':  'data/sgt/',
    'num_reps':     5,
    'rep_time':     5,
    'auto_advance': True,
})
gui.start_gui()
streamer.stop()
```

Next, parse those recordings with an `OfflineDataHandler`, fit your model, and — crucially — save the SGT data **as a memory slice**. That slice seeds the adaptation manager so early online updates don't wander away from a known-good starting point.

```Python
# Parse the SGT recordings.
offdh = libemg.data_handler.OfflineDataHandler()
offdh.get_data('data/sgt/', regex_filters, metadata_fetchers, delimiter=',')

# Fit the initial within-subject model (implements predict/adapt/save/load).
model = MyModel(...)
model.calibrate(offdh)

# Seed the adaptation with the SGT data as the first memory slice.
initial_memory = MyMemory(...)          # subclass of libemg.adaptation.memory.Memory
initial_memory.load_from_odh(offdh)
initial_memory.save('data/adapt/sgt_memory.pkl')
```

The same `model` object is handed to both the live regressor and the adaptation manager below; because they run in separate processes each gets its own copy.

# Step 2 — Launch the Online Adaptive Environment
Now assemble the four processes. Start by requesting the pre-wired shared-memory items and output writers.

```Python
from libemg.adaptation._base import get_edil_adaptation_objects, produce_tciil_feedback
from libemg.adaptation.managers import MemoryManager, AdaptationManager
from libemg.environments.controllers import RegressorController
from libemg.environments.curricular_fitts import (
    CurricularFitts, CurricularFittsConfig, RadiusTargetGenerator,
)

NUM_FEATURES, NUM_DOFS, NUM_TRIALS = 64, 2, 80
ADAPT_DIR = 'data/adapt/'   # adapted models (mdl<N>.pkl) AND memory slices (memory_<trial>.pkl)

(model_smi, model_ow,
 environment_smi, environment_ow,
 adaptation_manager_smi, adaptation_manager_ow,
 memory_manager_smi, memory_manager_ow) = get_edil_adaptation_objects(
    num_features=NUM_FEATURES, num_outputs=NUM_DOFS)
```

**The live model.** Pass `model_ow`/`model_smi` so it publishes `model_input` and exposes the `active_flag`/`adapt_flag`. Its `file_path` is where it reloads adapted models from.

```Python
streamer, sm_items = myo_streamer()
odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=sm_items)

emg_regressor = libemg.emg_predictor.EMGRegressor(model)   # the SGT-trained model
online_regressor = libemg.emg_predictor.OnlineEMGRegressor(
    offline_regressor   = emg_regressor,
    online_data_handler = odh,
    window_size = 100, window_increment = 40,
    features = ['WENG'],
    output_writers = model_ow,          # publishes model_input (features + timestamp)
    smm = True, smm_items = model_smi,  # exposes active_flag / adapt_flag / model_input
    file_path = ADAPT_DIR,              # reloads mdl<N>.pkl from here when adapt_flag is set
)
online_regressor.run(block=False)
```

**The environment.** The `feedback_handle` converts cursor/target geometry into a pseudo-label; `environment_ow` writes it out for the memory manager.

```Python
controller = RegressorController()
config = CurricularFittsConfig(
    feedback_handle = produce_tciil_feedback,
    num_trials      = NUM_TRIALS,
    controller_map  = [1, -1],
)
env = CurricularFitts(
    controller, config,
    target_generator = RadiusTargetGenerator(config, F=0, P=0),
    environment_ow   = environment_ow,   # writes environment_feedback (label + timestamp + trial)
    save_file        = ADAPT_DIR,
)
```

**The memory and adaptation managers.** Two path linkages must line up: the adaptation manager's `load_dir` equals the memory manager's `save_dir` (where slices are written), and its `save_dir` equals the online regressor's `file_path` (where adapted models are read back). Setting `notify=True` is what closes the loop by raising the `adapt_flag`.

```Python
memory_manager = MemoryManager(
    memory   = MyMemory(...),        # empty memory of the same type as the SGT seed
    smi      = memory_manager_smi,
    ow       = memory_manager_ow,
    save_dir = ADAPT_DIR,            # writes memory_<trial>.pkl
)

adaptation_manager = AdaptationManager(
    model              = model,      # adapts its own copy of the SGT model
    smi                = adaptation_manager_smi,
    ow                 = adaptation_manager_ow,
    initial_memory_loc = ADAPT_DIR + 'sgt_memory.pkl',   # SGT seed for stability
    load_dir           = ADAPT_DIR,  # reads memory_<trial>.pkl  (== MemoryManager.save_dir)
    save_dir           = ADAPT_DIR,  # writes mdl<N>.pkl         (== OnlineEMGRegressor.file_path)
    stop_condition     = lambda n: n >= NUM_TRIALS - 1,
    notify             = True,       # raise adapt_flag so the live model hot-swaps
)
```

Finally, launch them. Run the environment and memory manager in the background and block on the adaptation manager; it returns once `stop_condition` is met. Then tear everything down.

```Python
env.run_helper(block=False)               # spawn the game
memory_manager.run_helper(block=False)    # spawn the memory assembler
adaptation_manager.run_helper(block=True) # adapt until stop_condition; blocks here

# Cleanup once adaptation ends.
env.process.join()
memory_manager.signal.set(); memory_manager.join()
online_regressor.odh.stop_all()
online_regressor.stop_running()
streamer.stop()
```

# Result
As the user completes trials, memory slices accumulate, the adaptation manager retrains on them (seeded by the SGT slice), and the live model is swapped out mid-session — so control quality improves *during* use rather than only between sessions. To make the model non-adaptive for a baseline comparison, build everything identically but pass `notify=False` to the `AdaptationManager`: memories are still collected, but the `adapt_flag` is never raised and the live model stays fixed.
