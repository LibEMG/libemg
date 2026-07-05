[View Source Code](https://github.com/ECEEvanCampbell/CoAdaptUnderCurriculum)

The offline performance of a myoelectric model (e.g., $R^2$, mean absolute error on screen-guided training data) does not necessarily reflect how usable it is once a person is actually in the loop. A model that looks excellent on a calibration set can still drift or feel unresponsive online. `LibEMG` provides an **adaptation suite** that keeps improving a model *while the user operates it*, using the ongoing interaction as a source of labels. This is the mechanism behind context-informed incremental learning (see [Context-Informed Incremental Learning Improves Throughput and Reduces Drift in Regression-Based Myoelectric Control, Morrell et al. 2025](https://github.com/ECEEvanCampbell/CoAdaptUnderCurriculum)).

This tutorial builds the minimal pipeline: a **within-subject initialization** from screen-guided training (SGT) data, followed by an **online environment that adapts the model** as the user plays through it.

# Architecture
Adaptation runs as **four cooperating processes** that communicate over shared memory. The `libemg.adaptation._base.get_edil_adaptation_objects` helper pre-builds every shared-memory item and `OutputWriter` needed to wire them together (each modality buffer and its sample counter are paired on a shared lock so readers always see a consistent snapshot), so you never allocate them by hand.

1. **`OnlineEMGRegressor`** — the live model. It windows the incoming EMG, extracts features, predicts, and publishes each feature vector (`model_input`) with a timestamp to shared memory. It also watches an `adapt_flag`; when the adaptation manager raises it, the regressor hot-swaps in the newly adapted model.
2. **Environment** (`CurricularFitts`) — the task the user performs. Every frame it turns the current cursor/target geometry into a *pseudo-label* through a `feedback_handle`, and writes that label (`environment_feedback`) with the same timestamp and trial number.
3. **`MemoryManager`** — joins each `environment_feedback` row to the `model_input` row with the matching timestamp, appends the pair to a `Memory`, and saves one memory slice (`memory_<trial>.pkl`) at the end of every trial.
4. **`AdaptationManager`** — loads memory slices (seeded by the SGT data for stability), calls `model.adapt(memory)`, saves the updated model (`mdl<N>.pkl`), and — if `notify=True` — sets the `adapt_flag` so the live model reloads it.

The loop is therefore: **predict → act → pseudo-label → remember → adapt → reload → predict …**, all without pausing the user.

**You supply three things.** The suite is model-agnostic; it only assumes:
- a **model** with `predict(features)`, `adapt(memory)`, `save(path)`, and `load(path)` (the reference repository uses a small MLP / Transformer),
- a **memory** subclassing `libemg.adaptation.memory.Memory` (implementing `append`, `reset`, `save`, `load`, and `__add__`),
- a **feedback function** mapping game state to a pseudo-label — `libemg.adaptation._base.produce_tciil_feedback` is a ready-made regression example.

**Note:** The snippets below use the `Myo Armband` and a 2-DOF wrist regressor (flexion/extension and radial/ulnar deviation). Any hardware works by switching the `streamer`, `window_size`, and `window_increment`.

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
