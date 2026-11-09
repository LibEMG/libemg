This example builds a four-stage pipeline where each stage wakes the next, and then reads back the debug log to see exactly why each stage fired. It runs against a mock streamer, so no hardware is needed.

The pipeline is `emg` to `filtered_emg` to `features` to `predictions`. Nothing polls. Each stage subscribes to the item before it and is notified when that item is written.

The point of the example is the middle two stages, because they observe the same kind of item and disagree about what a change means. The filter is dirty on a single new sample. The feature stage is not dirty until a full window increment has accrued.

# Setting up

```Python
import numpy as np
from multiprocessing import Lock

from libemg.shared_memory_manager import SharedMemoryManager, assign_shared_memory_locks
from libemg.reactive import (ReactiveGraph, FilterHook, FeatureHook, ProbeHook,
                             Hook, Input, Output, OnSamples, LATEST, WINDOW,
                             default_notifier_pool)
from libemg.event_log import EventLog
from libemg.filtering import Filter

FS, CHANNELS = 1000, 8
WINDOW_SIZE, INCREMENT = 200, 50
FEATURES = ['MAV', 'RMS', 'WL', 'ZC']
```

Everything is assembled in one process so that the streamer and the graph share a notifier pool. That is what lets a write in the streamer's process wake a hook in another.

```Python
pool = default_notifier_pool()

items = [['emg', (2000, CHANNELS), np.double],
         ['emg_count', (1, 1), np.int32]]
assign_shared_memory_locks(items)

writer = SharedMemoryManager(notifier_pool=pool)
for item in items:
    writer.create_variable(*item)
```

# A model stage

The classifier is written as a hook. It observes `features` and writes `predictions`. It loads the model in `setup` rather than in `__init__`, because the executor is a spawned process and a fitted model is not always picklable.

```Python
class ModelHook(Hook):
    def __init__(self, model_path, num_features):
        super().__init__(
            'model',
            inputs=[Input('features', OnSamples(1), mode=LATEST)],
            outputs=[Output('predictions', (100, 2), np.double)],
        )
        self.model_path = model_path
        self.model = None

    def setup(self, context):
        import pickle
        with open(self.model_path, 'rb') as handle:
            self.model = pickle.load(handle)

    def step(self, data, snapshots):
        features = data['features']
        probabilities = self.model.predict_proba(features)
        prediction = float(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities))
        return {'predictions': np.array([[prediction, confidence]])}
```

# Wiring the stages

```Python
fi = Filter(sampling_frequency=FS)
fi.install_common_filters()

log = EventLog(path='reactive.log', keep=50000)
graph = ReactiveGraph(items, log=log, notifier_pool=pool)

# dirty on a single new sample
graph.add(FilterHook('filter', 'emg', 'filtered_emg', fi=fi,
                     shape=(2000, CHANNELS), window=400, increment=1))

# dirty only once an increment of samples has accrued
graph.add(FeatureHook('features', 'filtered_emg', 'features',
                      feature_list=FEATURES,
                      window_size=WINDOW_SIZE,
                      window_increment=INCREMENT,
                      num_features=len(FEATURES) * CHANNELS),
          executor='features')

graph.add(ModelHook('mdl.pkl', len(FEATURES) * CHANNELS), executor='model')

# A probe cannot stall what it watches, so it is safe to leave attached.
# in_process=True runs it as a thread here rather than in its own process,
# which a lambda requires, since a lambda cannot be pickled to reach one.
graph.add(ProbeHook('scope', 'predictions',
                    fn=lambda data, snaps: print('prediction', data['predictions']),
                    hz=10),
          executor='probe', in_process=True)

print(graph.describe())
graph.start()
```

`describe` prints the wiring, including each observer's criterion, which is the quickest way to check a pipeline is hooked up the way you meant:

```
executor 'main':
  filter: observes emg [OnSamples(1), window(400)] -> writes filtered_emg
executor 'features':
  features: observes filtered_emg [OnSamples(50), window(200)] -> writes features
executor 'model':
  model: observes features [OnSamples(1), latest] -> writes predictions
executor 'probe':
  scope: observes predictions [Periodic(10), latest] -> writes -
```

# Feeding it

```Python
rng = np.random.default_rng(0)
for _ in range(2000):
    writer.commit('emg', rng.standard_normal((1, CHANNELS)))
    time.sleep(1.0 / FS)

graph.stop()
```

# Reading back what happened

The counters on each item show the cascade converging, and they show the two stages disagreeing about what a change is. Two thousand samples produce two thousand filter runs and forty feature extractions.

```Python
reader = SharedMemoryManager()
for item in graph.shared_memory_items:
    reader.find_variable(*item)

for tag in ['emg', 'filtered_emg', 'features', 'predictions']:
    state = reader.snapshot(tag)
    print(tag, 'generation', state.generation, 'total', state.total_samples)
```

| Item | Total samples | Why |
| --- | --- | --- |
| `emg` | 2000 | one per commit |
| `filtered_emg` | 2000 | the filter is dirty on every sample |
| `features` | 40 | 2000 divided by an increment of 50 |
| `predictions` | 40 | one per feature row |

The log carries the reasoning, not just the outcome:

```Python
print(log.summary())
for event in log.events(observer='features')[:4]:
    print(event.format())
```

Each recorded decision names the item, the observer, the criterion applied and the counters it was applied to, so a stage that fires too often or never fires can be diagnosed without adding print statements.

# Turning it off

Nothing here is mandatory. Passing no log at all installs a null object that absorbs every call, and an existing script that uses `OnlineEMGClassifier` gets the event-driven streaming loop without any of this, because the classifier hooks its data handler itself.
