"""Coverage for libemg.deep: one way to train, run, stream and adapt deep models.

What has to hold for every architecture is the same, so most tests run over
all of them: a model learns a problem it obviously can, a stream fed a window
at a time gets exactly what it would have got all at once, it survives a
pickle, and it reports every epoch as it trains. The data is generated here and
made easy on purpose -- what is under test is the tooling, not the networks.
"""
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from libemg.deep import (ARCHITECTURES, DeepModel, EpochResult, Recording, SequenceMemory,
                         TrainingSet)
from libemg.deep import networks

CLASSES = 3


def _feature_recordings(count=18, steps=30, seed=0):
    """Each class lifts its own feature, every window: separable at every step."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        label = i % CLASSES
        inputs = rng.normal(size=(steps, 6))
        inputs[:, label] += 3.0
        out.append(Recording(inputs, np.full(steps, label)))
    return out


def _window_recordings(count=18, steps=12, seed=0):
    """Raw windows: each class a sine at its own frequency on every channel."""
    rng = np.random.default_rng(seed)
    t = np.arange(32)
    out = []
    for i in range(count):
        label = i % CLASSES
        wave = np.sin(2 * np.pi * (label + 1) * 2 * t / 32)
        inputs = wave[None, None, :] * 1.5 + rng.normal(scale=0.3, size=(steps, 4, 32))
        out.append(Recording(inputs, np.full(steps, label)))
    return out


def _gestures(count=30, steps=24, seed=0):
    """A discrete gesture: rest everywhere except a short burst, somewhere, of its class."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        label = i % CLASSES
        inputs = rng.normal(scale=0.5, size=(steps, 6))
        start = rng.integers(2, steps - 6)
        inputs[start:start + 4, label] += 4.0
        out.append(Recording(inputs, np.asarray(label)))
    return out


def _model(architecture, **options):
    settings = dict(epochs=40, sequence_length=12, batch_size=8, horizon=4, hidden_size=32,
                    layers=2, embedding_size=16, patience=None, learning_rate=3e-3)
    settings.update(options)
    return DeepModel(architecture, **settings)


def _data(architecture):
    return _window_recordings() if architecture == "cnn" else _feature_recordings()


# ======================================================================
# training sets
# ======================================================================
def test_arrays_laid_end_to_end_split_back_into_their_recordings():
    inputs = np.arange(20, dtype=float).reshape(10, 2)
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1])

    continuous = TrainingSet.from_arrays(inputs, labels, lengths=[3, 2, 5])
    assert [len(r) for r in continuous.recordings] == [3, 2, 5]
    np.testing.assert_array_equal(continuous.recordings[1].inputs, inputs[3:5])
    assert continuous.classes == [0, 1, 2]

    discrete = TrainingSet.from_arrays(inputs, labels, lengths=[3, 2, 5], timing="discrete")
    assert [int(r.targets) for r in discrete.recordings] == [0, 1, 2], "the most common label"

    with pytest.raises(ValueError, match="add up"):
        TrainingSet.from_arrays(inputs, labels, lengths=[3, 3])
    with pytest.raises(ValueError, match="regression is"):
        TrainingSet.from_arrays(inputs, labels, timing="discrete", task="regression")


def test_validation_holds_out_whole_recordings_and_every_class():
    training = TrainingSet(_gestures(), timing="discrete")

    kept, held = training.split(0.2)

    assert len(kept.recordings) + len(held.recordings) == len(training.recordings)
    assert not {id(r) for r in kept.recordings} & {id(r) for r in held.recordings}
    assert held.classes == [0, 1, 2], "stratified: no class left out by chance"


def test_batches_are_crops_for_continuous_and_whole_gestures_for_discrete():
    continuous = TrainingSet(_feature_recordings(count=4, steps=30))
    batches = list(continuous.batches(sequence_length=8, batch_size=5))
    inputs, targets, mask = batches[0]
    assert inputs.shape == (5, 8, 6) and targets.shape == (5, 8) and mask.all()
    assert sum(len(b[0]) for b in batches) == 4 * 4, "about one pass over every window"

    gestures = [Recording(np.ones((n, 6)), np.asarray(1)) for n in (5, 9)]
    inputs, targets, mask = next(TrainingSet(gestures, timing="discrete").batches(8, 4,
                                                                                  shuffle=False))
    assert inputs.shape == (2, 9, 6)
    assert mask.sum(axis=1).tolist() == [5, 9], "padding is masked out"
    assert targets.tolist() == [1, 1]


# ======================================================================
# every architecture learns, streams, pickles and reports
# ======================================================================
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_learns_a_held_motion(architecture):
    training = TrainingSet(_data(architecture))
    held = _data(architecture)[:3]

    model = _model(architecture).fit(training, validation=None)

    right = sum((model.predict(r.inputs) == r.targets).sum() for r in held)
    assert right / sum(len(r) for r in held) > 0.9
    assert model.history[-1].train_loss < model.history[0].train_loss


@pytest.mark.parametrize("architecture", ["transformer", "gru", "lstm", "mlp"])
def test_a_stream_fed_a_window_at_a_time_gets_what_it_would_all_at_once(architecture):
    model = _model(architecture, epochs=2).fit(TrainingSet(_feature_recordings()))
    inputs = _feature_recordings(count=1, steps=20, seed=9)[0].inputs
    whole = model.predict_proba(inputs)

    model.streaming = True
    model.reset()
    streamed = np.vstack([model.predict_proba(inputs[i:i + 1]) for i in range(len(inputs))])

    np.testing.assert_allclose(streamed, whole, atol=1e-5)
    model.reset()
    np.testing.assert_allclose(model.predict_proba(inputs[:5]), whole[:5], atol=1e-5), \
        "reset starts a new stream"


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_a_trained_model_survives_a_pickle_and_exposes_its_latent_space(architecture,
                                                                        tmp_path):
    model = _model(architecture, epochs=2).fit(TrainingSet(_data(architecture)))
    inputs = _data(architecture)[0].inputs

    model.save(tmp_path / "model.pkl")
    restored = DeepModel.load(tmp_path / "model.pkl")

    np.testing.assert_allclose(restored.predict_proba(inputs), model.predict_proba(inputs),
                               atol=1e-6)
    assert restored.latent(inputs).shape == (len(inputs), 16)
    assert restored.on_epoch is None


def test_every_epoch_is_reported_and_stopping_keeps_the_best_so_far():
    seen = []
    calls = {"batches": 0}

    def stop():
        calls["batches"] += 1
        return len(seen) >= 3

    model = _model("gru", epochs=30).fit(TrainingSet(_feature_recordings()), validation=0.25,
                                        on_epoch=seen.append, should_stop=stop)

    assert 3 <= len(seen) < 30, "stopped within an epoch of being asked"
    assert all(isinstance(e, EpochResult) for e in seen)
    assert [e.epoch for e in seen] == list(range(1, len(seen) + 1))
    assert all(e.validation_loss is not None and e.score_name == "accuracy" for e in seen)
    best = min(seen, key=lambda e: e.validation_loss)
    assert best.best, "the best epoch says so"
    assert "validation loss" in seen[0].describe()


# ======================================================================
# discrete gestures and regression
# ======================================================================
@pytest.mark.parametrize("architecture, pooling", [("mlp", "max"), ("gru", "last"),
                                                   ("transformer", "max")])
def test_a_gesture_labelled_once_is_recognised_wherever_it_happened(architecture, pooling):
    training = TrainingSet(_gestures(count=60), timing="discrete")
    model = _model(architecture, timing="discrete", pooling=pooling, epochs=60)

    model.fit(training, validation=None)

    _, accuracy, name = model.evaluate(TrainingSet(_gestures(count=15, seed=3),
                                                   timing="discrete"))
    assert name == "accuracy" and accuracy > 0.8, "scored per gesture, on gestures it never saw"


def test_regression_learns_a_mapping_and_scores_rmse():
    rng = np.random.default_rng(1)
    recordings = []
    for _ in range(12):
        inputs = rng.normal(size=(40, 4))
        recordings.append(Recording(inputs, np.column_stack([inputs[:, 0], -inputs[:, 1]])))
    model = _model("mlp", task="regression", epochs=80)

    model.fit(TrainingSet(recordings, task="regression"), validation=0.25)

    assert model.history[-1].score_name == "rmse"
    assert model.history[-1].score < 0.35
    assert model.predict(recordings[0].inputs).shape == (40, 2)
    with pytest.raises(AttributeError):
        model.predict_proba(recordings[0].inputs)


def test_a_cnn_refuses_features_and_says_what_it_needs():
    with pytest.raises(ValueError, match="raw windows"):
        _model("cnn").fit(TrainingSet(_feature_recordings()))


def test_the_model_and_its_data_have_to_agree():
    model = _model("mlp", epochs=1).fit(TrainingSet(_feature_recordings()))
    with pytest.raises(ValueError, match="continuous classification model"):
        model.adapt(TrainingSet(_gestures(), timing="discrete"))


# ======================================================================
# live adaptation
# ======================================================================
def test_adapting_on_a_memory_nudges_the_model_and_returns_its_losses(tmp_path):
    model = _model("gru", epochs=15).fit(TrainingSet(_feature_recordings()))
    before = {k: v.clone() for k, v in model.network.state_dict().items()}
    reports = []
    model.on_epoch = reports.append

    memory = SequenceMemory()
    for trial, recording in enumerate(_feature_recordings(count=3, seed=4), start=1):
        for row, label in zip(recording.inputs, recording.targets):
            memory.append([label], row, trial)      # what MemoryManager.append_to_memory does
    memory.append([7], np.zeros(6), 9)              # a class the model has no output for

    losses = model.adapt(memory)

    assert len(losses) == model.adapt_epochs and all(np.isfinite(losses))
    assert [r.phase for r in reports] == ["adapt"] * model.adapt_epochs
    changed = any(not torch.equal(before[k], v) for k, v in model.network.state_dict().items())
    assert changed, "adapting changes the weights"
    model.save(tmp_path / "mdl1.pkl")               # what the adaptation side does next
    assert DeepModel.load(tmp_path / "mdl1.pkl").fitted


def test_memory_slices_add_up_save_and_load_as_the_managers_handle_them(tmp_path):
    first, second = SequenceMemory(), SequenceMemory()
    for row in range(4):
        first.append([0.5, -0.5], np.full(3, row), trial=1)
    second.append([1.0, 0.0], np.ones((2, 3)), trial=2)
    second.append([1.0, 0.0], np.empty((0, 3)), trial=2)     # no matching model input

    first.save(tmp_path / "memory_1.pkl")
    loaded = SequenceMemory().load(tmp_path / "memory_1.pkl")
    combined = loaded + second

    assert len(combined) == 6
    assert combined.processed_data[0].shape == (6, 3)
    regression = combined.training_set("regression")
    assert [len(r) for r in regression.recordings] == [4, 2]
    assert regression.dofs == 2


# ======================================================================
# it slots into what already takes a model
# ======================================================================
def test_wrapped_in_an_emg_classifier_it_runs_and_the_feature_viewer_sees_its_latent_space():
    from libemg._gui._pipeline.feature_viewer import model_spaces
    from libemg.emg_predictor import EMGClassifier

    model = _model("transformer", epochs=2).fit(TrainingSet(_feature_recordings()))
    predictor = EMGClassifier(model)
    inputs = _feature_recordings(count=1, seed=5)[0].inputs

    assert predictor._predict(inputs).shape == (len(inputs),)
    spaces = model_spaces(predictor, inputs)
    assert spaces[0][0] == "Latent space" and spaces[0][1].shape == (len(inputs), 16)
    assert [title for title, _, _, _ in spaces][-1] == "Class probabilities"


def test_networks_and_the_package_agree_on_the_architectures():
    assert tuple(ARCHITECTURES) == networks.ARCHITECTURES


# ======================================================================
# in the pipeline
# ======================================================================
def _recording_folder(root, classes=3, sets=3, samples=400, seed=0):
    rng = np.random.default_rng(seed)
    for trial in range(1, sets + 1):
        folder = root / f"trial_{trial}"
        folder.mkdir(parents=True, exist_ok=True)
        for label in range(classes):
            t = np.arange(samples) / 200.0
            block = (1 + label) * np.sin(2 * np.pi * (10 + 15 * label) * t)[:, None] \
                * np.linspace(0.6, 1.4, 4)[None, :] + rng.normal(0, 0.05, (samples, 4))
            np.savetxt(folder / f"C_{label}_R_0_emg.csv", block, delimiter=",")
    return root


def _deep_document(folder, model_path, architecture="GRU", windows=False, **params):
    from libemg.pipeline import PipelineDocument
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder), "regex_filters": "/trial_|/|1,2,3|sets\nC_|_R|0,1,2|classes",
        "label_key": "classes", "sampling_rate": 200, "delimiter": ","})
    window = document.add_node("window.enframe", params={"window_size": 20,
                                                         "window_increment": 10})
    settings = {"architecture": architecture, "model_path": str(model_path), "epochs": 6,
                "sequence_length": 8, "horizon": 4, "hidden_size": 16, "patience": 0}
    settings.update(params)
    model = document.add_node("model.deep", params=settings)
    document.connect(source, "emg", window, "input")
    if windows:
        document.connect(window, "output", model, "windows")
    else:
        features = document.add_node("features.extract", params={"feature_group": "HTD"})
        document.connect(window, "output", features, "input")
        document.connect(features, "output", model, "input")
    metrics = document.add_node("sink.metrics", params={"metrics": ["CA"]})
    document.connect(model, "output", metrics, "input")
    document.connect(source, "labels", metrics, "labels")
    return document


@pytest.mark.parametrize("architecture, windows", [("GRU", False), ("CNN", True),
                                                   ("Transformer", False)])
def test_the_pipeline_trains_a_deep_model_reporting_every_epoch(tmp_path, architecture,
                                                                windows):
    from libemg.pipeline import compile_pipeline
    folder = _recording_folder(tmp_path / "data")
    path = tmp_path / "deep.pkl"
    document = _deep_document(folder, path, architecture, windows=windows)
    seen = []

    pipeline = compile_pipeline(document)
    trained = pipeline.train(on_epoch=seen.append)

    assert [e.epoch for e in seen] == list(range(1, 7)) == [e.epoch for e in pipeline.epochs]
    assert trained["recordings"] == 9, "one recording per file, never stitched together"
    assert trained["model_path"] == str(path) and path.exists()
    results = compile_pipeline(document).run()
    assert 0.0 <= results["CA"] <= 1.0
    with open(path, "rb") as handle:
        predictor = pickle.load(handle)
    assert predictor.model.architecture == architecture.lower()


def test_stopping_training_keeps_and_saves_the_best_epoch_so_far(tmp_path):
    from libemg.pipeline import compile_pipeline
    folder = _recording_folder(tmp_path / "data")
    path = tmp_path / "deep.pkl"
    pipeline = compile_pipeline(_deep_document(folder, path, epochs=200))

    def stop_after_two(result):
        if result.epoch == 2:
            pipeline.request_stop()

    trained = pipeline.train(on_epoch=stop_after_two)

    assert trained["stopped"] and trained["epochs"] == 2
    assert path.exists(), "a network's best epoch is a finished model, so it is saved"


def test_a_live_pipeline_streams_a_deep_model_through_the_classifier_hook(tmp_path):
    from libemg._gui._pipeline.compile import _ClassifierHook
    model = _model("gru", epochs=1).fit(TrainingSet(_feature_recordings()))
    from libemg.emg_predictor import EMGClassifier
    path = tmp_path / "gru.pkl"
    with open(path, "wb") as handle:
        pickle.dump(EMGClassifier(model), handle)

    hook = _ClassifierHook("deep", "features", "out", str(path), 2)
    hook.setup(None)

    assert hook._predictor.model.streaming, "memory carries from one live window to the next"
    rows = _feature_recordings(count=1, seed=8)[0].inputs
    outputs = [hook._predict(rows[i:i + 1]) for i in range(5)]
    assert all(len(o) == 2 for o in outputs)


def test_the_deep_block_offers_what_libemg_deep_understands():
    from libemg._gui._pipeline.registry import build_registry
    spec = build_registry()["model.deep"]
    choices = [c.lower() for c in spec.param("architecture").choices]
    assert choices == list(ARCHITECTURES)
    assert spec.param("timing").choices == ["Continuous", "Discrete"]


# ======================================================================
# training records: export, load, look at later
# ======================================================================
def test_a_training_record_round_trips_through_json(tmp_path):
    from libemg.deep import TrainingOutcome
    model = _model("gru", epochs=3).fit(TrainingSet(_feature_recordings()), validation=0.25)

    outcome = model.outcome(model_path="gru.pkl", name="baseline", data={"folder": "data/"})
    written = outcome.save(tmp_path / "run.training.json")
    loaded = TrainingOutcome.load(written)

    assert loaded.to_dict() == outcome.to_dict()
    assert loaded.name == "baseline" and loaded.architecture == "gru"
    assert [e.epoch for e in loaded.epochs] == [1, 2, 3]
    assert loaded.data["recordings"] == 18 and loaded.data["validation_recordings"] > 0
    assert loaded.data["classes"] == [0, 1, 2] and loaded.data["folder"] == "data/"
    assert loaded.options["learning_rate"] == model.training_options["learning_rate"]
    assert loaded.best.epoch == outcome.best.epoch
    assert "baseline" in loaded.describe()


def test_a_saved_model_carries_its_own_training_record(tmp_path):
    from libemg.deep import TrainingOutcome
    from libemg.emg_predictor import EMGClassifier
    model = _model("mlp", epochs=2).fit(TrainingSet(_feature_recordings()))
    path = tmp_path / "mlp.pkl"
    with open(path, "wb") as handle:
        pickle.dump(EMGClassifier(model), handle)

    outcome = TrainingOutcome.load(path)

    assert len(outcome.epochs) == 2 and outcome.model_path == str(path)
    with pytest.raises(ValueError, match="not a deep model"):
        with open(tmp_path / "other.pkl", "wb") as handle:
            pickle.dump({"not": "a model"}, handle)
        TrainingOutcome.load(tmp_path / "other.pkl")


def test_the_pipeline_writes_a_training_record_beside_every_model(tmp_path):
    from libemg.deep import TrainingOutcome
    from libemg.pipeline import compile_pipeline
    folder = _recording_folder(tmp_path / "data")
    pipeline = compile_pipeline(_deep_document(folder, tmp_path / "models" / "gru.pkl"))

    trained = pipeline.train()

    assert trained["outcome_path"] == str(tmp_path / "models" / "gru.training.json")
    outcome = TrainingOutcome.load(trained["outcome_path"])
    assert len(outcome.epochs) == trained["epochs"]
    assert outcome.data["folder"] == str(folder)
    assert outcome.data["features"] == ["MAV", "ZC", "SSC", "WL"], "the HTD group, by name"
    assert outcome.model_path == str(tmp_path / "models" / "gru.pkl")


def test_the_history_window_loads_compares_and_exports_records(tmp_path):
    dpg = pytest.importorskip("dearpygui.dearpygui")
    from libemg._gui import _style
    from libemg._gui._pipeline.training_view import TrainingView
    paths = []
    for i, rate in enumerate((1e-3, 3e-3)):
        model = _model("gru", epochs=3, learning_rate=rate).fit(TrainingSet(_feature_recordings()))
        paths.append(model.outcome(name=f"lr {rate}").save(tmp_path / f"run{i}.training.json"))

    dpg.create_context()
    try:
        _style.apply()
        view = TrainingView().spawn_window()
        assert not dpg.is_item_enabled(view._export), "nothing to export yet"
        for path in paths:
            view.load(path)
        view.load(paths[0])                         # the same record twice is shown once

        assert [run.name for run in view.runs] == ["lr 0.001", "lr 0.003"]
        lines = dpg.get_item_children(view._loss_y, 1)
        assert len(lines) == 2, "one validation curve per run when comparing"
        assert dpg.is_item_enabled(view._export)

        view._export_chosen(None, {"file_path_name": str(tmp_path / "exported")})
        assert (tmp_path / "exported.json").exists()

        view.clear()
        assert view.runs == [] and not dpg.get_item_children(view._loss_y, 1)
        view.cleanup()
    finally:
        dpg.destroy_context()
