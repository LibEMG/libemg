"""Coverage for the Prediction Viewer block: the maths, and the run that feeds it.

The window is DearPyGui and is checked by looking at it; what is held here is
everything it draws from. Every recording has to be predicted on its own, with
a model's memory cleared between them; the recordings have to keep their
signal, labels and filename metadata; and which ones a model trained on and
which it never saw has to be right, since that is the comparison the block is
for.
"""
import numpy as np
import pytest

from libemg._gui._pipeline import prediction_viewer as viewer
from libemg.pipeline import PipelineDocument, compile_pipeline

CLASSES, SETS, CHANNELS, SAMPLES = 3, 3, 4, 400
WINDOW, INCREMENT, RATE = 20, 10, 200


def _recording_folder(root, seed=0):
    """trial_<n>/C_<c>_R_0_emg.csv, each class at its own amplitude and frequency."""
    rng = np.random.default_rng(seed)
    for trial in range(1, SETS + 1):
        folder = root / f"trial_{trial}"
        folder.mkdir(parents=True, exist_ok=True)
        for label in range(CLASSES):
            t = np.arange(SAMPLES) / RATE
            block = (1 + label) * np.sin(2 * np.pi * (10 + 15 * label) * t)[:, None] \
                * np.linspace(0.6, 1.4, CHANNELS)[None, :] + rng.normal(0, 0.05, (SAMPLES, CHANNELS))
            np.savetxt(folder / f"C_{label}_R_0_emg.csv", block, delimiter=",")
    return root


def _document(folder, model_path, model="LDA", validation=0.0, **viewer_params):
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder),
        "regex_filters": (f"/trial_|/|{','.join(str(s) for s in range(1, SETS + 1))}|sets\n"
                          f"C_|_R|{','.join(str(c) for c in range(CLASSES))}|classes"),
        "sampling_rate": RATE, "delimiter": ","})
    window = document.add_node("window.enframe", params={"window_size": WINDOW,
                                                         "window_increment": INCREMENT})
    features = document.add_node("features.extract", params={"feature_group": "HTD"})
    if model == "GRU":
        block = document.add_node("model.deep", params={
            "architecture": "GRU", "model_path": str(model_path), "epochs": 3,
            "sequence_length": 8, "hidden_size": 16, "patience": 0, "validation": 0.34})
    else:
        block = document.add_node("model.classifier", params={"model": model,
                                                              "model_path": str(model_path),
                                                              "validation": validation})
    shown = document.add_node("sink.prediction_viewer", params=viewer_params)
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", block, "input")
    document.connect(features, "output", block, "train_input")
    document.connect(window, "metadata", block, "train_labels")
    document.connect(block, "output", shown, "model")
    document.connect(window, "metadata", shown, "labels")
    return document


# ======================================================================
# the maths
# ======================================================================
def test_runs_split_a_label_sequence_where_it_changes():
    assert viewer.class_runs([0, 0, 1, 1, 1, 0]) == [(0, 2, 0), (2, 5, 1), (5, 6, 0)]
    assert viewer.class_runs([]) == []


def test_a_windows_output_is_placed_at_its_last_sample():
    np.testing.assert_allclose(viewer.window_times(3, WINDOW, INCREMENT, RATE),
                               [0.1, 0.15, 0.2])


def test_a_stream_lays_recordings_end_to_end_with_a_gap():
    def recording(index, samples, label):
        count = (samples - WINDOW) // INCREMENT + 1
        return {"index": index, "signal": np.zeros((samples, 2)), "truth": np.full(count, label),
                "outputs": np.zeros((count, 2)), "predicted": np.full(count, label)}

    joined = viewer.stream([recording(0, 200, 0), recording(1, 100, 1)], WINDOW, INCREMENT,
                           RATE, gap=0.5)

    assert [(round(a, 3), round(b, 3)) for a, b, _ in joined["bounds"]] == [(0, 1), (1.5, 2)]
    assert joined["signal"].shape == (300, 2)
    assert len(joined["window_time"]) == len(joined["predicted"]) == len(joined["truth"])
    # The second recording's windows start after the gap, not at the join.
    assert joined["window_time"][19] == pytest.approx(1.5 + WINDOW / RATE)


def test_accuracy_counts_windows_and_ignores_recordings_without_labels():
    right = {"truth": np.array([0, 0, 1]), "predicted": np.array([0, 1, 1])}
    unlabelled = {"truth": None, "predicted": np.array([2, 2])}
    assert viewer.accuracy([right, unlabelled]) == pytest.approx(2 / 3)
    assert viewer.accuracy([unlabelled]) is None


def test_each_recording_is_predicted_from_a_fresh_memory():
    class Remembering:
        """Predicts how many windows it has seen since it was last reset."""
        classes_ = np.array([0, 1])

        def __init__(self):
            self.seen = 0

        def reset(self):
            self.seen = 0

        def predict_proba(self, x):
            rows = []
            for _ in x:
                self.seen += 1
                rows.append([1.0, 0.0] if self.seen <= 2 else [0.0, 1.0])
            return np.array(rows)

    class Classifier:
        def __init__(self, model):
            self.model = model

        def _predict_proba(self, x):
            return self.model.predict_proba(x)

    results = viewer.predict_recordings(Classifier(Remembering()),
                                        [np.zeros((3, 1)), np.zeros((3, 1))])

    for probabilities, predicted in results:
        np.testing.assert_array_equal(predicted, [0, 0, 1])
        assert probabilities.shape == (3, 2)


# ======================================================================
# the run that feeds the block
# ======================================================================
def test_a_run_hands_the_viewer_every_recording_with_its_predictions(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document = _document(folder, tmp_path / "lda.pkl")
    compile_pipeline(document).train()

    pipeline = compile_pipeline(document)
    pipeline.run()

    view = pipeline.prediction_view
    assert view["task"] == "classification" and view["classes"] == list(range(CLASSES))
    assert (view["window_size"], view["window_increment"], view["sampling_rate"]) == \
        (WINDOW, INCREMENT, RATE)
    recordings = view["recordings"]
    assert len(recordings) == SETS * CLASSES
    windows = (SAMPLES - WINDOW) // INCREMENT + 1
    metas = set()
    for r in recordings:
        assert r["signal"].shape == (SAMPLES, CHANNELS)
        assert r["outputs"].shape == (windows, CLASSES)
        np.testing.assert_allclose(r["outputs"].sum(axis=1), 1.0, atol=1e-6)
        assert len(r["predicted"]) == len(r["truth"]) == windows
        assert len(set(r["truth"])) == 1, "a file is one class"
        assert str(r["truth"][0]) == r["meta"]["classes"]
        assert r["split"] == "training", "fitted on every recording: none was held out"
        metas.add((r["meta"]["sets"], r["meta"]["classes"]))
    assert len(metas) == SETS * CLASSES
    # These classes are easy: the viewer should see the model get them right.
    assert viewer.accuracy(recordings) > 0.9


def test_the_block_can_name_the_recordings_held_out(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document = _document(folder, tmp_path / "lda.pkl", held_out_field="sets",
                         held_out_values="3")
    compile_pipeline(document).train()

    pipeline = compile_pipeline(document)
    pipeline.run()

    splits = {(r["meta"]["sets"], r["split"]) for r in pipeline.prediction_view["recordings"]}
    assert splits == {("1", "training"), ("2", "training"), ("3", "held out")}
    assert "sets 3" in pipeline.prediction_view["note"]


def test_a_deep_model_remembers_which_recordings_it_validated_on(tmp_path):
    pytest.importorskip("torch")
    folder = _recording_folder(tmp_path / "data")
    document = _document(folder, tmp_path / "gru.pkl", model="GRU")
    compile_pipeline(document).train()

    pipeline = compile_pipeline(document)
    pipeline.run()

    recordings = pipeline.prediction_view["recordings"]
    held = [r["index"] for r in recordings if r["split"] == "held out"]
    import pickle
    with open(tmp_path / "gru.pkl", "rb") as handle:
        trained_on = pickle.load(handle).model.trained_on
    assert held == trained_on["validation_indices"] and held
    assert all(r["split"] in ("training", "held out") for r in recordings)


def test_the_viewer_needs_a_trained_model(tmp_path):
    from libemg._gui._pipeline.compile import CompileError
    folder = _recording_folder(tmp_path / "data")
    pipeline = compile_pipeline(_document(folder, tmp_path / "missing.pkl"))
    with pytest.raises(CompileError, match="does not exist"):
        pipeline.run()


def test_the_whole_session_saves_as_one_figure(tmp_path):
    pytest.importorskip("matplotlib")
    folder = _recording_folder(tmp_path / "data")
    document = _document(folder, tmp_path / "lda.pkl", held_out_field="sets",
                         held_out_values="3")
    compile_pipeline(document).train()
    pipeline = compile_pipeline(document)
    pipeline.run()

    path = viewer.save_grid_figure(pipeline.prediction_view, str(tmp_path / "grid.png"))

    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_the_block_is_offered_for_stored_data_only():
    from libemg._gui._pipeline.registry import build_registry
    spec = build_registry()["sink.prediction_viewer"]
    assert spec.offline_only
    assert not spec.port("model").optional and spec.port("labels").optional
