"""Splitting stored data, windowing its metadata, and training a model block.

Three things a stored-data pipeline does with no script:

- a Split block divides the recordings by a metadata field -- training
  repetitions from test ones -- whole recordings at a time;
- a Window block windows every metadata field alongside the signal, so each
  label is the label of one window and nothing has to be lined up afterwards;
- a model block is trained from its Train Features and Train Labels inputs,
  holds whole recordings out for validation, and reports how it did on both.

The parity test is the one that matters most: one document, split and trained
and scored, gives exactly the numbers the documented script does.
"""
import json
import os
import pickle
import threading

import numpy as np
import pytest

from libemg._gui._pipeline.compile import CompileError, compile_pipeline
from libemg._gui._pipeline.document import PipelineDocument
from libemg._gui._pipeline.offline_data import (Recordings, hold_out, parse_values,
                                                window_recordings)

CLASSES, SETS, CHANNELS, SAMPLES, RATE = 3, 4, 4, 400, 200
WINDOW, INCREMENT = 20, 10


def _recording_folder(root, seed=0):
    """trial_<set>/C_<class>_R_0_emg.csv, each class at its own amplitude and frequency."""
    rng = np.random.default_rng(seed)
    for trial in range(1, SETS + 1):
        folder = root / f"trial_{trial}"
        folder.mkdir(parents=True, exist_ok=True)
        for label in range(CLASSES):
            t = np.arange(SAMPLES) / RATE
            block = (1 + label) * np.sin(2 * np.pi * (10 + 15 * label) * t)[:, None] \
                * np.linspace(0.6, 1.4, CHANNELS)[None, :] \
                + rng.normal(0, 0.3, (SAMPLES, CHANNELS))
            np.savetxt(folder / f"C_{label}_R_0_emg.csv", block, delimiter=",")
    return root


FILTERS = [{"left_bound": "/trial_", "right_bound": "/",
            "values": [str(s) for s in range(1, SETS + 1)], "description": "sets"},
           {"left_bound": "C_", "right_bound": "_R",
            "values": [str(c) for c in range(CLASSES)], "description": "classes"}]
FILTER_TEXT = (f"/trial_|/|{','.join(str(s) for s in range(1, SETS + 1))}|sets\n"
               f"C_|_R|{','.join(str(c) for c in range(CLASSES))}|classes")


def _split_document(folder, model_path="", train_sets="1-3", model="LDA", validation=0.0,
                    block="model.classifier", **model_params):
    """Stored Data -> Split -> two windows -> two feature blocks; train on one, score the other."""
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder), "regex_filters": FILTER_TEXT, "sampling_rate": RATE})
    split = document.add_node("transform.split", params={"field": "sets",
                                                         "values": train_sets})
    document.connect(source, "emg", split, "input")
    nodes = {"source": source, "split": split}
    for side, port in (("train", "matching"), ("test", "rest")):
        window = document.add_node("window.enframe", params={
            "window_size": WINDOW, "window_increment": INCREMENT})
        features = document.add_node("features.extract", params={"feature_group": "HTD"})
        document.connect(split, port, window, "input")
        document.connect(window, "output", features, "input")
        nodes[f"{side}_window"], nodes[f"{side}_features"] = window, features
    params = {"model": model, "model_path": str(model_path), "validation": validation}
    params.update(model_params)
    model_node = document.add_node(block, params=params)
    document.connect(nodes["train_features"], "output", model_node, "train_input")
    document.connect(nodes["train_window"], "metadata", model_node, "train_labels")
    document.connect(nodes["test_features"], "output", model_node, "input")
    metrics = document.add_node("sink.metrics", params={"metrics": ["CA"]})
    document.connect(model_node, "output", metrics, "input")
    document.connect(nodes["test_window"], "metadata", metrics, "labels")
    nodes.update(model=model_node, metrics=metrics)
    return document, nodes


# ======================================================================
# splitting
# ======================================================================
def test_values_can_be_listed_or_given_as_an_inclusive_range():
    assert parse_values("0,1,2") == ["0", "1", "2"]
    assert parse_values("0-3") == ["0", "1", "2", "3"]
    assert parse_values("5, 1-2, open") == ["5", "1", "2", "open"]
    assert parse_values("3-1") == ["3", "2", "1"]
    assert parse_values("") == []


def test_a_split_divides_whole_recordings_by_what_their_filenames_say(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    recordings = Recordings.read(str(folder), FILTERS, sampling_rate=RATE)

    matching, rest = recordings.split("sets", parse_values("1-2"))

    assert len(matching) == 2 * CLASSES and len(rest) == (SETS - 2) * CLASSES
    assert {matching.value("sets", i) for i in range(len(matching))} == {"1", "2"}
    assert {rest.value("sets", i) for i in range(len(rest))} == {"3", "4"}
    # Nothing is lost or shared: together they are the recordings, file for file.
    assert sorted(matching.files + rest.files) == sorted(recordings.files)
    # Each side is still an OfflineDataHandler, metadata and all.
    windows, metadata = matching.parse_windows(WINDOW, INCREMENT)
    assert set(metadata) == {"sets", "classes"} and len(metadata["classes"]) == len(windows)


def test_a_split_on_a_field_the_recordings_lack_names_the_ones_they_have(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder)
    document.set_param(nodes["split"], "field", "reps")
    with pytest.raises(CompileError, match="splits on 'reps'.*sets, classes"):
        compile_pipeline(document).train(save=False)


# ======================================================================
# windowing the metadata
# ======================================================================
def test_every_field_is_windowed_one_value_per_window_in_window_order(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    recordings = Recordings.read(str(folder), FILTERS, sampling_rate=RATE)

    windows = window_recordings(recordings, WINDOW, INCREMENT)

    per_file = (SAMPLES - WINDOW) // INCREMENT + 1
    assert windows.lengths == [per_file] * len(recordings)
    for name in ("sets", "classes"):
        assert len(windows.metadata.field(name)) == len(windows)
    # Window by window, the label is the label of the recording that window is from.
    script_windows, script_metadata = recordings.parse_windows(WINDOW, INCREMENT)
    np.testing.assert_array_equal(windows.windows, script_windows)
    np.testing.assert_array_equal(windows.metadata.field("classes"), script_metadata["classes"])


def test_a_field_that_changes_within_a_recording_follows_the_rule_chosen(tmp_path):
    recordings = Recordings()
    recordings.data = [np.zeros((10, 1))]
    recordings.extra_attributes = ["labels"]
    recordings.labels = [np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3]).reshape(-1, 1)]

    def windowed(rule):
        return window_recordings(recordings, 4, 3, rule).metadata.field("labels").tolist()

    # Windows span samples 0-3, 3-6, 6-9.
    assert windowed("last sample") == [1, 2, 3]
    assert windowed("first sample") == [0, 1, 2]
    assert windowed("most common") == [0, 1, 2]
    assert windowed("mean") == pytest.approx([0.25, 1.5, 2.25])


def test_a_recording_shorter_than_a_window_is_left_out_of_both(tmp_path):
    recordings = Recordings()
    recordings.data = [np.zeros((50, 2)), np.zeros((5, 2)), np.zeros((30, 2))]
    recordings.extra_attributes = ["classes"]
    recordings.classes = [np.full((50, 1), 0), np.full((5, 1), 1), np.full((30, 1), 2)]

    windows = window_recordings(recordings, 20, 10)

    assert windows.files == [0, 2] and windows.lengths == [4, 2]
    assert windows.metadata.field("classes").tolist() == [0] * 4 + [2] * 2


def test_labels_from_other_windows_than_the_predictions_are_refused(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder)
    # Score the test predictions against the training windows' labels.
    document.disconnect(nodes["test_window"], "metadata", nodes["metrics"], "labels")
    document.connect(nodes["train_window"], "metadata", nodes["metrics"], "labels")
    pipeline = compile_pipeline(document)
    pipeline.train(save=False)
    with pytest.raises(CompileError, match="not the same windows"):
        pipeline.run()


# ======================================================================
# training a model block
# ======================================================================
def test_one_split_document_trains_and_scores_exactly_as_the_script_does(tmp_path):
    from libemg.data_handler import OfflineDataHandler, RegexFilter
    from libemg.emg_predictor import EMGClassifier
    from libemg.feature_extractor import FeatureExtractor
    from libemg.offline_metrics import OfflineMetrics

    folder = _recording_folder(tmp_path / "data")
    handler = OfflineDataHandler()
    handler.get_data(str(folder), [RegexFilter(**f) for f in FILTERS], delimiter=",")
    train = handler.isolate_data("sets", [0, 1, 2])
    test = handler.isolate_data("sets", [3])
    train_windows, train_meta = train.parse_windows(WINDOW, INCREMENT)
    test_windows, test_meta = test.parse_windows(WINDOW, INCREMENT)
    extractor = FeatureExtractor()
    predictor = EMGClassifier("LDA")
    predictor.fit({"training_features": extractor.extract_feature_group("HTD", train_windows,
                                                                       array=True),
                   "training_labels": train_meta["classes"]})
    predictions, _ = predictor.run(extractor.extract_feature_group("HTD", test_windows,
                                                                    array=True))
    expected = OfflineMetrics().extract_offline_metrics(["CA"], test_meta["classes"],
                                                        predictions)

    document, _ = _split_document(folder)
    pipeline = compile_pipeline(document)
    pipeline.train(save=False)
    got = pipeline.run()

    assert got["CA"] == expected["CA"]
    assert expected["CA"] > 0.5, "a check that both paths agree at chance proves nothing"


def test_training_holds_whole_recordings_out_and_reports_both_sides(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder, validation=0.34)
    pipeline = compile_pipeline(document)

    result = pipeline.train(save=False)

    stats = result["statistics"]
    assert stats["recordings"] == 3 * CLASSES
    assert len(stats["validation_files"]) == CLASSES, "a share of each class"
    assert not set(stats["training_files"]) & set(stats["validation_files"])
    for side in ("training", "validation"):
        assert 0.0 <= stats[side]["accuracy"] <= 1.0
        assert set(stats[side]["per_class"]) == set(range(CLASSES))
        confusion = np.asarray(stats[side]["confusion"])
        assert confusion.shape == (CLASSES, CLASSES)
        assert confusion.sum() == stats[side]["windows"]
    assert stats["training"]["windows"] + stats["validation"]["windows"] == stats["windows"]
    # Nothing was written; the model is held for the run.
    assert result["model_path"] == "" and nodes["model"] in pipeline.models
    assert 0.0 <= pipeline.run()["CA"] <= 1.0


def test_a_regressor_reports_its_error_per_degree_of_freedom(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder, model="LR", block="model.regressor",
                                      validation=0.34)
    pipeline = compile_pipeline(document)
    # A class index is a number too: enough to check the shape of the report.
    stats = pipeline.train(save=False)["statistics"]
    assert stats["task"] == "regression" and stats["dofs"] == 1
    for side in ("training", "validation"):
        assert len(stats[side]["rmse"]) == len(stats[side]["mae"]) == len(stats[side]["r2"]) == 1


def test_a_classifier_refuses_a_field_that_is_not_a_class(tmp_path):
    from libemg._gui._pipeline.fitting import FitError, fit_block
    folder = _recording_folder(tmp_path / "data")
    recordings = Recordings.read(str(folder), FILTERS, sampling_rate=RATE)
    windows = window_recordings(recordings, WINDOW, INCREMENT)
    windows.metadata.fields["force"] = np.linspace(0, 1, len(windows))
    with pytest.raises(FitError, match="not whole numbers"):
        fit_block("model.classifier", {"model": "LDA"}, windows, windows.metadata, "force")


def test_the_statistics_travel_inside_the_saved_model(tmp_path):
    from libemg._gui._pipeline.compile import load_model, save_model
    from libemg._gui._pipeline.fitting import STATISTICS
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder, validation=0.34)
    pipeline = compile_pipeline(document)
    pipeline.train(save=False)

    path = tmp_path / "saved" / "lda.pkl"
    save_model(pipeline.models[nodes["model"]], str(path))
    loaded = load_model(str(path))

    assert getattr(loaded, STATISTICS) == pipeline.fits[nodes["model"]].statistics
    # And a pipeline with nothing in memory runs the saved one.
    document.set_param(nodes["model"], "model_path", str(path))
    assert 0.0 <= compile_pipeline(document).run()["CA"] <= 1.0


def test_a_model_given_to_the_pipeline_is_used_before_its_file(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder, model_path=tmp_path / "nowhere.pkl")
    trained = compile_pipeline(document)
    trained.train(save=False)
    assert not os.path.exists(tmp_path / "nowhere.pkl")

    results = compile_pipeline(document, models=trained.models).run()

    assert 0.0 <= results["CA"] <= 1.0


def test_a_deep_model_trains_on_one_branch_and_runs_on_the_other(tmp_path):
    pytest.importorskip("torch")
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder, block="model.deep", architecture="GRU",
                                      epochs=3, sequence_length=8, hidden_size=16,
                                      patience=0, validation=0.34)
    pipeline = compile_pipeline(document)
    epochs = []

    result = pipeline.train(save=False, on_epoch=epochs.append)

    assert [e.epoch for e in epochs] == [1, 2, 3]
    stats = result["statistics"]
    assert stats["validation"] is not None and stats["epochs"] == 3
    assert pipeline.fits[nodes["model"]].outcome.data["statistics"] == stats
    assert 0.0 <= pipeline.run()["CA"] <= 1.0


# ======================================================================
# what a document allows
# ======================================================================
def test_a_model_that_is_only_trained_is_a_finished_pipeline(tmp_path):
    folder = _recording_folder(tmp_path / "data")
    document, nodes = _split_document(folder)
    document.remove_node(nodes["metrics"])
    document.disconnect(nodes["test_features"], "output", nodes["model"], "input")
    document.remove_node(nodes["test_features"])
    document.remove_node(nodes["test_window"])
    document.disconnect(nodes["split"], "rest", nodes["test_window"], "input")
    assert not any("produces something" in p for p in document.validate())


def test_training_inputs_come_as_a_pair():
    document = PipelineDocument()
    source = document.add_node("source.offline", params={"folder": "x",
                                                         "regex_filters": FILTER_TEXT})
    window = document.add_node("window.enframe")
    features = document.add_node("features.extract")
    model = document.add_node("model.classifier")
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(window, "metadata", model, "train_labels")
    assert any("Train Labels but nothing to learn" in p for p in document.validate())
    document.disconnect(window, "metadata", model, "train_labels")
    document.connect(features, "output", model, "train_input")
    assert any("no Train Labels" in p for p in document.validate())


def test_a_live_pipeline_has_no_metadata_and_trains_nothing():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    window = document.add_node("window.enframe")
    features = document.add_node("features.extract")
    model = document.add_node("model.classifier")
    sink = document.add_node("sink.console")
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", model, "input")
    document.connect(features, "output", model, "train_input")
    document.connect(window, "metadata", model, "train_labels")
    document.connect(model, "output", sink, "input")
    problems = " ".join(document.validate())
    assert "a live stream carries no labels" in problems
    assert "trained from stored data, but this pipeline is live" in problems


def test_a_version_one_file_is_rewired_to_take_its_labels_from_the_window(tmp_path):
    """Before, Stored Data had a Labels output and Train fitted the model on the run's data."""
    folder = _recording_folder(tmp_path / "data")
    old = {"schema_version": 1, "nodes": [
        {"id": "src", "spec_id": "source.offline", "params": {
            "folder": str(folder), "regex_filters": FILTER_TEXT, "label_key": "classes",
            "sampling_rate": RATE}},
        {"id": "win", "spec_id": "window.enframe", "params": {"window_size": WINDOW,
                                                              "window_increment": INCREMENT}},
        {"id": "fea", "spec_id": "features.extract", "params": {"feature_group": "HTD"}},
        {"id": "mdl", "spec_id": "model.classifier", "params": {
            "model": "LDA", "model_path": str(tmp_path / "m.pkl")}},
        {"id": "met", "spec_id": "sink.metrics", "params": {"metrics": ["CA"]}}],
        "links": [
            {"from_node": "src", "from_port": "emg", "to_node": "win", "to_port": "input"},
            {"from_node": "win", "from_port": "output", "to_node": "fea", "to_port": "input"},
            {"from_node": "fea", "from_port": "output", "to_node": "mdl", "to_port": "input"},
            {"from_node": "mdl", "from_port": "output", "to_node": "met", "to_port": "input"},
            {"from_node": "src", "from_port": "labels", "to_node": "met", "to_port": "labels"}]}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))

    document = PipelineDocument.load(str(path))

    links = {l.key() for l in document.links}
    assert ("win", "metadata", "met", "labels") in links
    assert ("fea", "output", "mdl", "train_input") in links
    assert ("win", "metadata", "mdl", "train_labels") in links
    assert not any(l.from_port == "labels" for l in document.links)
    assert document.nodes["mdl"].params["label_field"] == "classes"
    assert document.validate() == []
    compile_pipeline(document).train()
    assert compile_pipeline(document).run()["CA"] > 0.9


# ======================================================================
# the editor's buttons
# ======================================================================
def _editor(monkeypatch):
    from unittest.mock import MagicMock

    from libemg._gui import _style
    from libemg._gui._pipeline import editor_panel as module
    fake = MagicMock()
    fake.does_item_exist.return_value = False
    fake.does_alias_exist.return_value = False
    monkeypatch.setattr(module, "dpg", fake)
    monkeypatch.setattr(_style, "dpg", fake)
    return module.PipelineEditorPanel()


def _finish(panel):
    """Let a training started from the editor end, and the editor take it in."""
    thread = panel._offline_thread
    assert thread is not None
    thread.join(timeout=120)
    panel.poll()
    assert panel.pipeline is None, "the editor has taken the result in"


def test_a_model_trained_from_its_block_is_kept_saved_and_loaded(tmp_path, monkeypatch):
    folder = _recording_folder(tmp_path / "data")
    panel = _editor(monkeypatch)
    panel.document, nodes = _split_document(folder, validation=0.34)
    model = nodes["model"]

    panel._train_clicked(None, None, model)
    _finish(panel)

    assert model in panel.models, "trained and held in memory"
    statistics = panel.model_info[model]["statistics"]
    assert statistics["validation"] is not None
    assert panel._model_status(model)[0].startswith("Trained. Training")
    assert "Not saved yet" in panel._model_status(model)[0]

    # Save writes it to Fitted Model, once that is set.
    path = tmp_path / "models" / "lda.pkl"
    panel.document.set_param(model, "model_path", str(path))
    panel._save_model_clicked(None, None, model)
    assert path.exists() and panel.model_info[model]["saved"] == str(path)

    # Start runs the model trained here.
    panel._start()
    panel._offline_thread.join(timeout=120)
    assert 0.0 <= panel.pipeline.results["CA"] <= 1.0
    panel.poll()

    # Another session loads it, statistics and all.
    other = _editor(monkeypatch)
    other.document = panel.document
    other.load_model(model, str(path))
    assert other.model_info[model]["statistics"] == statistics
    assert other._model_status(model)[0].startswith("Loaded lda.pkl.")


def test_changing_a_trained_models_settings_asks_for_it_to_be_trained_again(tmp_path,
                                                                            monkeypatch):
    folder = _recording_folder(tmp_path / "data")
    panel = _editor(monkeypatch)
    panel.document, nodes = _split_document(folder)
    panel._train_clicked(None, None, nodes["model"])
    _finish(panel)

    panel._param_changed(None, "SVM", (nodes["model"], "model"))

    assert nodes["model"] not in panel.models
    assert "train it again" in panel._model_status(nodes["model"])[0]


def test_train_all_trains_every_block_with_training_inputs(tmp_path, monkeypatch):
    folder = _recording_folder(tmp_path / "data")
    panel = _editor(monkeypatch)
    panel.document, nodes = _split_document(folder)
    second = panel.document.add_node("model.classifier", params={"model": "SVM"})
    panel.document.connect(nodes["train_features"], "output", second, "train_input")
    panel.document.connect(nodes["train_window"], "metadata", second, "train_labels")

    panel._train()
    _finish(panel)

    assert set(panel.models) == {nodes["model"], second}
