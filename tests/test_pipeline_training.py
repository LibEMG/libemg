"""Fitting a model from the pipeline that describes it.

A pipeline document already says where the recordings are, how they are
filtered and windowed and which features to take. A model block is trained
from its Train Features and Train Labels inputs -- here the same features it
runs on, and the Window block's Metadata -- so the model is fitted on exactly
the data it is then scored on, and the document that trained it can be pointed
at a device and run.

The recordings here are synthesised into ``tmp_path`` so the tests carry their
own data and stay honest about what the labels are: two classes whose
amplitudes differ by a factor the features cannot miss, so a fit that silently
learned nothing would show up as chance accuracy rather than passing.
"""
import os
import pickle

import numpy as np
import pytest

from libemg._gui._pipeline import PipelineDocument, PortType
from libemg._gui._pipeline.compile import CompileError, compile_pipeline


def _json_from(text):
    """The JSON a command printed, past anything the library printed first."""
    import json
    return json.loads(text[text.index("{"):])


CHANNELS = 4
SAMPLES = 3000
RATE = 1000


@pytest.fixture
def recording(tmp_path):
    """Two classes, two repetitions, in the file naming the regex expects."""
    generator = np.random.default_rng(0)
    folder = tmp_path / "recording"
    folder.mkdir()
    for label in (0, 1):
        for rep in (0, 1):
            amplitude = 1.0 + 4.0 * label
            block = generator.normal(0.0, amplitude, (SAMPLES, CHANNELS))
            np.savetxt(folder / f"C_{label}_R_{rep}_emg.csv", block, delimiter=",")
    return str(folder)


def _document(folder, model_path, model="LDA", block="model.classifier",
              features=("MAV", "WL")):
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": folder,
        "regex_filters": "C_|_R|0,1|classes\n_R_|_emg.csv|0,1|reps",
        "sampling_rate": RATE})
    window = document.add_node("window.enframe",
                               params={"window_size": 200, "window_increment": 100})
    extract = document.add_node("features.extract", params={"features": list(features)})
    model_node = document.add_node(block, params={"model": model,
                                                  "model_path": str(model_path)})
    metrics = document.add_node("sink.metrics", params={"metrics": ["CA"]})
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", extract, "input")
    document.connect(extract, "output", model_node, "input")
    document.connect(extract, "output", model_node, "train_input")
    document.connect(window, "metadata", model_node, "train_labels")
    document.connect(model_node, "output", metrics, "input")
    document.connect(window, "metadata", metrics, "labels")
    return document


# ======================================================================
# what training produces
# ======================================================================
def test_untrained_pipeline_still_compiles(recording, tmp_path):
    """The pipeline you are about to train has no model file yet.

    Demanding one would make training impossible from the editor, because the
    only way to get the file is to run the training that is being refused.
    """
    target = tmp_path / "not_yet.pkl"
    document = _document(recording, target)
    assert document.mode() == "offline"
    assert document.validate() == []
    assert not os.path.exists(target)
    pipeline = compile_pipeline(document)
    assert [document.nodes[n].params["model"] for n in pipeline.trainable()] == ["LDA"]


def test_train_writes_a_usable_classifier(recording, tmp_path):
    target = tmp_path / "fitted.pkl"
    result = compile_pipeline(_document(recording, target)).train()

    assert os.path.exists(target)
    assert result["classes"] == [0, 1]
    assert result["windows"] > 0
    assert result["features"] == 2 * CHANNELS

    with open(target, "rb") as handle:
        predictor = pickle.load(handle)
    from libemg.emg_predictor import EMGClassifier
    assert isinstance(predictor, EMGClassifier)


def test_the_same_document_scores_what_it_trained(recording, tmp_path):
    """Training then running is one document and two clicks, not a script."""
    target = tmp_path / "fitted.pkl"
    compile_pipeline(_document(recording, target)).train()
    results = compile_pipeline(_document(recording, target)).run()
    # Two classes whose amplitudes differ fivefold are separable, so anything
    # near chance would mean the labels and the windows came apart.
    assert results["CA"] > 0.9


def test_progress_is_reported_and_reaches_one(recording, tmp_path):
    seen = []
    compile_pipeline(_document(recording, tmp_path / "m.pkl")).train(
        on_progress=seen.append)
    assert seen and seen[-1] == 1.0
    assert seen == sorted(seen)


def test_a_regressor_trains_on_a_single_degree_of_freedom(recording, tmp_path):
    """A label field holding one value per sample is one DOF.

    Scikit-learn's multi-output wrapper rejects a flat target outright, so
    without saying that explicitly a single-DOF recording could not be trained
    on at all.
    """
    target = tmp_path / "reg.pkl"
    result = compile_pipeline(
        _document(recording, target, model="LR", block="model.regressor")).train()
    assert result["dofs"] == 1
    assert result["classes"] is None
    with open(target, "rb") as handle:
        from libemg.emg_predictor import EMGRegressor
        assert isinstance(pickle.load(handle), EMGRegressor)


# ======================================================================
# what training refuses, and why
# ======================================================================
def test_no_save_path_is_refused_before_the_work(recording, tmp_path):
    document = _document(recording, "")
    with pytest.raises(CompileError, match="nowhere to save"):
        compile_pipeline(document).train()


def test_a_recording_without_labels_says_which_field_to_set(recording, tmp_path):
    document = _document(recording, tmp_path / "m.pkl")
    for node in document.nodes.values():
        if node.spec_id == "model.classifier":
            node.params["label_field"] = "not_a_field"
    with pytest.raises(CompileError, match="no metadata field called 'not_a_field'.*classes"):
        compile_pipeline(document).train()


def test_an_empty_folder_names_the_three_things_to_check(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CompileError, match="no windows"):
        compile_pipeline(_document(str(empty), tmp_path / "m.pkl")).train()


def test_a_stopped_training_leaves_the_model_file_alone(recording, tmp_path):
    """Half a recording makes a model indistinguishable from a whole one."""
    target = tmp_path / "untouched.pkl"
    target.write_bytes(b"previous model")
    pipeline = compile_pipeline(_document(recording, target))
    pipeline.request_stop()
    with pytest.raises(CompileError, match="stopped part way"):
        pipeline.train()
    assert target.read_bytes() == b"previous model"


def test_an_unknown_model_name_lists_the_ones_that_work(recording, tmp_path):
    document = _document(recording, tmp_path / "m.pkl")
    for node in document.nodes.values():
        if node.spec_id == "model.classifier":
            node.params["model"] = "NOT_A_MODEL"
    with pytest.raises(CompileError, match="not a model this version can fit"):
        compile_pipeline(document).train()


# ======================================================================
# a regressor has somewhere to send its output
# ======================================================================
def test_a_regressor_may_feed_a_sink():
    """Otherwise the regressor block could not be connected to anything.

    Every sink takes a prediction port, and a regressor produces a continuous
    one. The two share a drag category because a sink treats them the same, so
    the link is allowed while the types stay distinct for probe drawing.
    """
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    window = document.add_node("window.enframe")
    extract = document.add_node("features.extract")
    model = document.add_node("model.regressor")
    sink = document.add_node("sink.console")
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", extract, "input")
    document.connect(extract, "output", model, "input")
    assert document.why_not_connect(model, "output", sink, "input") is None
    document.connect(model, "output", sink, "input")


def test_types_that_share_no_category_are_still_refused():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    model = document.add_node("model.classifier")
    problem = document.why_not_connect(source, "emg", model, "input")
    assert problem and "samples" in problem and "features" in problem


def test_the_two_decision_types_stay_distinct_for_drawing():
    assert PortType.RENDER[PortType.PREDICTION] != PortType.RENDER[PortType.CONTINUOUS]
    assert PortType.accepts(PortType.CONTINUOUS, PortType.PREDICTION)
    assert not PortType.accepts(PortType.SAMPLES, PortType.FEATURES)


# ======================================================================
# the same thing without the editor
# ======================================================================
def test_the_command_line_trains_and_then_scores(recording, tmp_path, capsys):
    """``train`` then ``run`` is the whole loop with no GUI and no script."""
    import json

    from libemg.pipeline import main

    target = tmp_path / "cli.pkl"
    path = tmp_path / "pipeline.json"
    _document(recording, target).save(str(path))

    assert main(["train", str(path), "--quiet"]) == 0
    trained = _json_from(capsys.readouterr().out)
    assert trained["classes"] == [0, 1]
    assert os.path.exists(target)

    assert main(["run", str(path), "--quiet"]) == 0
    assert _json_from(capsys.readouterr().out)["CA"] > 0.9


def test_the_command_line_refuses_to_train_a_live_pipeline(recording, tmp_path,
                                                          capsys):
    """A device has no labelled recording to fit on, and is told so."""
    from libemg.pipeline import main

    # Train an ordinary pipeline first, so the live one has a model file and
    # fails for the reason under test rather than for a missing predictor.
    fitted = tmp_path / "fitted.pkl"
    stored = tmp_path / "stored.json"
    _document(recording, fitted).save(str(stored))
    assert main(["train", str(stored), "--quiet"]) == 0
    capsys.readouterr()

    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    window = document.add_node("window.enframe")
    extract = document.add_node("features.extract")
    model = document.add_node("model.classifier",
                              params={"model": "LDA", "model_path": str(fitted)})
    sink = document.add_node("sink.console")
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", extract, "input")
    document.connect(extract, "output", model, "input")
    document.connect(model, "output", sink, "input")
    path = tmp_path / "live.json"
    document.save(str(path))
    assert document.mode() == "online"

    assert main(["train", str(path), "--quiet"]) == 1
    assert "source is a device" in capsys.readouterr().err


# ======================================================================
# the window can be closed and opened again
# ======================================================================
def test_the_editor_can_be_closed_and_reopened():
    """Its file dialogs are parented to the viewport, not to its window.

    Deleting the window leaves them behind, so the second open used to fail
    partway through building the toolbar, having already created a window it
    would never finish.
    """
    dpg = pytest.importorskip("dearpygui.dearpygui")
    from libemg._gui._pipeline import editor_panel as module

    try:
        dpg.create_context()
        dpg.create_viewport(title="editor reopen", width=900, height=700)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    except Exception as error:
        pytest.skip(f"DearPyGui cannot create a context here: {error}")

    try:
        first = module.PipelineEditorPanel()
        first.spawn_window()
        dpg.render_dearpygui_frame()
        first.cleanup()
        for _ in range(2):
            dpg.render_dearpygui_frame()
        for tag in (module.WINDOW_TAG, module.EDITOR_TAG,
                    module.OPEN_TAG, module.SAVEAS_TAG):
            assert not dpg.does_alias_exist(tag), f"{tag} outlived the window"

        second = module.PipelineEditorPanel()
        second.spawn_window()
        dpg.render_dearpygui_frame()
        assert dpg.does_alias_exist(module.WINDOW_TAG)
        second.cleanup()
    finally:
        dpg.destroy_context()
