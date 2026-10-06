"""The same offline analysis, written as a script and built as a pipeline.

The documentation's two offline examples -- a classifier over the
OneSubjectMyoDataset and a regressor over the OneSubjectEMaGerDataset -- are
scripts. The pipeline editor is supposed to be able to express the same
analysis, and the only useful definition of "the same" is that the numbers
:class:`~libemg.offline_metrics.OfflineMetrics` reports come out identical.

These tests hold that. Each one writes a recording, runs the script form
exactly as the documentation does, builds the pipeline form, and requires the
two to agree to the last decimal place -- not approximately, because both paths
run the same estimator over the same windows and any difference at all means
one of them is reading the data differently.

The recordings are generated here rather than downloaded, so the check runs
without a network and in seconds. ``test_the_documented_datasets_agree_end_to_end``
covers the real thing and is marked slow.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from libemg.data_handler import FilePackager, OfflineDataHandler, RegexFilter
from libemg.emg_predictor import EMGClassifier, EMGRegressor
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
from libemg.pipeline import PipelineDocument, compile_pipeline

WINDOW_SIZE, WINDOW_INCREMENT, FEATURE_SET = 20, 10, "HTD"
CLASSES, SETS, CHANNELS, SAMPLES = 4, 6, 4, 400


def _write_classification_recording(root, seed=0):
    """A folder laid out the way the Myo dataset is: trial_<n>/C_<c>_R_0_emg.csv.

    Each class gets its own amplitude and frequency, so a classifier can
    separate them and the accuracy is not a coin toss -- what is under test is
    that two paths agree, and two paths agree trivially at chance.
    """
    rng = np.random.default_rng(seed)
    for trial in range(1, SETS + 1):
        folder = root / f"trial_{trial}"
        folder.mkdir(parents=True, exist_ok=True)
        for label in range(CLASSES):
            t = np.arange(SAMPLES) / 200.0
            signal = (1.0 + label) * np.sin(2 * np.pi * (10 + 15 * label) * t)
            block = signal[:, None] * np.linspace(0.6, 1.4, CHANNELS)[None, :]
            block = block + rng.normal(0, 0.05, block.shape)
            np.savetxt(folder / f"C_{label}_R_0_emg.csv", block, delimiter=",")
    return root


def _write_regression_recording(root, dofs=2, reps=5, seed=1):
    """A folder laid out the way the EMaGer dataset is: a labels file per folder.

    The point of this layout is that the target is a value per sample, so it
    cannot live in the filename. It lives in labels.txt beside the recording,
    which is what a FilePackager is for.
    """
    rng = np.random.default_rng(seed)
    for rep in range(reps):
        folder = root / "open-close" / f"rep_{rep}"
        folder.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0, 4 * np.pi, SAMPLES * 2)
        targets = np.column_stack([np.sin(t + dof) for dof in range(dofs)])
        # An agonist/antagonist pair of channels per DOF, each carrying the half
        # of that DOF it drives. Without the pair only the magnitude of the
        # target is in the signal and not its sign, and a regressor that cannot
        # recover the sign scores near zero however correct the wiring is.
        envelope = np.column_stack([half for dof in range(dofs)
                                    for half in (np.maximum(targets[:, dof], 0),
                                                 np.maximum(-targets[:, dof], 0))])
        block = rng.normal(0, 1, envelope.shape) * (0.05 + envelope)
        np.savetxt(folder / f"movement_R_{rep}_emg.csv", block, delimiter=",")
        np.savetxt(folder / "labels.txt", targets, delimiter=",")
    return root


# ======================================================================
# classification
# ======================================================================

def _script_classification(folder, train_sets, test_sets, model="LDA", null_label=2):
    """The documented script, verbatim in shape: isolate, window, fit, score."""
    filters = [
        RegexFilter(left_bound="/trial_", right_bound="/",
                    values=[str(i) for i in range(1, SETS + 1)], description="sets"),
        RegexFilter(left_bound="C_", right_bound="_R",
                    values=[str(i) for i in range(CLASSES)], description="classes"),
    ]
    handler = OfflineDataHandler()
    handler.get_data(folder_location=str(folder), regex_filters=filters, delimiter=",")
    train = handler.isolate_data("sets", train_sets)
    test = handler.isolate_data("sets", test_sets)
    train_windows, train_meta = train.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    test_windows, test_meta = test.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)

    extractor = FeatureExtractor()
    data_set = {
        "training_features": extractor.extract_feature_group(FEATURE_SET, train_windows),
        "training_labels": train_meta["classes"]}
    test_features = extractor.extract_feature_group(FEATURE_SET, test_windows)

    predictor = EMGClassifier(model)
    predictor.fit(data_set.copy())
    predictions, _ = predictor.run(test_features)
    return OfflineMetrics().extract_common_metrics(
        test_meta["classes"], predictions, null_label), len(test_windows)


def _classification_document(folder, sets, model_path, model="LDA", null_label=2):
    """The same analysis as a document: what the editor builds by dragging."""
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder),
        "regex_filters": (f"/trial_|/|{','.join(sets)}|sets\n"
                          f"C_|_R|{','.join(str(i) for i in range(CLASSES))}|classes"),
        "label_key": "classes", "sampling_rate": 200, "delimiter": ","})
    window = document.add_node("window.enframe", params={
        "window_size": WINDOW_SIZE, "window_increment": WINDOW_INCREMENT})
    features = document.add_node("features.extract",
                                 params={"feature_group": FEATURE_SET})
    predictor = document.add_node("model.classifier", params={
        "model": model, "model_path": str(model_path)})
    metrics = document.add_node("sink.metrics", params={
        "metrics": ["CA", "AER", "INS"], "null_label": null_label})
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", predictor, "input")
    document.connect(predictor, "output", metrics, "input")
    document.connect(source, "labels", metrics, "labels")
    return document


def test_a_classification_pipeline_scores_what_the_script_scores(tmp_path):
    folder = _write_classification_recording(tmp_path / "myo")
    model_path = tmp_path / "lda.pkl"

    expected, n_test = _script_classification(folder, [0, 1, 2], [3, 4, 5])
    compile_pipeline(_classification_document(
        folder, ["1", "2", "3"], model_path)).train()
    got = compile_pipeline(_classification_document(
        folder, ["4", "5", "6"], model_path)).run()

    assert set(got) == {"CA", "AER", "INS"}
    for metric in ("CA", "AER", "INS"):
        assert got[metric] == expected[metric], f"{metric} differs from the script"
    assert expected["CA"] > 0.5, "a check that both paths agree at chance proves nothing"


def test_the_two_paths_agree_for_every_model_the_examples_compare(tmp_path):
    # The documented example loops over several classifiers. Each one has to
    # come out the same both ways, not just the default.
    folder = _write_classification_recording(tmp_path / "myo")
    for model in ("LDA", "KNN", "SVM"):
        expected, _ = _script_classification(folder, [0, 1, 2], [3, 4, 5], model=model)
        path = tmp_path / f"{model}.pkl"
        compile_pipeline(_classification_document(
            folder, ["1", "2", "3"], path, model=model)).train()
        got = compile_pipeline(_classification_document(
            folder, ["4", "5", "6"], path, model=model)).run()
        assert got["CA"] == expected["CA"], f"{model} differs between the two paths"


def test_a_pipeline_saved_and_reopened_still_scores_the_same(tmp_path):
    """What the editor actually does: the document goes to a file and back."""
    folder = _write_classification_recording(tmp_path / "myo")
    model_path = tmp_path / "lda.pkl"
    compile_pipeline(_classification_document(
        folder, ["1", "2", "3"], model_path)).train()
    document = _classification_document(folder, ["4", "5", "6"], model_path)
    direct = compile_pipeline(document).run()

    saved = tmp_path / "scored.json"
    document.save(saved)
    reopened = compile_pipeline(PipelineDocument.load(saved)).run()

    assert reopened == direct
    # Every parameter survived the round trip, including the ones added for
    # stored-data work.
    stored = [node for node in json.loads(saved.read_text())["nodes"]
              if node["spec_id"] == "source.offline"][0]
    assert stored["params"]["label_key"] == "classes"
    assert "sets" in stored["params"]["regex_filters"]


# ======================================================================
# regression
# ======================================================================

def _script_regression(folder, train_reps, test_reps, dofs=2, model="LR"):
    """The documented regression script: packaged labels, last-sample windowing."""
    filters = [RegexFilter(left_bound="_R_", right_bound="_emg.csv",
                           values=[str(i) for i in range(5)], description="reps")]
    packager = FilePackager(
        RegexFilter(left_bound="/", right_bound=".txt", values=["labels"],
                    description="labels"),
        lambda meta, data: Path(meta).parent.absolute() == Path(data).parent.absolute())
    handler = OfflineDataHandler()
    handler.get_data(folder_location=str(folder), regex_filters=filters,
                     metadata_fetchers=[packager], delimiter=",")
    train = handler.isolate_data("reps", train_reps)
    test = handler.isolate_data("reps", test_reps)
    operations = {"labels": lambda x: x[-1]}
    train_windows, train_meta = train.parse_windows(
        WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=operations)
    test_windows, test_meta = test.parse_windows(
        WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=operations)

    extractor = FeatureExtractor()
    training_set = {
        "training_features": extractor.extract_feature_group(
            FEATURE_SET, train_windows, array=True),
        "training_labels": train_meta["labels"]}
    predictor = EMGRegressor(model)
    predictor.fit(training_set.copy())
    predictions = predictor.run(extractor.extract_feature_group(
        FEATURE_SET, test_windows, array=True))
    return OfflineMetrics().extract_offline_metrics(
        ["R2", "NRMSE", "MAE"], test_meta["labels"], predictions), test_meta["labels"]


def _regression_document(folder, reps, model_path, model="LR"):
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder),
        "regex_filters": f"_R_|_emg.csv|{','.join(reps)}|reps",
        "label_files": "/|.txt|labels|labels",
        "package_by": "same folder",
        "label_key": "labels", "sampling_rate": 200, "delimiter": ","})
    window = document.add_node("window.enframe", params={
        "window_size": WINDOW_SIZE, "window_increment": WINDOW_INCREMENT})
    features = document.add_node("features.extract",
                                 params={"feature_group": FEATURE_SET})
    predictor = document.add_node("model.regressor", params={
        "model": model, "model_path": str(model_path)})
    metrics = document.add_node("sink.metrics",
                                params={"metrics": ["R2", "NRMSE", "MAE"]})
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", predictor, "input")
    document.connect(predictor, "output", metrics, "input")
    document.connect(source, "labels", metrics, "labels")
    return document


def test_a_regression_pipeline_scores_what_the_script_scores(tmp_path):
    folder = _write_regression_recording(tmp_path / "emager")
    model_path = tmp_path / "lr.pkl"

    expected, labels = _script_regression(folder, [0, 1, 2, 3], [4])
    assert np.shape(labels)[1] == 2, "the example's target has two DOFs"

    trained = compile_pipeline(_regression_document(
        folder, ["0", "1", "2", "3"], model_path)).train()
    got = compile_pipeline(_regression_document(folder, ["4"], model_path)).run()

    assert trained["dofs"] == 2, "a pipeline must fit both DOFs, not the first one"
    for metric in ("R2", "NRMSE", "MAE"):
        np.testing.assert_array_equal(
            np.asarray(got[metric]), np.asarray(expected[metric]),
            err_msg=f"{metric} differs from the script")
    assert np.mean(expected["R2"]) > 0.2, "a regressor that learned nothing proves nothing"


def test_labels_in_a_file_of_their_own_keep_every_degree_of_freedom(tmp_path):
    """The gap that made the regression example inexpressible.

    A filename filter can only label a whole recording, so a per-sample target
    has to be read from a packaged file -- and windowing it must keep one value
    per DOF rather than collapsing to the first column.
    """
    folder = _write_regression_recording(tmp_path / "emager", dofs=3)
    pipeline = compile_pipeline(_regression_document(
        folder, ["0", "1"], tmp_path / "m.pkl"))

    windows, features, truth = pipeline._gather()

    assert truth.ndim == 2 and truth.shape[1] == 3
    assert len(truth) == len(windows)
    # The label of a window is the label at its end, the same convention the
    # script's metadata_operations asks for.
    assert np.isfinite(truth).all()


def test_a_single_column_label_still_arrives_as_a_flat_vector(tmp_path):
    """Classification must not be turned into a one-DOF regression problem."""
    folder = _write_classification_recording(tmp_path / "myo")
    pipeline = compile_pipeline(_classification_document(
        folder, ["1"], tmp_path / "m.pkl"))

    _, _, truth = pipeline._gather()

    assert truth.ndim == 1
    assert set(np.unique(truth)) <= set(range(CLASSES))


def test_a_stored_data_block_without_label_files_asks_for_no_packagers(tmp_path):
    from libemg._gui._pipeline.compile import _metadata_fetchers
    assert _metadata_fetchers({"label_files": [], "package_by": "same folder"}) is None
    fetchers = _metadata_fetchers({
        "label_files": [{"left_bound": "/", "right_bound": ".txt",
                         "values": ["labels"], "description": "labels"}],
        "package_by": "same stem"})
    assert len(fetchers) == 1 and fetchers[0].description == "labels"
    # 'same stem' pairs a recording with the metadata file named after it.
    assert fetchers[0].package_function("a/C_0_R_0_emg.txt", "a/C_0_R_0_emg.csv")
    assert not fetchers[0].package_function("a/other.txt", "a/C_0_R_0_emg.csv")


# ======================================================================
# the real datasets
# ======================================================================

@pytest.mark.slow
def test_the_documented_datasets_agree_end_to_end(tmp_path):
    """The documentation's own datasets, both examples, script against pipeline.

    Downloads on first run, which is why it is slow. It is the only test that
    proves the parity on the data the documentation actually shows.
    """
    from libemg.datasets import OneSubjectEMaGerDataset, OneSubjectMyoDataset

    myo = OneSubjectMyoDataset()
    myo.prepare_data()
    emager = OneSubjectEMaGerDataset()
    emager.prepare_data()

    # -- classification, as docs/source/examples/simple_offline_example ----
    expected, _ = _script_classification_on(
        myo.dataset_folder, [0, 1, 2], [3, 4, 5], sets=[str(i) for i in range(1, 7)],
        classes=[str(i) for i in range(5)], window=20, increment=10, null_label=2)
    document = _myo_document(myo.dataset_folder, ["1", "2", "3"], tmp_path / "lda.pkl")
    compile_pipeline(document).train()
    got = compile_pipeline(
        _myo_document(myo.dataset_folder, ["4", "5", "6"], tmp_path / "lda.pkl")).run()
    for metric in ("CA", "AER", "INS"):
        assert got[metric] == expected[metric]

    # -- regression, as docs/source/examples/offline_regression_example ----
    expected_regression = _emager_script(emager.dataset_folder)
    compile_pipeline(_emager_document(
        emager.dataset_folder, ["0", "1", "2", "3"], tmp_path / "lr.pkl")).train()
    got = compile_pipeline(_emager_document(
        emager.dataset_folder, ["4"], tmp_path / "lr.pkl")).run()
    for metric in ("R2", "NRMSE", "MAE"):
        np.testing.assert_array_equal(np.asarray(got[metric]),
                                      np.asarray(expected_regression[metric]))


def _script_classification_on(folder, train_sets, test_sets, sets, classes,
                              window, increment, null_label):
    filters = [
        RegexFilter(left_bound="/trial_", right_bound="/", values=sets,
                    description="sets"),
        RegexFilter(left_bound="C_", right_bound=".csv", values=classes,
                    description="classes"),
        RegexFilter(left_bound="R_", right_bound="_", values=["0", "1"],
                    description="reps")]
    handler = OfflineDataHandler()
    handler.get_data(folder_location=folder, regex_filters=filters, delimiter=",")
    train = handler.isolate_data("sets", train_sets)
    test = handler.isolate_data("sets", test_sets)
    train_windows, train_meta = train.parse_windows(window, increment)
    test_windows, test_meta = test.parse_windows(window, increment)
    extractor = FeatureExtractor()
    predictor = EMGClassifier("LDA")
    predictor.fit({"training_features": extractor.extract_feature_group(
        "HTD", train_windows), "training_labels": train_meta["classes"]})
    predictions, _ = predictor.run(extractor.extract_feature_group("HTD", test_windows))
    return OfflineMetrics().extract_common_metrics(
        test_meta["classes"], predictions, null_label), len(test_windows)


def _myo_document(folder, sets, model_path):
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": folder,
        "regex_filters": (f"/trial_|/|{','.join(sets)}|sets\n"
                          "C_|.csv|0,1,2,3,4|classes\nR_|_|0,1|reps"),
        "label_key": "classes", "sampling_rate": 200, "delimiter": ","})
    window = document.add_node("window.enframe",
                               params={"window_size": 20, "window_increment": 10})
    features = document.add_node("features.extract", params={"feature_group": "HTD"})
    predictor = document.add_node("model.classifier", params={
        "model": "LDA", "model_path": str(model_path)})
    metrics = document.add_node("sink.metrics", params={
        "metrics": ["CA", "AER", "INS"], "null_label": 2})
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", predictor, "input")
    document.connect(predictor, "output", metrics, "input")
    document.connect(source, "labels", metrics, "labels")
    return document


def _emager_script(folder):
    filters = [
        RegexFilter(left_bound="/", right_bound="/",
                    values=["open-close", "pro-sup"], description="movements"),
        RegexFilter(left_bound="_R_", right_bound="_emg.csv",
                    values=[str(i) for i in range(5)], description="reps")]
    packager = FilePackager(
        RegexFilter(left_bound="/", right_bound=".txt", values=["labels"],
                    description="labels"),
        lambda meta, data: Path(meta).parent.absolute() == Path(data).parent.absolute())
    handler = OfflineDataHandler()
    handler.get_data(folder_location=folder, regex_filters=filters,
                     metadata_fetchers=[packager])
    operations = {"labels": lambda x: x[-1]}
    train_windows, train_meta = handler.isolate_data(
        "reps", [0, 1, 2, 3]).parse_windows(150, 40, metadata_operations=operations)
    test_windows, test_meta = handler.isolate_data(
        "reps", [4]).parse_windows(150, 40, metadata_operations=operations)
    extractor = FeatureExtractor()
    predictor = EMGRegressor("LR")
    predictor.fit({"training_features": extractor.extract_feature_group(
        "HTD", train_windows, array=True), "training_labels": train_meta["labels"]})
    predictions = predictor.run(extractor.extract_feature_group(
        "HTD", test_windows, array=True))
    return OfflineMetrics().extract_offline_metrics(
        ["R2", "NRMSE", "MAE"], test_meta["labels"], predictions)


def _emager_document(folder, reps, model_path):
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": folder,
        "regex_filters": ("/|/|open-close,pro-sup|movements\n"
                          f"_R_|_emg.csv|{','.join(reps)}|reps"),
        "label_files": "/|.txt|labels|labels", "package_by": "same folder",
        "label_key": "labels", "sampling_rate": 1010, "delimiter": ","})
    window = document.add_node("window.enframe",
                               params={"window_size": 150, "window_increment": 40})
    features = document.add_node("features.extract", params={"feature_group": "HTD"})
    predictor = document.add_node("model.regressor", params={
        "model": "LR", "model_path": str(model_path)})
    metrics = document.add_node("sink.metrics",
                                params={"metrics": ["R2", "NRMSE", "MAE"]})
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", predictor, "input")
    document.connect(predictor, "output", metrics, "input")
    document.connect(source, "labels", metrics, "labels")
    return document


# ======================================================================
# the editor
# ======================================================================

def _editor(monkeypatch):
    """The editor panel with a stand-in DearPyGui, so its callbacks can be driven."""
    from unittest.mock import MagicMock

    from libemg._gui._pipeline import editor_panel as editor_module
    fake = MagicMock()
    fake.does_item_exist.return_value = False
    fake.does_alias_exist.return_value = False
    monkeypatch.setattr(editor_module, "dpg", fake)
    return editor_module.PipelineEditorPanel(), fake


def _drag(panel, source, source_port, target, target_port):
    """What dragging one port onto another does, without the drag."""
    assert not panel.document.why_not_connect(source, source_port,
                                              target, target_port)
    panel.document.connect(source, source_port, target, target_port)


def test_a_pipeline_built_through_the_editor_scores_what_the_script_scores(
        tmp_path, monkeypatch):
    """The whole point: the same numbers, whether typed or dragged.

    This drives the editor's own callbacks -- adding blocks from the palette,
    changing parameters, linking ports -- rather than building a document
    directly, so what is under test is the path a person actually takes.
    """
    folder = _write_classification_recording(tmp_path / "myo")
    model_path = tmp_path / "lda.pkl"
    expected, _ = _script_classification(folder, [0, 1, 2], [3, 4, 5])

    def build(panel, sets):
        added = {}
        for spec_id in ("source.offline", "window.enframe", "features.extract",
                        "model.classifier", "sink.metrics"):
            before = set(panel.document.nodes)
            panel._add_from_palette(None, None, spec_id)
            added[spec_id] = (set(panel.document.nodes) - before).pop()
        source, window = added["source.offline"], added["window.enframe"]
        features = added["features.extract"]
        model, metrics = added["model.classifier"], added["sink.metrics"]

        for name, value in (
                ("folder", str(folder)),
                ("regex_filters",
                 f"/trial_|/|{','.join(sets)}|sets\n"
                 f"C_|_R|{','.join(str(i) for i in range(CLASSES))}|classes"),
                ("label_key", "classes"), ("sampling_rate", 200)):
            panel._param_changed(None, value, (source, name))
        panel._param_changed(None, WINDOW_SIZE, (window, "window_size"))
        panel._param_changed(None, WINDOW_INCREMENT, (window, "window_increment"))
        panel._param_changed(None, FEATURE_SET, (features, "feature_group"))
        panel._param_changed(None, "LDA", (model, "model"))
        panel._param_changed(None, str(model_path), (model, "model_path"))
        panel._param_changed(None, ["CA", "AER", "INS"], (metrics, "metrics"))
        panel._param_changed(None, 2, (metrics, "null_label"))

        _drag(panel, source, "emg", window, "input")
        _drag(panel, window, "output", features, "input")
        _drag(panel, features, "output", model, "input")
        _drag(panel, model, "output", metrics, "input")
        _drag(panel, source, "labels", metrics, "labels")
        return panel

    trainer, _ = _editor(monkeypatch)
    build(trainer, ["1", "2", "3"])
    assert trainer.document.mode() == "offline"
    compile_pipeline(trainer.document).train()

    scorer, _ = _editor(monkeypatch)
    build(scorer, ["4", "5", "6"])
    got = compile_pipeline(scorer.document).run()

    for metric in ("CA", "AER", "INS"):
        assert got[metric] == expected[metric], f"{metric} differs from the script"


def test_the_editor_draws_a_box_for_every_stored_data_parameter(monkeypatch):
    """A parameter the editor cannot show is one the GUI cannot configure."""
    panel, fake = _editor(monkeypatch)
    node_id = panel.document.add_node("source.offline")

    panel._draw_node(node_id)

    drawn = {call.kwargs.get("tag") for call in fake.add_input_text.call_args_list}
    drawn |= {call.kwargs.get("tag") for call in fake.add_combo.call_args_list}
    drawn |= {call.kwargs.get("tag") for call in fake.add_input_int.call_args_list}
    spec = panel.registry.get("source.offline")
    for param in spec.params:
        assert any(tag and tag.endswith(param.name) for tag in drawn), \
            f"the editor draws no widget for {param.name}"
    # The two parameters written one entry per line get a box with room for
    # them, not a single-line field that swallows the newline.
    multiline = {call.kwargs.get("tag") for call in fake.add_input_text.call_args_list
                 if call.kwargs.get("multiline")}
    assert any(tag.endswith("regex_filters") for tag in multiline)
    assert any(tag.endswith("label_files") for tag in multiline)
