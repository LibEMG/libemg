"""Coverage for turning a pipeline document into something that runs.

Most of this is arithmetic on plain data: the filter dictionary a block
compiles to, the regex filters a line of text parses into, the feature list a
block resolves. Those need nothing running and are checked directly.

Two things are not arithmetic and are the reason the module exists. The first
is window fusion: a window block must not become a stage of its own, it must
become *the terms on which the next stage observes the one before it*, and the
compiled graph is inspected to prove that. The second is that the whole thing
actually runs, which is checked end to end on the synthetic source so no
hardware is needed.

Everything spawned is torn down by a fixture rather than at the end of a test,
so a failing assertion still leaves no executor and no shared memory behind.
"""
import itertools
import os
import pickle
import time

import numpy as np
import pytest

from libemg._gui._pipeline import PipelineDocument
from libemg._gui._pipeline.compile import (CompileError, OfflinePipeline,
                                           OnlinePipeline, _filter_dictionary,
                                           _parse_regex_filters,
                                           _resolve_features, _upstream_stream,
                                           _window_feeding, compile_pipeline,
                                           tag_for)
from libemg.reactive import LATEST, WINDOW as WINDOW_MODE, reset_default_notifier_pool

OFFLINE_DATA = r"N:\libemg\Data\2026-08-25-1"
FEATURES = ["MAV", "RMS"]
CHANNELS, WINDOW_SIZE, INCREMENT = 4, 200, 50

_COUNTER = itertools.count()


# ======================================================================
# helpers
# ======================================================================
def _unique(stem):
    """A node id nothing else in this process will use.

    Node ids become shared-memory tags, so two tests that both placed a block
    called ``filter_1`` would be talking about the same buffer.
    """
    return f"pipe{os.getpid()}x{next(_COUNTER)}_{stem}"


def _wait_until(predicate, timeout=60.0, interval=0.05):
    """Poll ``predicate`` until it holds or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _hooks(graph):
    """Every hook in a compiled graph, keyed by name."""
    return {hook.name: hook for hooks in graph._groups.values() for hook in hooks}


def _online_document(model_path, sink_path=None, probes=(), channels=CHANNELS,
                     window_size=WINDOW_SIZE, increment=INCREMENT):
    """source -> filter -> window -> features -> classifier -> file sink."""
    document = PipelineDocument()
    source = document.add_node(
        "source.synthetic_streamer", node_id=_unique("src"),
        params={"num_channels": channels, "sampling_rate": 1000,
                "pattern": "bursts"})
    filtered = document.add_node(
        "transform.filter", node_id=_unique("flt"),
        params={"name": "bandpass", "cutoff": "20,450", "order": 4,
                "sampling_rate": 1000})
    window = document.add_node(
        "window.enframe", node_id=_unique("win"),
        params={"window_size": window_size, "window_increment": increment})
    features = document.add_node(
        "features.extract", node_id=_unique("fea"),
        params={"feature_group": "(custom)", "features": FEATURES})
    model = document.add_node(
        "model.classifier", node_id=_unique("mdl"),
        params={"model": "LDA", "model_path": model_path})
    sink = document.add_node(
        "sink.file", node_id=_unique("out"),
        params={"file_path": str(sink_path) if sink_path else "out.log"})
    for pair in ((source, "emg", filtered, "input"),
                 (filtered, "output", window, "input"),
                 (window, "output", features, "input"),
                 (features, "output", model, "input"),
                 (model, "output", sink, "input")):
        document.connect(*pair)
    for node_id, port in probes:
        document.add_probe({"source": source, "filter": filtered,
                            "window": window, "features": features,
                            "model": model}[node_id], port, hz=20)
    return document, {"source": source, "filter": filtered, "window": window,
                      "features": features, "model": model, "sink": sink}


def _offline_document(model_path, folder=OFFLINE_DATA, metrics=("CA", "RECALL")):
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder),
        "regex_filters": "C_|_R|0,1,2|classes\n_R_|_emg.csv|0,1|reps",
        "label_key": "classes", "sampling_rate": 1000})
    filtered = document.add_node("transform.filter", params={
        "name": "bandpass", "cutoff": "20,450", "order": 4, "sampling_rate": 1000})
    window = document.add_node("window.enframe", params={
        "window_size": WINDOW_SIZE, "window_increment": INCREMENT})
    features = document.add_node("features.extract",
                                 params={"features": list(FEATURES)})
    model = document.add_node("model.classifier",
                              params={"model_path": str(model_path)})
    sink = document.add_node("sink.metrics", params={"metrics": list(metrics)})
    for pair in ((source, "emg", filtered, "input"),
                 (filtered, "output", window, "input"),
                 (window, "output", features, "input"),
                 (features, "output", model, "input"),
                 (model, "output", sink, "input"),
                 (source, "labels", sink, "labels")):
        document.connect(*pair)
    return document, {"source": source, "filter": filtered, "window": window,
                      "features": features, "model": model, "sink": sink}


# ======================================================================
# fixtures
# ======================================================================
@pytest.fixture(scope="module")
def synthetic_model(tmp_path_factory):
    """An LDA fitted on made-up data with the shape the online document uses."""
    from libemg.emg_predictor import EMGClassifier
    from libemg.feature_extractor import FeatureExtractor
    from libemg.utils import get_windows

    rng = np.random.default_rng(0)
    training = np.vstack([rng.standard_normal((3000, CHANNELS)) * (c + 1)
                          for c in range(2)])
    labels = np.repeat([0, 1], 3000).reshape(-1, 1)
    windows = get_windows(training, WINDOW_SIZE, INCREMENT)
    windowed_labels = get_windows(labels, WINDOW_SIZE, INCREMENT)[:, 0, -1].astype(int)
    classifier = EMGClassifier("LDA")
    classifier.fit({"training_features": FeatureExtractor().extract_features(
        FEATURES, windows, array=True), "training_labels": windowed_labels})
    path = tmp_path_factory.mktemp("models") / "synthetic_lda.pkl"
    with open(path, "wb") as handle:
        pickle.dump(classifier, handle)
    return str(path)


@pytest.fixture(scope="module")
def recorded_model(tmp_path_factory):
    """An LDA fitted the ordinary way on the recordings the offline run uses."""
    if not os.path.isdir(OFFLINE_DATA):
        pytest.skip(f"no recordings at {OFFLINE_DATA}")
    from libemg.data_handler import OfflineDataHandler, RegexFilter
    from libemg.emg_predictor import EMGClassifier
    from libemg.feature_extractor import FeatureExtractor

    handler = OfflineDataHandler()
    handler.get_data(folder_location=OFFLINE_DATA, regex_filters=[
        RegexFilter("C_", "_R", ["0", "1", "2"], "classes"),
        RegexFilter("_R_", "_emg.csv", ["0", "1"], "reps")])
    if not handler.data:
        pytest.skip(f"no matching recordings in {OFFLINE_DATA}")
    windows, metadata = handler.parse_windows(WINDOW_SIZE, INCREMENT)
    classifier = EMGClassifier("LDA")
    classifier.fit({"training_features": FeatureExtractor().extract_features(
        FEATURES, windows, array=True),
        "training_labels": metadata["classes"].astype(int)})
    path = tmp_path_factory.mktemp("recorded") / "recorded_lda.pkl"
    with open(path, "wb") as handle:
        pickle.dump(classifier, handle)
    return str(path)


@pytest.fixture
def running():
    """Owns every pipeline a test started, and tears it down even on failure."""
    started = []
    try:
        yield started
    finally:
        for pipeline in reversed(started):
            handles = list(getattr(pipeline, "streamers", []))
            try:
                pipeline.stop(timeout=15)
            except Exception:
                pass
            for handle in handles:
                try:
                    handle.join(timeout=10)
                    if handle.is_alive():
                        handle.terminate()
                        handle.join(timeout=5)
                except Exception:
                    pass
            # stop() joins the executors but leaves the items the graph itself
            # created for hook outputs, so the test releases them.
            owned = getattr(pipeline.graph, "_smm", None)
            if owned is not None:
                try:
                    owned.cleanup(parent=True)
                except Exception:
                    pass
        if started:
            # Notifier slots are claimed per executor and never handed back, so
            # a module that starts several graphs would exhaust the pool.
            reset_default_notifier_pool()


# ======================================================================
# 17. the dictionary a filter block compiles to
# ======================================================================
def test_a_bandpass_compiles_to_two_cutoffs_and_an_order():
    assert _filter_dictionary({"name": "bandpass", "cutoff": "20,450", "order": 4}) == \
        {"name": "bandpass", "cutoff": [20.0, 450.0], "order": 4}


def test_a_single_cutoff_filter_compiles_to_one_number():
    assert _filter_dictionary({"name": "lowpass", "cutoff": "450", "order": 2}) == \
        {"name": "lowpass", "cutoff": 450.0, "order": 2}
    assert _filter_dictionary({"name": "highpass", "cutoff": " 20 ", "order": 3}) == \
        {"name": "highpass", "cutoff": 20.0, "order": 3}


def test_a_notch_compiles_to_one_cutoff_and_a_bandwidth_and_no_order():
    dictionary = _filter_dictionary({"name": "notch", "cutoff": "60",
                                     "bandwidth": 3.0, "order": 4})

    assert dictionary == {"name": "notch", "cutoff": 60.0, "bandwidth": 3.0}
    assert "order" not in dictionary, "a notch is not an order-n filter"


def test_a_notch_takes_only_the_first_of_several_cutoffs():
    assert _filter_dictionary({"name": "notch", "cutoff": "60,120"})["cutoff"] == 60.0


def test_standardize_compiles_to_nothing_but_its_name():
    assert _filter_dictionary({"name": "standardize", "cutoff": "20,450",
                               "order": 4}) == {"name": "standardize"}


def test_an_unreadable_cutoff_falls_back_to_the_usual_band():
    assert _filter_dictionary({"name": "bandpass", "cutoff": "wide"}) == \
        {"name": "bandpass", "cutoff": [20.0, 450.0], "order": 4}


def test_a_cutoff_can_be_given_as_numbers_rather_than_text():
    assert _filter_dictionary({"name": "bandpass", "cutoff": [20, 450]})["cutoff"] == \
        [20.0, 450.0]


def test_the_compiled_dictionary_is_one_libemg_s_filter_accepts():
    from libemg.filtering import Filter

    for params in ({"name": "bandpass", "cutoff": "20,450", "order": 4},
                   {"name": "lowpass", "cutoff": "450", "order": 2},
                   {"name": "highpass", "cutoff": "20", "order": 2},
                   {"name": "bandstop", "cutoff": "58,62", "order": 2},
                   {"name": "notch", "cutoff": "60", "bandwidth": 3.0}):
        conditioner = Filter(sampling_frequency=1000)
        conditioner.install_filters(_filter_dictionary(params))
        conditioner.filter(np.random.default_rng(0).standard_normal((1000, 2)))


def test_standardize_needs_data_and_so_cannot_be_installed_bare():
    # The premise behind the two tests below: standardizing means subtracting a
    # mean and dividing by a deviation, and those have to be measured from
    # something. Installing the compiled dictionary on its own is rejected by
    # the filter itself.
    from libemg.filtering import Filter

    with pytest.raises(AssertionError):
        Filter(sampling_frequency=1000).install_filters(
            _filter_dictionary({"name": "standardize"}))


def test_standardize_is_refused_on_a_live_pipeline_with_a_reason(synthetic_model,
                                                                 tmp_path):
    # A live stream has nothing to measure statistics from before it starts, so
    # this is refused while compiling. Letting it through would fail inside a
    # spawned executor, where the reason is much harder to see.
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")
    document.set_param(nodes["filter"], "name", "standardize")

    with pytest.raises(CompileError) as raised:
        compile_pipeline(document)

    message = str(raised.value)
    assert "standardize" in message
    assert "stored-data" in message, "the message should say where it does work"


def test_standardize_works_offline_because_the_recording_supplies_the_statistics(
        recorded_model):
    # Offline the recording is already loaded, so it can supply the mean and
    # deviation the filter needs.
    document, nodes = _offline_document(recorded_model)
    document.set_param(nodes["filter"], "name", "standardize")

    pipeline = compile_pipeline(document)
    assert pipeline.plan["filters"] == [{"name": "standardize"}]

    results = pipeline.run()
    assert pipeline.progress == 1.0
    assert set(results) == {"CA", "RECALL"}


# ======================================================================
# 18. the regex filters a block's text compiles to
# ======================================================================
def test_one_line_becomes_one_regex_filter():
    assert _parse_regex_filters("C_|_R|0,1,2|classes") == [
        {"left_bound": "C_", "right_bound": "_R", "values": ["0", "1", "2"],
         "description": "classes"}]


def test_several_lines_become_several_filters_and_blanks_are_ignored():
    text = "\n\nC_|_R|0,1|classes\n   \n_R_|_emg.csv|0,1|reps\n\n"

    parsed = _parse_regex_filters(text)

    assert [f["description"] for f in parsed] == ["classes", "reps"]
    assert parsed[1]["right_bound"] == "_emg.csv"


def test_surrounding_whitespace_is_trimmed_from_every_field():
    parsed = _parse_regex_filters("  C_ | _R | 0 , 1 | classes  ")[0]

    assert parsed == {"left_bound": "C_", "right_bound": "_R",
                      "values": ["0", "1"], "description": "classes"}


def test_empty_text_parses_to_no_filters():
    assert _parse_regex_filters("") == []
    assert _parse_regex_filters("   \n  \n") == []


def test_a_line_that_is_missing_a_field_is_refused_with_an_example():
    with pytest.raises(CompileError) as raised:
        _parse_regex_filters("C_|_R|classes")

    message = str(raised.value)
    assert "C_|_R|classes" in message, "the offending line has to be quoted back"
    assert "left|right|values|description" in message
    assert "for example" in message


def test_only_the_malformed_line_has_to_be_wrong_for_the_whole_thing_to_be_refused():
    with pytest.raises(CompileError):
        _parse_regex_filters("C_|_R|0,1|classes\noops")


# ======================================================================
# 19. the features a block resolves to
# ======================================================================
def test_a_named_group_wins_over_the_custom_list():
    from libemg.feature_extractor import FeatureExtractor

    groups = FeatureExtractor().get_feature_groups()
    resolved = _resolve_features({"feature_group": "HTD",
                                  "features": ["MAV", "RMS"]})

    assert resolved == list(groups["HTD"])
    assert resolved != ["MAV", "RMS"]


def test_the_custom_list_is_used_when_no_group_is_chosen():
    assert _resolve_features({"feature_group": "(custom)",
                              "features": ["MAV", "RMS"]}) == ["MAV", "RMS"]
    assert _resolve_features({"features": ["ZC"]}) == ["ZC"]


def test_a_group_this_library_does_not_have_falls_back_to_the_custom_list():
    assert _resolve_features({"feature_group": "NOTAGROUP",
                              "features": ["WL"]}) == ["WL"]


def test_asking_for_nothing_at_all_still_yields_a_feature():
    # An empty feature matrix would make every downstream shape zero-wide, so
    # the fallback is one feature rather than none.
    assert _resolve_features({}) == ["MAV"]
    assert _resolve_features({"feature_group": "(custom)", "features": []}) == ["MAV"]


# ======================================================================
# 20. the window is fused into whatever it feeds
# ======================================================================
def test_the_window_block_does_not_become_a_stage_of_its_own(synthetic_model, tmp_path):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")

    pipeline = compile_pipeline(document)

    assert isinstance(pipeline, OnlinePipeline)
    hooks = _hooks(pipeline.graph)
    assert nodes["window"] not in hooks, "the window compiled to a stage"
    assert nodes["window"] not in pipeline.graph._groups, \
        "the window compiled to an executor"
    assert not any(tag_for(nodes["window"], "output") == item[0]
                   for item in pipeline.shared_memory_items), \
        "the window compiled to a shared-memory item"
    assert set(hooks) == {nodes["filter"], nodes["features"], nodes["model"],
                          nodes["sink"]}


def test_the_features_stage_observes_the_filter_on_the_window_s_terms(
        synthetic_model, tmp_path):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log",
                                       window_size=256, increment=64)

    pipeline = compile_pipeline(document)

    features = _hooks(pipeline.graph)[nodes["features"]]
    assert len(features.inputs) == 1
    observed = features.inputs[0]
    assert observed.tag == tag_for(nodes["filter"], "output"), \
        "the features stage should read the filter directly, not the window"
    assert observed.criterion.describe() == "OnSamples(64)", \
        "the window's increment is what wakes the features stage"
    assert observed.mode == WINDOW_MODE
    assert observed.size == 256, "the window's size is how many rows arrive"

    described = pipeline.graph.describe()
    assert f"{tag_for(nodes['filter'], 'output')} [OnSamples(64), window(256)]" in described


def test_the_helpers_that_fuse_the_window_say_so_directly(synthetic_model, tmp_path):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")

    window = _window_feeding(document, nodes["features"])
    assert window is not None and window.id == nodes["window"]
    assert _window_feeding(document, nodes["filter"]) is None

    # The features block reads past the window to the item actually carrying
    # samples, which is the filter's output.
    assert _upstream_stream(document, nodes["features"]) == \
        tag_for(nodes["filter"], "output")
    assert _upstream_stream(document, nodes["filter"]) == \
        tag_for(nodes["source"], "emg")


def test_a_model_fed_features_reads_the_newest_row_rather_than_a_window(
        synthetic_model, tmp_path):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")

    hooks = _hooks(compile_pipeline(document).graph)

    model = hooks[nodes["model"]]
    assert model.inputs[0].tag == tag_for(nodes["features"], "output")
    assert model.inputs[0].mode == LATEST
    assert model.inputs[0].criterion.describe() == "OnSamples(1)"


def test_a_model_cannot_be_fed_windows_directly():
    # The compiler has a branch for a model that reads windows rather than
    # features, and it inherits the window's terms. Nothing in the editor can
    # reach it: a model's input carries features and a window's output carries
    # windows, so the link is refused before anything is compiled.
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer", node_id=_unique("src"))
    window = document.add_node("window.enframe", node_id=_unique("win"))
    model = document.add_node("model.classifier", node_id=_unique("mdl"))
    document.connect(source, "emg", window, "input")

    why = document.why_not_connect(window, "output", model, "input")
    assert why is not None
    assert "windows" in why and "features" in why


def test_each_stage_gets_its_own_executor(synthetic_model, tmp_path):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")

    groups = compile_pipeline(document).graph._groups

    assert set(groups) == {nodes["filter"], nodes["features"], nodes["model"],
                           nodes["sink"]}
    assert all(len(hooks) == 1 for hooks in groups.values())


def test_the_feature_item_is_as_wide_as_features_times_channels(
        synthetic_model, tmp_path):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")

    pipeline = compile_pipeline(document)

    shapes = {item[0]: item[1] for item in pipeline.shared_memory_items}
    assert shapes[tag_for(nodes["source"], "emg")][1] == CHANNELS
    assert shapes[tag_for(nodes["filter"], "output")][1] == CHANNELS
    assert shapes[tag_for(nodes["features"], "output")][1] == len(FEATURES) * CHANNELS
    assert shapes[tag_for(nodes["model"], "output")][1] == 2, "class and confidence"


def test_a_channel_mask_narrows_every_shape_after_it(synthetic_model, tmp_path):
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer", node_id=_unique("src"),
                               params={"num_channels": 8})
    mask = document.add_node("transform.channel_mask", node_id=_unique("msk"),
                             params={"channels": "0,2,5"})
    window = document.add_node("window.enframe", node_id=_unique("win"))
    features = document.add_node("features.extract", node_id=_unique("fea"),
                                 params={"features": list(FEATURES)})
    document.connect(source, "emg", mask, "input")
    document.connect(mask, "output", window, "input")
    document.connect(window, "output", features, "input")
    document.add_probe(features, "output")

    pipeline = compile_pipeline(document)

    shapes = {item[0]: item[1] for item in pipeline.shared_memory_items}
    assert shapes[tag_for(mask, "output")][1] == 3
    assert shapes[tag_for(features, "output")][1] == len(FEATURES) * 3
    assert _hooks(pipeline.graph)[mask].channels == [0, 2, 5]


# ======================================================================
# 21. a pipeline that cannot run fails at compile time, not half way through
# ======================================================================
def test_compiling_a_document_that_does_not_validate_is_refused():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    window = document.add_node("window.enframe")
    document.connect(source, "emg", window, "input")

    with pytest.raises(CompileError) as raised:
        compile_pipeline(document)

    message = str(raised.value)
    assert "cannot run yet" in message
    for problem in document.validate():
        assert problem in message, "every reason has to reach the user"


def test_compiling_a_document_with_an_unresolved_block_is_refused():
    document = PipelineDocument.from_dict({"nodes": [
        {"id": "src", "spec_id": "source.synthetic_streamer"},
        {"id": "future", "spec_id": "transform.time_machine"}]})

    with pytest.raises(CompileError, match="not recognised"):
        compile_pipeline(document)


def test_a_live_model_with_no_fitted_model_file_is_refused(tmp_path):
    document, nodes = _online_document("", tmp_path / "out.log")
    assert document.validate() == [], "this has to fail at compile, not validate"

    with pytest.raises(CompileError) as raised:
        compile_pipeline(document)

    message = str(raised.value)
    assert nodes["model"] in message
    assert "Fitted Model" in message


def test_a_live_model_pointing_at_a_file_that_is_not_there_is_refused(tmp_path):
    document, nodes = _online_document(str(tmp_path / "absent.pkl"),
                                       tmp_path / "out.log")

    with pytest.raises(CompileError, match="needs a fitted model"):
        compile_pipeline(document)


def test_a_stored_data_block_with_no_folder_is_refused(recorded_model):
    document, _ = _offline_document(recorded_model, folder="")

    with pytest.raises(CompileError, match="needs a folder"):
        compile_pipeline(document)


def test_a_stored_data_block_with_no_regex_filters_is_refused(recorded_model,
                                                              tmp_path):
    document, nodes = _offline_document(recorded_model)
    document.set_param(nodes["source"], "regex_filters", "")

    with pytest.raises(CompileError, match="at least one regex filter"):
        compile_pipeline(document)


# ======================================================================
# 22. an offline document compiles to a plan, not a graph
# ======================================================================
def test_stored_data_compiles_to_an_offline_pipeline_carrying_the_whole_plan(
        recorded_model):
    document, nodes = _offline_document(recorded_model, metrics=["CA", "RECALL"])
    document.set_param(nodes["window"], "window_size", 256)
    document.set_param(nodes["window"], "window_increment", 64)

    pipeline = compile_pipeline(document)

    assert isinstance(pipeline, OfflinePipeline)
    plan = pipeline.plan
    assert plan["window_size"] == 256
    assert plan["window_increment"] == 64
    assert plan["features"] == FEATURES
    assert plan["metrics"] == ["CA", "RECALL"]
    assert plan["model_path"] == recorded_model
    assert plan["folder"] == OFFLINE_DATA
    assert plan["sampling_rate"] == 1000
    assert plan["filters"] == [{"name": "bandpass", "cutoff": [20.0, 450.0],
                                "order": 4}]
    assert [f["description"] for f in plan["regex_filters"]] == ["classes", "reps"]
    assert plan["null_label"] is None, "the -1 placeholder means no null class"
    assert plan["channels"] is None


def test_an_offline_plan_picks_up_a_channel_mask_and_a_null_label(recorded_model):
    document, nodes = _offline_document(recorded_model)
    mask = document.add_node("transform.channel_mask", params={"channels": "0,1"})
    document.disconnect(nodes["source"], "emg", nodes["filter"], "input")
    document.connect(nodes["source"], "emg", mask, "input")
    document.connect(mask, "output", nodes["filter"], "input")
    document.set_param(nodes["sink"], "null_label", 2)

    plan = compile_pipeline(document).plan

    assert plan["channels"] == [0, 1]
    assert plan["null_label"] == 2


def test_an_offline_document_compiles_without_touching_the_disk(recorded_model):
    # Compiling is a description of the run, so it must not read the folder.
    document, _ = _offline_document(recorded_model)
    pipeline = compile_pipeline(document)

    assert pipeline.progress == 0.0
    assert pipeline.results == {}


# ======================================================================
# 23. the offline run, over real recordings
# ======================================================================
def test_an_offline_run_reports_real_progress_and_returns_the_metrics(
        recorded_model):
    document, _ = _offline_document(recorded_model, metrics=["CA", "RECALL"])
    pipeline = compile_pipeline(document)
    seen = []

    results = pipeline.run(on_progress=seen.append)

    assert seen, "a run over stored data knows its own total and has to report it"
    assert all(later >= earlier for earlier, later in zip(seen, seen[1:])), \
        f"progress went backwards: {seen}"
    assert seen[-1] == pytest.approx(1.0)
    assert pipeline.progress == pytest.approx(1.0)
    assert set(results) == {"CA", "RECALL"}
    assert 0.0 <= float(results["CA"]) <= 1.0
    assert results is pipeline.results


def test_an_offline_run_asked_for_a_different_metric_returns_that_one(
        recorded_model):
    document, _ = _offline_document(recorded_model, metrics=["CA", "CONF_MAT"])

    results = compile_pipeline(document).run()

    assert set(results) == {"CA", "CONF_MAT"}
    assert np.asarray(results["CONF_MAT"]).ndim == 2


# ======================================================================
# 24. a run can be asked to stop
# ======================================================================
def test_a_stop_request_ends_a_run_before_it_starts_working(recorded_model):
    document, _ = _offline_document(recorded_model)
    pipeline = compile_pipeline(document)
    pipeline.request_stop()

    results = pipeline.run()

    assert results == {}
    assert pipeline.progress == 0.0, "nothing should have been processed"


def test_a_stop_request_part_way_through_ends_the_run_early(recorded_model):
    document, _ = _offline_document(recorded_model)
    baseline = compile_pipeline(document)
    every_file = []
    baseline.run(on_progress=every_file.append)
    assert len(every_file) > 2, "this only means something with several recordings"

    pipeline = compile_pipeline(document)
    seen = []

    def stop_after_the_first_file(fraction):
        seen.append(fraction)
        pipeline.request_stop()

    results = pipeline.run(on_progress=stop_after_the_first_file)

    # One report, from the one recording the loop got through. A cancelled run
    # must not also report 1.0: that would leave the progress bar full and make
    # a cancelled run indistinguishable from a complete one, even though the
    # metrics below cover only part of the recording.
    assert len(seen) == 1, f"the run kept reading recordings after being asked to stop: {seen}"
    assert seen[0] < 1.0
    assert seen[0] == pytest.approx(every_file[0]), \
        "the run should have stopped after exactly the first recording"
    assert pipeline.stopped is True
    assert pipeline.progress < 1.0
    assert pipeline.files_read < pipeline.files_total
    # A completed run says so, which is the comparison that makes the flag useful.
    assert baseline.stopped is False and baseline.progress == 1.0
    # It still scores what it managed to read, rather than throwing it away.
    assert set(results) == set(baseline.results)


# ======================================================================
# 25. end to end, on a source that needs no hardware
# ======================================================================
def test_a_compiled_pipeline_carries_data_through_every_stage(
        synthetic_model, tmp_path, running):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")
    pipeline = compile_pipeline(document)
    running.append(pipeline)

    pipeline.start()
    assert pipeline.running is True

    source_tag = tag_for(nodes["source"], "emg")
    filter_tag = tag_for(nodes["filter"], "output")
    feature_tag = tag_for(nodes["features"], "output")
    model_tag = tag_for(nodes["model"], "output")

    def enough():
        status = pipeline.status()
        return status.get(model_tag) is not None and \
            status[model_tag].total_samples >= 20

    assert _wait_until(enough, timeout=120), \
        f"predictions never arrived: {pipeline.graph.describe()}"

    status = pipeline.status()
    for tag in (source_tag, filter_tag, feature_tag, model_tag):
        assert tag in status, tag
        assert status[tag].total_samples > 0, f"{tag} never produced anything"

    samples = status[source_tag].total_samples
    rows = status[feature_tag].total_samples
    assert rows == pytest.approx(samples / INCREMENT, abs=10), \
        f"{samples} samples should give about {samples // INCREMENT} rows, got {rows}"
    assert status[model_tag].total_samples == pytest.approx(rows, abs=5), \
        "every feature row should reach the model"
    assert status[filter_tag].total_samples == pytest.approx(samples, abs=200)

    pipeline.stop(timeout=15)
    assert pipeline.running is False
    assert pipeline.streamers == []
    assert not any(executor.is_alive() for executor in pipeline.graph._executors)


def test_the_newest_prediction_is_a_class_and_a_confidence(
        synthetic_model, tmp_path, running):
    from libemg.shared_memory_manager import SharedMemoryManager

    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")
    pipeline = compile_pipeline(document)
    running.append(pipeline)
    pipeline.start()
    model_tag = tag_for(nodes["model"], "output")

    assert _wait_until(
        lambda: (pipeline.status().get(model_tag) is not None
                 and pipeline.status()[model_tag].total_samples >= 5),
        timeout=120), "no prediction was ever published"

    reader = SharedMemoryManager()
    try:
        declared = {item[0]: item for item in pipeline.shared_memory_items}
        assert reader.find_variable(*declared[model_tag])
        row, _ = reader.read_window(model_tag, 1, chronological=False)
        row = np.asarray(row).ravel()
        assert row.size == 2
        assert float(row[0]) in (0.0, 1.0), f"not one of the fitted classes: {row}"
        assert 0.0 <= float(row[1]) <= 1.0, f"confidence out of range: {row}"
    finally:
        reader.cleanup(parent=False)


# ======================================================================
# 26. probes compile to items of their own
# ======================================================================
def test_a_probe_compiles_to_its_own_item_drawn_as_its_port_type_says(
        synthetic_model, tmp_path):
    document, nodes = _online_document(
        synthetic_model, tmp_path / "out.log",
        probes=[("filter", "output"), ("features", "output"), ("model", "output")])

    pipeline = compile_pipeline(document)

    assert set(pipeline.probe_items) == {(nodes["filter"], "output"),
                                         (nodes["features"], "output"),
                                         (nodes["model"], "output")}
    renders = {node: entry["render"] for (node, _), entry in pipeline.probe_items.items()}
    assert renders[nodes["filter"]] == "timeseries"
    assert renders[nodes["features"]] == "bars"
    assert renders[nodes["model"]] == "probabilities"
    for probe in document.probes:
        assert pipeline.probe_items[(probe.node, probe.port)]["render"] == \
            document.probe_render(probe)


def test_a_probe_item_is_declared_and_hooked_without_disturbing_the_stage_it_watches(
        synthetic_model, tmp_path):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log",
                                       probes=[("filter", "output")])

    pipeline = compile_pipeline(document)

    entry = pipeline.probe_items[(nodes["filter"], "output")]
    declared = {item[0]: item for item in pipeline.shared_memory_items}
    assert entry["tag"] in declared
    assert declared[entry["tag"]][1] == (entry["rows"], entry["width"])
    assert entry["width"] == CHANNELS

    hooks = _hooks(pipeline.graph)
    probe = hooks[entry["tag"]]
    assert probe.inputs[0].tag == tag_for(nodes["filter"], "output")
    assert probe.inputs[0].criterion.describe() == "Periodic(20)", \
        "a probe runs on its own clock so it cannot stall what it watches"
    # The stage it watches is untouched: same input, same criterion.
    assert hooks[nodes["filter"]].inputs[0].tag == tag_for(nodes["source"], "emg")
    assert f"probe_{nodes['filter']}" in pipeline.graph._groups


def test_a_timeseries_probe_keeps_rows_and_a_snapshot_probe_keeps_one(
        synthetic_model, tmp_path):
    document, nodes = _online_document(
        synthetic_model, tmp_path / "out.log",
        probes=[("filter", "output"), ("model", "output")])

    pipeline = compile_pipeline(document)

    assert pipeline.probe_items[(nodes["filter"], "output")]["rows"] > 1
    assert pipeline.probe_items[(nodes["model"], "output")]["rows"] == 1


def test_a_probe_on_a_port_nothing_publishes_is_refused_at_the_document(
        synthetic_model, tmp_path):
    # A window has no item of its own, so a probe on one would have nothing to
    # copy. It is refused when it is asked for rather than dropped silently at
    # compile time, so the user finds out while they can still act on it.
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log")

    with pytest.raises(ValueError, match="cannot be probed"):
        document.add_probe(nodes["window"], "output")

    pipeline = compile_pipeline(document)
    assert pipeline.probe_items == {}


def test_a_running_probe_publishes_what_it_watches(synthetic_model, tmp_path,
                                                   running):
    document, nodes = _online_document(synthetic_model, tmp_path / "out.log",
                                       probes=[("filter", "output"),
                                               ("model", "output")])
    pipeline = compile_pipeline(document)
    running.append(pipeline)
    pipeline.start()

    tags = [entry["tag"] for entry in pipeline.probe_items.values()]
    assert _wait_until(
        lambda: all(pipeline.status().get(tag) is not None
                    and pipeline.status()[tag].total_samples > 0 for tag in tags),
        timeout=120), "a probe never published anything"
