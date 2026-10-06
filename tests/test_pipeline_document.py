"""Coverage for the two layers of the visual pipeline that need no display.

The registry describes what can be placed and the document is the pipeline as
plain data, so everything here runs in one process with nothing spawned and
nothing written outside ``tmp_path``.

Two things are worth more than the rest. The registry claims to be *generated*
from the library rather than hand-listed, so the assertions about its choices
compare against the live LibEMG values -- a hardcoded copy would pass a test
written against a literal list and fail these. And a saved pipeline has to
survive being opened by a LibEMG whose registry has moved on, so an unknown
block is checked to come back with its parameters and its position intact and
to be written out again unchanged.
"""
import json

import pytest

from libemg._gui._pipeline import (PipelineDocument, SCHEMA_VERSION, NodeSpec,
                                   ParamSpec, PortSpec, PortType,
                                   build_registry, default_registry)
from libemg._gui._pipeline.document import Link, Node, Probe, ValidationError, migrate
from libemg._gui._pipeline.registry import (BOOL, ENUM, FLOAT, FOLDER, INT,
                                            MULTI_SELECT, PATH, SINK, SOURCE,
                                            STR, _streamer_functions)


# ======================================================================
# helpers
# ======================================================================
def _online_document():
    """source -> filter -> window -> features -> classifier -> console."""
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer", params={"num_channels": 4})
    filtered = document.add_node("transform.filter")
    window = document.add_node("window.enframe")
    features = document.add_node("features.extract")
    model = document.add_node("model.classifier")
    sink = document.add_node("sink.console")
    document.connect(source, "emg", filtered, "input")
    document.connect(filtered, "output", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", model, "input")
    document.connect(model, "output", sink, "input")
    return document, (source, filtered, window, features, model, sink)


def _offline_document():
    """stored data -> filter -> window -> features -> classifier -> metrics."""
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": "somewhere", "regex_filters": "C_|_R|0,1|classes"})
    filtered = document.add_node("transform.filter")
    window = document.add_node("window.enframe")
    features = document.add_node("features.extract")
    model = document.add_node("model.classifier")
    metrics = document.add_node("sink.metrics")
    document.connect(source, "emg", filtered, "input")
    document.connect(filtered, "output", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", model, "input")
    document.connect(model, "output", metrics, "input")
    document.connect(source, "labels", metrics, "labels")
    return document, (source, filtered, window, features, model, metrics)


# ======================================================================
# 1. a parameter knows what a valid value looks like
# ======================================================================
def test_an_int_parameter_takes_a_string_and_clamps_to_its_bounds():
    spec = ParamSpec("window_size", INT, default=200, minimum=2, maximum=1000)

    assert spec.coerce("350") == 350
    assert spec.coerce(350.9) == 350
    assert spec.coerce(1) == 2, "below the minimum has to come back at the minimum"
    assert spec.coerce(99999) == 1000
    assert spec.coerce("not a number") == 200, "unreadable falls back to the default"


def test_a_float_parameter_takes_a_string_and_clamps_to_its_bounds():
    spec = ParamSpec("threshold", FLOAT, default=0.5, minimum=0.0, maximum=1.0)

    assert spec.coerce("0.25") == pytest.approx(0.25)
    assert spec.coerce(-3.0) == pytest.approx(0.0)
    assert spec.coerce(7.5) == pytest.approx(1.0)
    assert spec.coerce("banana") == pytest.approx(0.5)


def test_a_bool_parameter_reads_the_words_a_widget_hands_back():
    spec = ParamSpec("velocity", BOOL, default=False)

    for truthy in ("true", "True", "1", "yes", "on", "  TRUE  "):
        assert spec.coerce(truthy) is True, truthy
    for falsy in ("false", "0", "no", "off", ""):
        assert spec.coerce(falsy) is False, falsy
    assert spec.coerce(1) is True
    assert spec.coerce(0) is False


def test_an_enum_falls_back_to_its_default_when_given_something_unknown():
    spec = ParamSpec("name", ENUM, choices=["bandpass", "notch"], default="bandpass")

    assert spec.coerce("notch") == "notch"
    assert spec.coerce("wavelet") == "bandpass"


def test_a_multi_select_drops_values_that_are_not_on_offer():
    spec = ParamSpec("features", MULTI_SELECT, choices=["MAV", "ZC", "WL"],
                     default=["MAV"])

    assert spec.coerce(["MAV", "NOTAFEATURE", "WL"]) == ["MAV", "WL"]
    assert spec.coerce([]) == []
    assert spec.coerce(7) == ["MAV"], "an unreadable value falls back to the default"


def test_none_always_comes_back_as_the_default():
    for spec in (ParamSpec("a", INT, default=3),
                 ParamSpec("b", FLOAT, default=1.5),
                 ParamSpec("c", BOOL, default=True),
                 ParamSpec("d", STR, default="hello"),
                 ParamSpec("e", PATH, default=""),
                 ParamSpec("f", FOLDER, default=""),
                 ParamSpec("g", ENUM, choices=["x", "y"], default="y"),
                 ParamSpec("h", MULTI_SELECT, choices=["x"], default=["x"])):
        assert spec.coerce(None) == spec.default, spec.name


def test_a_string_parameter_stringifies_whatever_it_is_given():
    spec = ParamSpec("cutoff", STR, default="20,450")

    assert spec.coerce(60) == "60"
    assert spec.coerce("20,450") == "20,450"


# ======================================================================
# 2. a spec that could not be drawn is refused at construction
# ======================================================================
@pytest.mark.parametrize("kind", [ENUM, MULTI_SELECT])
def test_a_choice_parameter_without_choices_is_refused(kind):
    with pytest.raises(ValueError, match="no choices"):
        ParamSpec("name", kind, default=None)
    with pytest.raises(ValueError, match="no choices"):
        ParamSpec("name", kind, choices=[], default=None)


def test_a_parameter_label_defaults_to_a_prettified_name():
    assert ParamSpec("window_increment", INT).label == "Window Increment"
    assert ParamSpec("ip", STR).label == "Ip"
    assert ParamSpec("window_size", INT, label="Size").label == "Size"


# ======================================================================
# 3. a port that carries nothing the toolkit understands is refused
# ======================================================================
def test_a_port_with_an_unknown_type_is_refused():
    with pytest.raises(ValueError, match="unknown type"):
        PortSpec("input", "spaghetti")


def test_a_port_label_defaults_to_a_prettified_name():
    assert PortSpec("input", PortType.SAMPLES).label == "Input"
    assert PortSpec("feature_matrix", PortType.FEATURES).label == "Feature Matrix"


# ======================================================================
# 4. the registry is generated from the library, not copied out of it
# ======================================================================
def test_the_feature_choices_are_the_feature_extractor_s_own_list():
    from libemg.feature_extractor import FeatureExtractor

    spec = default_registry()["features.extract"]
    assert list(spec.param("features").choices) == list(
        FeatureExtractor().get_feature_list())


def test_the_feature_group_choices_are_the_published_groups():
    from libemg.feature_extractor import FeatureExtractor

    spec = default_registry()["features.extract"]
    expected = ["(custom)"] + list(FeatureExtractor().get_feature_groups().keys())
    assert list(spec.param("feature_group").choices) == expected


def test_the_metric_choices_are_the_offline_metrics_own_list():
    from libemg.offline_metrics import OfflineMetrics

    spec = default_registry()["sink.metrics"]
    assert list(spec.param("metrics").choices) == list(
        OfflineMetrics().get_available_metrics())


def test_the_model_choices_are_the_predictors_own_tables():
    from libemg.emg_predictor import CLASSIFIER_MODELS, REGRESSOR_MODELS

    registry = default_registry()
    assert list(registry["model.classifier"].param("model").choices) == \
        list(CLASSIFIER_MODELS)
    assert list(registry["model.regressor"].param("model").choices) == \
        list(REGRESSOR_MODELS)


def test_rebuilding_the_registry_gives_the_same_description():
    # If any of the above were hardcoded, a rebuild would still agree with
    # itself; this only guards the caching in default_registry().
    assert build_registry() == default_registry()


# ======================================================================
# 5. what is offered as a source is what could actually be connected
# ======================================================================
def test_every_generated_source_accepts_shared_memory_items():
    import inspect

    from libemg import streamers

    registry = default_registry()
    generated = [spec_id for spec_id in registry
                 if spec_id.startswith("source.")
                 and spec_id not in ("source.offline", "source.synthetic_streamer")]
    assert generated, "no device sources were generated at all"
    for spec_id in generated:
        function = getattr(streamers, spec_id.split(".", 1)[1])
        parameters = inspect.signature(function).parameters
        assert "shared_memory_items" in parameters, spec_id


def test_the_udp_mock_stream_is_not_offered_as_a_source():
    from libemg import streamers

    # It exists, and it is deliberately left out: it publishes over a socket
    # rather than into shared memory, so a pipeline could never read it.
    assert hasattr(streamers, "mock_emg_stream")
    assert "mock_emg_stream" not in _streamer_functions()
    assert "source.mock_emg_stream" not in default_registry()


def test_the_synthetic_source_is_offered_even_though_it_is_not_a_device():
    spec = default_registry()["source.synthetic_streamer"]
    assert spec.category == SOURCE
    assert spec.online_only is True
    assert spec.port("emg", "output").type == PortType.SAMPLES


# ======================================================================
# 6. the registry is internally consistent
# ======================================================================
def test_every_block_has_a_unique_id_that_matches_its_key():
    registry = default_registry()
    assert len(registry) == len({spec.id for spec in registry.values()})
    for key, spec in registry.items():
        assert key == spec.id


def test_every_port_carries_a_type_the_toolkit_knows():
    for spec in default_registry().values():
        for port in tuple(spec.inputs) + tuple(spec.outputs):
            assert port.type in PortType.ALL, f"{spec.id}.{port.name}"


def test_every_port_type_knows_how_a_probe_should_draw_it():
    for port_type in PortType.ALL:
        assert port_type in PortType.RENDER, port_type


def test_every_block_declares_a_category_the_compiler_understands():
    from libemg._gui._pipeline.registry import (FEATURES, MODEL, TRANSFORM,
                                                WINDOW)

    known = {SOURCE, TRANSFORM, WINDOW, FEATURES, MODEL, SINK}
    for spec in default_registry().values():
        assert spec.category in known, spec.id


# ======================================================================
# 7. placing and removing blocks
# ======================================================================
def test_placing_the_same_block_twice_gives_two_different_nodes():
    document = PipelineDocument()
    first = document.add_node("transform.filter")
    second = document.add_node("transform.filter")

    assert first != second
    assert set(document.nodes) == {first, second}


def test_a_new_node_starts_with_its_spec_s_defaults():
    document = PipelineDocument()
    node_id = document.add_node("window.enframe")

    assert document.nodes[node_id].params == {"window_size": 200, "window_increment": 50}


def test_parameters_handed_to_add_node_are_coerced_to_their_declared_kinds():
    document = PipelineDocument()
    node_id = document.add_node("window.enframe",
                                params={"window_size": "512", "window_increment": 0})

    params = document.nodes[node_id].params
    assert params["window_size"] == 512
    assert params["window_increment"] == 1, "the increment has a minimum of one"


def test_set_param_coerces_too():
    document = PipelineDocument()
    node_id = document.add_node("model.classifier")
    document.set_param(node_id, "model", "NOT_A_MODEL")
    document.set_param(node_id, "velocity", "yes")

    assert document.nodes[node_id].params["model"] == "LDA"
    assert document.nodes[node_id].params["velocity"] is True


def test_placing_an_unknown_block_is_refused():
    with pytest.raises(KeyError):
        PipelineDocument().add_node("source.telepathy")


def test_reusing_a_node_id_is_refused():
    document = PipelineDocument()
    document.add_node("transform.filter", node_id="only")
    with pytest.raises(ValueError, match="already exists"):
        document.add_node("transform.filter", node_id="only")


def test_removing_a_node_takes_its_links_and_probes_with_it():
    document, (source, filtered, window, features, model, sink) = _online_document()
    document.add_probe(filtered, "output")
    document.add_probe(features, "output")

    document.remove_node(filtered)

    assert filtered not in document.nodes
    assert not [l for l in document.links
                if filtered in (l.from_node, l.to_node)], "a dangling link survived"
    assert [p.node for p in document.probes] == [features]


# ======================================================================
# 8. every reason a link is refused, in the words the editor shows
# ======================================================================
def test_a_link_between_mismatched_types_is_refused():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    features = document.add_node("features.extract")

    why = document.why_not_connect(source, "emg", features, "input")
    assert why is not None
    assert "samples" in why and "windows" in why


def test_a_link_to_a_port_that_does_not_exist_is_refused():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")

    assert "no output called 'nope'" in document.why_not_connect(
        source, "nope", filtered, "input")
    assert "no input called 'nope'" in document.why_not_connect(
        source, "emg", filtered, "nope")


def test_a_block_cannot_feed_itself():
    document = PipelineDocument()
    filtered = document.add_node("transform.filter")

    assert document.why_not_connect(filtered, "output", filtered, "input") == \
        "A block cannot feed itself."


def test_a_link_that_is_already_there_is_refused():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    document.connect(source, "emg", filtered, "input")

    assert document.why_not_connect(source, "emg", filtered, "input") == \
        "Those are already connected."


def test_a_second_link_into_a_single_link_input_is_refused():
    document = PipelineDocument()
    first = document.add_node("source.synthetic_streamer")
    second = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    document.connect(first, "emg", filtered, "input")

    why = document.why_not_connect(second, "emg", filtered, "input")
    assert "already has a connection" in why
    assert "interleave" in why


def test_a_link_that_would_close_a_loop_is_refused():
    # Both ends carry samples on purpose. A cycle built out of mismatched
    # ports would be refused by the type check first and would prove nothing
    # about loop detection.
    document = PipelineDocument()
    first = document.add_node("transform.filter")
    second = document.add_node("transform.filter")
    document.connect(first, "output", second, "input")

    why = document.why_not_connect(second, "output", first, "input")
    assert why is not None, "closing the loop was allowed"
    assert "loop" in why.lower()


def test_a_longer_loop_is_refused_too():
    document = PipelineDocument()
    chain = [document.add_node("transform.filter") for _ in range(4)]
    for before, after in zip(chain, chain[1:]):
        document.connect(before, "output", after, "input")

    assert "loop" in document.why_not_connect(chain[-1], "output", chain[0], "input").lower()


def test_a_link_involving_a_node_that_is_not_on_the_canvas_is_refused():
    document = PipelineDocument()
    filtered = document.add_node("transform.filter")

    assert "not on the canvas" in document.why_not_connect(
        "ghost", "output", filtered, "input")


def test_an_acceptable_link_has_no_reason_against_it():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")

    assert document.why_not_connect(source, "emg", filtered, "input") is None


# ======================================================================
# 9. connect raises with exactly the reason why_not_connect gives
# ======================================================================
def test_connect_raises_with_the_reason_the_editor_would_have_shown():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    features = document.add_node("features.extract")
    why = document.why_not_connect(source, "emg", features, "input")

    with pytest.raises(ValueError) as raised:
        document.connect(source, "emg", features, "input")

    assert str(raised.value) == why
    assert document.links == [], "a refused link must not be recorded"


def test_disconnect_removes_a_link_and_ignores_one_that_is_not_there():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    document.connect(source, "emg", filtered, "input")

    document.disconnect(source, "emg", filtered, "input")
    document.disconnect(source, "emg", filtered, "input")

    assert document.links == []


# ======================================================================
# 10. the mode is inferred from the sources
# ======================================================================
def test_an_empty_document_has_no_mode():
    assert PipelineDocument().mode() == "empty"
    document = PipelineDocument()
    document.add_node("transform.filter")
    assert document.mode() == "empty", "a transform alone is still no source"


def test_a_document_fed_by_a_device_is_online():
    document = PipelineDocument()
    document.add_node("source.synthetic_streamer")
    assert document.mode() == "online"


def test_a_document_fed_by_recordings_is_offline():
    document = PipelineDocument()
    document.add_node("source.offline")
    assert document.mode() == "offline"


def test_a_document_fed_by_both_is_neither():
    document = PipelineDocument()
    document.add_node("source.synthetic_streamer")
    document.add_node("source.offline")
    assert document.mode() == "mixed"


# ======================================================================
# 11. validation names every reason a pipeline could not run
# ======================================================================
def test_a_complete_online_pipeline_validates():
    document, _ = _online_document()
    assert document.validate() == []
    assert document.check() is True


def test_a_complete_offline_pipeline_validates():
    document, _ = _offline_document()
    assert document.validate() == []


def test_a_pipeline_with_no_source_is_reported():
    document = PipelineDocument()
    document.add_node("sink.console")

    problems = document.validate()
    assert any("no source" in p for p in problems)


def test_a_pipeline_with_two_kinds_of_source_is_reported():
    document = PipelineDocument()
    live = document.add_node("source.synthetic_streamer")
    stored = document.add_node("source.offline")

    problems = " ".join(document.validate())
    assert "mixes live sources" in problems
    assert live in problems and stored in problems


def test_a_required_input_with_nothing_connected_is_reported():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    sink = document.add_node("sink.console")
    document.connect(source, "emg", filtered, "input")

    problems = document.validate()
    assert any(sink in p and "Input input" in p for p in problems)


def test_an_online_only_block_in_an_offline_pipeline_is_reported():
    document, (source, filtered, window, features, model, metrics) = _offline_document()
    socket = document.add_node("sink.socket")
    document.connect(model, "output", socket, "input")

    problems = document.validate()
    assert any(socket in p and "live stream" in p for p in problems)


def test_an_offline_only_block_in_an_online_pipeline_is_reported():
    document, (source, filtered, window, features, model, sink) = _online_document()
    metrics = document.add_node("sink.metrics")
    document.connect(model, "output", metrics, "input")

    problems = document.validate()
    assert any(metrics in p and "stored data" in p for p in problems)


def test_a_producer_nothing_reads_is_reported():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    window = document.add_node("window.enframe")
    sink = document.add_node("sink.console")
    document.connect(source, "emg", filtered, "input")
    document.connect(filtered, "output", window, "input")

    problems = document.validate()
    assert any(window in p and "nothing reads" in p for p in problems)


def test_a_probe_is_enough_of_a_reader_to_settle_that_complaint():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    document.connect(source, "emg", filtered, "input")
    document.add_probe(filtered, "output")

    assert not any("nothing reads" in p for p in document.validate())


def test_a_sink_s_own_unconnected_output_is_not_a_complaint():
    # The metrics block's output is the table the user reads once the run has
    # finished. Nothing downstream of it is the normal case, and reporting it
    # was a false positive.
    document, (source, filtered, window, features, model, metrics) = _offline_document()

    assert default_registry()["sink.metrics"].outputs, \
        "this test is pointless unless the metrics sink has an output"
    assert document.links_out_of(metrics) == []
    assert document.validate() == []


def test_a_pipeline_that_computes_nothing_anybody_sees_is_reported():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    document.connect(source, "emg", filtered, "input")

    problems = document.validate()
    assert any("no model, output or probe" in p for p in problems)


def test_a_probe_is_enough_to_make_a_pipeline_worth_running():
    # Watching a stage live is a legitimate pipeline on its own while
    # exploring, so a probe counts as something the user can see. Requiring a
    # model or a sink would mean a pipeline could not be built up a block at a
    # time, since every intermediate state would be refused.
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    document.connect(source, "emg", filtered, "input")
    document.add_probe(filtered, "output")

    problems = document.validate()
    assert not any("anybody could see" in p for p in problems)
    assert problems == []


def test_check_raises_a_validation_error_carrying_every_problem():
    document = PipelineDocument()
    document.add_node("sink.console")

    with pytest.raises(ValidationError) as raised:
        document.check()

    assert len(raised.value.problems) == len(document.validate()) >= 2
    for problem in raised.value.problems:
        assert problem in str(raised.value)


# ======================================================================
# 12. ordering
# ======================================================================
def test_order_puts_every_node_after_everything_that_feeds_it():
    document, nodes = _online_document()

    order = document.order()
    assert sorted(order) == sorted(document.nodes)
    position = {node_id: index for index, node_id in enumerate(order)}
    for link in document.links:
        assert position[link.from_node] < position[link.to_node], link


def test_order_covers_disconnected_islands_too():
    document = PipelineDocument()
    lonely = document.add_node("sink.console")
    source = document.add_node("source.synthetic_streamer")
    filtered = document.add_node("transform.filter")
    document.connect(source, "emg", filtered, "input")

    order = document.order()
    assert set(order) == {lonely, source, filtered}
    assert order.index(source) < order.index(filtered)


# ======================================================================
# 13. probes attach to a port, not to the canvas
# ======================================================================
def test_probing_the_same_port_twice_replaces_the_probe():
    document, (source, filtered, *_) = _online_document()
    document.add_probe(filtered, "output", hz=10)
    document.add_probe(filtered, "output", hz=60)

    assert len(document.probes) == 1
    assert document.probes[0].hz == 60
    assert document.has_probe(filtered, "output")


def test_probing_a_port_that_does_not_exist_is_refused():
    document, (source, filtered, *_) = _online_document()

    with pytest.raises(ValueError, match="no output called 'nope'"):
        document.add_probe(filtered, "nope")
    with pytest.raises(ValueError, match="no output called 'input'"):
        document.add_probe(filtered, "input")


def test_removing_a_probe_that_is_not_there_is_harmless():
    document, (source, filtered, *_) = _online_document()
    document.remove_probe(filtered, "output")

    assert document.probes == []
    assert document.has_probe(filtered, "output") is False


def test_how_a_probe_draws_follows_from_the_type_of_the_port():
    document, (source, filtered, window, features, model, sink) = _online_document()
    document.add_probe(filtered, "output")
    document.add_probe(features, "output")
    document.add_probe(model, "output")

    rendered = {p.node: document.probe_render(p) for p in document.probes}
    assert rendered[filtered] == "timeseries"
    assert rendered[features] == "bars"
    assert rendered[model] == "probabilities"


def test_a_window_cannot_be_probed_because_it_publishes_nothing():
    # A window is folded into the block it feeds rather than becoming a stage,
    # so it has no item to watch. Refusing here beats accepting the probe and
    # dropping it at compile time, which left an empty plot with no
    # explanation and let the probe count as a reader of a dangling branch.
    document, (source, filtered, window, features, model, sink) = _online_document()

    with pytest.raises(ValueError) as raised:
        document.add_probe(window, "output")

    message = str(raised.value)
    assert "cannot be probed" in message
    assert "folded into" in message
    assert "before it" in message and "after it" in message
    assert document.probes == []


def test_a_regressor_probe_draws_as_a_timeseries():
    document = PipelineDocument()
    source = document.add_node("source.synthetic_streamer")
    window = document.add_node("window.enframe")
    features = document.add_node("features.extract")
    model = document.add_node("model.regressor")
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", model, "input")
    document.add_probe(model, "output")

    assert document.probe_render(document.probes[0]) == "timeseries"


# ======================================================================
# 14. a pipeline comes back off disk as the pipeline that went on
# ======================================================================
def test_saving_and_loading_preserves_everything(tmp_path):
    document, (source, filtered, window, features, model, sink) = _online_document()
    document.set_param(window, "window_size", 128)
    document.set_param(features, "features", ["MAV", "RMS"])
    document.nodes[source].position = (17, 23)
    document.nodes[model].position = (900, 41)
    document.add_probe(filtered, "output", hz=12.5)
    document.add_probe(model, "output")
    document.canvas = {"zoom": 1.25}
    path = tmp_path / "pipeline.json"
    document.save(path)

    loaded = PipelineDocument.load(path)

    assert loaded.to_dict() == document.to_dict()
    assert set(loaded.nodes) == set(document.nodes)
    assert loaded.nodes[source].position == (17, 23)
    assert loaded.nodes[window].params["window_size"] == 128
    assert loaded.nodes[features].params["features"] == ["MAV", "RMS"]
    assert [l.key() for l in loaded.links] == [l.key() for l in document.links]
    assert {(p.node, p.port, p.hz) for p in loaded.probes} == \
        {(filtered, "output", 12.5), (model, "output", 30.0)}
    assert loaded.canvas == {"zoom": 1.25}
    assert loaded.validate() == []


def test_a_saved_file_records_the_schema_version(tmp_path):
    document, _ = _online_document()
    path = tmp_path / "pipeline.json"
    document.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION


# ======================================================================
# 15. a file written by a LibEMG this one has never met
# ======================================================================
def _future_file(tmp_path):
    """A file naming a block this registry does not have."""
    raw = {
        "schema_version": SCHEMA_VERSION,
        "nodes": [
            {"id": "src", "spec_id": "source.synthetic_streamer",
             "params": {"num_channels": 4, "sampling_rate": 1000,
                        "pattern": "bursts", "amplitude": 1.0,
                        "retired_option": "kept anyway"},
             "position": [10, 20]},
            {"id": "future", "spec_id": "transform.neural_denoiser",
             "params": {"depth": 7, "flavour": "quantum"},
             "position": [240, 60]},
        ],
        "links": [
            {"from_node": "src", "from_port": "emg",
             "to_node": "future", "to_port": "input"},
            {"from_node": "src", "from_port": "emg",
             "to_node": "vanished", "to_port": "input"},
        ],
        "probes": [{"node": "src", "port": "emg", "hz": 15.0},
                   {"node": "vanished", "port": "emg", "hz": 15.0}],
        "canvas": {"zoom": 2.0},
    }
    path = tmp_path / "from_the_future.json"
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path, raw


def test_a_block_this_version_does_not_know_loads_unresolved_with_everything_intact(tmp_path):
    path, raw = _future_file(tmp_path)

    document = PipelineDocument.load(path)

    node = document.nodes["future"]
    assert node.unresolved is True
    assert node.spec_id == "transform.neural_denoiser"
    assert node.params == {"depth": 7, "flavour": "quantum"}, "parameters were lost"
    assert node.position == (240, 60), "the layout was lost"
    assert document.spec("future") is None


def test_an_unresolved_block_is_reported_rather_than_ignored(tmp_path):
    path, _ = _future_file(tmp_path)
    document = PipelineDocument.load(path)

    problems = document.validate()
    assert any("not recognised by this version" in p and "future" in p
               for p in problems)


def test_an_unresolved_block_is_written_back_out_unchanged(tmp_path):
    path, raw = _future_file(tmp_path)
    document = PipelineDocument.load(path)

    written = {node["id"]: node for node in document.to_dict()["nodes"]}
    original = {node["id"]: node for node in raw["nodes"]}
    assert written["future"] == original["future"]

    # And it survives a second trip, so repeatedly opening and saving a file
    # from a newer LibEMG does not erode it.
    again = tmp_path / "again.json"
    document.save(again)
    assert PipelineDocument.load(again).to_dict() == document.to_dict()


def test_a_link_to_a_node_that_is_not_in_the_file_is_dropped(tmp_path):
    path, _ = _future_file(tmp_path)
    document = PipelineDocument.load(path)

    assert [l.key() for l in document.links] == [("src", "emg", "future", "input")]
    assert [p.key() for p in document.probes] == [("src", "emg")]


def test_a_parameter_the_spec_no_longer_declares_is_kept(tmp_path):
    path, _ = _future_file(tmp_path)
    document = PipelineDocument.load(path)

    params = document.nodes["src"].params
    assert params["retired_option"] == "kept anyway"
    assert params["num_channels"] == 4, "the declared parameters still get coerced"


def test_an_unresolved_block_refuses_to_be_linked_or_probed(tmp_path):
    path, _ = _future_file(tmp_path)
    document = PipelineDocument.load(path)
    document.add_node("sink.console", node_id="out")

    assert "does not recognise" in document.why_not_connect(
        "future", "output", "out", "input")
    with pytest.raises(ValueError):
        document.add_probe("future", "output")


def test_removing_the_unresolved_block_is_all_it_takes_to_move_on(tmp_path):
    path, _ = _future_file(tmp_path)
    document = PipelineDocument.load(path)
    document.remove_node("future")

    assert not any("not recognised" in p for p in document.validate())


# ======================================================================
# 16. a file from a newer schema is tried, not refused
# ======================================================================
def test_a_newer_schema_version_is_carried_forward_rather_than_rejected():
    raw = {"schema_version": SCHEMA_VERSION + 5, "nodes": [], "links": [],
           "probes": [], "canvas": {}}

    migrated = migrate(raw)

    assert migrated["schema_version"] == SCHEMA_VERSION
    assert raw["schema_version"] == SCHEMA_VERSION + 5, "migrate must not mutate its input"


def test_a_document_from_a_newer_schema_still_loads(tmp_path):
    path, _ = _future_file(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = SCHEMA_VERSION + 5
    path.write_text(json.dumps(raw), encoding="utf-8")

    document = PipelineDocument.load(path)

    assert set(document.nodes) == {"src", "future"}
    assert document.to_dict()["schema_version"] == SCHEMA_VERSION


def test_a_file_missing_a_schema_version_is_treated_as_the_first_one():
    # migrate() only stamps a version when it had to move the file, so an
    # unversioned file comes back as-is; what matters is that it is accepted.
    migrated = migrate({"nodes": [], "links": []})
    assert migrated.get("schema_version", SCHEMA_VERSION) == SCHEMA_VERSION

    document = PipelineDocument.from_dict({"nodes": [
        {"id": "src", "spec_id": "source.synthetic_streamer"}]})
    assert set(document.nodes) == {"src"}
    assert document.to_dict()["schema_version"] == SCHEMA_VERSION


# ======================================================================
# odds and ends the rest leans on
# ======================================================================
def test_a_document_can_be_given_a_registry_of_its_own():
    tiny = {"only.block": NodeSpec(id="only.block", category=SOURCE, title="Only",
                                   outputs=[PortSpec("out", PortType.SAMPLES)])}
    document = PipelineDocument(registry=tiny)
    node_id = document.add_node("only.block")

    assert document.mode() == "online"
    with pytest.raises(KeyError):
        document.add_node("transform.filter")
    assert document.nodes[node_id].unresolved is False


def test_node_and_link_and_probe_round_trip_through_plain_dicts():
    node = Node(id="a", spec_id="transform.filter", params={"order": 4},
                position=(3, 4))
    assert Node.from_dict(node.to_dict(), default_registry()).position == (3, 4)
    link = Link("a", "output", "b", "input")
    assert Link.from_dict(link.to_dict()).key() == link.key()
    probe = Probe("a", "output", 12.0)
    assert Probe.from_dict(probe.to_dict()).key() == probe.key()
    assert Probe.from_dict({"node": "a", "port": "out"}).hz == 30.0
