"""Coverage for the Feature Viewer block: the maths, and the run that feeds it.

The window is DearPyGui and is checked by looking at it; what is held here is
everything it draws from. The columns have to be named in the order the matrix
holds them, the projections have to return what they say, the colours have to
split the points the way the labels do, and a stored-data run has to hand the
block the same matrix a model would be fitted on.
"""
import numpy as np
import pytest

from libemg._gui._pipeline import feature_viewer as viewer
from libemg.feature_extractor import FeatureExtractor
from libemg.pipeline import PipelineDocument, compile_pipeline

CLASSES, SETS, CHANNELS, SAMPLES = 4, 3, 4, 400


def _write_classification_recording(root, seed=0):
    """trial_<n>/C_<c>_R_0_emg.csv, each class at its own amplitude and frequency."""
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


# ======================================================================
# naming the columns
# ======================================================================
def test_columns_are_named_in_the_order_array_true_stacks_them():
    windows = np.random.default_rng(0).normal(size=(30, 4, 40))
    feature_list = ["MAV", "AR", "WL"]
    extractor = FeatureExtractor()
    by_name = extractor.extract_features(feature_list, windows)
    stacked = extractor.extract_features(feature_list, windows, array=True)

    names = viewer.column_names(feature_list, by_name, channels=4)

    assert len(names) == stacked.shape[1]
    assert names[:4] == ["MAV ch1", "MAV ch2", "MAV ch3", "MAV ch4"]
    assert names[4] == "AR [1]" and names[-1] == "WL ch4"
    np.testing.assert_array_equal(np.hstack([by_name[f] for f in feature_list]), stacked)


# ======================================================================
# projections
# ======================================================================
@pytest.mark.parametrize("method", [viewer.PCA, viewer.ICA, viewer.ISOMAP])
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_every_projection_returns_a_coordinate_per_window_per_dimension(method, dims):
    features = np.random.default_rng(1).normal(size=(120, 6))

    coords, axes, _ = viewer.project(features, method, dims)

    assert coords.shape == (120, dims)
    assert len(axes) == dims and np.isfinite(coords).all()


def test_pca_says_how_much_variance_its_components_explain():
    rng = np.random.default_rng(2)
    strong = rng.normal(size=(200, 1)) * 10
    features = np.hstack([strong, strong + rng.normal(size=(200, 1)), rng.normal(size=(200, 3))])

    _, axes, note = viewer.project(features, viewer.PCA, 2, standardize=False)

    assert axes[0].startswith("PC1 (") and "%" in note


def test_columns_are_taken_as_they_are_and_a_projection_needs_enough_of_them():
    features = np.arange(20, dtype=float).reshape(5, 4)

    coords, axes, _ = viewer.project(features, viewer.COLUMNS, 2, columns=(3, 1, 0))

    np.testing.assert_array_equal(coords, features[:, [3, 1]])
    assert axes is None
    with pytest.raises(ValueError, match="at least 3"):
        viewer.project(features[:, :2], viewer.PCA, 3)


def test_a_sample_is_even_repeatable_and_never_larger_than_asked():
    assert list(viewer.subsample(5, 10)) == [0, 1, 2, 3, 4]
    picked = viewer.subsample(10000, 300)
    assert len(picked) == 300 and len(set(picked)) == 300
    np.testing.assert_array_equal(picked, viewer.subsample(10000, 300))


# ======================================================================
# colours
# ======================================================================
def test_class_labels_get_a_colour_each_and_a_quantity_gets_a_heat_scale():
    classes = np.array([0, 1, 2, 1, 0, 2])
    groups, scale = viewer.colour_groups(classes, viewer.AUTO)
    assert [name for name, _, _ in groups] == ["Class 0", "Class 1", "Class 2"]
    assert scale is None
    assert groups[1][1].tolist() == [False, True, False, True, False, False]

    values = np.linspace(-1, 1, 50)
    groups, scale = viewer.colour_groups(values, viewer.AUTO)
    assert scale == (-1.0, 1.0)
    assert sum(mask.sum() for _, mask, _ in groups) == 50, "every point in exactly one step"

    # A second degree of freedom is coloured by its own column.
    two = np.column_stack([values, values * 3])
    _, scale = viewer.colour_groups(two, viewer.BY_VALUE, dof=1)
    assert scale == (-3.0, 3.0)

    groups, scale = viewer.colour_groups(classes, viewer.NO_COLOUR)
    assert len(groups) == 1 and groups[0][1] is None


def test_turning_a_3d_view_keeps_points_where_the_maths_puts_them():
    up = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(viewer.rotate(up, 0, 0), [[0, 1]], atol=1e-12)
    np.testing.assert_allclose(viewer.rotate(up, 0, 90), [[0, 0]], atol=1e-12)
    right = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(viewer.rotate(right, 90, 0), [[0, 0]], atol=1e-12)
    scaled = viewer.normalise_3d(np.array([[0, 0, 0], [10, 2, -4], [-10, -2, 4]], float))
    assert np.abs(scaled).max(axis=0).tolist() == [1, 1, 1]


# ======================================================================
# the run that feeds the block
# ======================================================================
def _viewer_document(folder, with_labels=True):
    document = PipelineDocument()
    source = document.add_node("source.offline", params={
        "folder": str(folder),
        "regex_filters": (f"/trial_|/|{','.join(str(s) for s in range(1, SETS + 1))}|sets\n"
                          f"C_|_R|{','.join(str(i) for i in range(CLASSES))}|classes"),
        "sampling_rate": 200, "delimiter": ","})
    window = document.add_node("window.enframe", params={"window_size": 20,
                                                         "window_increment": 10})
    features = document.add_node("features.extract", params={"feature_group": "HTD"})
    view = document.add_node("sink.feature_viewer", params={"method": "PCA",
                                                            "dimensions": "3D"})
    document.connect(source, "emg", window, "input")
    document.connect(window, "output", features, "input")
    document.connect(features, "output", view, "features")
    if with_labels:
        document.connect(window, "metadata", view, "labels")
    return document


def test_a_stored_data_run_hands_the_viewer_named_features_and_labels(tmp_path):
    folder = _write_classification_recording(tmp_path / "data")
    pipeline = compile_pipeline(_viewer_document(folder))

    pipeline.run()

    view = pipeline.feature_view
    assert view is not None, "a run with a Feature Viewer block keeps what it draws"
    features, names, labels = view["features"], view["names"], view["labels"]
    assert features.shape[1] == len(names) == 4 * 4, "HTD is four features on four channels"
    assert names[0] == "MAV ch1"
    assert len(labels) == len(features)
    assert set(np.unique(labels)) == set(range(CLASSES))
    assert view["params"]["dimensions"] == "3D"
    # Every other metadata field comes along, to colour by instead.
    assert set(view["fields"]) == {"sets", "classes"} and view["label_field"] == "classes"
    assert all(len(v) == len(features) for v in view["fields"].values())
    # What the block draws is the matrix a model would be fitted on.
    window = next(n for n, node in pipeline.document.nodes.items()
                  if node.spec_id == "window.enframe")
    windows = pipeline.value(window, "output").windows
    np.testing.assert_array_equal(
        features, FeatureExtractor().extract_feature_group("HTD", windows, array=True))


def test_without_a_labels_link_the_viewer_draws_one_colour(tmp_path):
    folder = _write_classification_recording(tmp_path / "data")
    pipeline = compile_pipeline(_viewer_document(folder, with_labels=False))

    pipeline.run()

    assert pipeline.feature_view["labels"] is None


def test_the_block_is_offered_for_stored_data_only():
    from libemg._gui._pipeline.registry import build_registry
    spec = build_registry()["sink.feature_viewer"]
    assert spec.offline_only
    assert spec.port("labels").optional and not spec.port("features").optional


def test_columns_are_ranked_by_what_they_say_about_the_labels():
    rng = np.random.default_rng(4)
    classes = np.repeat([0, 1, 2], 100)
    telling = classes * 5.0 + rng.normal(size=300)
    features = np.column_stack([rng.normal(size=300), telling, np.ones(300)])

    ranked = viewer.rank_columns(features, classes)
    assert [index for index, _ in ranked][0] == 1
    assert ranked[-1][1] == 0.0, "a constant column says nothing, and scores nothing"

    target = np.linspace(-1, 1, 300)
    features = np.column_stack([rng.normal(size=300), -3 * target + 0.01 * rng.normal(size=300)])
    ranked = viewer.rank_columns(features, np.column_stack([target, target]), dof=1)
    assert ranked[0][0] == 1 and ranked[0][1] > 0.99, "anti-correlation counts as much"

    assert viewer.rank_columns(features, None) == []
    assert viewer.rank_columns(features, np.zeros(300)) == []


# ======================================================================
# a model's own spaces
# ======================================================================
def _fitted(name, kind="classifier"):
    from libemg.emg_predictor import EMGClassifier, EMGRegressor
    rng = np.random.default_rng(5)
    classes = np.repeat([0, 1, 2], 60)
    features = rng.normal(size=(180, 6)) + classes[:, None] * np.array([2, 0, 1, 0, 0, 0])
    if kind == "classifier":
        predictor = EMGClassifier(name)
        predictor.fit({"training_features": features, "training_labels": classes})
    else:
        predictor = EMGRegressor(name)
        predictor.fit({"training_features": features,
                       "training_labels": np.column_stack([classes, -classes]).astype(float)})
    return predictor, features


@pytest.mark.parametrize("name, titles", [
    ("LDA", ["Discriminant axes (LDA)", "Class probabilities"]),
    ("MLP", ["Last hidden layer (126 units)", "Class probabilities"]),
    ("SVM", ["Decision scores", "Class probabilities"]),
    ("RF", ["Class probabilities"]),
])
def test_each_classifier_offers_the_spaces_it_has(name, titles):
    predictor, features = _fitted(name)

    spaces = viewer.model_spaces(predictor, features)

    assert [title for title, _, _, _ in spaces] == titles
    for title, matrix, columns, note in spaces:
        assert matrix.shape[0] == len(features)
        assert matrix.shape[1] == len(columns) and note
    probabilities = spaces[-1]
    assert probabilities[2] == ["P(class 0)", "P(class 1)", "P(class 2)"]
    np.testing.assert_allclose(probabilities[1].sum(axis=1), 1.0)


def test_lda_axes_are_one_fewer_than_the_classes_and_mlp_layers_are_its_own():
    predictor, features = _fitted("LDA")
    assert viewer.model_spaces(predictor, features)[0][1].shape[1] == 2

    mlp, features = _fitted("MLP")
    hidden = viewer.model_spaces(mlp, features)[0][1]
    model = mlp.model
    # The output layer applied to what was drawn gives the model's own output.
    logits = hidden @ model.coefs_[-1] + model.intercepts_[-1]
    np.testing.assert_array_equal(logits.argmax(axis=1), model.predict(features))


def test_a_regressor_offers_its_outputs():
    predictor, features = _fitted("LR", kind="regressor")

    spaces = viewer.model_spaces(predictor, features)

    assert [title for title, _, _, _ in spaces] == ["Model outputs"]
    assert spaces[0][1].shape == (len(features), 2)


def test_a_model_of_ones_own_names_its_latent_space():
    class Mine:
        def predict(self, x):
            return np.zeros(len(x), int)

        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5)

        def latent(self, x):
            return np.asarray(x)[:, :3] * 2

    features = np.random.default_rng(6).normal(size=(40, 5))
    spaces = viewer.model_spaces(Mine(), features)

    assert spaces[0][0] == "Latent space"
    np.testing.assert_array_equal(spaces[0][1], features[:, :3] * 2)
    assert spaces[0][2] == ["Latent 1", "Latent 2", "Latent 3"]


def test_a_run_hands_the_viewer_the_trained_models_spaces(tmp_path):
    folder = _write_classification_recording(tmp_path / "data")
    document = _viewer_document(folder)
    view = next(n for n in document.nodes.values() if n.spec_id == "sink.feature_viewer").id
    source = next(n for n in document.nodes.values() if n.spec_id == "source.offline").id
    features = next(n for n in document.nodes.values() if n.spec_id == "features.extract").id
    window = next(n for n in document.nodes.values() if n.spec_id == "window.enframe").id
    model = document.add_node("model.classifier", params={
        "model": "LDA", "model_path": str(tmp_path / "lda.pkl")})
    document.connect(features, "output", model, "input")
    document.connect(features, "output", model, "train_input")
    document.connect(window, "metadata", model, "train_labels")
    document.connect(model, "output", view, "model")

    untrained = compile_pipeline(document)
    untrained.run()
    assert untrained.feature_view["spaces"] == []
    assert "not been trained" in untrained.feature_view["model_note"]

    compile_pipeline(document).train()
    trained = compile_pipeline(document)
    trained.run()
    titles = [title for title, _, _, _ in trained.feature_view["spaces"]]
    assert titles == ["Discriminant axes (LDA)", "Class probabilities"]
    assert trained.feature_view["spaces"][0][1].shape == (len(trained.feature_view["features"]),
                                                          CLASSES - 1)


def test_a_column_whose_classes_never_overlap_ranks_first_with_an_infinite_score():
    classes = np.repeat([0, 1], 20)
    features = np.column_stack([np.random.default_rng(7).normal(size=40), classes.astype(float)])

    ranked = viewer.rank_columns(features, classes)

    assert ranked[0][0] == 1 and np.isinf(ranked[0][1])
