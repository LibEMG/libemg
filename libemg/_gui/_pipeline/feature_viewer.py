"""See what a pipeline's features look like before a model is fitted to them.

A Feature Viewer block takes the feature matrix a pipeline extracts from stored
data -- one row per window, one column per feature and channel -- and the
labels read with it, and draws them once the run has walked the recording. The
question it answers is the one asked before any model: do the classes come
apart in this feature space, and which features pull them apart?

Three ways to look, chosen in the window rather than on the block, because the
point is to flip between them without reading the recording again:

    One column.       A single feature, as overlaid histograms per class (or,
                      for a regression target, the feature against the label).
    Two or three.     Any columns against each other as a scatter.
    A projection.     PCA, ICA or Isomap down to 1, 2 or 3 dimensions, over
                      every column at once.

Points are coloured by class, or by the label's value on a heat scale for a
regression target. DearPyGui has no 3D plot, so a 3D view is a 2D plot of the
points turned by an azimuth and an elevation, which can spin.

With a fitted model connected, the same views apply to the spaces the model
works in, not only to the features it reads (see :func:`model_spaces`): an
LDA's discriminant axes, an MLP's last hidden layer, an SVM's decision scores,
any classifier's class probabilities, and the latent space of a model of one's
own that defines ``latent(features)``. Whether the classes come apart there is
what the model actually achieved, as opposed to what the features allowed.

The maths is kept apart from the window, in plain functions at the top of the
module, so it can be checked without a display.
"""

import math
import threading

import numpy as np

import dearpygui.dearpygui as dpg

from libemg._gui import _style as style

COLUMNS, PCA, ICA, ISOMAP = "Columns", "PCA", "ICA", "Isomap"
METHODS = (COLUMNS, PCA, ICA, ISOMAP)
DIMENSIONS = ("1D", "2D", "3D")
AUTO, BY_CLASS, BY_VALUE, NO_COLOUR = "Auto", "Class", "Label value", "None"
COLOURINGS = (AUTO, BY_CLASS, BY_VALUE, NO_COLOUR)

#: More distinct values than this and a label is a quantity, not a class.
MAX_CLASSES = 24
#: Steps on the heat scale a regression target is drawn in.
HEAT_BINS = 10
#: Isomap and ICA are fitted to a sample of the windows, not all of them:
#: Isomap's cost grows with the square of the count.
DEFAULT_POINTS = 4000

#: Readable on the dark background and distinct from one another, in order.
CLASS_COLOURS = [
    (96, 165, 250), (251, 146, 60), (74, 222, 128), (248, 113, 113),
    (192, 132, 252), (250, 204, 21), (45, 212, 191), (244, 114, 182),
    (163, 230, 53), (148, 163, 184), (56, 189, 248), (251, 191, 36),
]
#: Matplotlib's viridis at ten points, the scale DearPyGui's colormap bar shows.
_VIRIDIS = [(68, 1, 84), (72, 40, 120), (62, 73, 137), (49, 104, 142),
            (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
            (181, 222, 43), (253, 231, 37)]

WINDOW_TAG = "__feature_viewer"
PLOT_AREA = "__fv_plot_area"
STATUS = "__fv_status"


# ======================================================================
# the maths
# ======================================================================
def column_names(feature_list, by_name, channels):
    """A name for every column of the feature matrix, in its order.

    The matrix is each feature's block in the order the features were asked
    for -- what FeatureExtractor's ``array=True`` produces -- so naming follows
    the same order. A feature with one column per channel is named by channel;
    one with another width (AR coefficients, wavelet bands) by position.
    """
    names = []
    for feature in feature_list:
        values = np.asarray(by_name[feature])
        width = values.reshape(values.shape[0], -1).shape[1]
        if width == channels:
            names += [f"{feature} ch{c + 1}" for c in range(width)]
        elif width == 1:
            names.append(feature)
        else:
            names += [f"{feature} [{j + 1}]" for j in range(width)]
    return names


def subsample(count, max_points, seed=0):
    """Which rows to draw: all of them, or an even random sample of ``max_points``."""
    if max_points is None or count <= max_points:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, int(max_points), replace=False))


def is_classes(labels):
    """Whether labels are class indices (few whole numbers) rather than a quantity."""
    if labels is None:
        return False
    values = np.asarray(labels, dtype=float)
    if values.ndim > 1 and values.shape[1] > 1:
        return False
    values = values.ravel()
    values = values[np.isfinite(values)]
    return (len(values) > 0 and np.all(values == np.round(values))
            and len(np.unique(values)) <= MAX_CLASSES)


def project(features, method, dims, columns=(0, 1, 2), standardize=True, seed=0):
    """Reduce ``features`` to ``dims`` coordinates.

    Parameters
    ----------
    features: array (windows, columns)
    method: str
        One of :data:`METHODS`. Columns takes ``columns[:dims]`` as they are.
    dims: int
        1, 2 or 3.
    standardize: bool
        Scale every column to zero mean and unit variance first. Feature
        columns are in different units -- a mean absolute value next to a zero
        crossing count -- and without this a projection is mostly a measure of
        which column has the biggest numbers.

    Returns
    ----------
    coords: array (windows, dims)
    axes: list of str
        What each coordinate is, or None for Columns (the caller has names).
    note: str
        What the projection found, such as the variance PCA explains.
    """
    data = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if method == COLUMNS:
        return data[:, list(columns[:dims])], None, ""
    if standardize:
        spread = data.std(axis=0)
        data = (data - data.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    if data.shape[1] < dims:
        raise ValueError(f"{method} to {dims}D needs at least {dims} feature columns; "
                         f"there are {data.shape[1]}.")
    if method == PCA:
        from sklearn.decomposition import PCA as _PCA
        model = _PCA(n_components=dims, random_state=seed).fit(data)
        ratios = model.explained_variance_ratio_
        return (model.transform(data),
                [f"PC{i + 1} ({r:.0%})" for i, r in enumerate(ratios)],
                f"These {dims} components explain {ratios.sum():.0%} of the variance.")
    if method == ICA:
        from sklearn.decomposition import FastICA
        model = FastICA(n_components=dims, whiten="unit-variance", random_state=seed,
                        max_iter=1000)
        return model.fit_transform(data), [f"IC{i + 1}" for i in range(dims)], ""
    if method == ISOMAP:
        from sklearn.manifold import Isomap
        import warnings
        neighbours = max(2, min(10, len(data) - 1))
        model = Isomap(n_neighbors=neighbours, n_components=dims)
        # A neighbour graph in more than one piece is joined up inside
        # scikit-learn by editing a sparse matrix, which scipy warns about once
        # per edit: hundreds of lines in the console for nothing the user can act on.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = model.fit_transform(data)
        return (coords, [f"Isomap {i + 1}" for i in range(dims)],
                f"Isomap over {neighbours} neighbours.")
    raise ValueError(f"Unknown method '{method}'. Choose one of {METHODS}.")


def colour_groups(labels, colouring, dof=0, class_names=None):
    """Split the points into coloured groups.

    Returns
    ----------
    groups: list of (name, mask, colour)
    scale: (low, high) or None
        The label range a heat scale spans, when colouring by value.
    """
    count = 0 if labels is None else len(labels)
    if labels is None or colouring == NO_COLOUR:
        return [("All windows", None, style.ACCENT)], None
    values = np.asarray(labels, dtype=float)
    if values.ndim > 1:
        values = values[:, min(dof, values.shape[1] - 1)]
    if colouring == AUTO:
        colouring = BY_CLASS if is_classes(values) else BY_VALUE
    if colouring == BY_CLASS:
        groups = []
        for i, value in enumerate(np.unique(values[np.isfinite(values)])[:MAX_CLASSES * 2]):
            name = (class_names or {}).get(int(value), f"Class {value:g}")
            groups.append((name, values == value, CLASS_COLOURS[i % len(CLASS_COLOURS)]))
        return groups, None
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [("All windows", np.ones(count, bool), style.ACCENT)], None
    low, high = float(finite.min()), float(finite.max())
    edges = np.linspace(low, high, HEAT_BINS + 1)
    which = np.clip(np.digitize(values, edges[1:-1]), 0, HEAT_BINS - 1)
    groups = []
    for i in range(HEAT_BINS):
        mask = (which == i) & np.isfinite(values)
        if mask.any():
            low_edge, high_edge = (0.0 if abs(e) < 1e-9 * max(1.0, high - low) else e
                                   for e in (edges[i], edges[i + 1]))
            groups.append((f"{low_edge:.3g} to {high_edge:.3g}", mask, _VIRIDIS[i]))
    return groups, (low, high)


def rank_columns(features, labels, dof=0):
    """Every column, best first, by how much it says about the labels on its own.

    For class labels the ANOVA F score: how far apart the class means sit
    against how spread each class is. For a regression target the absolute
    correlation with it.

    Returns
    ----------
    list of (column index, score), or [] when there is nothing to rank against.
    """
    if labels is None:
        return []
    data = np.nan_to_num(np.asarray(features, dtype=float))
    values = np.asarray(labels, dtype=float)
    if values.ndim > 1:
        values = values[:, min(dof, values.shape[1] - 1)]
    keep = np.isfinite(values)
    data, values = data[keep], values[keep]
    if len(np.unique(values)) < 2:
        return []
    if is_classes(values):
        import warnings
        from sklearn.feature_selection import f_classif
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # a constant column scores nan, not an error
            scores, _ = f_classif(data, values.astype(int))
    else:
        centred = data - data.mean(axis=0)
        target = values - values.mean()
        spread = np.sqrt((centred ** 2).sum(axis=0) * (target ** 2).sum())
        scores = np.abs(centred.T @ target) / np.where(spread > 0, spread, np.inf)
    # A column whose classes do not overlap at all has no spread within a class,
    # so its F score is infinite. That is the best score there is, and stays one.
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=np.inf)
    order = np.argsort(-scores, kind="stable")
    return [(int(i), float(scores[i])) for i in order]


#: The method a model of one's own defines to expose its latent space: it
#: takes the model's input and returns one row per window.
LATENT_METHOD = "latent"


def model_spaces(predictor, data):
    """The spaces a fitted model works in, as extra views on the same windows.

    Parameters
    ----------
    predictor: EMGClassifier, EMGRegressor or a model
        Anything a Classifier or Regressor block loads. LibEMG's wrappers keep
        the fitted estimator as ``.model``; a bare model is used as it is.
    data: array (windows, columns)
        What the model reads, as it read it when trained.

    Returns
    ----------
    list of (title, matrix, column names, note)
        Most telling first: a latent space where the model has one, then its
        decision scores, then what it outputs. Empty when it offers none.
        Each matrix has one row per window of ``data``.
    """
    model = getattr(predictor, "model", predictor)
    spaces = []

    def add(title, values, prefix, note="", names=None):
        values = _as_rows(values, len(data))
        if values is None:
            return
        names = names or [f"{prefix} {i + 1}" for i in range(values.shape[1])]
        spaces.append((title, values, list(names), note))

    # A model of one's own says what its latent space is. Checked first, so
    # that defining it wins over anything inferred below.
    for owner in (predictor, model):
        hook = getattr(owner, LATENT_METHOD, None)
        if callable(hook):
            add("Latent space", hook(data), "Latent",
                f"From the model's own {LATENT_METHOD}() method.")
            break
    else:
        hidden = _mlp_hidden(model, data)
        if hidden is not None:
            add(f"Last hidden layer ({hidden.shape[1]} units)", hidden, "Unit",
                "Activations of the MLP's last hidden layer: the representation its "
                "output layer separates linearly.")
        elif hasattr(model, "transform") and hasattr(model, "scalings_"):
            add("Discriminant axes (LDA)", model.transform(data), "LD",
                "LDA's projection onto the axes that best separate the classes it was "
                "trained on: at most one fewer than the classes.")
    if hasattr(model, "decision_function") and not hasattr(model, "scalings_"):
        try:
            add("Decision scores", model.decision_function(data), "Score",
                "The margin the model gives each class before deciding.",
                names=_class_names(model, "Score"))
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        try:
            add("Class probabilities", model.predict_proba(data), "P",
                "What the model outputs: one probability per class, so classes it "
                "separates sit near the corners.",
                names=_class_names(model, "P"))
        except Exception:
            pass
    elif hasattr(model, "predict"):
        add("Model outputs", model.predict(data), "Output",
            "What the regressor outputs, one column per degree of freedom.")
    return spaces


def _as_rows(values, count):
    """A (count, columns) float matrix from whatever a model returned, or None."""
    if values is None:
        return None
    if hasattr(values, "detach"):          # a torch tensor, from a model of one's own
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values = values.reshape(values.shape[0], -1)
    return values if values.shape[0] == count and values.shape[1] else None


def _class_names(model, prefix):
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    return [f"{prefix}(class {c})" for c in classes]


def _mlp_hidden(model, data):
    """An sklearn MLP's last hidden layer for ``data``, or None for any other model."""
    weights = getattr(model, "coefs_", None)
    biases = getattr(model, "intercepts_", None)
    if weights is None or biases is None or len(weights) < 2:
        return None
    activations = {"relu": lambda x: np.maximum(x, 0), "tanh": np.tanh,
                   "logistic": lambda x: 1 / (1 + np.exp(-x)), "identity": lambda x: x}
    act = activations.get(getattr(model, "activation", "relu"), activations["relu"])
    layer = np.asarray(data, dtype=float)
    # Every layer but the output one, which is what the model decides from.
    for w, b in zip(weights[:-1], biases[:-1]):
        layer = act(layer @ w + b)
    return layer


def normalise_3d(coords):
    """Centre each axis and scale them alike, so a turn shows shape rather than units."""
    coords = np.asarray(coords, dtype=float)
    centred = coords - coords.mean(axis=0)
    spread = np.abs(centred).max(axis=0)
    return centred / np.where(spread > 0, spread, 1.0)


def rotate(coords, azimuth, elevation):
    """Where 3D points land on screen, seen from ``azimuth`` round and ``elevation`` up (degrees)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    x, y, z = np.asarray(coords, dtype=float).T
    across = x * math.cos(a) - y * math.sin(a)
    depth = x * math.sin(a) + y * math.cos(a)
    return np.column_stack([across, z * math.cos(e) - depth * math.sin(e)])


# ======================================================================
# the window
# ======================================================================
class FeatureViewer:
    """A window drawing a feature matrix and its labels.

    Parameters
    ----------
    features: array (windows, columns)
    names: list of str
        One per column.
    labels: array or None
        One row per window: a class index, or one value per degree of freedom.
    params: dict or None
        The block's settings, as the starting view: ``method``,
        ``dimensions``, ``colour_by``, ``max_points``, ``standardize``.
    spaces: list or None
        Further spaces over the same windows, as :func:`model_spaces` returns
        them: ``(title, matrix, column names, note)``. The features are always
        the first space; these are offered beside them.
    """

    window_tag = WINDOW_TAG

    def __init__(self, features, names, labels=None, params=None, width=1000, height=700,
                 pos=(60, 60), spaces=None):
        params = dict(params or {})
        features = np.asarray(features, dtype=float)
        names = list(names) if names else [f"Column {i + 1}" for i in range(features.shape[1])]
        self.spaces = [("Features", features, names, "")] + [
            (title, np.asarray(matrix, dtype=float), list(columns), note)
            for title, matrix, columns, note in (spaces or [])]
        self.space = 0
        self.features, self.names = features, names
        self.labels = None if labels is None else np.asarray(labels)
        self.width, self.height, self.pos = width, height, pos
        self.method = params.get("method", PCA) if params.get("method") in METHODS else PCA
        self.dims = int(str(params.get("dimensions", "2D"))[0]) \
            if str(params.get("dimensions", "2D")) in DIMENSIONS else 2
        self.colouring = params.get("colour_by", AUTO) if params.get("colour_by") in COLOURINGS else AUTO
        self.max_points = min(int(params.get("max_points") or DEFAULT_POINTS),
                              len(self.features))
        self.standardize = bool(params.get("standardize", True))
        width_count = self.features.shape[1]
        self.columns = [0, min(1, width_count - 1), min(2, width_count - 1)]
        self.dof = 0
        self.azimuth, self.elevation, self.spin = 35.0, 20.0, False
        self._cache = {}
        self._pending = None        # key being computed
        self._result = None         # (key, coords, axes, note) or (key, error)
        self._thread = None
        self._series = []           # (series tag, coords of its points) in 3D
        self._fit_next = None       # axes to fit to their data once drawn
        self._axis_lines = []

    # -- life cycle --------------------------------------------------------
    def spawn_window(self):
        self.cleanup()
        count, width = self.features.shape
        has_dofs = self.labels is not None and self.labels.ndim > 1 and self.labels.shape[1] > 1
        with dpg.window(label="Feature Viewer", tag=WINDOW_TAG, width=self.width,
                        height=self.height, pos=self.pos, on_close=self.cleanup):
            labels = ("no labels" if self.labels is None else
                      "class labels" if is_classes(self.labels) else "regression labels")
            spaces = (f" Also from the model: {', '.join(t for t, _, _, _ in self.spaces[1:])}."
                      if len(self.spaces) > 1 else "")
            style.note(f"{count} windows, {width} feature columns, {labels}.{spaces} Change the "
                       "view here; the recording is not read again.", wrap=self.width - 40)
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingFixedFit,
                           borders_innerV=True):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=320)
                dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                with dpg.table_row():
                    with dpg.group():
                        self._build_controls(has_dofs)
                    dpg.add_group(tag=PLOT_AREA)
        self._sync_controls()
        self._fill_ranking()
        self._request()
        return self

    def _fill_ranking(self, shown=8):
        """List the columns that best separate the labels, each one a click away."""
        if not dpg.does_item_exist("__fv_ranking"):
            return
        dpg.delete_item("__fv_ranking", children_only=True)
        ranked = rank_columns(self.features, self.labels, self.dof)[:shown]
        if not ranked:
            dpg.add_text("Nothing to rank: the labels hold a single value.",
                         color=style.MUTED, parent="__fv_ranking")
            return
        with dpg.table(header_row=True, parent="__fv_ranking", row_background=True,
                       policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="Column", init_width_or_weight=3.0)
            dpg.add_table_column(label="Score", init_width_or_weight=1.0)
            for index, score in ranked:
                with dpg.table_row():
                    dpg.add_selectable(label=self.names[index], user_data=index,
                                       callback=self._ranked_chosen)
                    shown_score = dpg.add_text(f"{score:.3g}" if np.isfinite(score) else "∞",
                                               color=style.MUTED)
                    if not np.isfinite(score):
                        with dpg.tooltip(shown_score):
                            dpg.add_text("The classes do not overlap at all in this column.")

    def _ranked_chosen(self, sender, app_data, index):
        dpg.set_value(sender, False)        # a click, not a lasting selection
        previous = self.columns[0]
        self.columns[0] = index
        if previous != index:
            self.columns[1] = previous
        self.method = COLUMNS
        if dpg.does_item_exist("__fv_method"):
            dpg.set_value("__fv_method", COLUMNS)
        for axis, slot in (("X", 0), ("Y", 1)):
            if dpg.does_item_exist(f"__fv_col_{axis}"):
                dpg.set_value(f"__fv_col_{axis}", self.names[self.columns[slot]])
        self._sync_controls()
        self._request()

    def cleanup(self):
        if dpg.does_item_exist(WINDOW_TAG):
            dpg.delete_item(WINDOW_TAG)
        self._series, self._axis_lines = [], []

    def poll(self):
        """Pick up a finished projection, and turn a spinning 3D view. Call once a frame."""
        if not dpg.does_item_exist(WINDOW_TAG):
            return
        if self._fit_next is not None:
            axes, self._fit_next = self._fit_next, None
            for axis in axes:
                if dpg.does_item_exist(axis):
                    dpg.fit_axis_data(axis)
        if self._result is not None:
            result, self._result = self._result, None
            key = result[0]
            if len(result) == 2:
                style.set_status(STATUS, f"Could not compute it: {result[1]}", "error")
            else:
                self._cache[key] = result[1:]
                if key == self._key():
                    self._draw()
            self._pending = None
            if self._key() not in self._cache:
                self._request()
        if self.spin and self.dims == 3 and self._series:
            self.azimuth = (self.azimuth + 0.5) % 360
            if dpg.does_item_exist("__fv_azimuth"):
                dpg.set_value("__fv_azimuth", self.azimuth)
            self._turn()

    # -- controls ----------------------------------------------------------
    def _build_controls(self, has_dofs):
        style.section("View", first=True)
        with style.form(label_width=95):
            if len(self.spaces) > 1:
                with style.row("Space", help="Features: what the model reads. The others are "
                               "spaces the connected model works in: whether the classes come "
                               "apart there is what it actually learned."):
                    dpg.add_combo([title for title, _, _, _ in self.spaces],
                                  default_value=self.spaces[0][0], width=-1,
                                  tag="__fv_space", callback=self._space_chosen)
            with style.row("Method", help="Columns: plot feature columns as they are. PCA: the "
                           "directions of greatest variance. ICA: statistically independent "
                           "directions. Isomap: a nonlinear unfolding that keeps neighbours "
                           "close."):
                dpg.add_combo(list(METHODS), default_value=self.method, width=-1,
                              tag="__fv_method", callback=lambda s, a: self._set(method=a))
            with style.row("Dimensions"):
                dpg.add_radio_button(list(DIMENSIONS), default_value=f"{self.dims}D",
                                     horizontal=True, tag="__fv_dims",
                                     callback=lambda s, a: self._set(dims=int(a[0])))
            for axis, index in (("X", 0), ("Y", 1), ("Z", 2)):
                with style.row(f"{axis} feature", tag=f"__fv_row_{axis}"):
                    dpg.add_combo(self.names, default_value=self.names[self.columns[index]],
                                  width=-1, tag=f"__fv_col_{axis}", user_data=index,
                                  callback=self._column_chosen)
            with style.row("Standardize", tag="__fv_row_standardize",
                           help="Scale every column to zero mean and unit variance before "
                                "projecting, so a column is not important just for having "
                                "large values."):
                dpg.add_checkbox(default_value=self.standardize,
                                 callback=lambda s, a: self._set(standardize=bool(a)))
        style.section("Colour")
        with style.form(label_width=95):
            with style.row("Colour by", help="Class: one colour per class. Label value: a heat "
                           "scale for a regression target. Auto picks from the labels."):
                dpg.add_combo(list(COLOURINGS), default_value=self.colouring, width=-1,
                              enabled=self.labels is not None,
                              callback=lambda s, a: self._set(colouring=a, recompute=False))
            if has_dofs:
                with style.row("Label column"):
                    dpg.add_combo([f"DOF {i + 1}" for i in range(self.labels.shape[1])],
                                  default_value="DOF 1", width=-1,
                                  callback=lambda s, a: self._set(dof=int(a.split()[-1]) - 1,
                                                                  recompute=False))
            with style.row("Points", help="How many windows to draw, chosen at random. "
                           "Projections are fitted to the same sample; Isomap slows down "
                           "with the square of this."):
                dpg.add_input_int(default_value=self.max_points, step=0, width=90,
                                  min_value=100, min_clamped=True, on_enter=True,
                                  callback=lambda s, a: self._set(max_points=int(a)))
                style.unit(f"of {self.features.shape[0]}")
        if self.labels is not None:
            style.section("Most telling columns")
            style.note("Each column on its own, ranked by how well it separates the labels: "
                       "ANOVA F for classes, |correlation| for a target. Click one to plot "
                       "it; the one plotted before moves to Y.", wrap=300)
            dpg.add_group(tag="__fv_ranking")
        with dpg.group(tag="__fv_turn"):
            style.section("Turn")
            with style.form(label_width=95):
                with style.row("Azimuth"):
                    dpg.add_slider_float(default_value=self.azimuth, min_value=0, max_value=360,
                                         width=-1, format="%.0f deg", tag="__fv_azimuth",
                                         callback=lambda s, a: self._set_view(azimuth=a))
                with style.row("Elevation"):
                    dpg.add_slider_float(default_value=self.elevation, min_value=-90,
                                         max_value=90, width=-1, format="%.0f deg",
                                         callback=lambda s, a: self._set_view(elevation=a))
                with style.row("Spin"):
                    dpg.add_checkbox(default_value=self.spin,
                                     callback=lambda s, a: self._set_view(spin=bool(a)))
        dpg.add_spacer(height=4)
        style.note("", tag="__fv_space_note", wrap=300, show=False)
        style.note("", tag=STATUS, wrap=300)

    def _sync_controls(self):
        columns = self.method == COLUMNS
        shown = {"__fv_row_X": columns, "__fv_row_Y": columns and self.dims >= 2,
                 "__fv_row_Z": columns and self.dims == 3,
                 "__fv_row_standardize": not columns, "__fv_turn": self.dims == 3}
        for tag, show in shown.items():
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=show)

    def _space_chosen(self, sender, title):
        """Look at another space over the same windows."""
        self.space = next(i for i, (t, _, _, _) in enumerate(self.spaces) if t == title)
        _, self.features, self.names, note = self.spaces[self.space]
        width = self.features.shape[1]
        self.columns = [0, min(1, width - 1), min(2, width - 1)]
        for axis, slot in (("X", 0), ("Y", 1), ("Z", 2)):
            if dpg.does_item_exist(f"__fv_col_{axis}"):
                dpg.configure_item(f"__fv_col_{axis}", items=self.names)
                dpg.set_value(f"__fv_col_{axis}", self.names[self.columns[slot]])
        if dpg.does_item_exist("__fv_space_note"):
            dpg.set_value("__fv_space_note", note)
            dpg.configure_item("__fv_space_note", show=bool(note))
        self._fill_ranking()
        self._request()

    def _column_chosen(self, sender, app_data, index):
        self.columns[index] = self.names.index(app_data)
        self._request()

    def _set(self, recompute=True, **changes):
        for name, value in changes.items():
            setattr(self, name, value)
        self._sync_controls()
        if "dof" in changes:
            self._fill_ranking()        # a different target ranks the columns differently
        if recompute:
            self._request()
        else:
            self._draw()

    def _set_view(self, **changes):
        for name, value in changes.items():
            setattr(self, name, value)
        self._turn()

    # -- computing ---------------------------------------------------------
    def _rows(self):
        return subsample(len(self.features), self.max_points)

    def _key(self):
        columns = tuple(self.columns[:self.dims]) if self.method == COLUMNS else None
        return (self.space, self.method, self.dims, columns, self.standardize, self.max_points)

    def _request(self):
        key = self._key()
        if key in self._cache:
            self._draw()
            return
        if self._pending is not None:
            return      # poll() asks again once the one running finishes
        rows = self._rows()
        if self.method == COLUMNS:
            self._cache[key] = project(self.features[rows], COLUMNS, self.dims, self.columns)
            self._draw()
            return
        # Off the render thread: Isomap over a few thousand windows takes
        # seconds, and the window has to keep drawing meanwhile.
        self._pending = key
        style.set_status(STATUS, f"Computing {self.method}...", "busy")
        data = self.features[rows]

        def work():
            try:
                self._result = (key,) + tuple(project(data, key[1], key[2],
                                                      standardize=key[4]))
            except Exception as error:
                self._result = (key, f"{type(error).__name__}: {error}")

        self._thread = threading.Thread(target=work, daemon=True, name="libemg-feature-viewer")
        self._thread.start()

    # -- drawing -----------------------------------------------------------
    def _draw(self):
        cached = self._cache.get(self._key())
        if cached is None or not dpg.does_item_exist(PLOT_AREA):
            return
        coords, axes, note = cached
        if axes is None:
            axes = [self.names[c] for c in self.columns[:self.dims]]
        rows = self._rows()
        labels = None if self.labels is None else self.labels[rows]
        groups, scale = colour_groups(labels, self.colouring, self.dof)
        dpg.delete_item(PLOT_AREA, children_only=True)
        self._series, self._axis_lines = [], []
        with dpg.group(horizontal=True, parent=PLOT_AREA):
            plot_width = -95 if scale else -1
            with dpg.plot(height=-1, width=plot_width, tag="__fv_plot",
                          equal_aspects=self.dims == 3) as plot:
                dpg.bind_item_theme(plot, _plot_theme())
                if not scale:
                    # A heat scale has its colour bar beside the plot; a legend
                    # listing its ten steps would only repeat it.
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                if self.dims == 1:
                    self._draw_1d(coords[:, 0], axes[0], labels, groups, scale)
                elif self.dims == 2:
                    self._draw_2d(coords, axes, groups)
                else:
                    self._draw_3d(coords, axes, groups)
            if scale:
                dpg.add_colormap_scale(min_scale=scale[0], max_scale=scale[1], height=-1,
                                       width=80, colormap=dpg.mvPlotColormap_Viridis,
                                       label="Label")
        message = note or f"Showing {len(rows)} of {len(self.features)} windows."
        if note and len(rows) < len(self.features):
            message += f" Fitted to {len(rows)} of {len(self.features)} windows."
        style.set_status(STATUS, message, "info")

    def _draw_1d(self, values, axis, labels, groups, scale):
        if scale is not None:
            # A quantity rather than classes: the feature against the label,
            # which shows whether it tracks the target at all.
            target = labels if labels.ndim == 1 else labels[:, min(self.dof, labels.shape[1] - 1)]
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label=axis)
            with dpg.plot_axis(dpg.mvYAxis, label="Label value"):
                for name, mask, colour in groups:
                    self._scatter(values[mask], np.asarray(target, float)[mask], name, colour)
            dpg.fit_axis_data(x_axis)
            return
        low, high = float(np.min(values)), float(np.max(values))
        dpg.add_plot_axis(dpg.mvXAxis, label=axis)
        with dpg.plot_axis(dpg.mvYAxis, label="Windows") as y_axis:
            for name, mask, colour in groups:
                chosen = values if mask is None else values[mask]
                if len(chosen):
                    series = dpg.add_histogram_series(chosen.tolist(), bins=40, label=name,
                                                      min_range=low, max_range=high)
                    dpg.bind_item_theme(series, _fill_theme(colour))
        dpg.fit_axis_data(y_axis)

    def _draw_2d(self, coords, axes, groups):
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label=axes[0])
        with dpg.plot_axis(dpg.mvYAxis, label=axes[1]) as y_axis:
            for name, mask, colour in groups:
                chosen = coords if mask is None else coords[mask]
                self._scatter(chosen[:, 0], chosen[:, 1], name, colour)
        # Fitted to the points, not left where the last view had it: a new pair
        # of columns can sit in a different range altogether. DearPyGui only
        # fits an axis that has been drawn, so poll() does it a frame later.
        self._fit_next = (x_axis, y_axis)

    def _draw_3d(self, coords, axes, groups):
        points = normalise_3d(coords)
        # The screen axes mean nothing once the points are turned: no ticks.
        bare = dict(no_tick_labels=True, no_gridlines=True, no_tick_marks=True)
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, **bare)
        lines = []
        with dpg.plot_axis(dpg.mvYAxis, **bare) as y_axis:
            # The three axes as lines from the centre, so a turn can be read.
            # Unlabelled, so they stay out of the legend.
            for i in range(3):
                end = np.zeros((2, 3))
                end[1, i] = 1.1
                line = dpg.add_line_series([0, 0], [0, 0])
                dpg.bind_item_theme(line, _line_theme((150, 156, 167)))
                lines.append((line, end))
            for name, mask, colour in groups:
                chosen = points if mask is None else points[mask]
                self._series.append((self._scatter(chosen[:, 0], chosen[:, 1], name, colour),
                                     chosen))
        # Annotations belong to the plot, not to an axis.
        for (line, end), name in zip(lines, axes):
            note = dpg.add_plot_annotation(label=name, default_value=(0, 0),
                                           color=(70, 76, 88, 255), parent="__fv_plot")
            self._axis_lines.append((line, note, end))
        self._turn()
        # A unit cube's corners reach sqrt(3) from the centre when turned, so
        # fixed limits just past that keep every point on screen at any angle.
        dpg.set_axis_limits(x_axis, -1.8, 1.8)
        dpg.set_axis_limits(y_axis, -1.8, 1.8)

    def _turn(self):
        if self.dims != 3:
            return
        for series, points in self._series:
            if dpg.does_item_exist(series):
                flat = rotate(points, self.azimuth, self.elevation)
                dpg.set_value(series, [flat[:, 0].tolist(), flat[:, 1].tolist()])
        for line, note, end in self._axis_lines:
            if dpg.does_item_exist(line):
                flat = rotate(end, self.azimuth, self.elevation)
                dpg.set_value(line, [flat[:, 0].tolist(), flat[:, 1].tolist()])
                dpg.set_value(note, (float(flat[1, 0]), float(flat[1, 1])))

    def _scatter(self, xs, ys, name, colour):
        series = dpg.add_scatter_series(np.asarray(xs, float).tolist(),
                                        np.asarray(ys, float).tolist(), label=name)
        dpg.bind_item_theme(series, _marker_theme(colour))
        return series


# ----------------------------------------------------------------------
# series themes, one per colour, made once per DearPyGui context
# ----------------------------------------------------------------------
def _marker_theme(colour):
    alias = "__fv_marker_{}_{}_{}".format(*colour[:3])
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (*colour[:3], 170),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (*colour[:3], 220),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 2.5,
                                category=dpg.mvThemeCat_Plots)
    return alias


def _plot_theme():
    """A margin round fitted data. Features such as zero crossings are whole
    numbers, so many windows sit exactly on the extremes, and with no margin
    they are drawn half outside the plot."""
    alias = "__fv_plot_theme"
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_style(dpg.mvPlotStyleVar_FitPadding, 0.06, 0.06,
                                category=dpg.mvThemeCat_Plots)
    return alias


def _fill_theme(colour):
    alias = "__fv_fill_{}_{}_{}".format(*colour[:3])
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvHistogramSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (*colour[:3], 110),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 255),
                                category=dpg.mvThemeCat_Plots)
    return alias


def _line_theme(colour):
    alias = "__fv_line_{}_{}_{}".format(*colour[:3])
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (*colour[:3], 200),
                                category=dpg.mvThemeCat_Plots)
    return alias
