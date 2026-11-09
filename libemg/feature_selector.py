"""Feature-space evaluation and feature-group selection.

See the feature selection documentation for definitions, units, and recording
metadata required by individual metrics.
"""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from libemg._feature_selection_metrics import METRICS, MetricEvaluation, resolve_metric


class FeatureSelector:
    """Evaluate usability metrics and rank named groups of feature columns.

    Parameters
    ----------
    random_state : int or None, default=0
        Seed for stratified classification folds. Input arrays are never shuffled
        in place. Recording-based metrics require real recording groups.
    regularization : float, default=1e-9
        Relative covariance eigenvalue floor for singular feature spaces.
    classifier : sklearn estimator or None
        A fresh clone is fitted per fold. The default is shrinkage LDA.
    prism_resolution : int, default=31
        Maximum number of partitions per axis (0..31 by default).
    cdm_min_samples : int or None
        Minimum adaptive cell size; None uses 2 * (dimensions + 1).
    cdm_max_depth : int, default=20
        Maximum adaptive partition depth.
    allow_approximations : bool, default=False
        Enable the explicitly documented C/rC concentration and CDM partition
        variants. Their original reference algorithms could not be verified.
    """

    def __init__(self, random_state=0, regularization=1e-9, classifier=None,
                 prism_resolution=31, cdm_min_samples=None, cdm_max_depth=20,
                 allow_approximations=False):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be positive and finite.")
        for name, value in (("prism_resolution", prism_resolution), ("cdm_max_depth", cdm_max_depth)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1.")
        if cdm_min_samples is not None and (
                not isinstance(cdm_min_samples, Integral) or isinstance(cdm_min_samples, bool)
                or cdm_min_samples < 2):
            raise ValueError("cdm_min_samples must be an integer >= 2.")
        self.random_state = random_state
        self.regularization = regularization
        self.classifier = classifier
        self.prism_resolution = prism_resolution
        self.cdm_min_samples = cdm_min_samples
        self.cdm_max_depth = cdm_max_depth
        self.allow_approximations = allow_approximations

    def get_metrics(self, *, abbreviations=False):
        """Return all supported names, retaining long names by default.

        Set ``abbreviations=True`` for the 32 unique names from the original
        comment plus RI and AE. Both spellings work in every public method.
        """
        return [code if abbreviations else spec.name for code, spec in METRICS.items()]

    def get_metric_info(self, metric):
        """Return the canonical name, objective, default strategy and requirements."""
        code = resolve_metric(metric)
        spec = METRICS[code]
        return dict(abbreviation=code, name=spec.name, objective=spec.objective,
                    strategy=spec.strategy, requires=spec.requires,
                    reference_status=spec.reference_status,
                    approximation=code in ("C", "rC", "CDM"))

    @staticmethod
    def _vector(value, length, name):
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1 or len(array) != length:
            raise ValueError(f"{name} must have one value per sample ({length}).")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite values.")
        if array.dtype.kind == "O" and any(v is None or v != v for v in array):
            raise ValueError(f"{name} must not contain missing values.")
        return array

    @staticmethod
    def _features(data):
        groups = data if isinstance(data, Mapping) else {"features": data}
        if not groups:
            raise ValueError("data must contain at least one feature group.")
        arrays = {}
        length = None
        for name, value in groups.items():
            if not isinstance(name, str):
                raise ValueError("Feature group names must be strings.")
            raw = np.asarray(value)
            if raw.dtype.kind not in "biuf":
                raise ValueError(f"Feature group {name!r} must contain real numeric values.")
            array = np.asarray(raw, dtype=float)
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2 or 0 in array.shape:
                raise ValueError(f"Feature group {name!r} must be a nonempty samples x features array.")
            if not np.isfinite(array).all():
                raise ValueError(f"Feature group {name!r} contains NaN or infinite values.")
            if length is not None and len(array) != length:
                raise ValueError("All feature groups must have the same number of samples.")
            length = len(array)
            arrays[name] = array
        return arrays, length

    def _context(self, length, class_var, crossvalidation_var, nm_class, metadata):
        if class_var is None:
            raise ValueError("class_var is required.")
        labels = self._vector(class_var, length, "class_var")
        if len(np.unique(labels)) < 2:
            raise ValueError("At least two classes are required.")
        cv = {} if crossvalidation_var is None else crossvalidation_var
        if not isinstance(cv, Mapping) or set(cv) - {"var", "crossval_amount"}:
            raise ValueError("crossvalidation_var accepts 'var' or 'crossval_amount'.")
        if "var" in cv and "crossval_amount" in cv:
            raise ValueError("Specify either 'var' or 'crossval_amount', not both.")
        allowed = {"repetition_var", "trial_var", "raw_emg", "window_size_ms",
                   "window_increment_ms", "recording_var", "training_time_ms", "adaptive_memory", "iif_mode"}
        if set(metadata) - allowed:
            raise TypeError(f"Unknown metric metadata: {sorted(set(metadata) - allowed)}")
        context = dict(labels=labels, nm=nm_class, **metadata)
        context["crossval"] = (self._vector(cv["var"], length, "crossvalidation_var['var']")
                               if "var" in cv else None)
        context["crossval_amount"] = cv.get("crossval_amount", 5)
        for key in ("repetition_var", "trial_var", "recording_var"):
            if context.get(key) is not None:
                context[key] = self._vector(context[key], length, key)
        # Preserve the original leave-one-repetition-out API while allowing
        # repetitions and trials to be supplied independently.
        for key in ("repetition_var", "trial_var"):
            if context.get(key) is None:
                context[key] = context["crossval"]
        context["selector"] = self
        return context

    def _folds(self, context):
        labels = context["labels"]
        groups = context["crossval"]
        if groups is not None:
            unique = np.unique(groups)
            if len(unique) < 2:
                raise ValueError("Cross-validation requires at least two groups.")
            folds = [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in unique]
        else:
            amount = context["crossval_amount"]
            if not isinstance(amount, Integral) or isinstance(amount, bool) or amount < 2:
                raise ValueError("crossval_amount must be an integer >= 2.")
            if np.unique(labels, return_counts=True)[1].min() < amount:
                raise ValueError("Each class must have at least crossval_amount samples.")
            cv = StratifiedKFold(amount, shuffle=True, random_state=self.random_state)
            folds = list(cv.split(np.zeros(len(labels)), labels))
        for train, _ in folds:
            if len(train) <= len(np.unique(labels)) or not np.array_equal(
                    np.unique(labels[train]), np.unique(labels)):
                raise ValueError("Each training fold must contain all classes and more samples than classes.")
        return folds

    def _classifier(self):
        return (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                if self.classifier is None else clone(self.classifier))

    def evaluate(self, data, metric="accuracy", class_var=None,
                 crossvalidation_var=None, nm_class=2, **metadata):
        """Evaluate one metric on a matrix or concatenated feature dictionary.

        Metadata keywords are ``repetition_var``, ``trial_var``, ``raw_emg``
        (one raw window per feature row), ``window_size_ms``,
        ``window_increment_ms`` (hop, not overlap), ``recording_var``,
        ``training_time_ms``, ``adaptive_memory`` (default 4 trials for UD), and
        ``iif_mode`` (``'classwise'`` or the printed equation, ``'paper'``).
        CD and UD use sorted repetition/trial IDs as chronological order.
        MAV and TT describe the recording, so their scores do not change with
        the selected feature subset. See the documentation for singular limits.
        """
        return self.evaluate_metrics([metric], data, class_var, crossvalidation_var,
                                     nm_class, **metadata)[metric]

    def evaluate_metrics(self, metrics, data, class_var=None,
                         crossvalidation_var=None, nm_class=2, **metadata):
        """Evaluate metrics together, sharing statistics and expensive calculations.

        ``metrics=None`` evaluates every metric and requires all associated
        metadata. Dictionary keys retain the caller's metric spellings.
        """
        arrays, length = self._features(data)
        context = self._context(length, class_var, crossvalidation_var, nm_class, metadata)
        requested = self.get_metrics(abbreviations=True) if metrics is None else metrics
        if isinstance(requested, str):
            requested = [requested]
        codes = [(name, resolve_metric(name)) for name in requested]
        evaluation = MetricEvaluation(np.hstack(list(arrays.values())), context)
        return {name: evaluation.score(code) for name, code in codes}

    def run_selection(self, data=None, metric="accuracy", class_var=None,
                      crossvalidation_var=None, verbose=False, early_stop=None,
                      nm_class=2, *, strategy=None, objective=None, **metadata):
        """Rank feature groups independently or by sequential forward selection.

        The original positional arguments and ``(scores, feature_order)`` return
        value are preserved. ``strategy`` can be ``'filter'`` or ``'sequential'``;
        the default is filter for FE, MAV and TT, and sequential otherwise.
        ``objective`` optionally overrides ``'min'``/``'max'``. Default objectives
        describe feature-space quality, not a guarantee of online usability.

        Filter scores are sorted alongside the names. Sequential scores have
        shape (number of input groups, number of selected groups), columns follow
        selection order, and uncomputed entries are NaN.
        """
        if not isinstance(data, Mapping):
            raise ValueError("run_selection data must be a dictionary of feature groups.")
        arrays, length = self._features(data)
        code = resolve_metric(metric)
        spec = METRICS[code]
        context = self._context(length, class_var, crossvalidation_var, nm_class, metadata)
        strategy = spec.strategy if strategy is None else strategy
        strategy = "filter" if strategy == "metric" else strategy
        objective = spec.objective if objective is None else objective
        if strategy not in ("filter", "sequential") or objective not in ("min", "max"):
            raise ValueError("strategy must be 'filter' or 'sequential'; objective must be 'min' or 'max'.")
        names = list(arrays)
        if early_stop is not None and (not isinstance(early_stop, Integral)
                or isinstance(early_stop, bool) or not 1 <= early_stop <= len(names)):
            raise ValueError("early_stop must be between 1 and the number of feature groups.")
        count = len(names) if early_stop is None else early_stop
        if code in ("CA", "ACA", "AE"):
            context["folds"] = self._folds(context)

        def score(matrix):
            value = MetricEvaluation(matrix, context).score(code)
            if np.isnan(value):
                raise ValueError(f"{metric} is undefined for this candidate feature space.")
            return value

        if strategy == "filter":
            values = np.array([score(arrays[name]) for name in names])
            order = np.argsort(values if objective == "min" else -values, kind="stable")[:count]
            results, feature_order = values[order], [names[i] for i in order]
        else:
            results = np.full((len(names), len(names)), np.nan)
            selected = []
            for step in range(count):
                for index, name in enumerate(names):
                    if index not in selected:
                        matrix = np.hstack([arrays[names[i]] for i in selected + [index]])
                        results[step, index] = score(matrix)
                choose = np.nanargmin if objective == "min" else np.nanargmax
                selected.append(int(choose(results[step])))
            results = results[:, selected]
            feature_order = [names[i] for i in selected]
        if verbose:
            self.print(metric, results, feature_order)
        return results, feature_order

    def _get_metric_type(self, metric="accuracy"):
        """Compatibility helper for the original selector."""
        spec = METRICS[resolve_metric(metric)]
        return ("metric" if spec.strategy == "filter" else "sequential",
                np.nanargmax if spec.objective == "max" else np.nanargmin)

    def print(self, metric, results, fs):
        """Print filter or sequential results, including early-stopped results."""
        resolve_metric(metric)
        values = np.asarray(results)
        print("iteration | " + " | ".join(fs))
        for index, row in enumerate(np.atleast_2d(values)):
            if index >= len(fs) and values.ndim == 2:
                break
            print(f"{index + 1:9d} | " + " | ".join(
                "" if np.isnan(value) else f"{value:.6g}" for value in row))

    def _legacy_metric(self, metric, dictionary):
        return self.evaluate(dictionary["features"], metric, dictionary["labels"],
                             {"var": dictionary["crossval"]} if "crossval" in dictionary else None,
                             dictionary.get("nm", 2))

    def _get_accuracy(self, dictionary):
        return self._legacy_metric("CA", dictionary)

    def _get_activeerror(self, dictionary):
        return self._legacy_metric("AE", dictionary)

    def _get_meansemiprincipalaxis(self, dictionary):
        return self._legacy_metric("MSA", dictionary)

    def _get_featureefficiency(self, dictionary):
        return self._legacy_metric("FE", dictionary)

    def _get_repeatability(self, dictionary):
        return self._legacy_metric("RI", dictionary)

    def _get_separability(self, dictionary):
        return self._legacy_metric("SI", dictionary)
