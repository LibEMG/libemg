"""Feature-space evaluation and feature-group selection.

See the feature selection documentation for definitions, units, and recording
metadata required by individual metrics.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from libemg._feature_selection_metrics import METRICS, MetricEvaluation, resolve_metric

#: Metrics scored by cross-validating a classifier.
_CLASSIFICATION = ("CA", "ACA", "AE")

#: The evaluation schemes of :meth:`FeatureSelector.run_study`, with how to name them.
SCHEME_TITLES = {
    "within_subject": "Within subject",
    "within_subject_shared": "Within subject, shared features",
    "subject_independent": "Subject independent",
    "cross_subject": "Cross subject",
}
SCHEMES = tuple(SCHEME_TITLES)


class SelectionStopped(Exception):
    """Raised by :meth:`FeatureSelector.run_study` when ``should_stop`` asks it to."""


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
        trace = self.select(data, metric, class_var, crossvalidation_var, early_stop, nm_class,
                            strategy=strategy, objective=objective, **metadata)
        results = trace.legacy_results()
        if verbose:
            self.print(metric, results, trace.selected)
        return results, list(trace.selected)

    def select(self, data, metric="accuracy", class_var=None, crossvalidation_var=None,
               early_stop=None, nm_class=2, *, strategy=None, objective=None,
               skip_undefined=False, **metadata):
        """Run :meth:`run_selection`, keeping every score it computed.

        Each iteration of a sequential selection scores every remaining group
        alongside those already chosen; the returned trace keeps all of those
        scores, so each choice can be compared with the alternatives it beat.
        A filter ranking is one iteration that scores every group alone.

        Parameters
        ----------
        skip_undefined: bool, default=False
            Score a candidate whose metric is undefined (or raises) as NaN and
            record why in ``trace.notes``, instead of raising. Selection stops
            early once no remaining candidate can be scored.

        Returns
        ----------
        SelectionTrace
        """
        if not isinstance(data, Mapping):
            raise ValueError("run_selection data must be a dictionary of feature groups.")
        arrays, length = self._features(data)
        code = resolve_metric(metric)
        context = self._context(length, class_var, crossvalidation_var, nm_class, metadata)
        strategy, objective = self._plan(code, strategy, objective)
        names = list(arrays)
        count = self._count(early_stop, len(names))
        if code in _CLASSIFICATION:
            context["folds"] = self._folds(context)
        columns = [arrays[name] for name in names]

        def score(chosen):
            return MetricEvaluation(np.hstack([columns[i] for i in chosen]), context).score(code), None

        return self._search(names, score, code, strategy, objective, count, skip_undefined, metric)

    @staticmethod
    def _plan(code, strategy, objective):
        spec = METRICS[code]
        strategy = spec.strategy if strategy is None else strategy
        strategy = "filter" if strategy == "metric" else strategy
        objective = spec.objective if objective is None else objective
        if strategy not in ("filter", "sequential") or objective not in ("min", "max"):
            raise ValueError("strategy must be 'filter' or 'sequential'; objective must be 'min' or 'max'.")
        return strategy, objective

    @staticmethod
    def _count(early_stop, available):
        if early_stop is not None and (not isinstance(early_stop, Integral)
                or isinstance(early_stop, bool) or not 1 <= early_stop <= available):
            raise ValueError("early_stop must be between 1 and the number of feature groups.")
        return available if early_stop is None else int(early_stop)

    def _search(self, names, score, code, strategy, objective, count, skip_undefined, label,
                tick=None):
        """Filter ranking or sequential forward selection over ``names``.

        ``score(indices)`` returns ``(value, parts)``: the criterion for the
        groups at ``indices`` together, and optionally the per-subject values
        it aggregates (None otherwise). ``tick`` is called after every score.
        """
        notes = []
        rows = 1 if strategy == "filter" else count
        scores = np.full((rows, len(names)), np.nan)
        parts = None

        def attempt(step, chosen):
            nonlocal parts
            described = " + ".join(names[i] for i in chosen)
            try:
                value, part = score(chosen)
                if np.isnan(value):
                    raise ValueError(f"{label} is undefined for this candidate feature space.")
            except ValueError as error:
                if not skip_undefined:
                    raise
                value, part = np.nan, None
                notes.append(f"{described}: {error}")
            scores[step, chosen[-1]] = value
            if part is not None:
                if parts is None:
                    parts = np.full((rows, len(names), len(part)), np.nan)
                parts[step, chosen[-1]] = part
            if tick is not None:
                tick()

        def unscorable():
            raise ValueError(f"No feature group could be scored with {label}. "
                             + (notes[0] if notes else ""))

        if strategy == "filter":
            for index in range(len(names)):
                attempt(0, [index])
            values = scores[0]
            if np.isnan(values).all():
                unscorable()
            # Stable, best first, undefined scores last.
            order = np.lexsort((values if objective == "min" else -values, np.isnan(values)))
            selected = [int(i) for i in order if not np.isnan(values[i])][:count]
        else:
            selected = []
            choose = np.nanargmin if objective == "min" else np.nanargmax
            for step in range(count):
                for index in range(len(names)):
                    if index not in selected:
                        attempt(step, selected + [index])
                if np.isnan(scores[step]).all():
                    if step == 0:
                        unscorable()
                    scores = scores[:step]
                    parts = None if parts is None else parts[:step]
                    break
                selected.append(int(choose(scores[step])))
        return SelectionTrace(metric=code, objective=objective, strategy=strategy,
                              candidates=list(names), scores=scores,
                              selected=[names[i] for i in selected], parts=parts, notes=notes)

    # ------------------------------------------------------------------
    # studies over subjects
    # ------------------------------------------------------------------
    def run_study(self, data, metric="accuracy", class_var=None, scheme="within_subject",
                  subject_var=None, repetition_var=None, test_repetitions=None,
                  early_stop=None, nm_class=2, *, strategy=None, objective=None,
                  cv_by="auto", aggregate="mean", on_progress=None, should_stop=None,
                  settings=None, **metadata):
        """Select feature groups for one of four evaluation schemes, and test the result.

        ``scheme`` decides who the features are selected for and what the
        selection may look at:

        ``'within_subject'``
            One selection per subject, from that subject's training
            repetitions only. Tested on that subject's held-out repetitions.
        ``'within_subject_shared'``
            One feature set for the whole cohort, but every model is still a
            subject's own: each candidate's criterion is computed per subject
            from that subject's training data, then aggregated (``aggregate``,
            mean or median). Tested per subject on held-out repetitions, with a
            model trained on that subject's training repetitions.
        ``'subject_independent'``
            One feature set and one model for the cohort, from every subject's
            training repetitions pooled. Tested on every subject's held-out
            repetitions.
        ``'cross_subject'``
            Leave one subject out: one selection per left-out subject, from the
            other subjects' data. The left-out subject provides no training
            data and is the test set. With ``test_repetitions``, the other
            subjects train on their non-test repetitions and only the left-out
            subject's test repetitions are tested, so every scheme can be
            tested on the same windows.

        Parameters
        ----------
        data: dict
            Feature group name to a (windows, columns) array, as for
            :meth:`run_selection`.
        class_var: array
            A class per window.
        subject_var: array or None
            A subject per window. None means one subject.
        repetition_var: array or None
            A repetition per window. Needed to hold repetitions out for testing,
            and passed to the metrics that use repetitions.
        test_repetitions: list or None
            Repetition values held out for testing. They are never seen by the
            selection. None selects on everything and tests nothing, except for
            the cross-subject scheme, whose test set is the left-out subject.
        early_stop: int or None
            Feature groups to select. None ranks all of them.
        cv_by: 'auto', 'repetition', 'subject' or int
            How a classification criterion (accuracy, active error...) is
            cross-validated inside the training data. ``'auto'`` mirrors the
            test: leave one repetition out within a subject or for a pooled
            model, leave one subject out across subjects; it falls back to a
            stratified 5-fold split when there are too few groups. An integer
            is a stratified k-fold split.
        on_progress: callable or None
            Called as ``on_progress(fraction, message)`` while it runs.
        should_stop: callable or None
            Polled between scores; when it returns True, :class:`SelectionStopped`
            is raised.
        settings: dict or None
            Anything else worth keeping with the results, such as where the data
            came from. Saved with them.
        **metadata
            Metric metadata as for :meth:`evaluate`. Per-window values
            (``raw_emg``, ``trial_var``, ``recording_var``) are subset with the
            windows; ``trial_var`` defaults to ``repetition_var``.

        Returns
        ----------
        SelectionStudy
        """
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {', '.join(SCHEMES)}.")
        if not isinstance(data, Mapping):
            raise ValueError("run_study data must be a dictionary of feature groups.")
        if aggregate not in ("mean", "median"):
            raise ValueError("aggregate must be 'mean' or 'median'.")
        if not (cv_by in ("auto", "repetition", "subject") or (
                isinstance(cv_by, Integral) and not isinstance(cv_by, bool) and cv_by >= 2)):
            raise ValueError("cv_by must be 'auto', 'repetition', 'subject' or an integer >= 2.")
        arrays, length = self._features(data)
        names = list(arrays)
        code = resolve_metric(metric)
        strategy, objective = self._plan(code, strategy, objective)
        count = self._count(early_stop, len(names))
        if class_var is None:
            raise ValueError("class_var is required.")
        labels = self._vector(class_var, length, "class_var")
        subjects = (np.zeros(length, dtype=int) if subject_var is None
                    else self._vector(subject_var, length, "subject_var"))
        reps = None if repetition_var is None else self._vector(repetition_var, length,
                                                                 "repetition_var")
        test = None
        if test_repetitions is not None and len(test_repetitions):
            if reps is None:
                raise ValueError("test_repetitions needs repetition_var.")
            test = _matches(reps, test_repetitions)
            if not test.any():
                raise ValueError(f"None of the test repetitions {list(test_repetitions)} "
                                 "appear in repetition_var.")
            if test.all():
                raise ValueError("Every window is in a test repetition, so none are left "
                                 "to select features with.")
        train = np.ones(length, dtype=bool) if test is None else ~test
        per_window = {}
        for key in ("raw_emg", "trial_var", "recording_var"):
            if metadata.get(key) is not None:
                value = np.asarray(metadata.pop(key))
                if len(value) != length:
                    raise ValueError(f"{key} must have one entry per window ({length}).")
                per_window[key] = value
            else:
                metadata.pop(key, None)
        if reps is not None:
            per_window["repetition_var"] = reps
            per_window.setdefault("trial_var", reps)
        cohort = list(np.unique(subjects))
        if scheme in ("within_subject_shared", "cross_subject") and len(cohort) < 2:
            raise ValueError(f"The {SCHEME_TITLES[scheme].lower()} scheme needs at least two "
                             "subjects in subject_var.")

        def subject_name(value):
            return f"Subject {_plain(value)}" if subject_var is not None else "All data"

        # (name, training rows per part, test rows per tested subject, natural folds)
        plans = []
        if scheme == "within_subject":
            for s in cohort:
                own = subjects == s
                plans.append((subject_name(s), [own & train],
                              {} if test is None else {s: [own & train, own & test]},
                              ("repetition",)))
        elif scheme == "within_subject_shared":
            plans.append(("Cohort", [(subjects == s) & train for s in cohort],
                          {} if test is None else
                          {s: [(subjects == s) & train, (subjects == s) & test] for s in cohort},
                          ("repetition",)))
        elif scheme == "subject_independent":
            plans.append(("Cohort", [train],
                          {} if test is None else
                          {s: [train, (subjects == s) & test] for s in cohort},
                          ("repetition", "subject")))
        else:
            for s in cohort:
                others = (subjects != s) & train
                held = (subjects == s) & (test if test is not None else True)
                plans.append((f"Left out: {subject_name(s).lower()}", [others],
                              {s: [others, held]}, ("subject", "repetition")))

        evaluations = (len(names) if strategy == "filter"
                       else sum(len(names) - k for k in range(count)))
        total = max(1, sum(evaluations * len(parts) for _, parts, _, _ in plans))
        done = 0

        def tick(unit_name, weight):
            nonlocal done
            if should_stop is not None and should_stop():
                raise SelectionStopped("Feature selection was stopped.")
            done += weight
            if on_progress is not None:
                on_progress(min(1.0, done / total), unit_name)

        units = []
        for unit_name, parts, tests, natural in plans:
            notes = []
            contexts, folds_used = [], []
            for rows in parts:
                if not rows.any():
                    raise ValueError(f"{unit_name} has no training windows.")
                groups, described = self._criterion_groups(
                    rows, natural, cv_by, {"repetition": reps, "subject": subjects})
                meta = dict(metadata)
                meta.update({key: value[rows] for key, value in per_window.items()})
                cv = ({"var": groups[rows]} if groups is not None
                      else {"crossval_amount": cv_by if isinstance(cv_by, Integral) else 5})
                context = self._context(int(rows.sum()), labels[rows], cv, nm_class, meta)
                if code in _CLASSIFICATION:
                    context["folds"] = self._folds(context)
                contexts.append(context)
                folds_used.append(described if code in _CLASSIFICATION else "")
            sliced = [[arrays[name][rows] for name in names] for rows in parts]
            part_labels = [subject_name(s) for s in cohort] if len(parts) > 1 else None

            def score(chosen, sliced=sliced, contexts=contexts, part_labels=part_labels,
                      notes=notes):
                values = []
                for position, (columns, context) in enumerate(zip(sliced, contexts)):
                    try:
                        values.append(MetricEvaluation(np.hstack([columns[i] for i in chosen]),
                                                       context).score(code))
                    except ValueError as error:
                        if part_labels is None:
                            raise
                        values.append(np.nan)
                        notes.append(f"{' + '.join(names[i] for i in chosen)}, "
                                     f"{part_labels[position]}: {error}")
                if part_labels is None:
                    return values[0], None
                values = np.asarray(values, dtype=float)
                if np.isnan(values).all():
                    return np.nan, values
                reduce = np.nanmean if aggregate == "mean" else np.nanmedian
                return float(reduce(values)), values

            trace = self._search(names, score, code, strategy, objective, count, True, metric,
                                 tick=lambda u=unit_name, w=len(parts): tick(u, w))
            trace.part_names = part_labels
            trace.notes = notes + trace.notes
            order = [names.index(name) for name in trace.selected]
            by_subject = {}
            for s, (fit_rows, test_rows) in tests.items():
                if test_rows.any():
                    by_subject[str(_plain(s))] = self._test_curve(arrays, names, order, labels,
                                                                  fit_rows, test_rows)
            curves = list(by_subject.values())
            units.append(SelectionUnit(
                name=unit_name, trace=trace,
                train_subjects=[str(_plain(s)) for s in cohort
                                if any((subjects[rows] == s).any() for rows in parts)],
                test_subjects=list(by_subject),
                train_windows=int(sum(rows.sum() for rows in parts)),
                test_windows=int(sum(rows.sum() for _, rows in tests.values())),
                criterion_folds=folds_used[0] if len(set(folds_used)) == 1 else "; ".join(folds_used),
                test_accuracy=_nanmean(np.vstack(curves)) if curves else None,
                test_accuracy_by_subject=by_subject))
        if on_progress is not None:
            on_progress(1.0, "done")
        recorded = dict(settings or {})
        recorded.update({
            "classifier": "shrinkage LDA" if self.classifier is None else repr(self.classifier),
            "test_repetitions": (None if test is None else [_plain(v) for v in test_repetitions]),
            "early_stop": early_stop, "cv_by": cv_by, "aggregate": aggregate,
            "nm_class": _plain(nm_class), "classes": [_plain(c) for c in np.unique(labels)],
            "subjects": [str(_plain(s)) for s in cohort] if subject_var is not None else [],
            "windows": int(length)})
        return SelectionStudy(scheme=scheme, metric=code, objective=objective, strategy=strategy,
                              candidates=names, units=units, settings=recorded)

    def _criterion_groups(self, rows, natural, cv_by, available):
        """The groups a classification criterion is cross-validated over, and how to say so."""
        if isinstance(cv_by, Integral):
            return None, f"stratified {cv_by}-fold"
        choices = natural if cv_by == "auto" else (cv_by,)
        for choice in choices:
            groups = available.get(choice)
            if groups is not None and len(np.unique(groups[rows])) >= 2:
                return groups, f"leave one {choice} out"
        if cv_by != "auto":
            raise ValueError(f"cv_by='{cv_by}' needs at least two {cv_by} values in the "
                             "training data.")
        return None, "stratified 5-fold"

    def _test_curve(self, arrays, names, order, labels, fit_rows, test_rows):
        """Test accuracy (%) with the first 1, 2, ... selected groups, trained on ``fit_rows``."""
        curve = np.full(len(order), np.nan)
        if len(np.unique(labels[fit_rows])) < 2:
            return curve
        for k in range(1, len(order) + 1):
            x = np.hstack([arrays[names[i]] for i in order[:k]])
            model = self._classifier()
            model.fit(x[fit_rows], labels[fit_rows])
            curve[k - 1] = 100 * np.mean(model.predict(x[test_rows]) == labels[test_rows])
        return curve

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


# ======================================================================
# results
# ======================================================================
@dataclass
class SelectionTrace:
    """Every score one selection computed, so that each choice can be explained.

    Attributes
    ----------
    metric: str
        The metric's abbreviation.
    objective: str
        ``'min'`` or ``'max'``: which end of the scores is better.
    strategy: str
        ``'filter'`` (every group scored alone, then ranked) or
        ``'sequential'`` (forward selection).
    candidates: list of str
        Every feature group that could be chosen, in input order.
    scores: ndarray
        (iterations, candidates). What each candidate scored at each
        iteration: alone for a filter (one row), together with the groups
        already chosen for sequential selection. NaN where a candidate was
        already chosen or could not be scored.
    selected: list of str
        The chosen groups in the order chosen, or the ranking for a filter.
    parts: ndarray or None
        (iterations, candidates, parts). For a criterion aggregated over
        subjects, each subject's own score; ``scores`` is their aggregate.
    part_names: list of str or None
        What each part is, such as 'Subject 3'.
    notes: list of str
        Why any candidate could not be scored.
    """

    metric: str
    objective: str
    strategy: str
    candidates: list
    scores: np.ndarray
    selected: list
    parts: np.ndarray = None
    part_names: list = None
    notes: list = field(default_factory=list)

    @property
    def iterations(self):
        """How many rows of scores there are: 1 for a filter, one per choice otherwise."""
        return len(self.scores)

    @property
    def selected_scores(self):
        """The score of each chosen group when it was chosen, in order.

        For sequential selection this is the criterion of the feature set
        after each addition; for a filter, each group's own score.
        """
        if self.strategy == "filter":
            return np.array([self.scores[0, self.candidates.index(n)] for n in self.selected])
        return np.array([self.scores[step, self.candidates.index(n)]
                         for step, n in enumerate(self.selected)])

    def ranking(self, iteration=0):
        """The candidates scored at ``iteration``, best first, and their scores.

        Returns
        ----------
        list of (str, float)
            Candidates that were not scored at that iteration (already chosen,
            or undefined) are left out.
        """
        values = self.scores[iteration]
        key = values if self.objective == "min" else -values
        order = np.lexsort((key, np.isnan(values)))
        return [(self.candidates[i], float(values[i])) for i in order if not np.isnan(values[i])]

    def chosen_before(self, iteration):
        """The groups already in the feature set when ``iteration`` was scored."""
        return [] if self.strategy == "filter" else list(self.selected[:iteration])

    def legacy_results(self):
        """Scores in :meth:`FeatureSelector.run_selection`'s original layout."""
        if self.strategy == "filter":
            return self.selected_scores
        count = len(self.candidates)
        full = np.full((count, count), np.nan)
        full[:len(self.scores)] = self.scores
        return full[:, [self.candidates.index(n) for n in self.selected]]

    def to_dict(self):
        return {"metric": self.metric, "objective": self.objective, "strategy": self.strategy,
                "candidates": list(self.candidates), "scores": _json_numbers(self.scores),
                "selected": list(self.selected),
                "parts": None if self.parts is None else _json_numbers(self.parts),
                "part_names": self.part_names, "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, record):
        candidates = list(record["candidates"])
        scores = _float_array(record["scores"]).reshape(-1, len(candidates))
        parts = record.get("parts")
        return cls(metric=record["metric"], objective=record["objective"],
                   strategy=record["strategy"], candidates=candidates, scores=scores,
                   selected=list(record["selected"]),
                   parts=None if parts is None else _float_array(parts),
                   part_names=record.get("part_names"), notes=list(record.get("notes", [])))


@dataclass
class SelectionUnit:
    """One selection within a study, and how its features fared on held-out data.

    Attributes
    ----------
    name: str
        Who it selected for: 'Subject 3', 'Cohort', 'Left out: subject 3'.
    trace: SelectionTrace
    train_subjects, test_subjects: list of str
    train_windows, test_windows: int
    criterion_folds: str
        How a classification criterion was cross-validated.
    test_accuracy: ndarray or None
        Accuracy (%) on held-out data with the first 1, 2, ... selected groups,
        averaged over the subjects tested. None when nothing was held out.
    test_accuracy_by_subject: dict
        The same curve for each tested subject.
    """

    name: str
    trace: SelectionTrace
    train_subjects: list = field(default_factory=list)
    test_subjects: list = field(default_factory=list)
    train_windows: int = 0
    test_windows: int = 0
    criterion_folds: str = ""
    test_accuracy: np.ndarray = None
    test_accuracy_by_subject: dict = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "trace": self.trace.to_dict(),
                "train_subjects": list(self.train_subjects),
                "test_subjects": list(self.test_subjects),
                "train_windows": self.train_windows, "test_windows": self.test_windows,
                "criterion_folds": self.criterion_folds,
                "test_accuracy": (None if self.test_accuracy is None
                                  else _json_numbers(self.test_accuracy)),
                "test_accuracy_by_subject": {k: _json_numbers(v) for k, v in
                                             self.test_accuracy_by_subject.items()}}

    @classmethod
    def from_dict(cls, record):
        accuracy = record.get("test_accuracy")
        return cls(name=record["name"], trace=SelectionTrace.from_dict(record["trace"]),
                   train_subjects=list(record.get("train_subjects", [])),
                   test_subjects=list(record.get("test_subjects", [])),
                   train_windows=int(record.get("train_windows", 0)),
                   test_windows=int(record.get("test_windows", 0)),
                   criterion_folds=record.get("criterion_folds", ""),
                   test_accuracy=None if accuracy is None else _float_array(accuracy),
                   test_accuracy_by_subject={k: _float_array(v) for k, v in
                                             record.get("test_accuracy_by_subject", {}).items()})


@dataclass
class SelectionStudy:
    """What :meth:`FeatureSelector.run_study` found, ready to save, load and draw.

    Attributes
    ----------
    scheme: str
        One of :data:`SCHEMES`.
    metric, objective, strategy: str
    candidates: list of str
        The feature bank.
    units: list of SelectionUnit
        One per subject (within subject), per left-out subject (cross
        subject), or one for the cohort.
    settings: dict
        How it was run, and anything the caller asked to keep with it.
    created: str
        ISO timestamp.
    """

    FORMAT = "libemg.feature_selection_study"
    VERSION = 1

    scheme: str
    metric: str
    objective: str
    strategy: str
    candidates: list
    units: list
    settings: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def title(self):
        return SCHEME_TITLES.get(self.scheme, self.scheme)

    @property
    def metric_name(self):
        return METRICS[self.metric].name if self.metric in METRICS else self.metric

    def rank_matrix(self):
        """When each unit chose each candidate.

        Returns
        ----------
        ndarray
            (units, candidates) of 1-based ranks; 0 where a unit never chose
            the candidate.
        """
        ranks = np.zeros((len(self.units), len(self.candidates)), dtype=int)
        for row, unit in enumerate(self.units):
            for rank, name in enumerate(unit.trace.selected, 1):
                ranks[row, self.candidates.index(name)] = rank
        return ranks

    def consensus(self, top=None):
        """Candidates ordered by how many units chose them within the first ``top`` places.

        Ties are broken by mean rank. Useful for a within-subject or
        cross-subject study, which selects one feature set per unit.

        Returns
        ----------
        list of (str, int, float)
            Name, how many units chose it within ``top``, and its mean rank
            among those that did (NaN if none did).
        """
        ranks = self.rank_matrix()
        if top is not None:
            ranks = np.where(ranks <= top, ranks, 0)
        rows = []
        for column, name in enumerate(self.candidates):
            chosen = ranks[:, column][ranks[:, column] > 0]
            rows.append((name, int(len(chosen)), float(chosen.mean()) if len(chosen) else math.nan))
        return sorted(rows, key=lambda r: (-r[1], r[2] if not math.isnan(r[2]) else math.inf))

    def mean_test_accuracy(self):
        """Test accuracy against the number of groups, averaged over units, or None."""
        curves = [u.test_accuracy for u in self.units if u.test_accuracy is not None]
        if not curves:
            return None
        padded = np.full((len(curves), max(len(c) for c in curves)), np.nan)
        for row, curve in enumerate(curves):
            padded[row, :len(curve)] = curve
        return _nanmean(padded)

    def summary(self):
        """A few lines saying what was selected and how well it tested."""
        lines = [f"{self.title}: {self.metric_name} ({self.metric}, best is "
                 f"{'lowest' if self.objective == 'min' else 'highest'}, {self.strategy}) "
                 f"over {len(self.candidates)} feature groups."]
        for unit in self.units:
            line = f"  {unit.name}: {', '.join(unit.trace.selected)}"
            accuracy = unit.test_accuracy
            if accuracy is not None and len(accuracy) and not np.isnan(accuracy).all():
                best = int(np.nanargmax(accuracy))
                line += (f"  (test {accuracy[-1]:.1f}% with all, best {accuracy[best]:.1f}% "
                         f"with {best + 1})")
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self):
        return {"format": self.FORMAT, "version": self.VERSION, "created": self.created,
                "scheme": self.scheme, "metric": self.metric, "objective": self.objective,
                "strategy": self.strategy, "candidates": list(self.candidates),
                "settings": _json_numbers(self.settings),
                "units": [unit.to_dict() for unit in self.units]}

    @classmethod
    def from_dict(cls, record):
        if record.get("format") != cls.FORMAT:
            raise ValueError("This is not a saved feature selection study.")
        if int(record.get("version", 0)) > cls.VERSION:
            raise ValueError("This study was saved by a newer version of libemg.")
        return cls(scheme=record["scheme"], metric=record["metric"],
                   objective=record["objective"], strategy=record["strategy"],
                   candidates=list(record["candidates"]),
                   units=[SelectionUnit.from_dict(u) for u in record["units"]],
                   settings=_restore_numbers(record.get("settings", {})),
                   created=record.get("created", ""))

    def save(self, path):
        """Write the study to ``path`` as JSON. Non-finite scores are written as strings."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1)
        return path

    @classmethod
    def load(cls, path):
        """Read a study written by :meth:`save`."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


# ----------------------------------------------------------------------
def _plain(value):
    """A numpy scalar as the Python value it holds."""
    return value.item() if hasattr(value, "item") else value


def _nanmean(array, axis=0):
    """Mean over ``axis`` ignoring NaN, and NaN (without a warning) where all are NaN."""
    array = np.asarray(array, dtype=float)
    counts = np.sum(~np.isnan(array), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, np.nansum(array, axis=axis) / np.maximum(counts, 1), np.nan)


def _matches(values, wanted):
    """Which entries of ``values`` are one of ``wanted``, '1' matching 1 and 1.0."""
    wanted = list(wanted)
    hits = np.zeros(len(values), dtype=bool)
    for value in np.unique(values):
        for candidate in wanted:
            same = str(_plain(value)) == str(candidate)
            if not same:
                try:
                    same = float(value) == float(candidate)
                except (TypeError, ValueError):
                    pass
            if same:
                hits |= values == value
                break
    return hits


_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _json_numbers(value):
    """``value`` with arrays as lists and NaN or infinity as strings, so strict JSON holds it."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numbers(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _restore_numbers(value):
    if isinstance(value, Mapping):
        return {k: _restore_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_numbers(v) for v in value]
    return _NONFINITE.get(value, value) if isinstance(value, str) else value


def _float_array(value):
    return np.asarray(_restore_numbers(value), dtype=float)
