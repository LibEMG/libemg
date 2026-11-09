"""Metric implementations used by :mod:`libemg.feature_selector`.

References and implementation conventions are recorded in the feature selection
documentation. Arrays reaching this module have been validated by the selector.
"""

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import linprog
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class MetricSpec:
    name: str
    objective: str = "max"
    strategy: str = "sequential"
    requires: tuple = ()
    reference_status: str = "defined"


METRICS = {
    "CA": MetricSpec("accuracy"),
    "AE": MetricSpec("activeerror", "min"),
    "MSA": MetricSpec("meansemiprincipalaxis", "min"),
    "FE": MetricSpec("featureefficiency", strategy="filter"),
    "RI": MetricSpec("repeatability", "min", requires=("repetition_var",)),
    "SI": MetricSpec("separability"),
    "mwRI": MetricSpec("meanwithinrepetitionrepeatability", "min", requires=("repetition_var",)),
    "swRI": MetricSpec("stdwithinrepetitionrepeatability", "min", requires=("repetition_var",)),
    "swSI": MetricSpec("stdwithintrialseparability", "min", requires=("trial_var",)),
    "CD": MetricSpec("centroiddrift", "min", requires=("repetition_var",)),
    "MAV": MetricSpec("meanabsolutevalue", strategy="filter", requires=("raw_emg",)),
    "mSI": MetricSpec("modifiedseparability"),
    "mwSI": MetricSpec("meanwithintrialseparability", requires=("trial_var",)),
    "BD": MetricSpec("bhattacharyyadistance"),
    "KLD": MetricSpec("kullbackleiblerdivergence"),
    "HD": MetricSpec("hellingerdistance"),
    "FDR": MetricSpec("fisherdiscriminantratio"),
    "VOR": MetricSpec("volumeofoverlapregion", "min"),
    "TSM": MetricSpec("traceofscattermatrices"),
    "DS": MetricSpec("desirabilityscore", requires=("repetition_var",)),
    "CDM": MetricSpec("classdiscriminability", "min", reference_status="partition variant"),
    "PU": MetricSpec("purity"),
    "rPU": MetricSpec("rescaledpurity"),
    "NS": MetricSpec("neighborhoodseparability"),
    "rNS": MetricSpec("rescaledneighborhoodseparability"),
    "CE": MetricSpec("collectiveentropy", reference_status="normalized entropy convention"),
    "rCE": MetricSpec("rescaledcollectiveentropy", reference_status="normalized entropy convention"),
    "C": MetricSpec("compactness", reference_status="concentration variant; original formula unverified"),
    "rC": MetricSpec("rescaledcompactness", reference_status="concentration variant; original formula unverified"),
    "ACA": MetricSpec("activeclassificationaccuracy"),
    "UD": MetricSpec("usabledata", requires=("trial_var",)),
    "ICF": MetricSpec("interclassfraction", "min"),
    "IIF": MetricSpec("intrainterfraction", "min"),
    "TT": MetricSpec("trainingtime", "min", "filter", ("timing",)),
}


def _normalize(name):
    return "".join(c for c in name.lower() if c.isalnum())


_ALIASES = {_normalize(name): code for code, spec in METRICS.items() for name in (code, spec.name)}
_ALIASES.update({"classificationaccuracy": "CA", "hellingerdistancesquared": "HD",
                 "fisherdiscriminantanalysis": "FDR", "classdiscriminabilitymeasure": "CDM",
                 "meansemiprincipalaxes": "MSA", "separabilityindex": "SI",
                 "modifiedseparabilityindex": "mSI", "repeatabilityindex": "RI"})
for _code in ("mwRI", "swRI", "mwSI", "swSI"):
    _ALIASES[METRICS[_code].name + "index"] = _code
_ALIASES.update({"standarddeviationwithinrepetitionrepeatabilityindex": "swRI",
                 "standarddeviationwithintrialseparabilityindex": "swSI"})


def resolve_metric(metric):
    if not isinstance(metric, str) or _normalize(metric) not in _ALIASES:
        raise ValueError(f"Unknown metric {metric!r}. Use FeatureSelector.get_metrics().")
    return _ALIASES[_normalize(metric)]


class MetricEvaluation:
    """One feature subset with lazily shared statistics and metric results."""

    def __init__(self, features, context):
        self.x = features
        self.context = context
        self.y = context["labels"]
        self.selector = context["selector"]
        self.classes = np.unique(self.y)
        self.samples = [self.x[self.y == label] for label in self.classes]
        self.cache = {}

    def score(self, code):
        if code not in self.cache:
            if code in ("C", "rC", "CDM") and not self.selector.allow_approximations:
                raise ValueError(f"{code}: the original reference algorithm is unverified. "
                                 "Set allow_approximations=True to use the documented variant.")
            for requirement in METRICS[code].requires:
                if requirement != "timing" and self.context.get(requirement) is None:
                    raise ValueError(f"{code} requires {requirement}; supply real recording metadata.")
            self.cache[code] = float(self._compute(code))
        return self.cache[code]

    @staticmethod
    def covariance(x):
        if len(x) < 2:
            raise ValueError("Covariance metrics require at least two samples per class/cell.")
        centered = x - x.mean(axis=0)
        return centered.T @ centered / (len(x) - 1)

    def positive_covariance(self, covariance):
        """Clip only small eigenvalues; retain ordinary nonsingular covariances."""
        values, vectors = np.linalg.eigh(covariance)
        scale = float(np.max(np.abs(values)))
        if scale == 0:
            scale = max(float(np.max(np.var(self.x, axis=0))), 1.0)
        values = np.maximum(values, self.selector.regularization * scale)
        return (vectors * values) @ vectors.T

    @cached_property
    def means(self):
        return np.array([sample.mean(axis=0) for sample in self.samples])

    @cached_property
    def covariances(self):
        return [self.positive_covariance(self.covariance(sample)) for sample in self.samples]

    @staticmethod
    def mahalanobis(delta, covariance):
        squared = np.einsum("...i,i...->...", delta, np.linalg.solve(covariance, delta.T))
        return np.sqrt(np.maximum(squared, 0))

    def groups(self, key, *, complete=True, minimum=1):
        values = self.context.get(key)
        if values is None:
            raise ValueError(f"This metric requires {key}.")
        groups = np.unique(values)
        if len(groups) < 2:
            raise ValueError(f"{key} must contain at least two groups.")
        if complete:
            for group in groups:
                for label in self.classes:
                    if np.count_nonzero((values == group) & (self.y == label)) < minimum:
                        raise ValueError(f"Each {key} group must contain at least {minimum} sample(s) of every class.")
        return values, groups

    def _compute(self, code):
        if code in ("CA", "ACA", "AE"):
            return self.classification[code]
        if code in ("RI", "CD", "mwRI", "swRI"):
            return self.repetition_metric(code)
        if code in ("mwSI", "swSI"):
            return self.trial_metrics[code]
        if code in ("SI", "mSI", "BD", "KLD", "HD", "FDR", "VOR", "FE"):
            return self.pairwise(code)
        if code in ("PU", "rPU", "NS", "rNS", "CE", "rCE", "C", "rC"):
            return self.prism[code]
        if code == "MSA":
            sizes = []
            for sample in self.samples:
                eigenvalues = np.linalg.eigvalsh(self.covariance(sample))
                tolerance = max(float(np.max(eigenvalues)), 0) * np.finfo(float).eps * self.x.shape[1]
                sizes.append(0.0 if np.any(eigenvalues <= tolerance)
                             else np.exp(np.mean(np.log(eigenvalues)) / 2))
            return np.mean(sizes)
        if code == "TSM":
            within = sum((sample - mean).T @ (sample - mean)
                         for sample, mean in zip(self.samples, self.means))
            between = sum(len(sample) * np.outer(mean - self.x.mean(axis=0), mean - self.x.mean(axis=0))
                          for sample, mean in zip(self.samples, self.means))
            return np.trace(np.linalg.solve(self.positive_covariance(within), between))
        if code == "DS":
            return self.ratio(self.score("SI"), self.score("RI") * self.score("MSA"), "DS")
        if code == "MAV":
            raw = np.asarray(self.context["raw_emg"], dtype=float)
            if raw.ndim < 2 or len(raw) != len(self.y) or raw.size == 0 or not np.isfinite(raw).all():
                raise ValueError("raw_emg must be finite raw windows with one window per feature row.")
            amplitude = np.abs(raw).reshape(len(raw), -1).mean(axis=1)
            return np.mean([amplitude[self.y == label].mean() for label in self.classes])
        if code == "TT":
            return self.training_time()
        if code in ("ICF", "IIF"):
            if code == "IIF" and self.context.get("iif_mode", "classwise") == "classwise":
                return self.classwise_intra_inter()
            if code == "IIF" and self.context.get("iif_mode") != "paper":
                raise ValueError("iif_mode must be 'classwise' or 'paper'.")
            distance, indices = self.nearest_neighbors
            same = self.y == self.y[indices]
            return (np.mean(~same) if code == "ICF" else
                    self.ratio(distance[same].sum(), distance[~same].sum(), "IIF"))
        if code == "UD":
            return self.usable_data()
        if code == "CDM":
            return self.class_discriminability()
        raise ValueError(f"No implementation for {code}.")

    @staticmethod
    def ratio(numerator, denominator, metric):
        if denominator == 0:
            if numerator == 0:
                raise ValueError(f"{metric} is undefined (0/0) for this feature space.")
            return np.inf
        return numerator / denominator

    def pairwise(self, code):
        count = len(self.classes)
        values = np.full((count, count), np.nan)
        for j, first in enumerate(self.samples):
            for i, second in enumerate(self.samples):
                if i == j:
                    continue
                delta = self.means[j] - self.means[i]
                if code in ("FE", "VOR"):
                    lower = np.maximum(first.min(axis=0), second.min(axis=0))
                    upper = np.minimum(first.max(axis=0), second.max(axis=0))
                    if code == "FE":
                        union = np.vstack((first, second))
                        overlap = ((union >= lower) & (union <= upper)).sum(axis=0)
                        value = np.max(1 - overlap / len(union))
                    else:
                        width = np.maximum(first.max(axis=0), second.max(axis=0)) - np.minimum(
                            first.min(axis=0), second.min(axis=0))
                        # Identical constant dimensions impose no separation.
                        ratios = np.divide(np.maximum(upper - lower, 0), width,
                                           out=np.ones_like(width), where=width > 0)
                        value = np.prod(ratios)
                elif code == "FDR":
                    if len(first) < 2 or len(second) < 2:
                        raise ValueError("FDR requires at least two samples per class.")
                    variance = first.var(axis=0, ddof=1) + second.var(axis=0, ddof=1)
                    ratios = np.divide(delta ** 2, variance, out=np.zeros_like(delta), where=variance > 0)
                    ratios[(variance == 0) & (delta != 0)] = np.inf
                    value = ratios.max()
                else:
                    a, b = self.covariances[j], self.covariances[i]
                    average = (a + b) / 2
                    if code == "SI":
                        value = self.mahalanobis(delta, a) / 2
                    elif code == "mSI":
                        value = self.mahalanobis(delta, average) / 2
                    elif code == "KLD":
                        value = max(0.0, (np.trace(np.linalg.solve(a, b))
                                    + self.mahalanobis(delta, a) ** 2 - self.x.shape[1]
                                    + np.linalg.slogdet(a)[1] - np.linalg.slogdet(b)[1]) / 2)
                    else:
                        # Standard Gaussian BD: the paper prints the opposite
                        # sign on the log determinant term. HD = 1 - exp(-BD).
                        bd = max(0.0, self.mahalanobis(delta, average) ** 2 / 8
                                 + (np.linalg.slogdet(average)[1]
                                    - (np.linalg.slogdet(a)[1] + np.linalg.slogdet(b)[1]) / 2) / 2)
                        value = bd if code == "BD" else -np.expm1(-bd)
                values[j, i] = value
        reduction = np.nanmax if code in ("FE", "VOR") else np.nanmin
        return np.mean(reduction(values, axis=1))

    def repetition_metric(self, code):
        values, groups = self.groups("repetition_var")
        centroids = []
        within = []
        repeatability = []
        for group in groups:
            group_means = []
            group_within = []
            for index, label in enumerate(self.classes):
                cell = self.x[(values == group) & (self.y == label)]
                other = self.x[(values != group) & (self.y == label)]
                mean = cell.mean(axis=0)
                group_means.append(mean)
                if code in ("mwRI", "swRI"):
                    group_within.append(np.mean(self.mahalanobis(cell - mean, self.covariances[index])) / 2)
                if code == "RI":
                    repeatability.append(self.mahalanobis(mean - other.mean(axis=0),
                                         self.positive_covariance(self.covariance(other))) / 2)
            centroids.append(group_means)
            if group_within:
                within.append(np.mean(group_within))
        if code == "CD":
            return np.linalg.norm(np.diff(np.array(centroids), axis=0), axis=2).sum(axis=0).mean()
        if code == "RI":
            return np.mean(repeatability)
        return np.mean(within) if code == "mwRI" else np.std(within, ddof=1)

    @cached_property
    def trial_metrics(self):
        values, groups = self.groups("trial_var", minimum=2)
        separability = []
        for group in groups:
            mask = values == group
            context = dict(self.context, labels=self.y[mask])
            separability.append(MetricEvaluation(self.x[mask], context).score("SI"))
        return dict(mwSI=np.mean(separability), swSI=np.std(separability, ddof=1))

    @cached_property
    def classification(self):
        folds = self.context.get("folds")
        if folds is None:
            folds = self.selector._folds(self.context)
        accuracy, active = [], []
        for train, test in folds:
            model = self.selector._classifier()
            model.fit(self.x[train], self.y[train])
            predictions = model.predict(self.x[test])
            correct = predictions == self.y[test]
            accuracy.append(correct.mean())
            active.append((correct | (predictions == self.context["nm"])).mean())
        ca, aca = 100 * np.mean(accuracy), 100 * np.mean(active)
        return dict(CA=ca, ACA=aca, AE=100 - aca)

    @cached_property
    def nearest_neighbors(self):
        # Chunking bounds memory while stable sorting resolves exact ties by
        # original sample index. Exclude self by identity, including duplicates.
        distances, indices = [], []
        for start in range(0, len(self.x), 256):
            block = cdist(self.x[start:start + 256], self.x)
            block[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
            index = np.argmin(block, axis=1)
            distances.extend(block[np.arange(len(block)), index])
            indices.extend(index)
        return np.array(distances), np.array(indices)

    def classwise_intra_inter(self):
        """Ratio of nearest same-class to nearest different-class distance sums.

        This follows the paper's prose and Ho & Basu's N2 definition. Set
        iif_mode='paper' to evaluate the literal indicator equation instead.
        """
        if any(len(sample) < 2 for sample in self.samples):
            raise ValueError("Classwise IIF requires at least two samples per class.")
        intra = inter = 0.0
        for start in range(0, len(self.x), 256):
            block = cdist(self.x[start:start + 256], self.x)
            same = self.y[start:start + len(block), None] == self.y[None, :]
            block[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
            intra += np.where(same, block, np.inf).min(axis=1).sum()
            inter += np.where(~same, block, np.inf).min(axis=1).sum()
        return self.ratio(intra, inter, "IIF")

    def training_time(self):
        duration = self.context.get("training_time_ms")
        if duration is not None:
            if not np.isscalar(duration) or not np.isfinite(duration) or duration <= 0:
                raise ValueError("training_time_ms must be positive and finite.")
            return duration
        width = self.context.get("window_size_ms")
        hop = self.context.get("window_increment_ms")
        if any(v is None or not np.isscalar(v) or not np.isfinite(v) or v <= 0 for v in (width, hop)):
            raise ValueError("TT requires training_time_ms or positive window_size_ms and window_increment_ms.")
        recordings = self.context.get("recording_var")
        counts = [len(self.x)] if recordings is None else np.unique(recordings, return_counts=True)[1]
        return sum(width + (count - 1) * hop for count in counts)

    def usable_data(self):
        values, groups = self.groups("trial_var", minimum=2)
        memory = self.context.get("adaptive_memory", 4)
        if not isinstance(memory, Integral) or isinstance(memory, bool) or memory < 1:
            raise ValueError("adaptive_memory must be a positive integer.")
        # The first full trial seeds the classifier. Subsequent trials are
        # predicted before adding correctly classified windows for retraining.
        accepted = [np.flatnonzero(values == groups[0])]
        used = len(accepted[0])
        for group in groups[1:]:
            train = np.concatenate(accepted[-memory:])
            if not np.array_equal(np.unique(self.y[train]), self.classes):
                raise ValueError("UD adaptive history no longer contains every class.")
            model = self.selector._classifier()
            model.fit(self.x[train], self.y[train])
            test = np.flatnonzero(values == group)
            keep = test[model.predict(self.x[test]) == self.y[test]]
            accepted.append(keep)
            used += len(keep)
        return 100 * used / len(self.x)

    @cached_property
    def prism(self):
        """Sparse multiresolution PRISM curves, without allocating a dense grid.

        Purity and neighborhood separability follow Singh (2003, TPAMI).
        Entropy uses local class-count normalization. Compactness uses the
        occupied-cell concentration convention documented with the public API;
        it is not a verified reproduction of Singh's compactness equation.
        """
        maximum = self.selector.prism_resolution
        span = np.ptp(self.x, axis=0)
        normalized = np.divide(self.x - self.x.min(axis=0), span,
                               out=np.zeros_like(self.x), where=span > 0)
        curves = {key: [] for key in ("PU", "NS", "CE", "C")}
        for resolution in range(maximum + 1):
            bins = np.minimum((normalized * (resolution + 1)).astype(np.int64), resolution)
            _, inverse = np.unique(bins, axis=0, return_inverse=True)
            purity = entropy = neighborhood = concentration = 0.0
            for cell in range(inverse.max() + 1):
                ids = np.flatnonzero(inverse == cell)
                size = len(ids)
                weight = size / len(self.x)
                _, counts = np.unique(self.y[ids], return_counts=True)
                probabilities = counts / size
                classes = len(counts)
                pure = (1.0 if classes == 1 else np.sqrt(
                    classes / (classes - 1) * np.sum((probabilities - 1 / classes) ** 2)))
                certainty = (1.0 if classes == 1 else
                             1 + np.sum(probabilities * np.log(probabilities)) / np.log(classes))
                purity += weight * pure
                entropy += weight * certainty
                concentration += weight ** 2
                if classes == 1:
                    neighborhood += weight
                else:
                    # Integrate all k=1..cell_size-1 proportions per point.
                    # Self is excluded even when another sample is identical.
                    local = self.x[ids]
                    proportions = []
                    for start in range(0, size, 128):
                        distances = cdist(local[start:start + 128], local)
                        distances[np.arange(len(distances)), np.arange(start, start + len(distances))] = np.inf
                        nearest = np.argsort(distances, axis=1, kind="stable")[:, :size - 1]
                        same = self.y[ids][nearest] == self.y[ids[start:start + len(distances)], None]
                        fractions = np.cumsum(same, axis=1) / np.arange(1, size)
                        proportions.extend(fractions.mean(axis=1) if size == 2 else
                                           trapezoid(fractions, dx=1 / (size - 2), axis=1))
                    neighborhood += weight * np.mean(proportions)
            curves["PU"].append(purity)
            curves["NS"].append(neighborhood)
            curves["CE"].append(entropy)
            curves["C"].append(concentration)
        # The unprefixed metric is an unweighted normalized-resolution AUC.
        # r* applies exponential resolution weighting and divides by its maximum
        # possible AUC. A perfect curve consequently has r*=1.
        resolution = np.linspace(0, 1, maximum + 1)
        weights = np.exp2(-np.arange(maximum + 1, dtype=float))
        maximum_area = trapezoid(weights, resolution)
        result = {}
        for code, curve in curves.items():
            result[code] = trapezoid(curve, resolution)
            result["r" + code] = trapezoid(np.asarray(curve) * weights, resolution) / maximum_area
        return result

    @staticmethod
    def linearly_separable(x, y):
        """Check hard-margin one-vs-rest feasibility; no fitted-label shortcut."""
        design = np.column_stack((x, np.ones(len(x))))
        for label in np.unique(y):
            signs = np.where(y == label, 1, -1)
            solution = linprog(np.zeros(design.shape[1]), A_ub=-signs[:, None] * design,
                               b_ub=-np.ones(len(x)), bounds=[(None, None)] * design.shape[1],
                               method="highs")
            if solution.status == 2:
                return False
            if not solution.success:
                raise ValueError(f"CDM linear separability check failed: {solution.message}")
        return True

    def class_discriminability(self):
        """CDM minority fraction with explicit, configurable median partition rules.

        The attached paper specifies the leaf statistic but defers its partition
        rules to Kohn et al. This implementation is an adaptive median variant,
        not a verified reproduction of the unavailable original stopping rules.
        """
        span = np.ptp(self.x, axis=0)
        normalized = np.divide(self.x - self.x.min(axis=0), span,
                               out=np.zeros_like(self.x), where=span > 0)
        minimum = self.selector.cdm_min_samples or 2 * (self.x.shape[1] + 1)
        pending = [(np.arange(len(self.x)), 0)]
        minority = 0
        while pending:
            ids, depth = pending.pop()
            _, counts = np.unique(self.y[ids], return_counts=True)
            if len(counts) == 1 or self.linearly_separable(normalized[ids], self.y[ids]):
                continue
            if len(ids) <= minimum or depth >= self.selector.cdm_max_depth:
                minority += len(ids) - counts.max()
                continue
            # Choose the widest nonconstant axis. Never split identical values
            # between children: they must remain in the same spatial cell.
            split = None
            for axis in np.argsort(-np.ptp(normalized[ids], axis=0), kind="stable"):
                values = normalized[ids, axis]
                unique = np.unique(values)
                if len(unique) < 2:
                    continue
                median = np.median(values)
                left = values <= median
                if left.all():
                    left = values < unique[-1]
                split = left
                break
            if split is None:
                minority += len(ids) - counts.max()
            else:
                pending.extend(((ids[split], depth + 1), (ids[~split], depth + 1)))
        return minority / len(self.x)
