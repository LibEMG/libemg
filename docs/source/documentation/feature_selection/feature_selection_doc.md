# Feature selection and feature-space metrics

`FeatureSelector` evaluates feature spaces and ranks named feature groups using
independent scores (`strategy="filter"`) or sequential forward selection
(`strategy="sequential"`). A group can contain multiple columns, such as one
MAV value per channel or several AR coefficients.

There are 34 metric codes: the 32 unique abbreviations in the original comment,
plus RI and AE. **Exact reproduction of C/rC and CDM is excluded from this
implementation's scope.** Their optional, explicitly defined variants require
`allow_approximations=True` and remain disabled by default. The Nawfel paper
defers their details to Singh and Kohn; the original algorithms remain
unverified. CE/rCE use the entropy normalization convention described below.

## API

```python
from libemg.feature_selector import FeatureSelector

selector = FeatureSelector(random_state=0)
selector.get_metrics()                    # long names, including legacy names
selector.get_metrics(abbreviations=True)  # paper abbreviations
selector.get_metric_info("SI")            # objective, requirements, reference status

# features: {name: samples x columns}; labels/trials: one value per sample
scores = selector.evaluate_metrics(
    ["CA", "ACA", "AE", "SI", "BD", "FE"], features, labels,
    crossvalidation_var={"var": trials}, nm_class=2,
)
si = selector.evaluate(features, "SI", labels)
results, order = selector.run_selection(
    features, "CA", labels, {"var": trials}, early_stop=2,
)
results, order = selector.run_selection(
    features, "BD", labels, strategy="filter", objective="max",
)
```

Abbreviations are case insensitive. Original names remain accepted: `accuracy`,
`activeerror`, `meansemiprincipalaxis`, `featureefficiency`, `repeatability`,
and `separability`. `evaluate` also accepts a matrix; dictionaries are
concatenated in insertion order. `evaluate_metrics(None, ...)` requests all
metrics, requiring all recording metadata and explicit opt-in to the variants.

`run_selection` preserves its original positional arguments and `(results,
feature_order)` return value. Filter scores form a sorted vector aligned with
the names. Sequential scores have one row per input group and one column per
selected group; columns follow selection order. Entry `[i,j]` scores the first
`i` selected groups plus candidate `j`. Previously selected candidates and
uncomputed rows are NaN. Ties preserve dictionary order. `early_stop` limits the
number of selected groups for either strategy. `verbose=True` and
`selector.print(...)` support either result shape.

## Metadata and validation

| Argument | Meaning |
| --- | --- |
| `class_var` | Class labels per row, including arbitrary numeric/string labels. All supplied classes participate; subset active motions first to exclude rest. |
| `crossvalidation_var={"var": groups}` | Leave-one-group-out validation for CA/ACA/AE. Every training fold must contain every class. |
| `crossvalidation_var={"crossval_amount": k}` | Stratified shuffled k-fold classification validation; default 5 folds, seed 0. Every class needs at least k samples. |
| `repetition_var` | Real chronological repetition IDs for RI/mwRI/swRI/CD/DS. Every group must contain every class. |
| `trial_var` | Real chronological trial IDs for mwSI/swSI/UD, with at least two windows per class/trial. |
| `nm_class` | No-motion label for ACA/AE, default 2 for the Myo example. |
| `raw_emg` | Raw windows aligned with feature rows, normally windows x channels x samples. Required for MAV. |
| `window_size_ms`, `window_increment_ms` | Window duration and **hop**, in milliseconds, for TT. |
| `recording_var` | Unique ID per continuous recording for TT; omit only for a single continuous recording. |
| `training_time_ms` | Optional measured collection duration, overriding window-based TT. |
| `adaptive_memory` | Preceding trials retained by UD, default 4. |
| `iif_mode` | `"classwise"` (default) or `"paper"` (literal printed indicator equation). |

For compatibility, explicit `crossvalidation_var["var"]` also supplies
repetition/trial IDs when the corresponding argument is omitted. Random folds
never substitute for real recording groups. CD and UD sort group IDs, so encode
chronology in the IDs. If repetition numbers reset each trial, combine trial
and repetition into a unique ID. Corresponding repetitions across classes share
an ID. Evaluate different subjects/sessions separately.

Geometric metrics use all supplied samples. Only CA/ACA/AE use cross-validation;
UD uses chronological adaptive retraining. Inputs are never modified. Default
classification uses LDA with automatic covariance shrinkage (`solver="lsqr",
shrinkage="auto"`). A custom `classifier=` must support scikit-learn cloning;
a fresh clone is fitted per fold/trial.

MAV and TT describe recordings and therefore give identical scores across
feature subsets. Their default strategy is filter, as is FE. All other defaults
are sequential. Objective directions describe conventional feature-space
quality and do not guarantee online usability; override with `objective=`.

## Definitions

Let N be the number of classes, D the feature dimension, mu_j the class centroid
and S_j its unbiased sample covariance. Class summaries are averaged equally.
Distance metrics select the nearest competing class before class averaging;
FE and VOR select the maximum competing-class value.

| Code | Definition | Objective |
| --- | --- | --- |
| RI | Average over classes and repetitions of half-Mahalanobis distance from held-out repetition centroid to the same-class centroid from all other repetitions, using the other-repetition covariance. | min |
| mwRI | Average half-Mahalanobis point distance to its repetition/class centroid, using whole-class covariance. Average points, classes, then repetitions. | min |
| swRI | Sample standard deviation (`ddof=1`) of the repetition means underlying mwRI. | min |
| swSI | Sample standard deviation of the within-trial SI scores. | min |
| MSA | Class-average geometric mean of covariance semi-axis lengths, `det(S_j)**(1/(2*D))`. Rank-deficient ellipsoids have zero volume. | min |
| CD | Class-average sum of Euclidean distances between successive repetition centroids. No division by number of transitions. | min |
| MAV | Absolute raw amplitude averaged across samples/channels/windows within class, then equally across classes. | max |
| SI | Half-Mahalanobis centroid distance using the reference class covariance. | max |
| mSI | SI using the mean covariance of both classes. | max |
| mwSI | Mean SI computed separately within each real trial. | max |
| BD | Standard Gaussian Bhattacharyya distance: `delta.T inv(S) delta/8 + log(det(S)/sqrt(det(S_i)*det(S_j)))/2`, where `S=(S_i+S_j)/2`. | max |
| KLD | Directed Gaussian KL divergence, with the reference class inverse covariance and compared class covariance in the trace term. | max |
| HD | Squared Gaussian Hellinger distance, `1-exp(-BD)` before nearest-class reduction. | max |
| FDR | Maximum across dimensions of squared centroid difference divided by summed class variances. | max |
| VOR | Product across dimensions of nonnegative overlap width / union width. Identical constant dimensions contribute 1. | min |
| FE | Maximum fraction of the two classes' points outside their inclusive overlap interval across feature dimensions; maximum competing class, then average reference classes. | max |
| TSM | `trace(inv(S_w) @ S_b)` using within-class residual scatter and class-size-weighted centroid scatter. Common class averaging factors cancel. | max |
| DS | `SI/(RI*MSA)`. | max |
| CDM | Minority fraction in mixed, non-linearly-separable adaptive terminal cells; partition variant below. | min |
| PU / rPU | PRISM cell purity / weighted-rescaled purity. | max |
| NS / rNS | PRISM within-cell nearest-neighbor separability / weighted-rescaled version. | max |
| CE / rCE | PRISM entropy certainty with local normalization / weighted-rescaled version. | max |
| C / rC | Explicit occupied-cell concentration variant / weighted-rescaled version. | max |
| CA | Mean held-out-fold classification accuracy in percent; folds weighted equally. | max |
| ACA | Mean fold percentage where prediction equals truth **or prediction equals no motion**. | max |
| AE | `100-ACA`, percent. | min |
| UD | Percent windows accepted during adaptive training: seed the first complete trial; predict later trials before adding correctly classified windows; retain accepted data from the preceding four trials by default. Seed windows count as accepted. | max |
| ICF | Fraction of samples whose nearest other sample belongs to a different class. | min |
| IIF | Sum of nearest same-class distances / sum of nearest different-class distances; equivalent to a ratio of means. | min |
| TT | Collection duration in ms: sum `window_size_ms+(n_recording-1)*window_increment_ms` across recordings, or explicit `training_time_ms`. | min |

### PRISM conventions and reference limitations

For B=0..31 by default, split each observed feature range into B+1 equal
intervals. Maximum endpoints occupy the final interval and constants interval
zero. Only occupied cells are stored. Cells are weighted by sample count.
For K observed classes with cell probabilities p_i, purity is
`sqrt(K/(K-1)*sum((p_i-1/K)**2))`; pure cells score 1.

NS excludes self by row identity and orders other samples in the cell by
Euclidean distance in the supplied feature units. It integrates same-class
proportions for every k=1..cell_size-1 against normalized k; singleton/pure cells
score 1. Distance ties follow original row index. NS is quadratic in cell size,
with chunked distance computation. Scale features deliberately for this and
other Euclidean-distance metrics.

Entropy certainty is `1+sum(p_i*log(p_i))/log(K)`, with pure cells assigned 1.
This logarithmic normalization keeps multiclass values in [0,1]; the provided
paper does not specify it. Unprefixed PRISM scores are trapezoidal AUCs against
normalized resolution. Rescaled scores apply weights `2**(-B)` and divide by
the maximum possible weighted AUC. A constant perfect curve scores 1.

**C/rC concentration variant:** per-resolution compactness is
`sum((n_cell/n_total)**2)`. This is an operational spread measure; Singh's
original compactness equation remains unverified. Enable explicitly with
`allow_approximations=True`.

**CDM partition variant:** min-max scale coordinates for numerical conditioning;
stop at homogeneous or hard-margin one-vs-rest linearly separable cells (checked
by linear programming). Otherwise split the widest nonconstant axis at its
median, keeping equal coordinates together. Stop at `cdm_min_samples` (default
`2*(D+1)`) or `cdm_max_depth` (default 20). Sum `cell_size-largest_class_count`
over remaining mixed cells and divide by total samples. These configurable
rules remain unverified against Kohn's original algorithm. Enable explicitly
with `allow_approximations=True`; report this as a variant in research comparisons.

### Formula ambiguities and numerical limits

The paper defines ACA as **active classification accuracy**, although the old
comment labelled it active error. ACA and the separate legacy AE are supported.
The printed BD logarithm has a negative sign; this implementation uses the
standard positive sign, preserving nonnegativity and `HD=1-exp(-BD)`.

IIF's prose describes separate within/between-class neighbors, while the printed
equation filters a single nearest-neighbor set by label agreement. The default
follows the prose and Ho/Basu ratio. With `iif_mode="paper"`, it computes
`sum(d_nearest[same_class])/sum(d_nearest[different_class])`. This alternative
can be infinite when all nearest neighbors share their query's class; choose
an appropriate objective explicitly for that version. Self is excluded even
with duplicate coordinates. Equal-distance ties use original row index.

Covariance eigenvalues are floored relative to their largest magnitude, using
`regularization` (default 1e-9). A completely zero covariance uses the overall
feature variance, with a unit lower bound, as its scale. MSA retains the
unregularized eigenvalues so flat ellipsoids have zero volume. Covariance
calculations require two samples per class; RI also needs two same-class
samples outside each repetition. CD only requires centroids. Positive-over-zero
ratios return infinity; 0/0 raises an informative error. NaNs, infinities,
inconsistent input lengths and invalid folds are rejected.

Raw MAV averages overlapping samples repeatedly when overlapping windows are
provided; use nonoverlapping windows for unweighted recording amplitude. TT
measures data collection time, not estimator fit runtime. Pauses and discarded
trailing samples are only represented by an explicit measured duration.

Numerical results intentionally change from the old selector: MSA uses axis
lengths instead of PCA direction-vector products; AE compares the actual
no-motion label; FE includes overlap endpoints and divides by N; RI averages
by the actual repetition count. MSA now defaults to minimization. Geometric
metrics use the full supplied data rather than overlapping training folds.

## Real-data example and verification

Run `python examples/feature_selection_metrics.py` from the repository root.
It loads `OneSubjectMyoDataset` through `libemg.datasets` (downloading the public
example dataset if absent), selects three classes/trials and two channels, and
extracts MAV, WL and ZC from nonoverlapping 250 ms windows. It evaluates all 34
metrics and runs two-step selection for each, explicitly enabling and labelling
the three variants.

```python
from examples.feature_selection_metrics import load_example
from libemg.feature_selector import FeatureSelector

features, labels, crossvalidation, options = load_example()
selector = FeatureSelector(allow_approximations=True)
scores = selector.evaluate_metrics(None, features, labels, crossvalidation, **options)
results, order = selector.run_selection(
    features, "mwRI", labels, crossvalidation, **options,
)
```

`pytest tests/test_feature_selector.py -q` covers analytic values, formula
identities, overlap boundaries, singular covariance, arbitrary labels,
classification/adaptive behavior, reproducibility, immutability, ordering and
input validation. Both selection strategies run for every metric on real data.
Integration tests stage a small subset of the repository's existing Myo
recordings into the `OneSubjectMyoDataset` layout and use its real loader;
they need no network access or separately downloaded dataset.

## Sources

- Nawfel, Englehart and Scheme (2021), [A Multi-Variate Approach to Predicting Myoelectric Control Usability](https://doi.org/10.1109/TNSRE.2021.3094324), Appendices A-E and Section II-C/Figure 1.
- Nawfel (2020), [author's thesis](https://unbscholar.lib.unb.ca/items/679a90d2-f44a-4e9e-b0cd-bf62fe8474ba), Appendix A.2.7 (FDR) and A.6.1 (collection duration). This API explicitly distinguishes hop from overlap.
- Singh (2003), [Multiresolution Estimates of Classification Complexity](https://sci2s.ugr.es/keel/pdf/specific/articulo/IEEEPAMI03.pdf), Section 2 (PRISM purity, neighborhoods and weighting).
- Singh (2003), [PRISM - A Novel Framework for Pattern Recognition](https://doi.org/10.1007/s10044-002-0186-2), deferred original PRISM reference; full algorithm text was unavailable during implementation.
- Kohn, Nakano and Silva (1996), [A Class Discriminality Measure Based on Feature Space Partitioning](https://doi.org/10.1016/0031-3203%2895%2900122-0), deferred CDM reference; original partition rules remain unverified.
