import pytest
import pickle
import numpy as np
from sklearn.metrics import *
from libemg.offline_metrics import OfflineMetrics, UsabilityMetrics

"""
Validation testing to ensure that our offline metrics are the same as sklearn. 
These tests are valid for now. 
"""
@pytest.fixture(scope='session') 
def om():
    return OfflineMetrics()

@pytest.fixture(scope='session')
def y_true():
    file = open('tests/data/test_labels','rb')
    return pickle.load(file)

@pytest.fixture(scope='session')
def y_predictions():
    file = open('tests/data/predictions','rb')
    return pickle.load(file)

def test_CA(om, y_true, y_predictions):
    assert om.get_CA(y_true, y_predictions) == accuracy_score(y_true, y_predictions)

def test_AER(om, y_true, y_predictions):
    null_label = 2
    null_idxs = np.where(y_predictions == null_label)
    y_true_cop = y_true.copy()
    aer = om.get_AER(y_true_cop, y_predictions, null_label)
    y_true_cop = np.delete(y_true_cop, null_idxs)
    y_predictions = np.delete(y_predictions, null_idxs)
    assert aer == (1- accuracy_score(y_true_cop, y_predictions))

def test_REJ_RATE(om, y_predictions):
    y_pred = y_predictions.copy()
    for i in range(0, len(y_pred), 2):
        y_pred[i] = -1 
    assert om.get_REJ_RATE(y_pred) == 0.5

def test_CONF_MAT(om, y_true, y_predictions):
    c_1 = om.get_CONF_MAT(y_true, y_predictions)
    c_2 = confusion_matrix(y_true, y_predictions)
    for r in range(0,len(c_1)):
        for c in range(0,len(c_2)):
            assert c_1[r,c] == c_2[r,c]

def test_RECALL(om, y_true, y_predictions):
    assert om.get_RECALL(y_true, y_predictions) == recall_score(y_true, y_predictions, average='weighted')

def test_PREC(om, y_true, y_predictions):
    # Assuming there is a rounding error
    assert om.get_PREC(y_true, y_predictions) - precision_score(y_true, y_predictions, average='weighted') < 0.0000000001

def test_F1(om, y_true, y_predictions):
    # Assuming there is a rounding error
    assert om.get_F1(y_true, y_predictions) - f1_score(y_true, y_predictions, average='weighted') < 0.0000000001

def test_REMOVE(om):
    preds = np.array([0,1,-1,-1,2,2,0,0,-1])
    labels = np.array([0,1,0,0,2,2,0,0,2])
    preds, labels = om._ignore_rejected(preds, labels)
    assert np.alltrue(preds == np.array([0,1,2,2,0,0]))
    assert np.alltrue(preds == np.array([0,1,2,2,0,0]))

def test_REMOVE2(om):
    preds = np.array([0,1,2,3,4,5,6,7,8,9,0])
    labels = np.array([0,1,2,3,4,5,6,7,8,9,0])
    preds2, labels2 = om._ignore_rejected(preds, labels)
    assert np.alltrue(preds2 == preds)
    assert np.alltrue(labels2 == labels)


# =============================================================================
# UsabilityMetrics tests
# Synthetic data: 3 well-separated Gaussian classes, 4 features, 3 repetitions.
# =============================================================================

RNG = np.random.default_rng(42)
N_PER_CLASS_PER_REP = 30
N_CLASSES = 3
N_REPS = 3
N_FEATURES = 4

# Class centroids well separated in feature space
_CENTROIDS = np.array([[0.0, 0.0, 0.0, 0.0],
                        [5.0, 5.0, 5.0, 5.0],
                        [10.0, 10.0, 10.0, 10.0]])


def _make_data():
    """Return (X, y, reps) for 3 well-separated classes × 3 reps × 30 samples."""
    X_parts, y_parts, r_parts = [], [], []
    for c, mu in enumerate(_CENTROIDS):
        for r in range(N_REPS):
            samples = RNG.normal(loc=mu, scale=0.3, size=(N_PER_CLASS_PER_REP, N_FEATURES))
            X_parts.append(samples)
            y_parts.append(np.full(N_PER_CLASS_PER_REP, c))
            r_parts.append(np.full(N_PER_CLASS_PER_REP, r))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    reps = np.concatenate(r_parts)
    return X, y, reps


def _make_mixed_data():
    """Return (X, y, reps) for 3 overlapping classes (poor separability)."""
    X_parts, y_parts, r_parts = [], [], []
    for c, mu in enumerate(_CENTROIDS):
        for r in range(N_REPS):
            # High noise so classes overlap
            samples = RNG.normal(loc=mu, scale=4.0, size=(N_PER_CLASS_PER_REP, N_FEATURES))
            X_parts.append(samples)
            y_parts.append(np.full(N_PER_CLASS_PER_REP, c))
            r_parts.append(np.full(N_PER_CLASS_PER_REP, r))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    reps = np.concatenate(r_parts)
    return X, y, reps


@pytest.fixture(scope='module')
def um():
    return UsabilityMetrics()


@pytest.fixture(scope='module')
def sep_data():
    return _make_data()


@pytest.fixture(scope='module')
def mixed_data():
    return _make_mixed_data()


# --- variability ---

def test_RI_low_for_consistent_reps(um, sep_data):
    X, y, reps = sep_data
    ri = um.get_RI(X, y, reps)
    assert ri >= 0, "RI must be non-negative"
    assert ri < 1.0, "RI should be small for tight, consistent repetitions"


def test_MSA_small_for_compact_classes(um, sep_data, mixed_data):
    X_sep, y_sep, _ = sep_data
    X_mix, y_mix, _ = mixed_data
    msa_sep = um.get_MSA(X_sep, y_sep)
    msa_mix = um.get_MSA(X_mix, y_mix)
    assert msa_sep < msa_mix, "Compact classes should have smaller MSA than spread-out ones"


def test_mwRI_non_negative(um, sep_data):
    X, y, reps = sep_data
    assert um.get_mwRI(X, y, reps) >= 0


def test_swRI_non_negative(um, sep_data):
    X, y, reps = sep_data
    assert um.get_swRI(X, y, reps) >= 0


def test_CD_zero_for_static_data(um):
    """CD should be ~0 when all reps share the same centroid."""
    X = np.tile(np.eye(4), (30, 1))  # identical rows repeated
    y = np.repeat([0, 1, 2, 3], [30, 30, 30, 30])
    reps = np.tile(np.repeat([0, 1, 2], 10), 4)
    cd = um.get_CD(X, y, reps)
    assert cd < 0.01, "Centroid drift should be near zero for static data"


def test_MAV_positive(um, sep_data):
    X, y, _ = sep_data
    assert um.get_MAV(X, y) > 0


def test_swSI_non_negative(um, sep_data):
    X, y, reps = sep_data
    assert um.get_swSI(X, y, reps) >= 0


# --- separability ---

def test_SI_positive_for_separable(um, sep_data):
    X, y, _ = sep_data
    si = um.get_SI(X, y)
    assert si > 0, "SI should be positive for separable classes"


def test_SI_higher_for_well_separated(um, sep_data, mixed_data):
    X_sep, y_sep, _ = sep_data
    X_mix, y_mix, _ = mixed_data
    assert um.get_SI(X_sep, y_sep) > um.get_SI(X_mix, y_mix)


def test_mSI_positive(um, sep_data):
    X, y, _ = sep_data
    assert um.get_mSI(X, y) > 0


def test_BD_positive(um, sep_data):
    X, y, _ = sep_data
    assert um.get_BD(X, y) > 0, "BD should be positive for distinct classes"


def test_KLD_positive(um, sep_data):
    X, y, _ = sep_data
    assert um.get_KLD(X, y) > 0


def test_HD_in_range(um, sep_data, mixed_data):
    X_sep, y_sep, _ = sep_data
    X_mix, y_mix, _ = mixed_data
    hd_sep = um.get_HD(X_sep, y_sep)
    hd_mix = um.get_HD(X_mix, y_mix)
    assert 0.0 <= hd_sep <= 1.0, "HD must be in [0, 1]"
    # HD → 1 for very different distributions, HD → 0 for identical ones
    assert hd_sep >= hd_mix, "Well-separated classes should have higher HD than mixed"


def test_VOR_lower_for_separated(um, sep_data, mixed_data):
    X_sep, y_sep, _ = sep_data
    X_mix, y_mix, _ = mixed_data
    vor_sep = um.get_VOR(X_sep, y_sep)
    vor_mix = um.get_VOR(X_mix, y_mix)
    assert vor_sep <= vor_mix, "Well-separated classes should have smaller VOR"


def test_FE_near_one_for_separated(um, sep_data):
    X, y, _ = sep_data
    fe = um.get_FE(X, y)
    assert fe > 0.9, "FE should approach 1.0 for fully separable data"


def test_TSM_positive(um, sep_data):
    X, y, _ = sep_data
    assert um.get_TSM(X, y) > 0


def test_FDR_positive(um, sep_data):
    X, y, _ = sep_data
    assert um.get_FDR(X, y) > 0


def test_mwSI_positive(um, sep_data):
    X, y, reps = sep_data
    assert um.get_mwSI(X, y, reps) > 0


def test_DS_positive(um, sep_data):
    X, y, reps = sep_data
    assert um.get_DS(X, y, reps) > 0


# --- classification ---

def test_ACA_perfect_predictions(um):
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert um.get_ACA(y_true, y_pred, null_class=0) == 1.0


def test_ACA_null_predictions_count_as_correct(um):
    y_true = np.array([1, 2, 1, 2])
    # All predictions are null class; none are truly correct but ACA treats them as OK
    y_pred = np.array([0, 0, 0, 0])
    aca = um.get_ACA(y_true, y_pred, null_class=0)
    assert aca == 1.0


def test_CA_equals_accuracy(um):
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    expected = np.mean(y_true == y_pred)
    assert um.get_CA(y_true, y_pred) == pytest.approx(expected)


def test_UD_equals_accuracy(um):
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    expected = np.mean(y_true == y_pred)
    assert um.get_UD(y_true, y_pred) == pytest.approx(expected)


# --- neighbourhood ---

def test_ICF_near_zero_for_separated(um, sep_data):
    X, y, _ = sep_data
    icf = um.get_ICF(X, y)
    assert icf < 0.05, "ICF should be near 0 for well-separated classes"


def test_ICF_higher_for_mixed(um, mixed_data):
    X, y, _ = mixed_data
    icf = um.get_ICF(X, y)
    assert icf > 0.1, "ICF should be noticeably above 0 for overlapping classes"


def test_IIF_lower_for_separated(um, sep_data, mixed_data):
    X_sep, y_sep, _ = sep_data
    X_mix, y_mix, _ = mixed_data
    iif_sep = um.get_IIF(X_sep, y_sep)
    iif_mix = um.get_IIF(X_mix, y_mix)
    # Well-separated: tiny intra-class NN / large inter-class NN → small IIF
    # Overlapping: intra ≈ inter → larger IIF
    assert iif_sep < iif_mix, "IIF should be smaller for well-separated classes"


# --- stubs raise NotImplementedError ---

@pytest.mark.parametrize("metric_name", [
    'CDM', 'PU', 'NS', 'CE', 'C', 'rPU', 'rNS', 'rCE', 'rC'
])
def test_stubs_raise(um, sep_data, metric_name):
    X, y, _ = sep_data
    method = getattr(um, 'get_' + metric_name)
    with pytest.raises(NotImplementedError):
        method(X, y)


# --- listing and extraction ---

def test_available_metrics_count(um):
    metrics = um.get_available_usability_metrics()
    assert len(metrics) == 32, "Should list all 32 metrics from Nawfel et al. (2021)"


def test_extract_usability_metrics(um, sep_data):
    X, y, reps = sep_data
    results = um.extract_usability_metrics(['SI', 'BD', 'FE', 'ICF'], X, y)
    assert set(results.keys()) == {'SI', 'BD', 'FE', 'ICF'}
    for v in results.values():
        assert np.isfinite(v)


def test_extract_requires_reps(um, sep_data):
    X, y, _ = sep_data
    with pytest.raises(ValueError):
        um.extract_usability_metrics(['RI'], X, y, reps=None)