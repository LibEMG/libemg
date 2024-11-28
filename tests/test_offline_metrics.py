import pytest
import pickle
import numpy as np
from sklearn.metrics import *
from libemg.offline_metrics import OfflineMetrics

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