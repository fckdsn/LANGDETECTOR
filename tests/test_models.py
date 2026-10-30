import numpy as np
import path_setup  # noqa: F401
import pytest
from scipy.sparse import csr_matrix

from models import (
    ALL_MODELS,
    DEFAULT_MODELS,
    FAST_MODELS,
    MODEL_REGISTRY,
    SoftmaxSVC,
    build_model,
    get_spec,
    resolve_model_names,
)


def toy_dataset(n_classes=4, per_class=25, block=6):
    """Zbior liniowo separowalny: kazda klasa ma wlasny blok cech."""
    rng = np.random.default_rng(0)
    n_features = n_classes * block
    rows, labels = [], []
    for cls in range(n_classes):
        chunk = np.zeros((per_class, n_features))
        chunk[:, cls * block:(cls + 1) * block] = rng.random((per_class, block)) + 0.5
        rows.append(chunk)
        labels.extend([cls] * per_class)
    return csr_matrix(np.vstack(rows)), np.array(labels)


def _test_overrides(name):
    """Zmniejsza modele do rozmiaru zbioru zabawkowego, by testy byly szybkie."""
    if name == 'random_forest':
        return {'n_estimators': 20}
    if name.startswith('mlp'):
        return {'hidden_layer_sizes': (32,), 'max_iter': 400,
                'batch_size': 16, 'early_stopping': False}
    return {}


# --- Rejestr modeli ---------------------------------------------------------

def test_registry_covers_required_algorithm_families():
    """Temat pracy wymaga Random Forest, SVM, Naive Bayes i sieci neuronowej."""
    families = {spec.family for spec in MODEL_REGISTRY.values()}
    assert {'Zespol drzew', 'SVM', 'Naive Bayes', 'Siec neuronowa'} <= families


def test_default_models_are_registered():
    for name in DEFAULT_MODELS:
        assert name in MODEL_REGISTRY


def test_get_spec_rejects_unknown_model():
    with pytest.raises(KeyError):
        get_spec('nie_ma_takiego_modelu')


def test_resolve_model_names_aliases():
    assert resolve_model_names(None) == DEFAULT_MODELS
    assert resolve_model_names('all') == ALL_MODELS
    assert resolve_model_names(['fast']) == FAST_MODELS
    assert resolve_model_names(['naive_bayes', 'linear_svm']) == ('naive_bayes', 'linear_svm')


def test_resolve_model_names_validates_entries():
    with pytest.raises(KeyError):
        resolve_model_names(['naive_bayes', 'brak'])


def test_build_model_applies_overrides():
    model = build_model('random_forest', n_estimators=7)
    assert model.n_estimators == 7


# --- Wspolny interfejs wszystkich algorytmow --------------------------------

@pytest.mark.parametrize('name', ALL_MODELS)
def test_every_model_fits_and_predicts(name):
    X, y = toy_dataset()
    model = build_model(name, **_test_overrides(name)).fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == y.shape

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (X.shape[0], len(np.unique(y)))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert (probabilities >= 0).all()

    # Zbior jest separowalny, wiec kazdy sensowny model powinien go opanowac.
    assert (predictions == y).mean() > 0.9


@pytest.mark.parametrize('name', ALL_MODELS)
def test_argmax_of_proba_matches_predict(name):
    X, y = toy_dataset(n_classes=3, per_class=20)
    model = build_model(name, **_test_overrides(name)).fit(X, y)
    argmax_classes = model.classes_[np.argmax(model.predict_proba(X), axis=1)]
    assert (argmax_classes == model.predict(X)).mean() > 0.95


# --- SoftmaxSVC -------------------------------------------------------------

def test_softmax_svc_exposes_decision_function():
    X, y = toy_dataset(n_classes=3, per_class=20)
    model = SoftmaxSVC().fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (X.shape[0], 3)


def test_softmax_svc_proba_ranking_follows_margins():
    """Softmax jest monotoniczny wobec marginesow, wiec ranking klas sie zgadza."""
    X, y = toy_dataset(n_classes=3, per_class=20)
    model = SoftmaxSVC().fit(X, y)
    margins = model.decision_function(X)
    probabilities = model.predict_proba(X)
    assert (np.argmax(margins, axis=1) == np.argmax(probabilities, axis=1)).all()


def test_softmax_svc_handles_binary_problem():
    X, y = toy_dataset(n_classes=2, per_class=20)
    model = SoftmaxSVC().fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (X.shape[0], 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
