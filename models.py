"""Zestaw algorytmow klasyfikacji porownywanych w projekcie.

Kazdy model wystawia jednolity interfejs (fit / predict / predict_proba), co
pozwala porownywac je w petli tym samym kodem ewaluacyjnym.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from config import RANDOM_STATE


class SoftmaxSVC(BaseEstimator, ClassifierMixin):
    """Liniowy SVM z rozkladem prawdopodobienstwa.

    ``LinearSVC`` skaluje sie na rzadkich macierzach o dziesiatkach tysiecy
    cech, ale nie udostepnia ``predict_proba``. Kalibracja Platta
    (``CalibratedClassifierCV``) wymagalaby walidacji krzyzowej dla kazdej z
    ~100 klas, dlatego marginesy decyzyjne przeliczamy funkcja softmax. Wynik
    jest monotoniczny wobec marginesow, wiec ranking klas i krzywe ROC sa
    identyczne jak dla surowego SVM, a interfejs pozostaje zgodny z reszta
    modeli.
    """

    def __init__(self, C=1.0, temperature=2.0, class_weight=None,
                 max_iter=3000, random_state=RANDOM_STATE):
        self.C = C
        self.temperature = temperature
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator_ = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            dual=True,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        scores = self.estimator_.decision_function(X)
        if scores.ndim == 1:
            # Problem binarny: LinearSVC zwraca jeden margines na probke.
            scores = np.column_stack([-scores, scores])
        scores = scores * self.temperature
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


@dataclass
class ModelSpec:
    """Opis jednego algorytmu w porownaniu."""

    name: str
    label: str
    family: str
    builder: Callable[..., BaseEstimator]
    notes: str = ""
    defaults: dict = field(default_factory=dict)

    def build(self, **overrides):
        params = {**self.defaults, **overrides}
        return self.builder(**params)


MODEL_REGISTRY = {
    'naive_bayes': ModelSpec(
        name='naive_bayes',
        label='Naive Bayes (Multinomial)',
        family='Naive Bayes',
        builder=MultinomialNB,
        defaults={'alpha': 0.05},
        notes='Model generatywny, zalozenie niezaleznosci cech; najszybszy w treningu.',
    ),
    'complement_nb': ModelSpec(
        name='complement_nb',
        label='Naive Bayes (Complement)',
        family='Naive Bayes',
        builder=ComplementNB,
        defaults={'alpha': 0.3},
        notes='Wariant NB odporniejszy na niezbalansowane liczebnosci klas.',
    ),
    'logistic_regression': ModelSpec(
        name='logistic_regression',
        label='Regresja logistyczna',
        family='Model liniowy',
        builder=LogisticRegression,
        defaults={
            'C': 10.0,
            'max_iter': 300,
            'solver': 'lbfgs',
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
        },
        notes='Liniowa granica decyzyjna, softmax na wyjsciu; punkt odniesienia.',
    ),
    'linear_svm': ModelSpec(
        name='linear_svm',
        label='SVM (liniowy)',
        family='SVM',
        builder=SoftmaxSVC,
        defaults={'C': 1.0, 'temperature': 2.0},
        notes='Maksymalizacja marginesu; skuteczny w przestrzeniach wysokowymiarowych.',
    ),
    'random_forest': ModelSpec(
        name='random_forest',
        label='Random Forest',
        family='Zespol drzew',
        builder=RandomForestClassifier,
        defaults={
            'n_estimators': 300,
            'max_depth': 40,
            'min_samples_leaf': 1,
            'class_weight': 'balanced_subsample',
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
        },
        notes='Zespol drzew decyzyjnych, nieliniowy, odporny na przeuczenie.',
    ),
    'mlp': ModelSpec(
        name='mlp',
        label='Siec neuronowa (MLP 256)',
        family='Siec neuronowa',
        builder=MLPClassifier,
        defaults={
            'hidden_layer_sizes': (256,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 1e-4,
            'batch_size': 256,
            'learning_rate_init': 1e-3,
            'max_iter': 60,
            'early_stopping': True,
            'n_iter_no_change': 5,
            'validation_fraction': 0.1,
            'random_state': RANDOM_STATE,
        },
        notes='Perceptron wielowarstwowy z jedna warstwa ukryta.',
    ),
    'mlp_deep': ModelSpec(
        name='mlp_deep',
        label='Siec neuronowa (MLP 512-256)',
        family='Siec neuronowa',
        builder=MLPClassifier,
        defaults={
            'hidden_layer_sizes': (512, 256),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 1e-4,
            'batch_size': 256,
            'learning_rate_init': 1e-3,
            'max_iter': 80,
            'early_stopping': True,
            'n_iter_no_change': 5,
            'validation_fraction': 0.1,
            'random_state': RANDOM_STATE,
        },
        notes='Glebsza siec - sprawdza, czy dodatkowa warstwa poprawia wynik.',
    ),
}

# Zestaw uzywany w pelnym porownaniu opisanym w pracy.
DEFAULT_MODELS = (
    'naive_bayes',
    'complement_nb',
    'logistic_regression',
    'linear_svm',
    'random_forest',
    'mlp',
)

# Modele o krotkim czasie treningu - do szybkiego sprawdzenia potoku.
FAST_MODELS = ('naive_bayes', 'complement_nb', 'linear_svm')

ALL_MODELS = tuple(MODEL_REGISTRY)


def get_spec(name):
    if name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY)
        raise KeyError(f"Nieznany model '{name}'. Dostepne: {available}")
    return MODEL_REGISTRY[name]


def build_model(name, **overrides):
    return get_spec(name).build(**overrides)


def resolve_model_names(names=None):
    """Zamienia nazwe zestawu ('all', 'fast', 'default') lub liste na krotke."""
    if not names:
        return DEFAULT_MODELS
    if isinstance(names, str):
        names = [names]
    if len(names) == 1:
        alias = names[0].lower()
        if alias == 'all':
            return ALL_MODELS
        if alias == 'fast':
            return FAST_MODELS
        if alias == 'default':
            return DEFAULT_MODELS
    for name in names:
        get_spec(name)
    return tuple(names)
