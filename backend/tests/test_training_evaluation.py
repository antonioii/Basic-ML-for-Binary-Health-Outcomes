import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from backend.dataset_store import DatasetEntry
from backend.services import training


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.threshold_: float | None = None

    def fit(self, X, y):  # type: ignore[override]
        X_array = np.asarray(X)
        self.threshold_ = float(np.median(X_array[:, 0]))
        return self

    def predict(self, X):  # type: ignore[override]
        X_array = np.asarray(X)
        threshold = self.threshold_ if self.threshold_ is not None else 0.0
        return (X_array[:, 0] >= threshold).astype(int)


def _build_dataset_entry() -> DatasetEntry:
    rows = 30
    df = pd.DataFrame({
        'id': np.arange(rows),
        'feature_a': np.linspace(0, 10, rows),
        'feature_b': np.random.RandomState(42).rand(rows),
        'target': [0] * (rows // 2) + [1] * (rows - rows // 2),
    })
    return DatasetEntry(
        dataset_id='test',
        file_name='test.csv',
        raw_df=df,
        id_column='id',
        target_column='target',
        feature_columns=['feature_a', 'feature_b'],
    )


def test_supervised_training_returns_std_metrics() -> None:
    entry = _build_dataset_entry()
    results = training.train_supervised_models(
        entry,
        [training.CLASSIFIER_NAMES['Logistic Regression']],
        svm_flexibility='Medium (Balanced)',
        processing_mode='light',
        custom_hyperparameters=None,
    )
    assert results, 'Expected at least one trained model'
    logistic = results[0]
    metrics = logistic['metrics']
    assert metrics is not None
    assert metrics['auc'] is not None
    assert metrics['roc_curve'] is not None
    assert metrics['auc_std'] is not None
    assert metrics['accuracy_std'] is not None
    assert metrics['sensitivity_std'] is not None


def test_auc_is_omitted_when_classifier_has_no_scores() -> None:
    df = pd.DataFrame({
        'feature': np.linspace(0, 1, 20),
    })
    y = pd.Series([0] * 10 + [1] * 10)
    pipeline = Pipeline([('classifier', ThresholdClassifier())])
    cv = training._build_cv(y)  # type: ignore[attr-defined]
    result = training._evaluate_classifier('Threshold', pipeline, df, y, cv)
    assert result['metrics']['auc'] is None
    assert result['roc_curve'] is None
    assert result['status_message'] is not None
