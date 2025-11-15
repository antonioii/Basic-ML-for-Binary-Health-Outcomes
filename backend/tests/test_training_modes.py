from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import pytest

from backend.services import training


def test_prepare_param_grid_light_returns_default():
    model_name = training.CLASSIFIER_NAMES['Logistic Regression']
    default = {'classifier__C': [0.01, 0.1]}
    result = training._prepare_param_grid(model_name, 'light', None, default)  # type: ignore[attr-defined]
    assert result['classifier__C'] == [0.01, 0.1]


def test_prepare_param_grid_hard_selects_hard_grid():
    model_name = training.CLASSIFIER_NAMES['Random Forest']
    default = {'classifier__n_estimators': [200]}
    hard = {'classifier__n_estimators': [200, 400, 600]}
    result = training._prepare_param_grid(model_name, 'hard', None, default, hard)  # type: ignore[attr-defined]
    assert result['classifier__n_estimators'] == [200, 400, 600]


def test_prepare_param_grid_custom_converts_and_prefixes():
    model_name = training.CLASSIFIER_NAMES['Gradient Boosting']
    custom_grid = {
        model_name: {
            'learning_rate': ['0.01', '0.1'],
            'max_depth': ['3', '5'],
        }
    }
    default = {'classifier__learning_rate': [0.05]}
    result = training._prepare_param_grid(model_name, 'custom', custom_grid, default)  # type: ignore[attr-defined]
    assert result['classifier__learning_rate'] == [0.01, 0.1]
    assert result['classifier__max_depth'] == [3, 5]


def test_normalize_balance_strategy_handles_disabled() -> None:
    assert training._normalize_balance_strategy(None) is None  # type: ignore[attr-defined]
    assert training._normalize_balance_strategy({'enabled': False, 'method': 'smote'}) is None  # type: ignore[attr-defined]
    cfg = {'enabled': True, 'method': 'OVERSAMPLE'}
    assert training._normalize_balance_strategy(cfg) == 'oversample'  # type: ignore[attr-defined]


def test_build_classifier_pipeline_attaches_sampler() -> None:
    pytest.importorskip('imblearn')
    preprocessor = ColumnTransformer([('identity', 'passthrough', [0])])
    classifier = LogisticRegression()
    pipeline = training._build_classifier_pipeline(preprocessor, classifier, 'smote')  # type: ignore[attr-defined]
    assert 'sampler' in pipeline.named_steps
