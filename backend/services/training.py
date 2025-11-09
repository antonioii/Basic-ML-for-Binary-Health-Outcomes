from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..dataset_store import DatasetEntry

CLASSIFIER_NAMES = {
    'Logistic Regression': 'Logistic Regression',
    'Elastic Net (Logistic Regression)': 'Elastic Net (Logistic Regression)',
    'K-Nearest Neighbors (KNN)': 'K-Nearest Neighbors (KNN)',
    'Support Vector Machine (SVM)': 'Support Vector Machine (SVM)',
    'Random Forest': 'Random Forest',
    'Gradient Boosting': 'Gradient Boosting',
    'XGBoost': 'XGBoost',
    'LightGBM': 'LightGBM',
    'CatBoost': 'CatBoost',
    'Naive Bayes (Gaussian)': 'Naive Bayes (Gaussian)',
    'Voting Classifier': 'Voting Classifier',
    'Stacking Classifier': 'Stacking Classifier',
    'K-Means Clustering': 'K-Means Clustering',
}

BASE_SUPERVISED_MODELS = [
    CLASSIFIER_NAMES['Logistic Regression'],
    CLASSIFIER_NAMES['Elastic Net (Logistic Regression)'],
    CLASSIFIER_NAMES['K-Nearest Neighbors (KNN)'],
    CLASSIFIER_NAMES['Support Vector Machine (SVM)'],
    CLASSIFIER_NAMES['Random Forest'],
    CLASSIFIER_NAMES['Gradient Boosting'],
    CLASSIFIER_NAMES['XGBoost'],
    CLASSIFIER_NAMES['LightGBM'],
    CLASSIFIER_NAMES['CatBoost'],
    CLASSIFIER_NAMES['Naive Bayes (Gaussian)'],
]

ENSEMBLE_MODELS = [
    CLASSIFIER_NAMES['Voting Classifier'],
    CLASSIFIER_NAMES['Stacking Classifier'],
]

RANDOM_STATE = 42


class MissingDependencyError(RuntimeError):
    def __init__(self, model_name: str, package_name: str):
        message = (
            f"{model_name} skipped because optional dependency '{package_name}' is not installed. "
            f"Install it with `pip install {package_name}` to enable this model."
        )
        super().__init__(message)
        self.model_name = model_name
        self.package_name = package_name


def _format_hyperparameters(best_params: Dict[str, Any], extras: Dict[str, Any] | None = None) -> Dict[str, Any]:
    formatted = {
        key.split('__', 1)[1] if '__' in key else key: value
        for key, value in best_params.items()
    }
    if extras:
        formatted.update(extras)
    return formatted


def _build_status_only_result(name: str, message: str) -> Dict[str, Any]:
    return {
        'name': name,
        'metrics': None,
        'roc_curve': None,
        'k_means_result': None,
        'hyperparameters': {},
        'status_message': message,
    }


def _import_optional_estimator(module_name: str, class_name: str, model_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(model_name, module_name) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise MissingDependencyError(model_name, module_name) from exc


def _sanitize_estimator_name(name: str, existing: set[str]) -> str:
    base = ''.join(ch for ch in name.lower() if ch.isalnum()) or 'model'
    candidate = base
    suffix = 1
    while candidate in existing:
        suffix += 1
        candidate = f'{base}_{suffix}'
    existing.add(candidate)
    return candidate


def _collect_base_estimators(
    selected_models: List[str],
    trained_pipelines: Dict[str, Pipeline],
) -> List[Tuple[str, BaseEstimator]]:
    estimators: List[Tuple[str, BaseEstimator]] = []
    existing_names: set[str] = set()
    seen_models: set[str] = set()
    for model in selected_models:
        if model in seen_models or model not in BASE_SUPERVISED_MODELS:
            continue
        seen_models.add(model)
        pipeline = trained_pipelines.get(model)
        if pipeline is None:
            continue
        estimator = clone(pipeline.named_steps['classifier'])
        estimator_name = _sanitize_estimator_name(model, existing_names)
        estimators.append((estimator_name, estimator))
    return estimators


def _split_feature_types(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
    binary_cols: List[str] = []
    continuous_cols: List[str] = []
    for column in columns:
        series = df[column]
        numeric_series = pd.to_numeric(series, errors='coerce')
        unique_values = numeric_series.dropna().unique()
        if len(unique_values) == 0:
            continue
        if set(unique_values).issubset({0, 1}):
            binary_cols.append(column)
        elif numeric_series.dtype.kind in {'i', 'u', 'f'}:
            continuous_cols.append(column)
        else:
            binary_cols.append(column)
    return {
        'binary': binary_cols,
        'continuous': continuous_cols,
    }


def _build_preprocessor(df: pd.DataFrame, columns: List[str]) -> ColumnTransformer:
    feature_types = _split_feature_types(df, columns)
    transformers = []
    if feature_types['continuous']:
        transformers.append(
            ('continuous', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), feature_types['continuous'])
        )
    if feature_types['binary']:
        transformers.append(
            ('binary', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
            ]), feature_types['binary'])
        )
    if not transformers:
        transformers.append(
            ('default', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
            ]), columns)
        )
    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)


def _build_cv(y: pd.Series) -> StratifiedKFold:
    class_counts = y.value_counts()
    max_splits = min(int(class_counts.min()), len(y), 10)
    if max_splits < 2:
        raise ValueError('Each class must contain at least two observations to perform stratified cross-validation.')
    return StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=42)


def _evaluate_classifier(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    hyperparameters: Dict[str, Union[str, int, float, None]] | None = None,
) -> Dict:
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method='predict')
    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        y_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
    else:
        decision = cross_val_predict(pipeline, X, y, cv=cv, method='decision_function')
        y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = recall_score(y, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y, y_pred, zero_division=0)
    vpn = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_value = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)

    return {
        'name': name,
        'metrics': {
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
            },
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'vpp': float(precision),
            'vpn': float(vpn),
            'f1_score': float(f1),
            'auc': float(auc_value),
            'accuracy': float(accuracy),
        },
        'roc_curve': [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr, tpr)],
        'hyperparameters': hyperparameters or {},
    }


def _grid_search(
    pipeline: Pipeline,
    param_grid: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Pipeline, Dict[str, Union[str, int, float]]]:
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def _train_logistic(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    preprocessor = _build_preprocessor(df, columns)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, solver='lbfgs')),
    ])
    param_grid = {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    classifier: LogisticRegression = best_pipeline.named_steps['classifier']
    hyperparameters = _format_hyperparameters(best_params, {
        'penalty': str(classifier.penalty),
        'solver': str(classifier.solver),
    })
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Logistic Regression'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    return result, best_pipeline


def _train_knn(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    preprocessor = _build_preprocessor(df, columns)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier()),
    ])
    n_samples = len(df)
    max_k = max(1, min(n_samples - 1, 51))
    k_values = [k for k in range(1, max_k + 1, 2)]
    param_grid = {'classifier__n_neighbors': k_values}
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    classifier: KNeighborsClassifier = best_pipeline.named_steps['classifier']
    hyperparameters = _format_hyperparameters(best_params, {
        'metric': str(classifier.metric),
        'weights': str(classifier.weights),
    })
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['K-Nearest Neighbors (KNN)'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    return result, best_pipeline


def _train_svm(df: pd.DataFrame, columns: List[str], y: pd.Series, flexibility: str, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    preprocessor = _build_preprocessor(df, columns)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True)),
    ])
    if 'Low' in flexibility:
        Cs = [0.1, 1]
        gammas = [0.001, 0.01]
    elif 'High' in flexibility:
        Cs = [10, 50, 100]
        gammas = [0.1, 1]
    else:
        Cs = [1, 5, 10]
        gammas = [0.01, 0.1]
    param_grid = {
        'classifier__C': Cs,
        'classifier__gamma': gammas,
        'classifier__kernel': ['rbf'],
    }
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    classifier: SVC = best_pipeline.named_steps['classifier']
    hyperparameters = _format_hyperparameters(best_params, {})
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Support Vector Machine (SVM)'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    return result, best_pipeline


def _train_random_forest(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    preprocessor = _build_preprocessor(df, columns)
    classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__n_estimators': [200, 400],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
    }
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    hyperparameters = _format_hyperparameters(best_params, {})
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Random Forest'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    feature_importances = _extract_feature_importances(best_pipeline)
    result['feature_importances'] = feature_importances
    return result, best_pipeline


def _train_gradient_boost(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    preprocessor = _build_preprocessor(df, columns)
    classifier = GradientBoostingClassifier(random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
    }
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    classifier: GradientBoostingClassifier = best_pipeline.named_steps['classifier']
    hyperparameters = _format_hyperparameters(best_params, {'loss': str(classifier.loss)})
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Gradient Boosting'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    feature_importances = _extract_feature_importances(best_pipeline)
    result['feature_importances'] = feature_importances
    return result, best_pipeline


def _train_elastic_net(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    preprocessor = _build_preprocessor(df, columns)
    classifier = LogisticRegression(
        max_iter=5000,
        solver='saga',
        penalty='elasticnet',
        l1_ratio=0.5,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0],
        'classifier__l1_ratio': [0.2, 0.5, 0.8],
    }
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    hyperparameters = _format_hyperparameters(best_params, {'solver': 'saga'})
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Elastic Net (Logistic Regression)'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    return result, best_pipeline


def _train_naive_bayes(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    preprocessor = _build_preprocessor(df, columns)
    classifier = GaussianNB()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]}
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    hyperparameters = _format_hyperparameters(best_params, {})
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Naive Bayes (Gaussian)'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    return result, best_pipeline


def _train_xgboost(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    XGBClassifier = _import_optional_estimator('xgboost', 'XGBClassifier', CLASSIFIER_NAMES['XGBoost'])
    preprocessor = _build_preprocessor(df, columns)
    classifier = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_estimators=200,
        verbosity=0,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__n_estimators': [200, 400],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__subsample': [0.8, 1.0],
    }
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['XGBoost'],
        best_pipeline,
        df[columns],
        y,
        cv,
        _format_hyperparameters(best_params, {}),
    )
    feature_importances = _extract_feature_importances(best_pipeline)
    result['feature_importances'] = feature_importances
    return result, best_pipeline


def _train_lightgbm(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    LGBMClassifier = _import_optional_estimator('lightgbm', 'LGBMClassifier', CLASSIFIER_NAMES['LightGBM'])
    preprocessor = _build_preprocessor(df, columns)
    classifier = LGBMClassifier(
        objective='binary',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__n_estimators': [200, 400],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [31, 63],
        'classifier__max_depth': [-1, 15],
    }
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['LightGBM'],
        best_pipeline,
        df[columns],
        y,
        cv,
        _format_hyperparameters(best_params, {}),
    )
    feature_importances = _extract_feature_importances(best_pipeline)
    result['feature_importances'] = feature_importances
    return result, best_pipeline


def _train_catboost(df: pd.DataFrame, columns: List[str], y: pd.Series, cv: StratifiedKFold) -> Tuple[Dict, Pipeline]:
    CatBoostClassifier = _import_optional_estimator('catboost', 'CatBoostClassifier', CLASSIFIER_NAMES['CatBoost'])
    preprocessor = _build_preprocessor(df, columns)
    classifier = CatBoostClassifier(
        loss_function='Logloss',
        verbose=0,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__iterations': [200, 400],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__depth': [4, 6],
    }
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['CatBoost'],
        best_pipeline,
        df[columns],
        y,
        cv,
        _format_hyperparameters(best_params, {}),
    )
    feature_importances = _extract_feature_importances(best_pipeline)
    result['feature_importances'] = feature_importances
    return result, best_pipeline


def _train_voting_classifier(
    df: pd.DataFrame,
    columns: List[str],
    y: pd.Series,
    cv: StratifiedKFold,
    trained_pipelines: Dict[str, Pipeline],
    requested_models: List[str],
) -> Tuple[Dict, Pipeline | None]:
    estimators = _collect_base_estimators(requested_models, trained_pipelines)
    if len(estimators) < 2:
        message = 'Voting Classifier requires at least two trained supervised models.'
        return _build_status_only_result(CLASSIFIER_NAMES['Voting Classifier'], message), None

    supports_soft_voting = all(hasattr(estimator, 'predict_proba') for _, estimator in estimators)
    voting_strategy = 'soft' if supports_soft_voting else 'hard'
    preprocessor = _build_preprocessor(df, columns)
    classifier = VotingClassifier(estimators=estimators, voting=voting_strategy)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {'classifier__weights': [None]}
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    hyperparameters = _format_hyperparameters(best_params, {
        'voting': voting_strategy,
        'estimators': ', '.join(name for name, _ in estimators),
    })
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Voting Classifier'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    return result, best_pipeline


def _train_stacking_classifier(
    df: pd.DataFrame,
    columns: List[str],
    y: pd.Series,
    cv: StratifiedKFold,
    trained_pipelines: Dict[str, Pipeline],
    requested_models: List[str],
) -> Tuple[Dict, Pipeline | None]:
    estimators = _collect_base_estimators(requested_models, trained_pipelines)
    if len(estimators) < 2:
        message = 'Stacking Classifier requires at least two trained supervised models.'
        return _build_status_only_result(CLASSIFIER_NAMES['Stacking Classifier'], message), None

    stacking_splits = min(5, cv.n_splits)
    stacking_cv = StratifiedKFold(n_splits=stacking_splits, shuffle=True, random_state=RANDOM_STATE)
    final_estimator = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        penalty='l2',
        random_state=RANDOM_STATE,
    )
    classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=False,
        cv=stacking_cv,
        n_jobs=-1,
    )
    preprocessor = _build_preprocessor(df, columns)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {'classifier__final_estimator__C': [0.5, 1.0]}
    best_pipeline, best_params = _grid_search(pipeline, param_grid, df[columns], y, cv)
    hyperparameters = _format_hyperparameters(best_params, {
        'estimators': ', '.join(name for name, _ in estimators),
    })
    result = _evaluate_classifier(
        CLASSIFIER_NAMES['Stacking Classifier'],
        best_pipeline,
        df[columns],
        y,
        cv,
        hyperparameters,
    )
    return result, best_pipeline


def _extract_feature_importances(pipeline: Pipeline) -> List[Dict[str, float]]:
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__', 1)[1] if '__' in name else name for name in raw_feature_names]
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        return [
            {'feature': feature_names[idx], 'importance': float(val)}
            for idx, val in enumerate(importances)
        ]
    return []


def train_supervised_models(entry: DatasetEntry, models: List[str], svm_flexibility: str) -> List[Dict]:
    df = entry.active_df.copy()
    feature_columns = entry.current_feature_columns or entry.feature_columns
    if not feature_columns:
        raise ValueError('No feature columns available for model training. Please review the cleaning step.')
    X = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    y = df[entry.target_column].astype(int)
    cv = _build_cv(y)

    base_requests = [model for model in models if model in BASE_SUPERVISED_MODELS]
    ensemble_requests = [model for model in models if model in ENSEMBLE_MODELS]

    results_by_model: Dict[str, Dict] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    for model_name in base_requests:
        try:
            if model_name == CLASSIFIER_NAMES['Logistic Regression']:
                result, pipeline = _train_logistic(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['Elastic Net (Logistic Regression)']:
                result, pipeline = _train_elastic_net(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['K-Nearest Neighbors (KNN)']:
                result, pipeline = _train_knn(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['Support Vector Machine (SVM)']:
                result, pipeline = _train_svm(X, feature_columns, y, svm_flexibility, cv)
            elif model_name == CLASSIFIER_NAMES['Random Forest']:
                result, pipeline = _train_random_forest(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['Gradient Boosting']:
                result, pipeline = _train_gradient_boost(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['XGBoost']:
                result, pipeline = _train_xgboost(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['LightGBM']:
                result, pipeline = _train_lightgbm(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['CatBoost']:
                result, pipeline = _train_catboost(X, feature_columns, y, cv)
            elif model_name == CLASSIFIER_NAMES['Naive Bayes (Gaussian)']:
                result, pipeline = _train_naive_bayes(X, feature_columns, y, cv)
            else:
                continue
        except MissingDependencyError as exc:
            results_by_model[model_name] = _build_status_only_result(model_name, str(exc))
            continue

        results_by_model[model_name] = result
        trained_pipelines[model_name] = pipeline

    for ensemble_name in ensemble_requests:
        if ensemble_name == CLASSIFIER_NAMES['Voting Classifier']:
            result, _ = _train_voting_classifier(X, feature_columns, y, cv, trained_pipelines, models)
        elif ensemble_name == CLASSIFIER_NAMES['Stacking Classifier']:
            result, _ = _train_stacking_classifier(X, feature_columns, y, cv, trained_pipelines, models)
        else:
            continue
        results_by_model[ensemble_name] = result

    ordered_results = [results_by_model[name] for name in models if name in results_by_model]
    return ordered_results


def compute_kmeans_elbow(entry: DatasetEntry, max_k: int = 10) -> List[Dict[str, float]]:
    df = entry.active_df.copy()
    feature_columns = entry.current_feature_columns or entry.feature_columns
    if not feature_columns:
        raise ValueError('No feature columns available for K-Means clustering.')
    X = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    max_clusters = min(max_k, X_scaled.shape[0])
    elbow = []
    for k in range(1, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        elbow.append({'k': int(k), 'inertia': float(model.inertia_)})
    return elbow


def run_kmeans(entry: DatasetEntry, n_clusters: int) -> Dict:
    df = entry.active_df.copy()
    feature_columns = entry.current_feature_columns or entry.feature_columns
    if not feature_columns:
        raise ValueError('No feature columns available for K-Means clustering.')
    X = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    n_clusters = max(1, min(n_clusters, X_scaled.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    id_values = df[entry.id_column].tolist()
    target_values = df[entry.target_column].tolist()

    cluster_mapping: Dict[str, List] = {}
    for cluster_label, sample_id in zip(clusters, id_values):
        cluster_mapping.setdefault(str(int(cluster_label)), []).append(sample_id)

    cluster_analysis = []
    for cluster_label in range(n_clusters):
        indices = np.where(clusters == cluster_label)[0]
        if len(indices) == 0:
            positive_rate = 0.0
        else:
            positives = np.take(target_values, indices)
            positive_rate = float(np.mean(positives))
        cluster_analysis.append({
            'cluster': int(cluster_label),
            'count': int(len(indices)),
            'positive_rate': positive_rate,
        })

    elbow = compute_kmeans_elbow(entry)

    return {
        'name': CLASSIFIER_NAMES['K-Means Clustering'],
        'metrics': None,
        'roc_curve': None,
        'k_means_result': {
            'elbow_plot': elbow,
            'clusters': cluster_mapping,
            'cluster_analysis': cluster_analysis,
        },
        'hyperparameters': {},
    }


def train_models(entry: DatasetEntry, models: List[str], svm_flexibility: str, kmeans_clusters: int) -> List[Dict]:
    results: List[Dict] = []
    supervised_models = [model for model in models if model != CLASSIFIER_NAMES['K-Means Clustering']]
    if supervised_models:
        results.extend(train_supervised_models(entry, supervised_models, svm_flexibility))
    if CLASSIFIER_NAMES['K-Means Clustering'] in models:
        results.append(run_kmeans(entry, kmeans_clusters))
    return results
