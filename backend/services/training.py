from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from ..dataset_store import DatasetEntry

CLASSIFIER_NAMES = {
    'Logistic Regression': 'Logistic Regression',
    'K-Nearest Neighbors (KNN)': 'K-Nearest Neighbors (KNN)',
    'Support Vector Machine (SVM)': 'Support Vector Machine (SVM)',
    'Random Forest': 'Random Forest',
    'Gradient Boosting': 'Gradient Boosting',
    'K-Means Clustering': 'K-Means Clustering',
}


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


def _evaluate_classifier(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Dict:
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
    }


def _grid_search(pipeline: Pipeline, param_grid: Dict, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Pipeline:
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    return grid.best_estimator_


def _train_logistic(df: pd.DataFrame, columns: List[str], y: pd.Series) -> Dict:
    preprocessor = _build_preprocessor(df, columns)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, solver='lbfgs')),
    ])
    param_grid = {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
    cv = _build_cv(y)
    best_pipeline = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(CLASSIFIER_NAMES['Logistic Regression'], best_pipeline, df[columns], y, cv)
    return result


def _train_knn(df: pd.DataFrame, columns: List[str], y: pd.Series) -> Dict:
    preprocessor = _build_preprocessor(df, columns)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier()),
    ])
    n_samples = len(df)
    max_k = max(1, n_samples - 1)
    k_values = [k for k in range(1, max_k + 1, 2)]
    param_grid = {'classifier__n_neighbors': k_values}
    cv = _build_cv(y)
    best_pipeline = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(CLASSIFIER_NAMES['K-Nearest Neighbors (KNN)'], best_pipeline, df[columns], y, cv)
    return result


def _train_svm(df: pd.DataFrame, columns: List[str], y: pd.Series, flexibility: str) -> Dict:
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
    cv = _build_cv(y)
    best_pipeline = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(CLASSIFIER_NAMES['Support Vector Machine (SVM)'], best_pipeline, df[columns], y, cv)
    return result


def _train_random_forest(df: pd.DataFrame, columns: List[str], y: pd.Series) -> Dict:
    preprocessor = _build_preprocessor(df, columns)
    classifier = RandomForestClassifier(random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__n_estimators': [200, 400],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
    }
    cv = _build_cv(y)
    best_pipeline = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(CLASSIFIER_NAMES['Random Forest'], best_pipeline, df[columns], y, cv)
    feature_importances = _extract_feature_importances(best_pipeline)
    result['feature_importances'] = feature_importances
    return result


def _train_gradient_boost(df: pd.DataFrame, columns: List[str], y: pd.Series) -> Dict:
    preprocessor = _build_preprocessor(df, columns)
    classifier = GradientBoostingClassifier(random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    param_grid = {
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
    }
    cv = _build_cv(y)
    best_pipeline = _grid_search(pipeline, param_grid, df[columns], y, cv)
    result = _evaluate_classifier(CLASSIFIER_NAMES['Gradient Boosting'], best_pipeline, df[columns], y, cv)
    feature_importances = _extract_feature_importances(best_pipeline)
    result['feature_importances'] = feature_importances
    return result


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

    results: List[Dict] = []
    if 'Logistic Regression' in models:
        results.append(_train_logistic(X, feature_columns, y))
    if 'K-Nearest Neighbors (KNN)' in models:
        results.append(_train_knn(X, feature_columns, y))
    if 'Support Vector Machine (SVM)' in models:
        results.append(_train_svm(X, feature_columns, y, svm_flexibility))
    if 'Random Forest' in models:
        results.append(_train_random_forest(X, feature_columns, y))
    if 'Gradient Boosting' in models:
        results.append(_train_gradient_boost(X, feature_columns, y))
    return results


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
    }


def train_models(entry: DatasetEntry, models: List[str], svm_flexibility: str, kmeans_clusters: int) -> List[Dict]:
    results: List[Dict] = []
    supervised_models = [model for model in models if model != CLASSIFIER_NAMES['K-Means Clustering']]
    if supervised_models:
        results.extend(train_supervised_models(entry, supervised_models, svm_flexibility))
    if CLASSIFIER_NAMES['K-Means Clustering'] in models:
        results.append(run_kmeans(entry, kmeans_clusters))
    return results
