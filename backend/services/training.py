from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import math

import re

import numpy as np
import pandas as pd
from sklearn.base import clone
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

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore

try:
    from catboost import CatBoostClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None  # type: ignore

RANDOM_STATE = 42


class MissingDependencyError(RuntimeError):
    """Raised when an optional estimator dependency is unavailable."""


class ModelName:
    LOGISTIC = "Logistic Regression"
    ELASTIC_NET = "Elastic Net Logistic Regression"
    KNN = "K-Nearest Neighbors (KNN)"
    SVM = "Support Vector Machine (SVM)"
    NAIVE_BAYES = "Naive Bayes (GaussianNB - no non-negative feature requirement)"
    RANDOM_FOREST = "Random Forest"
    GRADIENT_BOOSTING = "Gradient Boosting"
    XGBOOST = "XGBoost Classifier"
    LIGHTGBM = "LightGBM Classifier"
    CATBOOST = "CatBoost Classifier"
    VOTING = "Voting Classifier"
    STACKING = "Stacking Classifier"
    KMEANS = "K-Means Clustering"


CLASSIFIER_NAMES: Dict[str, str] = {
    ModelName.LOGISTIC: ModelName.LOGISTIC,
    ModelName.ELASTIC_NET: ModelName.ELASTIC_NET,
    ModelName.KNN: ModelName.KNN,
    ModelName.SVM: ModelName.SVM,
    ModelName.NAIVE_BAYES: ModelName.NAIVE_BAYES,
    ModelName.RANDOM_FOREST: ModelName.RANDOM_FOREST,
    ModelName.GRADIENT_BOOSTING: ModelName.GRADIENT_BOOSTING,
    ModelName.XGBOOST: ModelName.XGBOOST,
    ModelName.LIGHTGBM: ModelName.LIGHTGBM,
    ModelName.CATBOOST: ModelName.CATBOOST,
    ModelName.VOTING: ModelName.VOTING,
    ModelName.STACKING: ModelName.STACKING,
    ModelName.KMEANS: ModelName.KMEANS,
}

_SUPERVISED_ENSEMBLES = {ModelName.VOTING, ModelName.STACKING}


def _split_feature_types(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
    binary_cols: List[str] = []
    continuous_cols: List[str] = []
    for column in columns:
        series = df[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        unique_values = numeric_series.dropna().unique()
        if len(unique_values) == 0:
            continue
        if set(unique_values).issubset({0, 1}):
            binary_cols.append(column)
        elif numeric_series.dtype.kind in {"i", "u", "f"}:
            continuous_cols.append(column)
        else:
            binary_cols.append(column)
    return {
        "binary": binary_cols,
        "continuous": continuous_cols,
    }


def _build_preprocessor(df: pd.DataFrame, columns: List[str]) -> ColumnTransformer:
    feature_types = _split_feature_types(df, columns)
    transformers = []
    if feature_types["continuous"]:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_types["continuous"],
            )
        )
    if feature_types["binary"]:
        transformers.append(
            (
                "binary",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                feature_types["binary"],
            )
        )
    if not transformers:
        transformers.append(
            (
                "default",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def _build_cv(y: pd.Series) -> StratifiedKFold:
    class_counts = y.value_counts()
    if class_counts.size < 2:
        raise ValueError(
            "Target column must contain at least two distinct classes (0 and 1) for supervised model training."
        )
    max_splits = min(int(class_counts.min()), len(y), 10)
    if max_splits < 2:
        raise ValueError(
            "Each class must contain at least two observations to perform stratified cross-validation."
        )
    return StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=RANDOM_STATE)


def _clean_param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_clean_param_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_clean_param_value(v) for v in value]
    return value


def _format_hyperparameters(
    best_params: Dict[str, Any], classifier: Any, extras: Iterable[str] = ()
) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, value in best_params.items():
        if key.startswith("classifier__"):
            clean_key = key.split("__", 1)[1]
            formatted[clean_key] = _clean_param_value(value)
    for extra in extras:
        if extra not in formatted and hasattr(classifier, extra):
            formatted[extra] = _clean_param_value(getattr(classifier, extra))
    return formatted


def _evaluate_classifier(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        decision = cross_val_predict(pipeline, X, y, cv=cv, method="decision_function")
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
        "name": name,
        "metrics": {
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "vpp": float(precision),
            "vpn": float(vpn),
            "f1_score": float(f1),
            "auc": float(auc_value),
            "accuracy": float(accuracy),
        },
        "roc_curve": [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)],
        "hyperparameters": hyperparameters or {},
    }


def _grid_search(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Pipeline, Dict[str, Any]]:
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def _extract_feature_importances(pipeline: Pipeline) -> List[Dict[str, float]]:
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split("__", 1)[1] if "__" in name else name for name in raw_feature_names]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        return [
            {"feature": feature_names[idx], "importance": float(val)}
            for idx, val in enumerate(importances)
        ]
    return []


def _train_generic_classifier(
    name: str,
    estimator: Any,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    feature_columns: List[str],
    y: pd.Series,
    extra_hyperparameters: Iterable[str] = (),
    include_feature_importances: bool = False,
    cv: Optional[StratifiedKFold] = None,
) -> Tuple[Dict[str, Any], Pipeline]:
    preprocessor = _build_preprocessor(X, feature_columns)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    cv = cv or _build_cv(y)
    best_pipeline, best_params = _grid_search(pipeline, param_grid, X, y, cv)
    classifier = best_pipeline.named_steps["classifier"]
    hyperparameters = _format_hyperparameters(best_params, classifier, extras=extra_hyperparameters)
    result = _evaluate_classifier(name, best_pipeline, X, y, cv, hyperparameters)
    if include_feature_importances:
        feature_importances = _extract_feature_importances(best_pipeline)
        if feature_importances:
            result["feature_importances"] = feature_importances
    return result, best_pipeline

def _train_logistic(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    estimator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
    param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
    return _train_generic_classifier(
        ModelName.LOGISTIC,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        extra_hyperparameters=("penalty", "solver"),
    )


def _train_elastic_net(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    estimator = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__l1_ratio": [0.1, 0.5, 0.9],
    }
    return _train_generic_classifier(
        ModelName.ELASTIC_NET,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        extra_hyperparameters=("penalty", "solver", "l1_ratio"),
    )


def _train_knn(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    cv = _build_cv(y)
    estimator = KNeighborsClassifier()
    n_samples = len(X)
    min_train_size = n_samples - math.ceil(n_samples / cv.get_n_splits())
    max_k = max(1, min(min_train_size, n_samples - 1, 25))
    k_values = [k for k in range(1, max_k + 1, 2)]
    if not k_values:
        k_values = [1]
    param_grid = {"classifier__n_neighbors": k_values}
    return _train_generic_classifier(
        ModelName.KNN,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        extra_hyperparameters=("metric", "weights"),
        cv=cv,
    )


def _train_svm(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series, flexibility: str
) -> Tuple[Dict[str, Any], Pipeline]:
    estimator = SVC(probability=True, random_state=RANDOM_STATE)
    if "Low" in flexibility:
        Cs = [0.1, 1]
        gammas = [0.001, 0.01]
    elif "High" in flexibility:
        Cs = [10, 50, 100]
        gammas = [0.1, 1]
    else:
        Cs = [1, 5, 10]
        gammas = [0.01, 0.1]
    param_grid = {
        "classifier__C": Cs,
        "classifier__gamma": gammas,
        "classifier__kernel": ["rbf"],
    }
    return _train_generic_classifier(
        ModelName.SVM,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        extra_hyperparameters=("kernel",),
    )


def _train_random_forest(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    estimator = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }
    return _train_generic_classifier(
        ModelName.RANDOM_FOREST,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        include_feature_importances=True,
    )


def _train_gradient_boosting(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    estimator = GradientBoostingClassifier(random_state=RANDOM_STATE)
    param_grid = {
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
    }
    return _train_generic_classifier(
        ModelName.GRADIENT_BOOSTING,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        include_feature_importances=True,
    )


def _train_xgboost(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    if XGBClassifier is None:
        raise MissingDependencyError(
            "xgboost is not installed. Install it to enable the XGBoost Classifier."
        )
    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
    }
    return _train_generic_classifier(
        ModelName.XGBOOST,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        include_feature_importances=True,
    )


def _train_lightgbm(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    if LGBMClassifier is None:
        raise MissingDependencyError(
            "lightgbm is not installed. Install it to enable the LightGBM Classifier."
        )
    estimator = LGBMClassifier(random_state=RANDOM_STATE)
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__num_leaves": [31, 63],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
    }
    return _train_generic_classifier(
        ModelName.LIGHTGBM,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        include_feature_importances=True,
    )


def _train_catboost(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    if CatBoostClassifier is None:
        raise MissingDependencyError(
            "catboost is not installed. Install it to enable the CatBoost Classifier."
        )
    estimator = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    param_grid = {
        "classifier__depth": [4, 6],
        "classifier__iterations": [200, 400],
        "classifier__learning_rate": [0.03, 0.1],
    }
    return _train_generic_classifier(
        ModelName.CATBOOST,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
        include_feature_importances=True,
    )


def _train_naive_bayes(
    X: pd.DataFrame, feature_columns: List[str], y: pd.Series
) -> Tuple[Dict[str, Any], Pipeline]:
    estimator = GaussianNB()
    param_grid = {"classifier__var_smoothing": [1e-9, 1e-8, 1e-7]}
    return _train_generic_classifier(
        ModelName.NAIVE_BAYES,
        estimator,
        param_grid,
        X,
        feature_columns,
        y,
    )


def _build_error_result(name: str, message: str) -> Dict[str, Any]:
    return {
        "name": name,
        "metrics": None,
        "roc_curve": None,
        "k_means_result": None,
        "feature_importances": None,
        "hyperparameters": {"error": message},
    }


def _unique_estimator_name(base: str, existing: set[str]) -> str:
    slug_raw = "".join(ch if ch.isalnum() else "_" for ch in base.lower())
    slug = re.sub(r"_+", "_", slug_raw).strip("_") or "estimator"
    candidate = slug
    index = 1
    while candidate in existing:
        candidate = f"{slug}_{index}"
        index += 1
    existing.add(candidate)
    return candidate


def _prepare_base_estimators(
    selected_models: List[str],
    trained_pipelines: Dict[str, Pipeline],
) -> List[Tuple[str, Any]]:
    estimators: List[Tuple[str, Any]] = []
    used_names: set[str] = set()
    for model_name in selected_models:
        if model_name in _SUPERVISED_ENSEMBLES or model_name == ModelName.KMEANS:
            continue
        pipeline = trained_pipelines.get(model_name)
        if not pipeline:
            continue
        estimator_name = _unique_estimator_name(model_name, used_names)
        estimator = clone(pipeline.named_steps["classifier"])
        estimators.append((estimator_name, estimator))
    return estimators


def _generate_voting_weight_grid(count: int) -> List[List[float]]:
    if count < 2:
        return []
    base_weights = [[1.0] * count]
    if count <= 4:
        variations: List[List[float]] = []
        for idx in range(count):
            weights = [1.0] * count
            weights[idx] = 2.0
            variations.append(weights)
        base_weights.extend(variations)
    return base_weights


def _train_voting_classifier(
    X: pd.DataFrame,
    feature_columns: List[str],
    y: pd.Series,
    selected_models: List[str],
    trained_pipelines: Dict[str, Pipeline],
) -> Optional[Tuple[Dict[str, Any], Pipeline]]:
    estimators = _prepare_base_estimators(selected_models, trained_pipelines)
    if len(estimators) < 2:
        return None
    supports_soft = all(hasattr(estimator, "predict_proba") for _, estimator in estimators)
    voting = VotingClassifier(
        estimators=estimators,
        voting="soft" if supports_soft else "hard",
        n_jobs=-1,
    )
    weight_grid = _generate_voting_weight_grid(len(estimators))
    if not weight_grid:
        weight_grid = [[1.0] * len(estimators)]
    param_grid = {"classifier__weights": weight_grid}
    return _train_generic_classifier(
        ModelName.VOTING,
        voting,
        param_grid,
        X,
        feature_columns,
        y,
    )


def _train_stacking_classifier(
    X: pd.DataFrame,
    feature_columns: List[str],
    y: pd.Series,
    selected_models: List[str],
    trained_pipelines: Dict[str, Pipeline],
) -> Optional[Tuple[Dict[str, Any], Pipeline]]:
    estimators = _prepare_base_estimators(selected_models, trained_pipelines)
    if len(estimators) < 2:
        return None
    base_cv = _build_cv(y)
    stack_cv = max(2, min(5, base_cv.get_n_splits()))
    final_estimator = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=stack_cv,
        passthrough=False,
        n_jobs=-1,
    )
    param_grid = {"classifier__final_estimator__C": [0.1, 1.0, 10.0]}
    return _train_generic_classifier(
        ModelName.STACKING,
        stacking,
        param_grid,
        X,
        feature_columns,
        y,
    )


TrainCallable = Callable[[pd.DataFrame, List[str], pd.Series], Tuple[Dict[str, Any], Pipeline]]

_BASE_TRAINERS: Dict[str, TrainCallable] = {
    ModelName.LOGISTIC: _train_logistic,
    ModelName.ELASTIC_NET: _train_elastic_net,
    ModelName.KNN: _train_knn,
    ModelName.NAIVE_BAYES: _train_naive_bayes,
    ModelName.RANDOM_FOREST: _train_random_forest,
    ModelName.GRADIENT_BOOSTING: _train_gradient_boosting,
    ModelName.XGBOOST: _train_xgboost,
    ModelName.LIGHTGBM: _train_lightgbm,
    ModelName.CATBOOST: _train_catboost,
}


def _get_trainer(model_name: str, svm_flexibility: str) -> Optional[TrainCallable]:
    if model_name == ModelName.SVM:
        return lambda X, columns, y: _train_svm(X, columns, y, svm_flexibility)
    return _BASE_TRAINERS.get(model_name)


def train_supervised_models(entry: DatasetEntry, models: List[str], svm_flexibility: str) -> List[Dict[str, Any]]:
    df = entry.active_df.copy()
    feature_columns = entry.current_feature_columns or entry.feature_columns
    if not feature_columns:
        raise ValueError(
            "No feature columns available for model training. Please review the cleaning step."
        )
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df[entry.target_column].astype(int)

    model_order = {name: idx for idx, name in enumerate(models)}
    results_with_order: List[Tuple[int, Dict[str, Any]]] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    for model_name in models:
        if model_name in _SUPERVISED_ENSEMBLES or model_name == ModelName.KMEANS:
            continue
        trainer = _get_trainer(model_name, svm_flexibility)
        if trainer is None:
            continue
        try:
            result, pipeline = trainer(X, feature_columns, y)
        except MissingDependencyError as exc:
            results_with_order.append(
                (model_order[model_name], _build_error_result(model_name, str(exc)))
            )
            continue
        except ValueError as exc:
            results_with_order.append(
                (model_order[model_name], _build_error_result(model_name, str(exc)))
            )
            continue
        trained_pipelines[model_name] = pipeline
        results_with_order.append((model_order[model_name], result))

    if ModelName.VOTING in models:
        ensemble = _train_voting_classifier(X, feature_columns, y, models, trained_pipelines)
        if ensemble is not None:
            result, pipeline = ensemble
            results_with_order.append((model_order[ModelName.VOTING], result))
            trained_pipelines[ModelName.VOTING] = pipeline
    if ModelName.STACKING in models:
        ensemble = _train_stacking_classifier(X, feature_columns, y, models, trained_pipelines)
        if ensemble is not None:
            result, pipeline = ensemble
            results_with_order.append((model_order[ModelName.STACKING], result))
            trained_pipelines[ModelName.STACKING] = pipeline

    results_with_order.sort(key=lambda item: item[0])
    return [result for _, result in results_with_order]


def compute_kmeans_elbow(entry: DatasetEntry, max_k: int = 10) -> List[Dict[str, float]]:
    df = entry.active_df.copy()
    feature_columns = entry.current_feature_columns or entry.feature_columns
    if not feature_columns:
        raise ValueError("No feature columns available for K-Means clustering.")
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    max_clusters = min(max_k, X_scaled.shape[0])
    elbow: List[Dict[str, float]] = []
    for k in range(1, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(X_scaled)
        elbow.append({"k": int(k), "inertia": float(model.inertia_)})
    return elbow


def run_kmeans(entry: DatasetEntry, n_clusters: int) -> Dict[str, Any]:
    df = entry.active_df.copy()
    feature_columns = entry.current_feature_columns or entry.feature_columns
    if not feature_columns:
        raise ValueError("No feature columns available for K-Means clustering.")
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    n_clusters = max(1, min(n_clusters, X_scaled.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = model.fit_predict(X_scaled)

    id_values = df[entry.id_column].tolist()
    target_values = df[entry.target_column].tolist()

    cluster_mapping: Dict[str, List[Any]] = {}
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
        cluster_analysis.append(
            {
                "cluster": int(cluster_label),
                "count": int(len(indices)),
                "positive_rate": positive_rate,
            }
        )

    elbow = compute_kmeans_elbow(entry)

    return {
        "name": ModelName.KMEANS,
        "metrics": None,
        "roc_curve": None,
        "k_means_result": {
            "elbow_plot": elbow,
            "clusters": cluster_mapping,
            "cluster_analysis": cluster_analysis,
        },
        "hyperparameters": {},
    }


def train_models(entry: DatasetEntry, models: List[str], svm_flexibility: str, kmeans_clusters: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    supervised_models = [model for model in models if model != ModelName.KMEANS]
    if supervised_models:
        results.extend(train_supervised_models(entry, supervised_models, svm_flexibility))
    if ModelName.KMEANS in models:
        results.append(run_kmeans(entry, kmeans_clusters))
    return results
