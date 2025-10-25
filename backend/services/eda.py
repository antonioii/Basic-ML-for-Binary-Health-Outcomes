from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..dataset_store import DatasetEntry

CORRELATION_THRESHOLD = 0.8
IMBALANCE_THRESHOLD = 0.2
LOW_CARDINALITY_THRESHOLD = 15
HISTOGRAM_BINS = 20


def _infer_variable_type(series: pd.Series) -> str:
    if series.name is None:
        return 'Unknown'
    unique_values = series.dropna().unique()
    if series.dtype == object or series.dtype == 'category':
        return 'Categorical (Low Cardinality)'
    if set(unique_values).issubset({0, 1}):
        return 'Binary'
    if len(unique_values) <= LOW_CARDINALITY_THRESHOLD and np.issubdtype(series.dtype, np.integer):
        return 'Categorical (Low Cardinality)'
    return 'Continuous'


def _calculate_missing(df: pd.DataFrame, columns: List[str]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    total_rows = len(df)
    for column in columns:
        missing_count = df[column].isna().sum()
        if missing_count:
            results.append({
                'variable': column,
                'percentage': (missing_count / total_rows) * 100,
            })
    return results


def _detect_outliers(entry: DatasetEntry, df: pd.DataFrame) -> Tuple[List[Dict], Dict[str, List]]:
    outliers: List[Dict] = []
    outlier_ids_by_column: Dict[str, List] = {}
    id_col = entry.id_column

    for column in entry.current_feature_columns or entry.feature_columns:
        series = df[column]
        if series.dtype.kind not in {'i', 'u', 'f'}:
            continue
        if series.dropna().nunique() <= LOW_CARDINALITY_THRESHOLD:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if math.isclose(iqr, 0.0):
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (series < lower_bound) | (series > upper_bound)
        ids = df.loc[mask, id_col].tolist()
        if ids:
            outliers.append({
                'variable': column,
                'count': len(ids),
                'ids': ids,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
            })
            outlier_ids_by_column[column] = ids
    return outliers, outlier_ids_by_column


def _compute_correlations(df: pd.DataFrame, columns: List[str]) -> List[Dict]:
    if not columns:
        return []
    numeric_df = df[columns].select_dtypes(include=[np.number])
    if numeric_df.empty:
        return []
    corr_matrix = numeric_df.corr().replace([np.inf, -np.inf], np.nan).dropna(how='all', axis=0).dropna(how='all', axis=1)
    strong_pairs: List[Dict] = []
    for i, col_a in enumerate(corr_matrix.columns):
        for col_b in corr_matrix.columns[i + 1:]:
            value = corr_matrix.at[col_a, col_b]
            if pd.isna(value):
                continue
            if abs(value) >= CORRELATION_THRESHOLD:
                strong_pairs.append({
                    'pair': [col_a, col_b],
                    'value': float(value),
                    'drop_column': col_b,
                })
    return strong_pairs


def _build_histograms(df: pd.DataFrame, columns: List[str]) -> List[Dict]:
    histograms: List[Dict] = []
    for column in columns:
        series = df[column]
        if series.dtype.kind not in {'i', 'u', 'f'}:
            continue
        if series.dropna().nunique() <= 1:
            continue
        counts, bin_edges = np.histogram(series.dropna().values, bins=min(HISTOGRAM_BINS, series.dropna().nunique()))
        histograms.append({
            'variable': column,
            'bins': [float(edge) for edge in bin_edges.tolist()],
            'counts': counts.astype(int).tolist(),
        })
    return histograms


def _build_box_plots(df: pd.DataFrame, columns: List[str], outlier_ids: Dict[str, List]) -> List[Dict]:
    box_plots: List[Dict] = []
    for column in columns:
        series = df[column]
        if series.dtype.kind not in {'i', 'u', 'f'}:
            continue
        if series.dropna().nunique() <= 1:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        median = float(series.median())
        iqr = q3 - q1
        lower_whisker = float(q1 - 1.5 * iqr)
        upper_whisker = float(q3 + 1.5 * iqr)
        box_plots.append({
            'variable': column,
            'median': median,
            'q1': q1,
            'q3': q3,
            'lower_whisker': lower_whisker,
            'upper_whisker': upper_whisker,
            'outliers': outlier_ids.get(column, []),
        })
    return box_plots


def _example_values(series: pd.Series, limit: int = 3) -> List:
    values = series.dropna().unique().tolist()
    return values[:limit]


def perform_eda(entry: DatasetEntry) -> Dict:
    df = entry.raw_df.copy()
    feature_columns = entry.feature_columns
    id_col = entry.id_column
    target_col = entry.target_column

    variable_info = []
    for column in [id_col] + feature_columns + [target_col]:
        series = df[column]
        if column == id_col:
            v_type = 'ID'
        elif column == target_col:
            v_type = 'Target'
        else:
            v_type = _infer_variable_type(series)
        variable_info.append({
            'name': column,
            'type': v_type,
            'missing': float(series.isna().mean() * 100),
            'distinct_values': int(series.nunique()),
            'example_values': _example_values(series),
        })

    missing_info = _calculate_missing(df, feature_columns)
    outlier_info, outlier_ids = _detect_outliers(entry, df)
    correlation_info = _compute_correlations(df, feature_columns)

    target_counts = df[target_col].value_counts().to_dict()
    class0 = int(target_counts.get(0, 0))
    class1 = int(target_counts.get(1, 0))
    total = class0 + class1 if (class0 + class1) else 1
    imbalance = (min(class0, class1) / total) < IMBALANCE_THRESHOLD

    suggestions = []
    for missing in missing_info:
        suggestions.append({
            'id': f"missing-{missing['variable']}",
            'type': 'missing',
            'description': f"Remove rows with missing values in '{missing['variable']}' ({missing['percentage']:.2f}% missing).",
            'details': {
                'variable': missing['variable'],
                'reason': 'Missing values can bias model training and need explicit handling.',
            },
        })
    for outlier in outlier_info:
        suggestions.append({
            'id': f"outlier-{outlier['variable']}",
            'type': 'outlier',
            'description': f"Remove {outlier['count']} outlier(s) in '{outlier['variable']}' (IQR-based detection).",
            'details': {
                'variable': outlier['variable'],
                'ids': outlier['ids'],
                'reason': 'Extreme values may distort model coefficients and should be reviewed.',
            },
        })
    for correlation in correlation_info:
        suggestions.append({
            'id': f"corr-{correlation['pair'][0]}-{correlation['pair'][1]}",
            'type': 'correlation',
            'description': f"Remove '{correlation['pair'][1]}' due to high correlation with '{correlation['pair'][0]}' (|r|={abs(correlation['value']):.2f}).",
            'details': {
                'variables': correlation['pair'],
                'value': correlation['value'],
                'dropColumn': correlation['drop_column'],
                'reason': 'Highly correlated variables can introduce multicollinearity.',
            },
        })

    histograms = _build_histograms(df, feature_columns)
    box_plots = _build_box_plots(df, feature_columns, outlier_ids)

    entry.last_variable_info = variable_info
    entry.last_suggestions = {s['id']: s for s in suggestions}

    return {
        'total_rows': int(df.shape[0]),
        'total_cols': int(df.shape[1]),
        'variable_info': variable_info,
        'target_distribution': {
            '0': class0,
            '1': class1,
            'imbalance': imbalance,
        },
        'missing_info': missing_info,
        'outlier_info': outlier_info,
        'correlation_info': [{
            'pair': item['pair'],
            'value': item['value'],
        } for item in correlation_info],
        'cleaning_suggestions': [{**s, 'apply': True} for s in suggestions],
        'histograms': histograms,
        'box_plots': box_plots,
    }
