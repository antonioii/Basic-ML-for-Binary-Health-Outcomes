from __future__ import annotations

from typing import Dict, Iterable, List, Set

from ..dataset_store import DatasetEntry
from ..utils.datasets import build_dataset_payload


def _to_set(values: Iterable) -> Set:
    return set(v for v in values if v is not None)


def apply_cleaning(entry: DatasetEntry, suggestion_ids: List[str]) -> Dict:
    df = entry.raw_df.copy()
    original_rows = df.shape[0]
    removed_row_ids: Set = set()
    removed_columns: Set[str] = set()

    for suggestion_id in suggestion_ids:
        suggestion = entry.last_suggestions.get(suggestion_id)
        if not suggestion:
            continue
        details = suggestion.get('details', {})
        s_type = suggestion.get('type')
        if s_type == 'missing':
            variable = details.get('variable')
            if variable in df.columns:
                mask = df[variable].isna()
                removed_row_ids.update(df.loc[mask, entry.id_column].tolist())
        elif s_type == 'outlier':
            ids = details.get('ids', [])
            removed_row_ids.update(ids)
        elif s_type == 'correlation':
            drop_column = details.get('dropColumn') or details.get('drop_column')
            if not drop_column and isinstance(details.get('variables'), list) and len(details['variables']) == 2:
                drop_column = details['variables'][1]
            if drop_column in df.columns:
                removed_columns.add(drop_column)

    if removed_row_ids:
        df = df[~df[entry.id_column].isin(_to_set(removed_row_ids))]

    if removed_columns:
        df = df.drop(columns=list(removed_columns))

    current_features = [col for col in df.columns if col not in {entry.id_column, entry.target_column}]
    entry.cleaned_df = df
    entry.current_feature_columns = current_features

    summary = {
        'rows_removed': int(original_rows - df.shape[0]),
        'cols_removed': list(removed_columns),
        'final_rows': int(df.shape[0]),
        'final_cols': int(len(current_features)),
    }
    if removed_row_ids:
        summary['notes'] = f"Removed {len(_to_set(removed_row_ids))} rows based on selected cleaning suggestions."
    elif removed_columns:
        summary['notes'] = 'Removed highly correlated columns based on selected suggestions.'

    return {
        'cleaned_dataset': build_dataset_payload(entry, df),
        'summary': summary,
    }
