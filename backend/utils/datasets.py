from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..dataset_store import DatasetEntry


def _format_label(label: Any) -> str:
    if isinstance(label, (int,)):
        return str(label)
    if isinstance(label, float):
        if label.is_integer():
            return str(int(label))
        return f'{label:.4g}'
    return str(label)


def build_dataset_payload(entry: DatasetEntry, df: pd.DataFrame | None = None) -> Dict[str, Any]:
    frame = df.copy() if df is not None else entry.active_df.copy()
    feature_columns = entry.current_feature_columns or entry.feature_columns
    distribution = frame[entry.target_column].value_counts(dropna=False).to_dict()
    class_distribution = {
        _format_label(label): int(count)
        for label, count in distribution.items()
    }
    class_distribution.setdefault('0', 0)
    class_distribution.setdefault('1', 0)
    return {
        'dataset_id': entry.dataset_id,
        'file_name': entry.file_name,
        'rows': int(frame.shape[0]),
        'cols': int(frame.shape[1]),
        'id_column': entry.id_column,
        'target_column': entry.target_column,
        'feature_columns': feature_columns,
        'preview': frame.head(10).to_dict(orient='records'),
        'class_distribution': class_distribution,
    }

