from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DatasetEntry:
    dataset_id: str
    file_name: str
    raw_df: pd.DataFrame
    id_column: str
    target_column: str
    feature_columns: List[str]
    cleaned_df: Optional[pd.DataFrame] = None
    current_feature_columns: Optional[List[str]] = None
    last_variable_info: Optional[List[dict]] = None
    last_suggestions: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_feature_columns is None:
            self.current_feature_columns = list(self.feature_columns)

    @property
    def active_df(self) -> pd.DataFrame:
        return self.cleaned_df if self.cleaned_df is not None else self.raw_df

    def reset_cleaning(self) -> None:
        self.cleaned_df = None
        self.current_feature_columns = list(self.feature_columns)
        self.last_suggestions = {}
        self.last_variable_info = None


class DatasetStore:
    def __init__(self) -> None:
        self._datasets: Dict[str, DatasetEntry] = {}

    def add(self, entry: DatasetEntry) -> None:
        self._datasets[entry.dataset_id] = entry

    def get(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset '{dataset_id}' not found")
        return self._datasets[dataset_id]

    def remove(self, dataset_id: str) -> None:
        self._datasets.pop(dataset_id, None)

    def clear(self) -> None:
        self._datasets.clear()


dataset_store = DatasetStore()
