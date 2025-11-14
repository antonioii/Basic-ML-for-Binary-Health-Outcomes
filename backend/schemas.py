from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


DataValue = Union[str, float, int, None]
SuggestionDetailValue = Union[DataValue, List[DataValue], Dict[str, DataValue]]


class DataRow(BaseModel):
    __root__: Dict[str, DataValue]

    def dict(self, *args, **kwargs):  # type: ignore[override]
        return super().dict(*args, **kwargs)['__root__']


class DatasetResponse(BaseModel):
    datasetId: str = Field(..., alias='dataset_id')
    fileName: str = Field(..., alias='file_name')
    rows: int
    cols: int
    idColumn: str = Field(..., alias='id_column')
    targetColumn: str = Field(..., alias='target_column')
    featureColumns: List[str] = Field(..., alias='feature_columns')
    preview: List[Dict[str, DataValue]]
    classDistribution: Dict[str, int] = Field(..., alias='class_distribution')

    class Config:
        allow_population_by_field_name = True


class VariableInfo(BaseModel):
    name: str
    type: str
    missing: float
    distinctValues: int = Field(..., alias='distinct_values')
    exampleValues: Optional[List[DataValue]] = Field(default=None, alias='example_values')

    class Config:
        allow_population_by_field_name = True


class MissingInfo(BaseModel):
    variable: str
    percentage: float


class OutlierInfo(BaseModel):
    variable: str
    count: int
    ids: List[DataValue]


class CorrelationInfo(BaseModel):
    pair: List[str]
    value: float


class CleaningSuggestion(BaseModel):
    id: str
    type: str
    description: str
    details: Dict[str, SuggestionDetailValue]


class HistogramInfo(BaseModel):
    variable: str
    bins: List[float]
    counts: List[int]


class BoxPlotInfo(BaseModel):
    variable: str
    median: float
    q1: float
    q3: float
    lowerWhisker: float = Field(..., alias='lower_whisker')
    upperWhisker: float = Field(..., alias='upper_whisker')
    outliers: List[DataValue]

    class Config:
        allow_population_by_field_name = True


class EdaResponse(BaseModel):
    totalRows: int = Field(..., alias='total_rows')
    totalCols: int = Field(..., alias='total_cols')
    variableInfo: List[VariableInfo] = Field(..., alias='variable_info')
    targetDistribution: Dict[str, float] = Field(..., alias='target_distribution')
    missingInfo: List[MissingInfo] = Field(..., alias='missing_info')
    outlierInfo: List[OutlierInfo] = Field(..., alias='outlier_info')
    correlationInfo: List[CorrelationInfo] = Field(..., alias='correlation_info')
    cleaningSuggestions: List[CleaningSuggestion] = Field(..., alias='cleaning_suggestions')
    histograms: List[HistogramInfo]
    boxPlots: List[BoxPlotInfo] = Field(..., alias='box_plots')

    class Config:
        allow_population_by_field_name = True


class CleaningRequest(BaseModel):
    suggestionIds: List[str] = Field(..., alias='suggestionIds')


class CleaningSummary(BaseModel):
    rowsRemoved: int = Field(..., alias='rows_removed')
    colsRemoved: List[str] = Field(..., alias='cols_removed')
    finalRows: int = Field(..., alias='final_rows')
    finalCols: int = Field(..., alias='final_cols')
    notes: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class CleaningResponse(BaseModel):
    cleanedDataSet: DatasetResponse = Field(..., alias='cleaned_dataset')
    summary: CleaningSummary

    class Config:
        allow_population_by_field_name = True


CustomHyperparameterValue = Union[str, float, int, bool, None]
CustomHyperparameterGrid = Dict[str, Dict[str, List[CustomHyperparameterValue]]]


class TrainingConfigRequest(BaseModel):
    models: List[str]
    svmFlexibility: str = Field(..., alias='svmFlexibility')
    kMeansClusters: int = Field(..., alias='kMeansClusters')
    processingMode: str = Field(default='light', alias='processingMode')
    customHyperparameters: Optional[CustomHyperparameterGrid] = Field(default=None, alias='customHyperparameters')


class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int


class ModelMetrics(BaseModel):
    confusionMatrix: ConfusionMatrix = Field(..., alias='confusion_matrix')
    sensitivity: float
    specificity: float
    vpp: float
    vpn: float
    f1Score: float = Field(..., alias='f1_score')
    auc: float
    accuracy: float

    class Config:
        allow_population_by_field_name = True


class RocPoint(BaseModel):
    fpr: float
    tpr: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class KMeansClusterSummary(BaseModel):
    cluster: int
    count: int
    positiveRate: float = Field(..., alias='positive_rate')

    class Config:
        allow_population_by_field_name = True


class KMeansResult(BaseModel):
    elbowPlot: List[Dict[str, float]] = Field(..., alias='elbow_plot')
    clusters: Dict[str, List[DataValue]]
    clusterAnalysis: List[KMeansClusterSummary] = Field(..., alias='cluster_analysis')

    class Config:
        allow_population_by_field_name = True


class ModelResult(BaseModel):
    name: str
    metrics: Optional[ModelMetrics]
    rocCurve: Optional[List[RocPoint]] = Field(..., alias='roc_curve')
    kMeansResult: Optional[KMeansResult] = Field(default=None, alias='k_means_result')
    featureImportances: Optional[List[FeatureImportance]] = Field(default=None, alias='feature_importances')
    hyperparameters: Dict[str, Union[str, float, int, None]] = Field(default_factory=dict)
    statusMessage: Optional[str] = Field(default=None, alias='status_message')

    class Config:
        allow_population_by_field_name = True


class TrainingResponse(BaseModel):
    results: List[ModelResult]


class ElbowPoint(BaseModel):
    k: int
    inertia: float


class KMeansElbowResponse(BaseModel):
    __root__: List[ElbowPoint]

    def dict(self, *args, **kwargs):  # type: ignore[override]
        return [point.dict(*args, **kwargs) for point in self.__root__]
