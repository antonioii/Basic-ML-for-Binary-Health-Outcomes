from __future__ import annotations

import io
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from .dataset_store import DatasetEntry, dataset_store
from .schemas import (
    CleaningRequest,
    CleaningResponse,
    CleaningSummary,
    DatasetResponse,
    EdaResponse,
    ModelResult,
    TrainingConfigRequest,
    TrainingResponse,
)
from .services.cleaning import apply_cleaning
from .services.eda import perform_eda
from .services.training import CLASSIFIER_NAMES, compute_kmeans_elbow, train_models
from .utils.datasets import build_dataset_payload


app = FastAPI(title='Health ML Pipeline API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def _read_dataset(file: UploadFile) -> pd.DataFrame:
    suffix = Path(file.filename or '').suffix.lower()
    content = file.file.read()
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(io.BytesIO(content))
    if suffix == '.csv':
        return pd.read_csv(io.BytesIO(content))
    raise HTTPException(status_code=400, detail='Only .xlsx, .xls, or .csv files are supported.')


def _validate_dataset(df: pd.DataFrame, file_name: str) -> DatasetEntry:
    if df.shape[1] < 3:
        raise HTTPException(status_code=400, detail='Dataset must contain at least three columns (ID, feature(s), target).')

    id_column = df.columns[0]
    target_column = df.columns[-1]
    feature_columns = df.columns[1:-1].tolist()

    if df[id_column].isna().any():
        raise HTTPException(status_code=400, detail='ID column contains missing values. Please ensure all IDs are present.')
    if df[id_column].duplicated().any():
        duplicates = df.loc[df[id_column].duplicated(), id_column].head(5).tolist()
        raise HTTPException(status_code=400, detail=f'Duplicate IDs detected: {duplicates}. IDs must be unique.')

    target_values = set(df[target_column].dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        raise HTTPException(status_code=400, detail='Target column must contain only binary values 0 or 1 without NaN.')
    if df[target_column].isna().any():
        raise HTTPException(status_code=400, detail='Target column contains missing values. Please clean the data before upload.')

    for column in feature_columns:
        coerced = pd.to_numeric(df[column], errors='coerce')
        if coerced.isna().all() and not df[column].isna().all():
            raise HTTPException(status_code=400, detail=f"Feature column '{column}' contains non-numeric values.")
        df[column] = coerced

    df[target_column] = pd.to_numeric(df[target_column], errors='raise').astype(int)

    dataset_id = str(uuid4())
    entry = DatasetEntry(
        dataset_id=dataset_id,
        file_name=file_name or 'dataset',
        raw_df=df,
        id_column=id_column,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    dataset_store.add(entry)
    return entry


@app.get('/api/health')
def health_check() -> dict:
    return {'status': 'ok'}


@app.post('/api/datasets', response_model=DatasetResponse, response_model_by_alias=False)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetResponse:
    try:
        df = _read_dataset(file)
        entry = _validate_dataset(df, file.filename or 'dataset')
    finally:
        await file.close()

    response = DatasetResponse(**build_dataset_payload(entry, entry.raw_df))
    return response


@app.get('/api/datasets/{dataset_id}/eda', response_model=EdaResponse, response_model_by_alias=False)
def run_eda(dataset_id: str) -> EdaResponse:
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Dataset not found.')

    result = perform_eda(entry)
    return EdaResponse(**result)


@app.post('/api/datasets/{dataset_id}/clean', response_model=CleaningResponse, response_model_by_alias=False)
def clean_dataset(dataset_id: str, request: CleaningRequest) -> CleaningResponse:
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Dataset not found.')

    payload = apply_cleaning(entry, request.suggestionIds)
    dataset_response = DatasetResponse(**payload['cleaned_dataset'])
    summary = CleaningSummary(**payload['summary'])
    return CleaningResponse(cleaned_dataset=dataset_response, summary=summary)


@app.get('/api/datasets/{dataset_id}/kmeans/elbow', response_model=List[dict])
def kmeans_elbow(dataset_id: str):
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Dataset not found.')

    elbow = compute_kmeans_elbow(entry)
    return jsonable_encoder(elbow)


@app.post('/api/datasets/{dataset_id}/train', response_model=TrainingResponse, response_model_by_alias=False)
def train(dataset_id: str, config: TrainingConfigRequest) -> TrainingResponse:
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Dataset not found.')

    models = config.models
    for model in models:
        if model not in CLASSIFIER_NAMES.values():
            raise HTTPException(status_code=400, detail=f'Unsupported model: {model}')

    try:
        results = train_models(
            entry,
            models,
            config.svmFlexibility,
            config.kMeansClusters,
            config.processingMode,
            config.customHyperparameters,
            config.classBalance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_results = [ModelResult(**jsonable_encoder(res)) for res in results]
    return TrainingResponse(results=model_results)
