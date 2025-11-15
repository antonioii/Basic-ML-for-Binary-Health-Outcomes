import { apiClient } from './apiClient';
import { DataSet, KMeansResult, ModelResult, TrainingConfig } from '../types';

export interface TrainingResponse {
  results: ModelResult[];
}

export const trainModels = async (
  dataSet: DataSet,
  config: TrainingConfig
): Promise<ModelResult[]> => {
  const payload: Record<string, unknown> = {
    models: config.models,
    svmFlexibility: config.svmFlexibility,
    kMeansClusters: config.kMeansClusters,
    processingMode: config.processingMode,
  };

  if (config.customHyperparameters && Object.keys(config.customHyperparameters).length) {
    payload.customHyperparameters = config.customHyperparameters;
  }

  if (config.classBalance) {
    payload.classBalance = config.classBalance;
  }

  const response = await apiClient.post<TrainingResponse>(`/datasets/${dataSet.datasetId}/train`, {
    ...payload,
  });
  return response.results;
};

export const fetchKMeansElbow = async (datasetId: string): Promise<KMeansResult['elbowPlot']> => {
  return apiClient.get<KMeansResult['elbowPlot']>(`/datasets/${datasetId}/kmeans/elbow`);
};
