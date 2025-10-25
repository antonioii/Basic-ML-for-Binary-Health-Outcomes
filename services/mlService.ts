import { apiClient } from './apiClient';
import { DataSet, KMeansResult, ModelResult, TrainingConfig } from '../types';

export interface TrainingResponse {
  results: ModelResult[];
}

export const trainModels = async (
  dataSet: DataSet,
  config: TrainingConfig
): Promise<ModelResult[]> => {
  const response = await apiClient.post<TrainingResponse>(`/datasets/${dataSet.datasetId}/train`, {
    models: config.models,
    svmFlexibility: config.svmFlexibility,
    kMeansClusters: config.kMeansClusters,
  });
  return response.results;
};

export const fetchKMeansElbow = async (datasetId: string): Promise<KMeansResult['elbowPlot']> => {
  return apiClient.get<KMeansResult['elbowPlot']>(`/datasets/${datasetId}/kmeans/elbow`);
};
