import { apiClient } from './apiClient';
import { DataSet, EdaResult } from '../types';

export interface UploadResponse extends DataSet {}

export const uploadDataset = async (file: File): Promise<DataSet> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<UploadResponse>('/datasets', formData);
  return response;
};

export const fetchEda = async (datasetId: string): Promise<EdaResult> => {
  return apiClient.get<EdaResult>(`/datasets/${datasetId}/eda`);
};
