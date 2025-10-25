import { apiClient } from './apiClient';
import { CleaningSummary, DataSet } from '../types';

export interface CleaningResponse {
  cleanedDataSet: DataSet;
  summary: CleaningSummary;
}

export const applyCleaning = async (datasetId: string, suggestionIds: string[]): Promise<CleaningResponse> => {
  return apiClient.post<CleaningResponse>(`/datasets/${datasetId}/clean`, {
    suggestionIds,
  });
};
