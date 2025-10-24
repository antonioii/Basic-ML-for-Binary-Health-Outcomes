
import { DataSet, CleaningSuggestion, CleaningSummary, DataRow } from '../types';

export const applyCleaning = (dataSet: DataSet, suggestions: CleaningSuggestion[]): { cleanedData: DataSet, summary: CleaningSummary } => {
  let cleanedRows: DataRow[] = [...dataSet.data];
  let cleanedFeatureCols: string[] = [...dataSet.featureCols];
  const initialRows = dataSet.data.length;

  const appliedSuggestions = suggestions.filter(s => s.apply);

  // 1. Remove columns first to avoid unnecessary processing
  const colsToRemove = new Set<string>();
  appliedSuggestions.filter(s => s.type === 'correlation').forEach(s => {
    colsToRemove.add(s.details.variables![1]);
  });

  cleanedFeatureCols = cleanedFeatureCols.filter(col => !colsToRemove.has(col));

  // 2. Remove rows with outliers
  const outlierIdsToRemove = new Set<string | number>();
  appliedSuggestions.filter(s => s.type === 'outlier').forEach(s => {
    s.details.ids?.forEach(id => outlierIdsToRemove.add(id));
  });

  if (outlierIdsToRemove.size > 0) {
    cleanedRows = cleanedRows.filter(row => !outlierIdsToRemove.has(row[dataSet.idCol]));
  }

  // 3. Remove rows with missing values
  const missingVars = appliedSuggestions.filter(s => s.type === 'missing').map(s => s.details.variable!);
  if (missingVars.length > 0) {
    cleanedRows = cleanedRows.filter(row => {
      return missingVars.every(col => row[col] !== null && row[col] !== undefined);
    });
  }

  const finalRows = cleanedRows.length;

  const summary: CleaningSummary = {
    rowsRemoved: initialRows - finalRows,
    colsRemoved: Array.from(colsToRemove),
    finalRows: finalRows,
    finalCols: cleanedFeatureCols.length,
  };

  const cleanedData: DataSet = {
    ...dataSet,
    data: cleanedRows,
    featureCols: cleanedFeatureCols,
  };

  return { cleanedData, summary };
};
