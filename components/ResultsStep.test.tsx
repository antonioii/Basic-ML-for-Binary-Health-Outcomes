import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import ResultsStep from './ResultsStep';
import { DataSet, ModelResult, ProcessingMode, SvmFlexibility, TrainingConfig } from '../types';

const baseDataSet: DataSet = {
  datasetId: 'ds-1',
  fileName: 'dataset.csv',
  rows: 20,
  cols: 4,
  idColumn: 'id',
  targetColumn: 'target',
  featureColumns: ['f1', 'f2'],
  preview: [],
  classDistribution: { '0': 10, '1': 10 },
};

const baseConfig: TrainingConfig = {
  models: ['Logistic Regression'],
  svmFlexibility: SvmFlexibility.MEDIUM,
  kMeansClusters: 3,
  processingMode: ProcessingMode.LIGHT,
};

const sampleResults: ModelResult[] = [
  {
    name: 'Logistic Regression',
    metrics: {
      confusionMatrix: { tp: 7, fp: 1, tn: 9, fn: 3 },
      sensitivity: 0.7,
      sensitivityStd: 0.05,
      specificity: 0.9,
      specificityStd: 0.02,
      vpp: 0.875,
      vppStd: 0.03,
      vpn: 0.75,
      vpnStd: 0.04,
      f1Score: 0.778,
      f1ScoreStd: 0.02,
      auc: 0.9123,
      aucStd: 0.01,
      accuracy: 0.8,
      accuracyStd: 0.02,
    },
    rocCurve: null,
    hyperparameters: { C: 1.0 },
  },
];

describe('ResultsStep', () => {
  it('displays metric deviations and exports CSV with std columns', async () => {
    const onComplete = vi.fn();
    const onRestart = vi.fn();
    let reportedBlob: Blob | null = null;
    const urlSpy = vi.spyOn(URL, 'createObjectURL').mockImplementation((blob: Blob) => {
      reportedBlob = blob;
      return 'blob:test';
    });
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);

    render(
      <ResultsStep
        dataSet={baseDataSet}
        trainingConfig={baseConfig}
        results={sampleResults}
        onResultsComplete={onComplete}
        onRestart={onRestart}
      />,
    );

    await screen.findByText('Modeling Results');
    expect(screen.getByText('0.912 ± 0.010')).toBeInTheDocument();
    expect(screen.getByText('80.0% ± 2.0%')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /download report/i }));
    expect(urlSpy).toHaveBeenCalled();
    expect(reportedBlob).not.toBeNull();
    const csvText = await reportedBlob!.text();
    expect(csvText.split('\n')[0]).toContain('AUC,AUC_SD');
    expect(csvText).toContain('0.9123,0.0100');

    urlSpy.mockRestore();
    revokeSpy.mockRestore();
  });
});
