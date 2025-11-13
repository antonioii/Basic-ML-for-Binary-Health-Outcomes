import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import TrainingConfigStep from './TrainingConfigStep';
import { DataSet, ProcessingMode } from '../types';

vi.mock('../services/mlService', () => ({
  fetchKMeansElbow: vi.fn().mockResolvedValue([]),
}));

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Line: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  CartesianGrid: () => <div />,
  Tooltip: () => <div />,
  Legend: () => <div />,
}));

const baseDataSet: DataSet = {
  datasetId: 'test-dataset',
  fileName: 'test.csv',
  rows: 10,
  cols: 5,
  idColumn: 'id',
  targetColumn: 'target',
  featureColumns: ['f1', 'f2', 'f3'],
  preview: [],
  classDistribution: { '0': 5, '1': 5 },
};

describe('TrainingConfigStep', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('collects custom hyperparameters when processing mode is custom', async () => {
    const onConfigComplete = vi.fn();
    render(
      <TrainingConfigStep
        dataSet={baseDataSet}
        onConfigComplete={onConfigComplete}
        processingMode={ProcessingMode.CUSTOM}
        onProcessingModeChange={() => undefined}
      />,
    );

    await waitFor(() => expect(screen.getByText(/Custom Hyperparameters/i)).toBeInTheDocument());

    const logisticHeading = screen.getByRole('heading', { name: 'Logistic Regression' });
    const logisticCard = logisticHeading.parentElement?.parentElement as HTMLElement;
    const logisticInput = within(logisticCard).getByLabelText(/Regularization strength/i);
    fireEvent.change(logisticInput, { target: { value: '0.1, 1' } });

    const rfInput = screen.getByLabelText(/Number of Trees/i);
    fireEvent.change(rfInput, { target: { value: '200, 400' } });

    fireEvent.click(screen.getByText(/Train & Evaluate Models/i));

    expect(onConfigComplete).toHaveBeenCalledTimes(1);
    const payload = onConfigComplete.mock.calls[0][0];
    expect(payload.processingMode).toBe(ProcessingMode.CUSTOM);
    expect(payload.customHyperparameters).toBeDefined();
    expect(payload.customHyperparameters['Logistic Regression'].C).toEqual(['0.1', '1']);
    expect(payload.customHyperparameters['Random Forest'].n_estimators).toEqual(['200', '400']);
  });
});
