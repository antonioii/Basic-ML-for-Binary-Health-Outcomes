import { DataSet, ProcessingMode, TrainingConfig } from '../types';

export interface TrainingStageEstimate {
  label: string;
  seconds: number;
}

export interface TrainingEstimate {
  totalSeconds: number;
  stages: TrainingStageEstimate[];
}

const MODEL_COMPLEXITY_WEIGHTS: Record<string, number> = {
  'Logistic Regression': 1.0,
  'Elastic Net (Logistic Regression)': 1.2,
  'K-Nearest Neighbors (KNN)': 1.35,
  'Support Vector Machine (SVM)': 1.8,
  'Random Forest': 1.6,
  'Gradient Boosting': 1.7,
  'XGBoost': 2.05,
  'LightGBM': 1.7,
  'CatBoost': 2.0,
  'Naive Bayes (Gaussian)': 0.7,
  'Voting Classifier': 1.25,
  'Stacking Classifier': 1.4,
};

const STATIC_GRID_SIZES: Record<string, number> = {
  'Logistic Regression': 4,
  'Elastic Net (Logistic Regression)': 9,
  'Random Forest': 12,
  'Gradient Boosting': 8,
  'XGBoost': 16,
  'LightGBM': 16,
  'CatBoost': 8,
  'Naive Bayes (Gaussian)': 3,
  'Voting Classifier': 1,
  'Stacking Classifier': 2,
};

const COST_TO_SECONDS = 0.03;
const MIN_SECONDS = 8;
const MAX_SECONDS = 7200;

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const computeDatasetEffort = (rows: number, featureCount: number): number => {
  const safeRows = Math.max(1, rows);
  const safeFeatures = Math.max(1, featureCount);
  const rowFactor = Math.pow(safeRows, 0.55);
  const featureFactor = 0.9 + Math.log1p(safeFeatures) * 0.65;
  return rowFactor * featureFactor;
};

const inferFeatureCount = (dataSet: DataSet): number => {
  if (Array.isArray(dataSet.featureColumns) && dataSet.featureColumns.length > 0) {
    return dataSet.featureColumns.length;
  }
  // fallback to dataset metadata when featureColumns is empty
  return Math.max(1, dataSet.cols - 2);
};

const computeKnnGridSize = (rows: number): number => {
  const maxK = Math.max(1, Math.min(rows - 1, 51));
  return Math.max(1, Math.ceil(maxK / 2));
};

const computeSvmGridSize = (flexibility: string): number => {
  if (flexibility.toLowerCase().includes('low')) return 4;
  if (flexibility.toLowerCase().includes('high')) return 6;
  return 6;
};

const computeBaseGridSize = (modelName: string, rows: number, config: TrainingConfig): number => {
  if (modelName === 'K-Nearest Neighbors (KNN)') {
    return computeKnnGridSize(rows);
  }
  if (modelName === 'Support Vector Machine (SVM)') {
    return computeSvmGridSize(config.svmFlexibility);
  }
  return STATIC_GRID_SIZES[modelName] ?? 4;
};

const getCustomGridSize = (modelName: string, config: TrainingConfig): number | undefined => {
  const modelConfig = config.customHyperparameters?.[modelName];
  if (!modelConfig) {
    return undefined;
  }
  const combinations = Object.values(modelConfig).reduce((acc, values) => {
    if (!Array.isArray(values) || values.length === 0) {
      return acc;
    }
    return acc * Math.max(values.length, 1);
  }, 1);
  return Math.max(combinations, 1);
};

const getGridSize = (modelName: string, rows: number, config: TrainingConfig): number => {
  if (config.processingMode === ProcessingMode.CUSTOM) {
    const customSize = getCustomGridSize(modelName, config);
    if (typeof customSize === 'number' && customSize > 0) {
      return customSize;
    }
  }
  const baseSize = computeBaseGridSize(modelName, rows, config);
  if (config.processingMode === ProcessingMode.HARD) {
    return Math.max(Math.round(baseSize * 1.75), baseSize + 4);
  }
  return baseSize;
};

const formatModelStageLabel = (modelName: string, config: TrainingConfig): string => {
  if (modelName === 'Support Vector Machine (SVM)') {
    return `Training ${modelName} (${config.svmFlexibility})`;
  }
  if (modelName === 'Voting Classifier' || modelName === 'Stacking Classifier') {
    return `Training ${modelName} (ensemble)`;
  }
  return `Training ${modelName}`;
};

const computeKMeansCost = (rows: number, featureCount: number, clusters: number): number => {
  const safeClusters = clamp(clusters || 1, 1, 25);
  const clusterFactor = Math.log1p(safeClusters) * (1 + safeClusters * 0.05);
  const scaling = Math.pow(Math.max(1, rows), 0.5) * (0.8 + Math.log1p(Math.max(1, featureCount)));
  return scaling * clusterFactor;
};

const ensureClassCount = (distribution: Record<string, number> | undefined, label: string): number => {
  if (!distribution) return 0;
  if (label in distribution) return distribution[label];
  return 0;
};

const computeCvFolds = (dataSet: DataSet): number => {
  const zeroCount = ensureClassCount(dataSet.classDistribution, '0');
  const oneCount = ensureClassCount(dataSet.classDistribution, '1');
  const hasDistribution = zeroCount > 0 && oneCount > 0;
  if (hasDistribution) {
    const minClass = Math.min(zeroCount, oneCount);
    return clamp(minClass, 2, 10);
  }
  const heuristic = Math.round(Math.log10(Math.max(10, dataSet.rows + 10)) * 2.5);
  return clamp(heuristic || 3, 2, 10);
};

interface StageCost {
  label: string;
  cost: number;
}

export const buildTrainingEstimate = (dataSet: DataSet, config: TrainingConfig): TrainingEstimate => {
  if (!config?.models?.length) {
    return { totalSeconds: 0, stages: [] };
  }

  const rows = Math.max(1, dataSet.rows);
  const featureCount = inferFeatureCount(dataSet);
  const cvFolds = computeCvFolds(dataSet);
  const datasetEffort = computeDatasetEffort(rows, featureCount);

  const stageCosts: StageCost[] = [
    { label: 'Preparing preprocessing pipelines', cost: datasetEffort * 0.45 },
    { label: `Creating ${cvFolds}-fold stratified splits`, cost: datasetEffort * Math.max(0.9, Math.log1p(cvFolds)) },
  ];

  config.models.forEach((modelName) => {
    if (modelName === 'K-Means Clustering') {
      const kMeansCost = computeKMeansCost(rows, featureCount, config.kMeansClusters);
      stageCosts.push({
        label: `Running K-Means Clustering (${config.kMeansClusters} clusters)`,
        cost: kMeansCost,
      });
      return;
    }
    const complexity = MODEL_COMPLEXITY_WEIGHTS[modelName] ?? 1.0;
    const gridSize = getGridSize(modelName, rows, config);
    const gridFactor = Math.pow(Math.max(gridSize, 1), 0.62);
    const foldFactor = Math.log1p(cvFolds) * Math.pow(cvFolds, 0.35);
    const modelCost = datasetEffort * complexity * gridFactor * foldFactor;
    stageCosts.push({
      label: formatModelStageLabel(modelName, config),
      cost: modelCost,
    });
  });

  stageCosts.push({
    label: 'Aggregating metrics & visualizations',
    cost: Math.max(datasetEffort * 0.35, datasetEffort * 0.12 * config.models.length),
  });

  const totalCost = stageCosts.reduce((sum, stage) => sum + stage.cost, 0);
  if (totalCost <= 0) {
    return { totalSeconds: 0, stages: [] };
  }

  const rawSeconds = totalCost * COST_TO_SECONDS;
  const baseSeconds = clamp(Math.round(rawSeconds), MIN_SECONDS, MAX_SECONDS);
  const targetSeconds = Math.max(stageCosts.length, baseSeconds);

  const allocations = stageCosts.map((stage) => {
    const exact = (stage.cost / totalCost) * targetSeconds;
    const floored = Math.floor(exact);
    const seconds = Math.max(1, floored);
    const fraction = exact - floored;
    return { label: stage.label, seconds, fraction };
  });

  let assigned = allocations.reduce((sum, stage) => sum + stage.seconds, 0);
  let delta = targetSeconds - assigned;

  if (delta > 0) {
    const sorted = [...allocations].sort((a, b) => b.fraction - a.fraction);
    let idx = 0;
    while (delta > 0) {
      const candidate = sorted[idx % sorted.length];
      candidate.seconds += 1;
      delta -= 1;
      idx += 1;
    }
  } else if (delta < 0) {
    let deficit = -delta;
    const sorted = [...allocations].sort((a, b) => a.fraction - b.fraction);
    for (const candidate of sorted) {
      if (candidate.seconds <= 1) {
        continue;
      }
      const reducible = Math.min(deficit, candidate.seconds - 1);
      candidate.seconds -= reducible;
      deficit -= reducible;
      if (deficit === 0) {
        break;
      }
    }
  }

  const stages: TrainingStageEstimate[] = allocations.map(({ label, seconds }) => ({
    label,
    seconds,
  }));

  return { totalSeconds: targetSeconds, stages };
};

export const formatTrainingTime = (seconds: number): string => {
  if (seconds <= 0) {
    return 'a few seconds';
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  if (seconds < 3600) {
    const minutes = seconds / 60;
    return minutes >= 10 ? `${minutes.toFixed(0)} min` : `${minutes.toFixed(1)} min`;
  }
  return `${(seconds / 3600).toFixed(1)} h`;
};
