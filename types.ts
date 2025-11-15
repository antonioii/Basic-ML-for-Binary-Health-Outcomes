
export interface ClassDistribution {
  [label: string]: number;
}

export type DataValue = string | number | null;
export type DataRow = Record<string, DataValue>;

export interface DataSet {
  datasetId: string;
  fileName: string;
  rows: number;
  cols: number;
  idColumn: string;
  targetColumn: string;
  featureColumns: string[];
  preview: DataRow[];
  classDistribution?: ClassDistribution;
}

export enum AppStep {
  Upload = 'Upload',
  EDA = 'EDA',
  Cleaning = 'Cleaning',
  Configuration = 'Configuration',
  Results = 'Results',
}

export enum VariableType {
  ID = 'ID',
  BINARY = 'Binary',
  CONTINUOUS = 'Continuous',
  CATEGORICAL = 'Categorical (Low Cardinality)',
  TARGET = 'Target',
}

export interface VariableInfo {
  name: string;
  type: VariableType;
  missing: number;
  distinctValues: number;
  exampleValues?: DataValue[];
}

export interface OutlierInfo {
  variable: string;
  count: number;
  ids: (string | number)[];
}

export interface CorrelationInfo {
  pair: [string, string];
  value: number;
}

export interface EdaResult {
  totalRows: number;
  totalCols: number;
  variableInfo: VariableInfo[];
  targetDistribution: { '0': number; '1': number; imbalance: boolean };
  missingInfo: { variable: string; percentage: number }[];
  outlierInfo: OutlierInfo[];
  correlationInfo: CorrelationInfo[];
  cleaningSuggestions: CleaningSuggestion[];
  histograms: HistogramInfo[];
  boxPlots: BoxPlotInfo[];
}

export type SuggestionType = 'missing' | 'outlier' | 'correlation';

export interface CleaningSuggestion {
  id: string;
  type: SuggestionType;
  description: string;
  details: {
    variable?: string;
    variables?: [string, string];
    value?: number;
    ids?: (string | number)[];
    dropColumn?: string;
    reason?: string;
  };
  apply: boolean;
}

export interface CleaningSummary {
  rowsRemoved: number;
  colsRemoved: string[];
  finalRows: number;
  finalCols: number;
  notes?: string;
}

export enum SvmFlexibility {
    LOW = 'Low (Rigid)',
    MEDIUM = 'Medium (Balanced)',
    HIGH = 'High (Flexible)',
}

export enum ProcessingMode {
  LIGHT = 'light',
  HARD = 'hard',
  CUSTOM = 'custom',
}

export type CustomHyperparameterValues = (string | number | boolean | null)[];

export type CustomHyperparameterConfig = Record<string, Record<string, CustomHyperparameterValues>>;

export enum ClassBalanceMethod {
  SMOTE = 'smote',
  OVERSAMPLE = 'oversample',
}

export interface ClassBalanceConfig {
  enabled: boolean;
  method: ClassBalanceMethod;
}

export interface TrainingConfig {
  models: string[];
  svmFlexibility: SvmFlexibility;
  kMeansClusters: number;
  processingMode: ProcessingMode;
  customHyperparameters?: CustomHyperparameterConfig;
  classBalance?: ClassBalanceConfig;
}

export interface ConfusionMatrix {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
}

export interface ModelMetrics {
    confusionMatrix: ConfusionMatrix;
    sensitivity: number;
    sensitivityStd?: number | null;
    specificity: number;
    specificityStd?: number | null;
    vpp: number;
    vppStd?: number | null;
    vpn: number;
    vpnStd?: number | null;
    f1Score: number;
    f1ScoreStd?: number | null;
    auc: number | null;
    aucStd?: number | null;
    accuracy: number;
    accuracyStd?: number | null;
}

export interface RocPoint {
    fpr: number;
    tpr: number;
}

export type HyperparameterValue = string | number | boolean | null;

export interface KMeansResult {
    elbowPlot: { k: number; inertia: number }[];
    clusters: Record<string, (string|number)[]>;
    clusterAnalysis: { cluster: number; count: number; positiveRate: number }[];
}

export interface HistogramInfo {
  variable: string;
  bins: number[];
  counts: number[];
}

export interface BoxPlotInfo {
  variable: string;
  median: number;
  q1: number;
  q3: number;
  lowerWhisker: number;
  upperWhisker: number;
  outliers: (string | number)[];
}


export interface ModelResult {
    name: string;
    metrics: ModelMetrics | null; // Null for unsupervised models
    rocCurve: RocPoint[] | null; // Null for unsupervised models
    kMeansResult?: KMeansResult;
    featureImportances?: { feature: string; importance: number }[];
    hyperparameters: Record<string, HyperparameterValue>;
    statusMessage?: string;
}
