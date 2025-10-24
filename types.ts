
export type DataRow = Record<string, string | number>;

export interface DataSet {
  fileName: string;
  data: DataRow[];
  idCol: string;
  targetCol: string;
  featureCols: string[];
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
  };
  apply: boolean;
}

export interface CleaningSummary {
  rowsRemoved: number;
  colsRemoved: string[];
  finalRows: number;
  finalCols: number;
}

export enum SvmFlexibility {
    LOW = 'Low (Rigid)',
    MEDIUM = 'Medium (Balanced)',
    HIGH = 'High (Flexible)',
}

export interface TrainingConfig {
    models: string[];
    svmFlexibility: SvmFlexibility;
    kMeansClusters: number;
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
    specificity: number;
    vpp: number;
    vpn: number;
    f1Score: number;
    auc: number;
    accuracy: number;
}

export interface RocPoint {
    fpr: number;
    tpr: number;
}

export interface KMeansResult {
    elbowPlot: { k: number; inertia: number }[];
    clusters: Record<string, (string|number)[]>;
    clusterAnalysis: { cluster: number; count: number; positiveRate: number }[];
}


export interface ModelResult {
    name: string;
    metrics: ModelMetrics | null; // Null for unsupervised models
    rocCurve: RocPoint[] | null; // Null for unsupervised models
    kMeansResult?: KMeansResult;
    featureImportances?: { feature: string; importance: number }[];
}
