
import React, { useState, useMemo, useEffect } from 'react';
import { CustomHyperparameterConfig, DataSet, ProcessingMode, SvmFlexibility, TrainingConfig } from '../types';
import { fetchKMeansElbow } from '../services/mlService';
import Card from './common/Card';
import Button from './common/Button';
import { SlidersHorizontal, BarChart2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import ProcessingModeSelector from './common/ProcessingModeSelector';

interface TrainingConfigStepProps {
  dataSet: DataSet;
  onConfigComplete: (config: TrainingConfig) => void;
  processingMode: ProcessingMode;
  onProcessingModeChange: (mode: ProcessingMode) => void;
}

interface ModelOption {
  value: string;
  label: string;
  helper?: string;
  defaultSelected?: boolean;
}

const MODEL_OPTIONS: ModelOption[] = [
  { value: "Logistic Regression", label: "Logistic Regression", defaultSelected: true },
  { value: "Elastic Net (Logistic Regression)", label: "Elastic Net (Logistic Regression)", helper: "Uses saga solver with elastic net penalty" },
  { value: "K-Nearest Neighbors (KNN)", label: "K-Nearest Neighbors (KNN)", defaultSelected: true },
  { value: "Support Vector Machine (SVM)", label: "Support Vector Machine (SVM)", defaultSelected: true },
  { value: "Random Forest", label: "Random Forest", defaultSelected: true },
  { value: "Gradient Boosting", label: "Gradient Boosting", defaultSelected: true },
  { value: "XGBoost", label: "XGBoost", helper: "Pré-instalado pelo launcher (pacote `xgboost`)" },
  { value: "LightGBM", label: "LightGBM", helper: "Pré-instalado pelo launcher (pacote `lightgbm`)" },
  { value: "CatBoost", label: "CatBoost", helper: "Pré-instalado pelo launcher (pacote `catboost`)" },
  { value: "Naive Bayes (Gaussian)", label: "Naive Bayes (Gaussian)", helper: "Gaussian NB accepts negative features; no non-negative restriction" },
  { value: "Voting Classifier", label: "Voting Classifier", helper: "Requires ≥ 2 trained supervised models" },
  { value: "Stacking Classifier", label: "Stacking Classifier", helper: "Requires ≥ 2 trained supervised models" },
  { value: "K-Means Clustering", label: "K-Means Clustering", defaultSelected: true },
];

const DEFAULT_SELECTED_MODELS = MODEL_OPTIONS.filter(option => option.defaultSelected).map(option => option.value);

interface HyperparameterField {
  name: string;
  label: string;
  placeholder: string;
  helper?: string;
}

const MODEL_HYPERPARAMETERS: Record<string, HyperparameterField[]> = {
  'Logistic Regression': [
    { name: 'C', label: 'Regularization strength (C)', placeholder: '0.01, 0.1, 1, 10' },
  ],
  'Elastic Net (Logistic Regression)': [
    { name: 'C', label: 'Regularization strength (C)', placeholder: '0.01, 0.1, 1' },
    { name: 'l1_ratio', label: 'L1 Ratio', placeholder: '0.2, 0.5, 0.8' },
  ],
  'K-Nearest Neighbors (KNN)': [
    { name: 'n_neighbors', label: 'Neighbors (odd numbers recommended)', placeholder: '3, 5, 7, 9, 11' },
    { name: 'weights', label: 'Weights', placeholder: 'uniform, distance', helper: 'Comma-separated values' },
  ],
  'Support Vector Machine (SVM)': [
    { name: 'C', label: 'Regularization strength (C)', placeholder: '0.1, 1, 10, 50' },
    { name: 'gamma', label: 'Gamma', placeholder: '0.001, 0.01, 0.1' },
    { name: 'kernel', label: 'Kernel', placeholder: 'rbf, linear' },
  ],
  'Random Forest': [
    { name: 'n_estimators', label: 'Number of Trees', placeholder: '200, 400' },
    { name: 'max_depth', label: 'Max Depth', placeholder: 'None, 10, 20' },
    { name: 'min_samples_split', label: 'Min Samples Split', placeholder: '2, 5' },
    { name: 'min_samples_leaf', label: 'Min Samples Leaf', placeholder: '1, 2, 4' },
  ],
  'Gradient Boosting': [
    { name: 'learning_rate', label: 'Learning Rate', placeholder: '0.05, 0.1' },
    { name: 'n_estimators', label: 'Estimators', placeholder: '100, 200' },
    { name: 'max_depth', label: 'Max Depth', placeholder: '3, 5' },
    { name: 'subsample', label: 'Subsample', placeholder: '1.0, 0.8' },
  ],
  'XGBoost': [
    { name: 'n_estimators', label: 'Estimators', placeholder: '200, 400' },
    { name: 'max_depth', label: 'Max Depth', placeholder: '3, 5' },
    { name: 'learning_rate', label: 'Learning Rate', placeholder: '0.05, 0.1' },
    { name: 'subsample', label: 'Subsample', placeholder: '0.8, 1.0' },
    { name: 'colsample_bytree', label: 'Column Subsample', placeholder: '0.7, 0.9, 1.0' },
  ],
  'LightGBM': [
    { name: 'n_estimators', label: 'Estimators', placeholder: '200, 400' },
    { name: 'learning_rate', label: 'Learning Rate', placeholder: '0.05, 0.1' },
    { name: 'num_leaves', label: 'Num Leaves', placeholder: '31, 63' },
    { name: 'max_depth', label: 'Max Depth', placeholder: '-1, 15' },
    { name: 'subsample', label: 'Subsample', placeholder: '0.8, 1.0' },
  ],
  'CatBoost': [
    { name: 'iterations', label: 'Iterations', placeholder: '200, 400' },
    { name: 'learning_rate', label: 'Learning Rate', placeholder: '0.05, 0.1' },
    { name: 'depth', label: 'Depth', placeholder: '4, 6' },
    { name: 'l2_leaf_reg', label: 'L2 Leaf Reg', placeholder: '3, 5, 7' },
  ],
  'Naive Bayes (Gaussian)': [
    { name: 'var_smoothing', label: 'Variance Smoothing', placeholder: '1e-9, 1e-8, 1e-7' },
  ],
  'Stacking Classifier': [
    { name: 'final_estimator__C', label: 'Meta-estimator C', placeholder: '0.5, 1.0' },
  ],
};

type CustomHyperparamInputs = Record<string, Record<string, string>>;

const TrainingConfigStep: React.FC<TrainingConfigStepProps> = ({ dataSet, onConfigComplete, processingMode, onProcessingModeChange }) => {
  const [selectedModels, setSelectedModels] = useState<string[]>(DEFAULT_SELECTED_MODELS);
  const [svmFlexibility, setSvmFlexibility] = useState<SvmFlexibility>(SvmFlexibility.MEDIUM);
  const [customInputs, setCustomInputs] = useState<CustomHyperparamInputs>({});
  
  const [elbowData, setElbowData] = useState<{ k: number; inertia: number }[]>([]);
  const [elbowLoading, setElbowLoading] = useState<boolean>(false);
  const [elbowError, setElbowError] = useState<string | null>(null);

  useEffect(() => {
    const loadElbow = async () => {
      try {
        setElbowLoading(true);
        const data = await fetchKMeansElbow(dataSet.datasetId);
        setElbowData(data);
      } catch (error) {
        setElbowError((error as Error).message);
      } finally {
        setElbowLoading(false);
      }
    };
    loadElbow();
  }, [dataSet.datasetId]);

  const suggestedK = useMemo(() => {
    if (elbowData.length < 3) return 2;
    let maxDiff = 0;
    let bestK = 2;
    for (let i = 1; i < elbowData.length - 1; i++) {
      const diff = (elbowData[i-1].inertia - elbowData[i].inertia) - (elbowData[i].inertia - elbowData[i+1].inertia);
      if (diff > maxDiff) {
        maxDiff = diff;
        bestK = elbowData[i].k;
      }
    }
    return bestK;
  }, [elbowData]);

  const [kMeansClusters, setKMeansClusters] = useState<number>(suggestedK);

  useEffect(() => {
    setKMeansClusters(suggestedK);
  }, [suggestedK]);

  useEffect(() => {
    setCustomInputs(prev => {
      const next: CustomHyperparamInputs = { ...prev };
      Object.keys(next).forEach(model => {
        if (!selectedModels.includes(model)) {
          delete next[model];
        }
      });
      selectedModels.forEach(model => {
        const fields = MODEL_HYPERPARAMETERS[model];
        if (!fields || fields.length === 0) {
          return;
        }
        if (!next[model]) {
          next[model] = {};
        }
        fields.forEach(field => {
          if (!(field.name in next[model])) {
            next[model][field.name] = '';
          }
        });
      });
      return next;
    });
  }, [selectedModels]);

  const handleModelToggle = (model: string) => {
    setSelectedModels(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    );
  };

  const handleCustomInputChange = (model: string, field: string, value: string) => {
    setCustomInputs(prev => ({
      ...prev,
      [model]: {
        ...(prev[model] || {}),
        [field]: value,
      },
    }));
  };

  const buildCustomPayload = (): CustomHyperparameterConfig | undefined => {
    if (processingMode !== ProcessingMode.CUSTOM) {
      return undefined;
    }
    const payload: CustomHyperparameterConfig = {};
    Object.entries(customInputs).forEach(([model, params]) => {
      const fields = MODEL_HYPERPARAMETERS[model];
      if (!fields || fields.length === 0) {
        return;
      }
      const cleaned: Record<string, (string | number | boolean | null)[]> = {};
      Object.entries(params).forEach(([param, raw]) => {
        const tokens = raw.split(',').map(token => token.trim()).filter(Boolean);
        if (tokens.length > 0) {
          cleaned[param] = tokens;
        }
      });
      if (Object.keys(cleaned).length > 0) {
        payload[model] = cleaned;
      }
    });
    return Object.keys(payload).length > 0 ? payload : undefined;
  };
  
  const handleSubmit = () => {
    onConfigComplete({
      models: selectedModels,
      svmFlexibility,
      kMeansClusters,
      processingMode,
      customHyperparameters: buildCustomPayload(),
    });
  };

  return (
    <Card>
      <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center">
        <SlidersHorizontal className="w-7 h-7 mr-3 text-blue-600" />
        Configure Model Training
      </h2>
      <p className="text-gray-600 mb-8">
        Select the models to train and specify any required configurations.
      </p>

      <div className="mb-10">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Processing Intensity</h3>
        <ProcessingModeSelector value={processingMode} onChange={onProcessingModeChange} />
        <p className="mt-2 text-sm text-gray-500">
          Light maintains the curated defaults. Hard expands every grid for more exhaustive searches, which can take hours
          on large datasets. Custom lets you provide comma-separated hyperparameter values for each model below.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Select Models</h3>
          <div className="space-y-3">
            {MODEL_OPTIONS.map(option => (
              <div key={option.value} className="flex items-start bg-gray-50 p-3 rounded-lg border">
                <input
                  type="checkbox"
                  id={option.value}
                  checked={selectedModels.includes(option.value)}
                  onChange={() => handleModelToggle(option.value)}
                  className="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor={option.value} className="ml-3 text-sm text-gray-800 cursor-pointer flex-1">
                  <span className="font-medium">{option.label}</span>
                  {option.helper && <span className="block text-xs text-gray-500 mt-1">{option.helper}</span>}
                </label>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-8">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">SVM Flexibility</h3>
            <div className="flex flex-col space-y-2">
              {Object.values(SvmFlexibility).map(level => (
                <label key={level} className="flex items-center p-3 bg-gray-50 rounded-lg border cursor-pointer">
                  <input
                    type="radio"
                    name="svm-flexibility"
                    value={level}
                    checked={svmFlexibility === level}
                    onChange={() => setSvmFlexibility(level)}
                    className="h-4 w-4 border-gray-300 text-blue-600 focus:ring-blue-500"
                    disabled={!selectedModels.includes("Support Vector Machine (SVM)")}
                  />
                  <span className={`ml-3 text-sm font-medium ${!selectedModels.includes("Support Vector Machine (SVM)") ? 'text-gray-400' : 'text-gray-700'}`}>{level}</span>
                </label>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <BarChart2 className="w-5 h-5 mr-2" /> K-Means Elbow Method
            </h3>
            <div className="h-64 w-full bg-gray-50 p-2 rounded-lg border flex items-center justify-center">
              {elbowLoading ? (
                <p className="text-gray-500">Computing elbow curve...</p>
              ) : elbowError ? (
                <p className="text-red-600 text-sm">{elbowError}</p>
              ) : elbowData.length === 0 ? (
                <p className="text-gray-500 text-sm">Insufficient data to compute elbow curve.</p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={elbowData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="k" label={{ value: 'Number of Clusters (k)', position: 'insideBottom', offset: -5, fontSize: 12 }} tick={{fontSize: 12}}/>
                    <YAxis label={{ value: 'Inertia', angle: -90, position: 'insideLeft', fontSize: 12 }} tick={{fontSize: 12}} />
                    <Tooltip />
                    <Legend wrapperStyle={{fontSize: "12px"}}/>
                    <Line type="monotone" dataKey="inertia" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
            <div className="mt-4">
                <label htmlFor="kmeans-k" className="block text-sm font-medium text-gray-700">
                    Select number of clusters (k): <span className="text-blue-600 font-bold">(Suggested: {suggestedK})</span>
                </label>
                <input
                  type="number"
                  id="kmeans-k"
                  value={kMeansClusters}
                  onChange={(e) => setKMeansClusters(Number(e.target.value))}
                  min="2"
                  max="10"
                  className="mt-1 block w-full rounded-md border-gray-300 bg-gray-50 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2"
                  disabled={!selectedModels.includes("K-Means Clustering") || elbowData.length === 0}
                />
            </div>
          </div>
        </div>
      </div>

      {processingMode === ProcessingMode.CUSTOM && (
        <div className="mt-10 border-t border-gray-200 pt-8 space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Custom Hyperparameters</h3>
            <p className="text-sm text-gray-500 mt-1">
              Enter comma-separated values for each parameter (e.g., <span className="font-mono">0.01, 0.1, 1</span>). Use{' '}
              <span className="font-mono">None</span> to include null values or booleans like <span className="font-mono">true</span>.
            </p>
          </div>
          {selectedModels.map(model => {
            const fields = MODEL_HYPERPARAMETERS[model] || [];
            const values = customInputs[model] || {};
            return (
              <div key={model} className="rounded-xl border border-gray-200 bg-white p-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-base font-semibold text-gray-800">{model}</h4>
                  <span className="text-xs uppercase tracking-wide text-gray-500">
                    {fields.length ? 'Custom grid' : 'Defaults applied'}
                  </span>
                </div>
                {fields.length === 0 ? (
                  <p className="mt-3 text-sm text-gray-500">
                    This model currently uses a fixed configuration. No manual hyperparameters are exposed.
                  </p>
                ) : (
                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    {fields.map(field => {
                      const inputId = `${model}-${field.name}`.replace(/\s+/g, '-').toLowerCase();
                      return (
                        <div key={`${model}-${field.name}`}>
                          <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor={inputId}>
                            {field.label}
                          </label>
                          <input
                            id={inputId}
                            type="text"
                            value={values[field.name] ?? ''}
                            onChange={(event) => handleCustomInputChange(model, field.name, event.target.value)}
                            placeholder={field.placeholder}
                            className="w-full rounded-md border border-gray-300 bg-gray-50 p-2 text-sm focus:border-blue-500 focus:ring-blue-500"
                          />
                          <p className="mt-1 text-xs text-gray-500">
                            {field.helper || 'Comma-separated values'}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="flex justify-end mt-10">
        <Button onClick={handleSubmit} text="Train & Evaluate Models" disabled={selectedModels.length === 0} />
      </div>
    </Card>
  );
};

export default TrainingConfigStep;
