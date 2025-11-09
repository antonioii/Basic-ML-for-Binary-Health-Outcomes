
import React, { useState, useMemo, useEffect } from 'react';
import { DataSet, SvmFlexibility, TrainingConfig } from '../types';
import { fetchKMeansElbow } from '../services/mlService';
import Card from './common/Card';
import Button from './common/Button';
import { SlidersHorizontal, BarChart2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrainingConfigStepProps {
  dataSet: DataSet;
  onConfigComplete: (config: TrainingConfig) => void;
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

const TrainingConfigStep: React.FC<TrainingConfigStepProps> = ({ dataSet, onConfigComplete }) => {
  const [selectedModels, setSelectedModels] = useState<string[]>(DEFAULT_SELECTED_MODELS);
  const [svmFlexibility, setSvmFlexibility] = useState<SvmFlexibility>(SvmFlexibility.MEDIUM);
  
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

  const handleModelToggle = (model: string) => {
    setSelectedModels(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    );
  };
  
  const handleSubmit = () => {
    onConfigComplete({
      models: selectedModels,
      svmFlexibility,
      kMeansClusters,
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

      <div className="flex justify-end mt-10">
        <Button onClick={handleSubmit} text="Train & Evaluate Models" disabled={selectedModels.length === 0} />
      </div>
    </Card>
  );
};

export default TrainingConfigStep;
