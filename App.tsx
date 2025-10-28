
import React, { useState } from 'react';
import { AppStep, DataSet, EdaResult, ModelResult, TrainingConfig } from './types';
import UploadStep from './components/UploadStep';
import EdaStep from './components/EdaStep';
import CleaningStep from './components/CleaningStep';
import TrainingConfigStep from './components/TrainingConfigStep';
import ResultsStep from './components/ResultsStep';
import StepIndicator from './components/common/StepIndicator';
import { Github } from 'lucide-react';

const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<AppStep>(AppStep.Upload);
  const [dataSet, setDataSet] = useState<DataSet | null>(null);
  const [edaResult, setEdaResult] = useState<EdaResult | null>(null);
  const [cleanedDataSet, setCleanedDataSet] = useState<DataSet | null>(null);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig | null>(null);
  const [modelResults, setModelResults] = useState<ModelResult[] | null>(null);

  const handleDataUploaded = (data: DataSet) => {
    setDataSet(data);
    setCurrentStep(AppStep.EDA);
  };

  const handleEdaComplete = (result: EdaResult) => {
    setEdaResult(result);
    setCurrentStep(AppStep.Cleaning);
  };

  const handleCleaningComplete = (cleanedData: DataSet) => {
    setCleanedDataSet(cleanedData);
    setCurrentStep(AppStep.Configuration);
  };

  const handleConfigComplete = (config: TrainingConfig) => {
    setTrainingConfig(config);
    setCurrentStep(AppStep.Results);
  };

  const handleResultsComplete = (results: ModelResult[]) => {
    setModelResults(results);
  };

  const handleRestart = () => {
    setDataSet(null);
    setEdaResult(null);
    setCleanedDataSet(null);
    setTrainingConfig(null);
    setModelResults(null);
    setCurrentStep(AppStep.Upload);
  };

  const renderStep = () => {
    switch (currentStep) {
      case AppStep.Upload:
        return <UploadStep onDataUploaded={handleDataUploaded} />;
      case AppStep.EDA:
        if (dataSet) {
          return <EdaStep dataSet={dataSet} onEdaComplete={handleEdaComplete} />;
        }
        return null;
      case AppStep.Cleaning:
        if (dataSet && edaResult) {
          return <CleaningStep dataSet={dataSet} edaResult={edaResult} onCleaningComplete={handleCleaningComplete} />;
        }
        return null;
      case AppStep.Configuration:
        if (cleanedDataSet) {
          return <TrainingConfigStep dataSet={cleanedDataSet} onConfigComplete={handleConfigComplete} />;
        }
        return null;
      case AppStep.Results:
        if (cleanedDataSet && trainingConfig) {
          return <ResultsStep 
            dataSet={cleanedDataSet} 
            trainingConfig={trainingConfig} 
            onResultsComplete={handleResultsComplete} 
            onRestart={handleRestart}
            results={modelResults} />;
        }
        return null;
      default:
        return <UploadStep onDataUploaded={handleDataUploaded} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <header className="bg-white shadow-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-3 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600"><path d="M12 11V3"/><path d="m16 5-8 8"/><path d="M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18Z"/></svg>
            <h1 className="text-xl font-bold text-gray-800">Health Data Science ML Pipeline</h1>
          </div>
           <a href="https://github.com/antonioii/Basic-ML-for-Binary-Health-Outcomes" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-gray-800 transition-colors">
              <Github size={24} />
           </a>
        </div>
      </header>
      
      <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <StepIndicator currentStep={currentStep} />
        <div className="mt-8">
          {renderStep()}
        </div>
      </main>

      <footer className="bg-white border-t border-gray-200 py-4">
        <div className="container mx-auto text-center text-sm text-gray-500">
          <p>&copy; {new Date().getFullYear()} AI-Powered Data Analysis Tool. All Rights Reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
