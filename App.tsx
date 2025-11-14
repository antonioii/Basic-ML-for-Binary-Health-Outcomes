
import React, { useState } from 'react';
import { AppStep, DataSet, EdaResult, ModelResult, ProcessingMode, TrainingConfig } from './types';
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
  const [processingMode, setProcessingMode] = useState<ProcessingMode>(ProcessingMode.LIGHT);

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
        return (
          <UploadStep
            onDataUploaded={handleDataUploaded}
            processingMode={processingMode}
            onProcessingModeChange={setProcessingMode}
          />
        );
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
          return (
            <TrainingConfigStep
              dataSet={cleanedDataSet}
              onConfigComplete={handleConfigComplete}
              processingMode={processingMode}
              onProcessingModeChange={setProcessingMode}
            />
          );
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
        return (
          <UploadStep
            onDataUploaded={handleDataUploaded}
            processingMode={processingMode}
            onProcessingModeChange={setProcessingMode}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <header className="bg-white shadow-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-3 flex justify-between items-center">
          <div className="flex items-center space-x-3">

            <svg id="Camada_1" data-name="Camada 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" fill="rgb(33,90,255)" width="60" height="60">
		<path d="M1021.78,74.55l-.27,211.29c1.17,18.03,6.37,24.42,15.14,38.96,19.63,32.54,42.98,62.89,62.53,95.5,1.11-24.27-4.75-53.19,3.13-76.59,2.86-8.51,7.05-7.84,9.41-12.06,7.07-12.65,14.7-47.56,20.26-63.62,12.39-35.82,31.16-103.33,59.17-127.34,9.17-7.86,21.49-10.62,28.25-15.88,2.66-2.07,5.14-9.65,9.34-13.44,12.91-11.67,45.22-19.92,62.01-16.46,31.63,6.52,35.24,53.32,8.05,69.43-11.43,6.77-43.75,11.63-56.62,9.04-11.38-2.29-9.68-11.18-21.7-4.77-12.59,6.71-19.69,23.71-24.84,36.37-17.62,43.35-29.75,90.37-46.76,134.09,9.28,12.24.3,28.9-.57,43.53-6.09,102.46,119.71,157.01,188.65,79.02,13.44-15.2,17.71-37.15,37.46-45.12,14.33-35.46,31.7-69.76,45.46-105.46,4.67-12.12,14.51-34.41,14.7-46.63.05-3.26-4.49-22.28-5.9-23.41-.16-.13-2.36,1.04-4.17.69-19.38-3.68-45.23-29.08-52.95-46.77-14.07-32.25,19.46-66.14,51.39-51.39,15.44,7.14,38.48,38.28,39.98,55.4.46,5.29-1.42,11.98-1.01,15.49.49,4.22,10.59,20.96,12.28,31.63,6.47,40.93-27.2,103.81-43.74,142.49-5.06,11.83-18.47,35.41-21.02,46.02-.87,3.61,1.09,6.84.91,10.62-.87,17.75-41.73,60.87-56.58,71.69-36.58,26.66-70.64,31.53-115.12,30.23-4.51-.13-9.1-3.71-11.8.9-2.89,4.93-8.23,22.78-9.51,28.92-4.23,20.35-2.43,22.74,7.25,39.76,39.12,68.74,125.2,155.02,78.21,237.91-1.07,1.89-9.81,16.33-12.12,14.21-1.19-8.93-4.47-17.33-7.89-25.57-2.88-6.96-12.01-21.06-13.16-26.71s.34-11.37-.34-16.74c-2.22-17.33-8.37-33.03-17.89-47.6l-54.68-86.96c-.05,27.01,5.07,54.81,14.58,80.07,17.29,45.93,56.43,82.82,66.13,131.77,17.93,90.44-48.63,174.54-134.23,196.65-113.98,29.45-226.67-47.86-252.66-159.28h-329.59c-94.11-6.52-142.89-92.25-100.58-177.58,20.6-41.54,55.49-89.64,80.83-129.89,57.85-91.88,120.77-183.52,176.3-276.44,9.76-16.34,16.59-26.11,17.98-46.09,3.26-47.02.86-102.88,1.42-150.92.09-7.76.82-56.5-1.67-58.96-44.15-4.74-41.78-70.47,2.27-72.85,89.28,4.18,183.33-5.38,272.04,0,9.17.55,18.59,1.19,26.16,6.59,26.16,18.65,18.35,60.21-13.92,66.28ZM976.02,57.27h-195.05c1.53,8.57,3.77,17.56,4.33,26.28,3.18,49.35-.49,105.55-1.54,155.19-.89,42.02,5.17,78.33-22.12,114.56-79.65,129.92-168.51,254.9-246.08,386.06-13.72,23.2-25.35,41.58-22.03,70.2,3.37,29.05,34.22,54.97,63.2,54.97h321.05c-2.76-3.34-2.79-18.14-4.54-19.67-1.52-1.33-24.87-2.67-29.92-3.53-191.54-32.68-255.39-276.12-110-402.46,115.82-100.64,295.7-53.59,354.42,85.43,56.5,133.76-26.03,290.47-168.67,316.03l2.85,24.2,287.54-2.84c6.17,15.28,12.71,31.39,11.3,48.25l-24.71,4.47-259.88-.04c24.04,78.75,97.86,131.14,181.33,119.39,58.18-8.19,114.88-56.28,119.06-117.28,3.46-50.59-22.46-74.66-44.36-115.06-32.57-60.08-44.86-123.75-31.67-191.74-29.91-48.7-60.34-97.1-91.36-145.1-24.29-37.59-53.3-75.37-75.78-113.58-12.87-21.89-20.2-39.17-21.7-65.15-3.93-67.95,3.13-139.64.01-207.99,1.47-6.9.93-14.27,4.33-20.59ZM860.42,430.71c-170.47,14.27-222.45,242.82-84.16,336.73,116.62,79.2,269.13,3.78,283.76-134.29,12.45-117.57-82.16-212.27-199.6-202.44Z"/>
		<path d="M982.75,494.58c4.97-.81,9.22,4.19,8.91,8.92-26.25,46.34-56.04,91.06-84.65,136.11-40.05,26.08-81.63,66.49-122.73,89.41-3.76,2.1-8.55,5.06-12.69,1.81-5.68-4.47-1.76-10.67.61-15.32,21.92-43.1,55.88-84.62,78.54-127.89,13.32-13.43,29.7-23.64,45.11-34.62,28.4-20.23,57.67-39.38,86.9-58.41ZM890.14,607.28c-10.32-11.3-28.99,3.96-16.69,15.98,9.98,9.76,26.93-4.76,16.69-15.98Z"/>
		<path d="M873.85,767.37l-21.76-69.85,65.55-47.28,40.1-63.17c3.24-1.91,69.02,19.15,71.79,22.47,4.85,5.82.96,11.01-5.09,13.67-28.22,9.46-59.35,15.04-87.16,25.32-16.44,6.08-17.61,5.56-23.19,22.37-8.64,26.06-13.12,70.53-25.26,92.91-3.47,6.39-8.69,6.78-14.99,3.56Z"/>
		<path d="M728.67,620.71c-4.69-4.9-.55-11.2,4.76-13.72,24.14-11.46,60.67-16.75,86.72-25.76,15.51-5.37,17.13-7.87,22.42-23.14,10.48-30.27,16.51-63.54,27-94.02,9.03-12.26,14.76-4.26,18.56,6.43,6.26,17.59,9.51,37.31,16.49,54.85l-.43,3.14-62.75,46.98c-7.6,11.56-13.97,23.88-21.42,35.53-3.07,4.8-18.26,28.94-21.55,29.81-3.5.92-37.92-8.85-44.2-10.8-4.42-1.37-23.55-7.16-25.58-9.28Z"/>

            </svg>

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
