
import React from 'react';
import { AppStep } from '../../types';
import { Upload, Microscope, Wand2, SlidersHorizontal, BarChart3 } from 'lucide-react';

interface StepIndicatorProps {
  currentStep: AppStep;
}

const STEPS = [
  { id: AppStep.Upload, name: 'Upload Data', icon: Upload },
  { id: AppStep.EDA, name: 'EDA', icon: Microscope },
  { id: AppStep.Cleaning, name: 'Clean Data', icon: Wand2 },
  { id: AppStep.Configuration, name: 'Configure', icon: SlidersHorizontal },
  { id: AppStep.Results, name: 'Results', icon: BarChart3 },
];

const StepIndicator: React.FC<StepIndicatorProps> = ({ currentStep }) => {
  const currentStepIndex = STEPS.findIndex(step => step.id === currentStep);

  return (
    <nav aria-label="Progress">
      <ol role="list" className="flex items-center">
        {STEPS.map((step, stepIdx) => (
          <li key={step.name} className={`relative ${stepIdx !== STEPS.length - 1 ? 'flex-1' : ''}`}>
            {stepIdx < currentStepIndex ? (
              // Completed Step
              <>
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="h-0.5 w-full bg-blue-600" />
                </div>
                <a href="#" className="relative flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 hover:bg-blue-900">
                  <step.icon className="h-5 w-5 text-white" aria-hidden="true" />
                  <span className="absolute -bottom-6 text-xs text-center w-24 text-gray-600">{step.name}</span>
                </a>
              </>
            ) : stepIdx === currentStepIndex ? (
              // Current Step
              <>
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="h-0.5 w-full bg-gray-200" />
                </div>
                <a href="#" className="relative flex h-8 w-8 items-center justify-center rounded-full border-2 border-blue-600 bg-white" aria-current="step">
                  <span className="h-2.5 w-2.5 rounded-full bg-blue-600" aria-hidden="true" />
                  <span className="absolute -bottom-6 text-xs font-semibold text-center w-24 text-blue-600">{step.name}</span>
                </a>
              </>
            ) : (
              // Upcoming Step
              <>
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="h-0.5 w-full bg-gray-200" />
                </div>
                <a href="#" className="group relative flex h-8 w-8 items-center justify-center rounded-full border-2 border-gray-300 bg-white hover:border-gray-400">
                  <span className="h-2.5 w-2.5 rounded-full bg-transparent group-hover:bg-gray-300" aria-hidden="true" />
                  <span className="absolute -bottom-6 text-xs text-center w-24 text-gray-500">{step.name}</span>
                </a>
              </>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
};

export default StepIndicator;
