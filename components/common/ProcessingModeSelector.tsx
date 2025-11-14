import React from 'react';
import { Flame, Sliders, Zap } from 'lucide-react';
import { ProcessingMode } from '../../types';

interface ProcessingModeOption {
  value: ProcessingMode;
  title: string;
  description: string;
  helper: string;
  icon: React.ReactNode;
}

const PROCESSING_MODE_OPTIONS: ProcessingModeOption[] = [
  {
    value: ProcessingMode.LIGHT,
    title: 'Light Processing',
    description: 'Uses curated grids for balanced runtime and accuracy.',
    helper: 'Matches the previous default behavior.',
    icon: <Zap className="w-6 h-6 text-green-600" />,
  },
  {
    value: ProcessingMode.HARD,
    title: 'Hard Processing',
    description: 'Explores wider hyperparameter grids for maximum accuracy.',
    helper: 'Expect significantly longer training times.',
    icon: <Flame className="w-6 h-6 text-orange-500" />,
  },
  {
    value: ProcessingMode.CUSTOM,
    title: 'Custom Processing',
    description: 'Provide your own hyperparameter ranges per model.',
    helper: 'Ideal when you already know promising ranges.',
    icon: <Sliders className="w-6 h-6 text-blue-600" />,
  },
];

interface ProcessingModeSelectorProps {
  value: ProcessingMode;
  onChange: (mode: ProcessingMode) => void;
  disabled?: boolean;
  className?: string;
}

const ProcessingModeSelector: React.FC<ProcessingModeSelectorProps> = ({
  value,
  onChange,
  disabled = false,
  className = '',
}) => {
  return (
    <div className={`grid gap-4 md:grid-cols-3 ${className}`}>
      {PROCESSING_MODE_OPTIONS.map((option) => {
        const isActive = value === option.value;
        return (
          <label
            key={option.value}
            className={`relative flex cursor-pointer flex-col rounded-xl border p-4 transition-colors ${
              isActive ? 'border-blue-500 bg-blue-50 shadow-sm' : 'border-gray-200 bg-white hover:border-blue-300'
            } ${disabled ? 'opacity-60 cursor-not-allowed' : ''}`}
          >
            <input
              type="radio"
              name="processing-mode"
              className="sr-only"
              value={option.value}
              checked={isActive}
              disabled={disabled}
              onChange={() => onChange(option.value)}
            />
            <div className="flex items-center space-x-3">
              {option.icon}
              <div>
                <p className="text-sm font-semibold text-gray-800">{option.title}</p>
                <p className="text-xs text-gray-500">{option.helper}</p>
              </div>
            </div>
            <p className="mt-3 text-sm text-gray-600">{option.description}</p>
          </label>
        );
      })}
    </div>
  );
};

export default ProcessingModeSelector;
