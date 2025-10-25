
import React, { useState } from 'react';
import { DataSet, EdaResult, CleaningSuggestion, CleaningSummary } from '../types';
import { applyCleaning } from '../services/dataService';
import Card from './common/Card';
import Button from './common/Button';
import { Check, Trash2, ShieldCheck, ListChecks, ChevronDown, ChevronUp } from 'lucide-react';
import Spinner from './common/Spinner';

interface CleaningStepProps {
  dataSet: DataSet;
  edaResult: EdaResult;
  onCleaningComplete: (cleanedDataSet: DataSet) => void;
}

const SuggestionItem: React.FC<{suggestion: CleaningSuggestion, checked: boolean, onChange: () => void}> = ({ suggestion, checked, onChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    const hasDetails = suggestion.type === 'outlier' && suggestion.details.ids && suggestion.details.ids.length > 0;

    return (
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
            <div className="flex items-start">
                <input
                    type="checkbox"
                    id={suggestion.id}
                    checked={checked}
                    onChange={onChange}
                    className="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 mt-1"
                />
                <div className="ml-3 text-sm flex-1">
                    <label htmlFor={suggestion.id} className="font-medium text-gray-800 cursor-pointer">{suggestion.description}</label>
                    {suggestion.details.reason && (
                      <p className="text-xs text-gray-500 mt-1">{suggestion.details.reason}</p>
                    )}
                    {hasDetails && (
                        <button onClick={() => setIsOpen(!isOpen)} className="text-blue-600 text-xs ml-2 hover:underline flex items-center">
                            {isOpen ? 'Hide IDs' : 'Show IDs'}
                            {isOpen ? <ChevronUp className="w-3 h-3 ml-1" /> : <ChevronDown className="w-3 h-3 ml-1" />}
                        </button>
                    )}
                </div>
            </div>
            {isOpen && hasDetails && (
                <div className="mt-2 ml-8 p-2 bg-white rounded border text-xs text-gray-600 max-h-24 overflow-y-auto">
                    IDs affected: {suggestion.details.ids?.join(', ')}
                </div>
            )}
        </div>
    );
};


const CleaningStep: React.FC<CleaningStepProps> = ({ dataSet, edaResult, onCleaningComplete }) => {
  const [suggestions, setSuggestions] = useState<CleaningSuggestion[]>(edaResult.cleaningSuggestions.map(s => ({ ...s, apply: true })));
  const [cleaningSummary, setCleaningSummary] = useState<CleaningSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleToggle = (id: string) => {
    setSuggestions(prev =>
      prev.map(s => (s.id === id ? { ...s, apply: !s.apply } : s))
    );
  };
  
  const handleApplyCleaning = async () => {
    try {
      setLoading(true);
      setError(null);
      const selectedIds = suggestions.filter(s => s.apply).map(s => s.id);
      const response = await applyCleaning(dataSet.datasetId, selectedIds);
      setCleaningSummary(response.summary);
      setTimeout(() => onCleaningComplete(response.cleanedDataSet), 800);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
     return (
        <Card>
            <div className="flex flex-col items-center justify-center h-64">
            <Spinner />
            <p className="mt-4 text-lg font-medium text-gray-700">Applying data cleaning...</p>
            <p className="text-gray-500">Generating the analysis-ready dataset.</p>
            </div>
      </Card>
     );
  }

  if (cleaningSummary) {
    return (
        <Card>
            <div className="text-center">
                <ShieldCheck className="w-16 h-16 text-green-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Cleaning Complete</h2>
                <p className="text-gray-600 mb-6">The dataset is now ready for model training.</p>
                <div className="grid grid-cols-2 gap-4 text-left p-4 bg-gray-100 rounded-lg">
                    <p>Rows Removed:</p><p className="font-mono text-right">{cleaningSummary.rowsRemoved}</p>
                    <p>Columns Removed:</p><p className="font-mono text-right">{cleaningSummary.colsRemoved.length > 0 ? cleaningSummary.colsRemoved.join(', ') : '0'}</p>
                    <p>Final Sample Size:</p><p className="font-mono text-right">{cleaningSummary.finalRows}</p>
                    <p>Final Feature Count:</p><p className="font-mono text-right">{cleaningSummary.finalCols}</p>
                    {cleaningSummary.notes && <><p>Notes:</p><p className="text-left text-gray-600 col-span-2">{cleaningSummary.notes}</p></>}
                </div>
            </div>
        </Card>
    )
  }

  return (
    <Card>
      <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center"><ListChecks className="w-7 h-7 mr-3 text-blue-600"/>Data Cleaning Suggestions</h2>
      <p className="text-gray-600 mb-6">
        Based on the EDA, we recommend the following actions. Uncheck any you wish to skip.
      </p>

      {suggestions.length > 0 ? (
        <div className="space-y-4">
          {suggestions.map(suggestion => (
            <SuggestionItem 
                key={suggestion.id}
                suggestion={suggestion}
                checked={suggestion.apply}
                onChange={() => handleToggle(suggestion.id)}
            />
          ))}
        </div>
      ) : (
        <div className="text-center p-8 bg-green-50 border border-green-200 rounded-lg">
            <Check className="w-12 h-12 text-green-500 mx-auto mb-3" />
            <h3 className="text-lg font-medium text-green-800">No cleaning required!</h3>
            <p className="text-gray-600">Your dataset looks clean based on our automated checks.</p>
        </div>
      )}
      
      {error && (
        <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg" role="alert">
          {error}
        </div>
      )}

      <div className="flex justify-end mt-8">
        <Button
            onClick={handleApplyCleaning}
            text="Apply Cleaning & Continue"
            icon={Trash2}
            disabled={loading}
        />
      </div>
    </Card>
  );
};

export default CleaningStep;
