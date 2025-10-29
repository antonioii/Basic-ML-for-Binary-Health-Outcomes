
import React, { useState, useCallback } from 'react';
import { DataSet, DataRow } from '../types';
import { UploadCloud, FileCheck2, AlertTriangle } from 'lucide-react';
import Spinner from './common/Spinner';
import Card from './common/Card';
import { uploadDataset } from '../services/edaService';


interface UploadStepProps {
  onDataUploaded: (dataSet: DataSet) => void;
}

const UploadStep: React.FC<UploadStepProps> = ({ onDataUploaded }) => {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dataSetPreview, setDataSetPreview] = useState<DataRow[] | null>(null);

  const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setFileName(file.name);

    try {
      const dataset = await uploadDataset(file);
      setDataSetPreview(dataset.preview);
      onDataUploaded(dataset);
    } catch (err) {
      setError((err as Error).message);
      setFileName(null);
      setDataSetPreview(null);
    } finally {
      setLoading(false);
    }
  }, [onDataUploaded]);

  return (
    <Card>
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Your Dataset</h2>
        <p className="text-gray-600 mb-6">Select a pre-processed database (excel, .xlsx) file to begin the analysis.</p>
        
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 hover:border-blue-500 transition-colors duration-300 bg-gray-50">
          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept=".xlsx, .csv"
            onChange={handleFileChange}
            disabled={loading}
          />
          <label htmlFor="file-upload" className={`cursor-pointer ${loading ? 'cursor-not-allowed' : ''}`}>
            {loading ? (
                <div className="flex flex-col items-center">
                    <Spinner/>
                    <p className="mt-4 text-lg font-medium text-gray-700">Analyzing file structure...</p>
                    <p className="text-gray-500">{fileName}</p>
                </div>
            ) : fileName ? (
                <div className="flex flex-col items-center text-green-600">
                    <FileCheck2 className="w-16 h-16" />
                    <p className="mt-4 text-lg font-medium">{fileName} is valid.</p>
                    <p className="text-gray-500">Proceeding to analysis...</p>
                </div>
            ) : (
                <div className="flex flex-col items-center text-gray-500 hover:text-blue-600">
                    <UploadCloud className="w-16 h-16" />
                    <p className="mt-4 text-lg font-medium">Click to browse or drag & drop</p>
                    <p className="text-sm">XLSX or CSV files only</p>
                </div>
            )}
          </label>
        </div>

        {error && (
          <div className="mt-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative flex items-start" role="alert">
            <AlertTriangle className="w-5 h-5 mr-3 mt-1 flex-shrink-0" />
            <div>
              <strong className="font-bold">Validation Error:</strong>
              <span className="block sm:inline ml-2">{error}</span>
            </div>
          </div>
        )}

        <div className="mt-8 text-left bg-blue-50 border border-blue-200 p-6 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-800 mb-3">Input File Requirements</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li><strong>First Column:</strong> Unique IDs for each sample.</li>
                <li><strong>Last Column:</strong> Binary target variable (must be 0 or 1).</li>
                <li><strong>Intermediate Columns:</strong> Feature variables, must be fully numeric (continuous, binary 0/1, or one-hot encoded).</li>
                <li>No missing values are allowed in the ID or Target columns.</li>
            </ul>
        </div>
        {fileName && !loading && !error && (
          <div className="mt-8 text-left">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Dataset Preview</h3>
            <div className="overflow-x-auto border border-gray-200 rounded-lg">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys((dataSetPreview?.[0] || {})).map((col) => (
                      <th key={col} className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {dataSetPreview?.map((row, idx) => (
                    <tr key={idx}>
                      {Object.entries(row).map(([key, value]) => (
                        <td key={key} className="px-3 py-2 whitespace-nowrap text-sm text-gray-700">
                          {value ?? '—'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default UploadStep;
