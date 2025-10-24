
import React, { useState, useCallback } from 'react';
import * as XLSX from 'xlsx';
import { DataRow, DataSet } from '../types';
import { UploadCloud, FileCheck2, AlertTriangle } from 'lucide-react';
import Spinner from './common/Spinner';
import Card from './common/Card';


interface UploadStepProps {
  onDataUploaded: (dataSet: DataSet) => void;
}

const UploadStep: React.FC<UploadStepProps> = ({ onDataUploaded }) => {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setFileName(file.name);

    try {
      const data = await readFile(file);
      const validatedData = validateData(data, file.name);
      setTimeout(() => {
        onDataUploaded(validatedData);
        setLoading(false);
      }, 1500); // Simulate processing
    } catch (err) {
      setError((err as Error).message);
      setLoading(false);
      setFileName(null);
    }
  }, [onDataUploaded]);

  const readFile = (file: File): Promise<DataRow[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = e.target?.result;
          const workbook = XLSX.read(data, { type: 'binary' });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          const jsonData = XLSX.utils.sheet_to_json(worksheet) as DataRow[];
          resolve(jsonData);
        } catch (err) {
          reject(new Error('Failed to parse the file. Please ensure it is a valid .xlsx or .csv file.'));
        }
      };
      reader.onerror = () => reject(new Error('Failed to read the file.'));
      reader.readAsBinaryString(file);
    });
  };

  const validateData = (data: DataRow[], fileName: string): DataSet => {
    if (data.length === 0) throw new Error('File is empty.');
    
    const headers = Object.keys(data[0]);
    if (headers.length < 3) throw new Error('File must have at least 3 columns (ID, one feature, Target).');

    const idCol = headers[0];
    const targetCol = headers[headers.length - 1];
    const featureCols = headers.slice(1, -1);

    const idValues = new Set();
    data.forEach((row, index) => {
      const id = row[idCol];
      if (id === null || id === undefined) throw new Error(`Row ${index + 2} has a missing ID.`);
      if (idValues.has(id)) throw new Error(`Duplicate ID found: ${id}. The first column must contain unique identifiers.`);
      idValues.add(id);

      const target = row[targetCol];
      if (target !== 0 && target !== 1) {
        throw new Error(`Invalid target value in row ${index + 2}. The last column must only contain 0 or 1.`);
      }

      for (const col of featureCols) {
          const val = row[col];
          if (val !== undefined && val !== null && typeof val !== 'number') {
               throw new Error(`Invalid data type in feature column '${col}' at row ${index + 2}. All intermediate columns must be numeric.`);
          }
      }
    });

    return { fileName, data, idCol, targetCol, featureCols };
  };

  return (
    <Card>
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Your Dataset</h2>
        <p className="text-gray-600 mb-6">Select a pre-processed .xlsx or .csv file to begin the analysis.</p>
        
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
      </div>
    </Card>
  );
};

export default UploadStep;
