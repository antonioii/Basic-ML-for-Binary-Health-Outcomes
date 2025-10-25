
import React, { useState, useEffect } from 'react';
import { DataSet, EdaResult, HistogramInfo, BoxPlotInfo } from '../types';
import { fetchEda } from '../services/edaService';
import Spinner from './common/Spinner';
import Card from './common/Card';
import Button from './common/Button';
import { BarChart, Users, AlertCircle, Link2, CheckCircle } from 'lucide-react';
import { ResponsiveContainer, BarChart as ReBarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

interface EdaStepProps {
  dataSet: DataSet;
  onEdaComplete: (result: EdaResult) => void;
}

const EdaStep: React.FC<EdaStepProps> = ({ dataSet, onEdaComplete }) => {
  const [loading, setLoading] = useState(true);
  const [edaResult, setEdaResult] = useState<EdaResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const runEda = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await fetchEda(dataSet.datasetId);
        setEdaResult(result);
      } catch (error) {
        console.error(error);
        setError((error as Error).message);
        setEdaResult(null);
      } finally {
        setLoading(false);
      }
    };
    runEda();
  }, [dataSet]);

  const handleContinue = () => {
    if (edaResult) {
      onEdaComplete(edaResult);
    }
  };
  
  const renderSummaryCard = (Icon: React.ElementType, title: string, value: string, color: string) => (
    <div className={`bg-white p-4 rounded-lg shadow flex items-center space-x-4 border-l-4 ${color}`}>
        <Icon className="w-8 h-8 text-gray-500" />
        <div>
            <p className="text-sm text-gray-500">{title}</p>
            <p className="text-xl font-bold text-gray-800">{value}</p>
        </div>
    </div>
  );

  if (loading) {
    return (
      <Card>
        <div className="flex flex-col items-center justify-center h-64">
          <Spinner />
          <p className="mt-4 text-lg font-medium text-gray-700">Performing Exploratory Data Analysis...</p>
          <p className="text-gray-500">Analyzing distributions, correlations, and data quality.</p>
        </div>
      </Card>
    );
  }

  if (error) {
    return <Card><p className="text-center text-red-500">{error}</p></Card>;
  }

  if (!edaResult) {
    return <Card><p className="text-center text-red-500">Failed to perform EDA.</p></Card>;
  }

  const { targetDistribution, missingInfo, outlierInfo, correlationInfo } = edaResult;
  const totalTarget = targetDistribution['0'] + targetDistribution['1'];
  const positiveClassPct = ((targetDistribution['1'] / totalTarget) * 100).toFixed(1);

  const renderHistogram = (hist: HistogramInfo) => (
    <Card key={hist.variable}>
      <h3 className="text-md font-semibold text-gray-800 mb-3">{hist.variable} - Distribution</h3>
      <div className="h-52">
        <ResponsiveContainer>
          <ReBarChart data={hist.counts.map((count, index) => ({
            binLabel: `${hist.bins[index].toFixed(2)} - ${hist.bins[index + 1].toFixed(2)}`,
            count,
          }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="binLabel" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={60} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" />
          </ReBarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );

  const renderVariableTable = () => (
    <Card>
      <h3 className="text-lg font-semibold text-gray-800 mb-3">Variable Classification</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Variable</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Missing (%)</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Distinct Values</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Example Values</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200 text-sm">
            {edaResult.variableInfo.map((info) => (
              <tr key={info.name}>
                <td className="px-4 py-2 font-medium text-gray-800">{info.name}</td>
                <td className="px-4 py-2 text-gray-700">{info.type}</td>
                <td className="px-4 py-2 text-gray-700">{info.missing.toFixed(2)}</td>
                <td className="px-4 py-2 text-gray-700">{info.distinctValues}</td>
                <td className="px-4 py-2 text-gray-500">{info.exampleValues?.slice(0, 3).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );

  const histogramCards = edaResult.histograms.slice(0, 4).map(renderHistogram);
  const boxPlotCards = edaResult.boxPlots.slice(0, 4).map((box: BoxPlotInfo) => (
    <Card key={box.variable}>
      <h3 className="text-md font-semibold text-gray-800 mb-3">{box.variable} - Box Plot Summary</h3>
      <ul className="text-sm text-gray-700 space-y-1">
        <li><strong>Median:</strong> {box.median.toFixed(3)}</li>
        <li><strong>Q1 - Q3:</strong> {box.q1.toFixed(3)} - {box.q3.toFixed(3)}</li>
        <li><strong>Whiskers:</strong> {box.lowerWhisker.toFixed(3)} - {box.upperWhisker.toFixed(3)}</li>
        <li><strong>Outlier IDs:</strong> {box.outliers.length > 0 ? box.outliers.slice(0, 5).join(', ') : 'None detected'}</li>
      </ul>
    </Card>
  ));

  return (
    <div className="space-y-6">
      <Card>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Exploratory Data Analysis Report</h2>
        <p className="text-gray-600 mb-6">Here is an automated summary of your dataset. Based on this analysis, we will generate suggestions for data cleaning in the next step.</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {renderSummaryCard(Users, "Total Samples", edaResult.totalRows.toString(), "border-blue-500")}
            {renderSummaryCard(BarChart, "Features", (edaResult.totalCols - 2).toString(), "border-green-500")}
            {renderSummaryCard(AlertCircle, "Potential Issues", (missingInfo.length + outlierInfo.length + correlationInfo.length).toString(), "border-yellow-500")}
            {renderSummaryCard(CheckCircle, "Target Class (1) Balance", `${positiveClassPct}%`, "border-purple-500")}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center"><Users className="w-5 h-5 mr-2 text-blue-600"/>Target Variable Distribution</h3>
            <div className="w-full bg-gray-200 rounded-full h-6">
                <div className="bg-blue-600 h-6 rounded-l-full text-center text-white text-sm leading-6" style={{ width: `${100 - parseFloat(positiveClassPct)}%` }}>
                    Class 0: {targetDistribution['0']}
                </div>
                <div className="bg-purple-600 h-6 rounded-r-full text-center text-white text-sm leading-6" style={{ width: `${positiveClassPct}%`, marginLeft: `${100 - parseFloat(positiveClassPct)}%`, marginTop: '-1.5rem'}}>
                    Class 1: {targetDistribution['1']}
                </div>
            </div>
            {targetDistribution.imbalance && <p className="text-yellow-700 mt-3 text-sm">Note: Significant class imbalance detected. This may affect model performance.</p>}
        </Card>
        <Card>
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center"><AlertCircle className="w-5 h-5 mr-2 text-yellow-600"/>Missing Values</h3>
            {missingInfo.length > 0 ? (
                <ul className="space-y-1 text-sm">
                    {missingInfo.map(info => <li key={info.variable}>{info.variable}: <span className="font-medium">{info.percentage.toFixed(2)}% missing</span></li>)}
                </ul>
            ) : <p className="text-gray-600">No missing values found. Great!</p>}
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center"><AlertCircle className="w-5 h-5 mr-2 text-orange-600"/>Outlier Detection</h3>
            {outlierInfo.length > 0 ? (
                <ul className="space-y-1 text-sm max-h-40 overflow-y-auto">
                    {outlierInfo.map(info => <li key={info.variable}>{info.variable}: <span className="font-medium">{info.count} potential outliers</span></li>)}
                </ul>
            ) : <p className="text-gray-600">No significant outliers detected via IQR method.</p>}
        </Card>
        <Card>
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center"><Link2 className="w-5 h-5 mr-2 text-red-600"/>High Correlation Pairs (|r| ≥ 0.8)</h3>
            {correlationInfo.length > 0 ? (
                <ul className="space-y-1 text-sm max-h-40 overflow-y-auto">
                    {correlationInfo.map((info, i) => <li key={i}>{info.pair[0]} & {info.pair[1]}: <span className="font-medium">{info.value.toFixed(3)}</span></li>)}
                </ul>
            ) : <p className="text-gray-600">No highly correlated feature pairs found.</p>}
        </Card>
      </div>

      {renderVariableTable()}

      {histogramCards.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center"><BarChart className="w-5 h-5 mr-2 text-blue-600"/>Key Feature Distributions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {histogramCards}
          </div>
        </div>
      )}

      {boxPlotCards.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Box Plot Summaries</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {boxPlotCards}
          </div>
        </div>
      )}

      <div className="flex justify-end mt-8">
        <Button onClick={handleContinue} text="Continue to Data Cleaning" />
      </div>
    </div>
  );
};

export default EdaStep;
