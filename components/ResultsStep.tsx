
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { DataSet, TrainingConfig, ModelResult, ModelMetrics } from '../types';
import { trainModels } from '../services/mlService';
import { generateSummary } from '../services/geminiService';
import Card from './common/Card';
import Button from './common/Button';
import Spinner from './common/Spinner';
import { Award, BarChartHorizontal, Bot, BrainCircuit, Download, FileText, ImageDown, Redo, Sparkles } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ResultsStepProps {
  dataSet: DataSet;
  trainingConfig: TrainingConfig;
  results: ModelResult[] | null;
  onResultsComplete: (results: ModelResult[]) => void;
  onRestart: () => void;
}

const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088FE", "#00C49F"];

const MetricCard: React.FC<{ title: string, value: string | number }> = ({ title, value }) => (
    <div className="bg-gray-50 p-3 rounded-lg text-center border">
        <p className="text-xs text-gray-500 uppercase font-semibold">{title}</p>
        <p className="text-lg font-bold text-blue-800">{value}</p>
    </div>
);

const ConfusionMatrixDisplay: React.FC<{ cm: ModelMetrics['confusionMatrix'] }> = ({ cm }) => (
    <div className="grid grid-cols-2 gap-px bg-gray-300 border border-gray-300 rounded-lg overflow-hidden text-sm">
        <div className="bg-green-100 p-2 text-center"><span className="font-bold">{cm.tp}</span><br/><span className="text-xs text-green-800">True Pos</span></div>
        <div className="bg-red-100 p-2 text-center"><span className="font-bold">{cm.fp}</span><br/><span className="text-xs text-red-800">False Pos</span></div>
        <div className="bg-orange-100 p-2 text-center"><span className="font-bold">{cm.fn}</span><br/><span className="text-xs text-orange-800">False Neg</span></div>
        <div className="bg-blue-100 p-2 text-center"><span className="font-bold">{cm.tn}</span><br/><span className="text-xs text-blue-800">True Neg</span></div>
    </div>
);

const ResultsStep: React.FC<ResultsStepProps> = ({ dataSet, trainingConfig, results, onResultsComplete, onRestart }) => {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('comparison');
  const [geminiSummary, setGeminiSummary] = useState('');
  const [geminiLoading, setGeminiLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const rocContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if(!results){
        const runTraining = async () => {
          try {
            setLoading(true);
            setError(null);
            const trainedResults = await trainModels(dataSet, trainingConfig);
            onResultsComplete(trainedResults);
          } catch (err) {
            setError((err as Error).message);
          } finally {
            setLoading(false);
          }
        };
        runTraining();
    } else {
        setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleGenerateSummary = async () => {
    if (!results) return;
    setGeminiLoading(true);
    setGeminiSummary('');
    try {
        const summary = await generateSummary(results);
        setGeminiSummary(summary);
    } catch (error) {
        setGeminiSummary("Error generating summary. Please ensure your Gemini API key is configured.");
        console.error(error);
    } finally {
        setGeminiLoading(false);
    }
  };

  const getBestModel = useCallback(() => {
    if (!results) return null;
    return results
      .filter(r => r.metrics?.auc)
      .reduce((best, current) => (current.metrics!.auc > best.metrics!.auc ? current : best), results[0]);
  }, [results]);

  const formatHyperparameters = useCallback((params: ModelResult['hyperparameters']) => {
    const entries = Object.entries(params || {});
    if (!entries.length) {
      return 'N/A';
    }
    return entries
      .map(([key, value]) => `${key}: ${value ?? 'None'}`)
      .join(' | ');
  }, []);

  const handleDownloadRocSvg = useCallback(() => {
    if (!rocContainerRef.current) return;
    const svg =
      rocContainerRef.current.querySelector('.recharts-wrapper svg') ??
      rocContainerRef.current.querySelector('svg');
    if (!svg) return;
    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(svg);
    if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {
      source = source.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'roc_curve.svg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  const handleDownloadRocPng = useCallback(() => {
    if (!rocContainerRef.current) return;
    const svg =
      rocContainerRef.current.querySelector('.recharts-wrapper svg') ??
      rocContainerRef.current.querySelector('svg');
    if (!svg) return;
    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(svg);
    if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {
      source = source.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    const svgBlob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    const image = new Image();
    image.onload = () => {
      const rect = svg.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      const canvas = document.createElement('canvas');
      canvas.width = rect.width * scale;
      canvas.height = rect.height * scale;
      const context = canvas.getContext('2d');
      if (!context) {
        URL.revokeObjectURL(url);
        return;
      }
      context.setTransform(scale, 0, 0, scale, 0, 0);
      context.drawImage(image, 0, 0, rect.width, rect.height);
      const pngUrl = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.href = pngUrl;
      link.download = 'roc_curve.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
    };
    image.src = url;
  }, []);

  if (loading) {
    return (
      <Card>
        <div className="flex flex-col items-center justify-center h-80">
          <Spinner />
          <p className="mt-4 text-lg font-medium text-gray-700">Training Models...</p>
          <p className="text-gray-500">This may take a moment. We're running cross-validation for robust results.</p>
        </div>
      </Card>
    );
  }

  if (error) {
    return <Card><p className="text-center text-red-500">{error}</p></Card>;
  }

  if (!results) {
    return <Card><p className="text-center text-red-500">Failed to get model results.</p></Card>;
  }
  
  const supervisedResults = results.filter(r => r.metrics);
  const kMeansResult = results.find(r => r.name === 'K-Means Clustering')?.kMeansResult;
  const errorResults = results.filter(
    r => !r.metrics && typeof r.hyperparameters?.error === 'string'
  );

  const renderComparisonTable = () => (
    <div className="space-y-6">
        {errorResults.length > 0 && (
          <Card>
            <h3 className="text-xl font-bold text-gray-800 mb-3">Models Not Trained</h3>
            <p className="text-sm text-gray-600 mb-2">
              The following models were skipped during training:
            </p>
            <ul className="space-y-1 text-sm text-gray-700">
              {errorResults.map(result => (
                <li key={result.name} className="flex flex-col sm:flex-row sm:items-center sm:space-x-2">
                  <span className="font-semibold text-gray-800">{result.name}</span>
                  <span className="text-gray-600">{result.hyperparameters?.error as string}</span>
                </li>
              ))}
            </ul>
          </Card>
        )}
        {supervisedResults.map((result) => (
            <Card key={result.name}>
                 <h3 className="text-xl font-bold text-gray-800 mb-4">{result.name}</h3>
                 <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="lg:col-span-1">
                        <h4 className="text-sm font-semibold mb-2 text-center">Confusion Matrix</h4>
                        <ConfusionMatrixDisplay cm={result.metrics!.confusionMatrix} />
                    </div>
                    <div className="lg:col-span-3 grid grid-cols-2 sm:grid-cols-3 gap-3">
                        <MetricCard title="AUC" value={result.metrics!.auc.toFixed(3)} />
                        <MetricCard title="Accuracy" value={`${(result.metrics!.accuracy * 100).toFixed(1)}%`} />
                        <MetricCard title="F1-Score" value={result.metrics!.f1Score.toFixed(3)} />
                        <MetricCard title="Sensitivity" value={result.metrics!.sensitivity.toFixed(3)} />
                        <MetricCard title="Specificity" value={result.metrics!.specificity.toFixed(3)} />
                        <MetricCard title="VPP / Precision" value={result.metrics!.vpp.toFixed(3)} />
                        <MetricCard title="VPN" value={result.metrics!.vpn.toFixed(3)} />
                    </div>
                    {Object.keys(result.hyperparameters || {}).length > 0 && (
                      <div className="lg:col-span-4 mt-4">
                        <h4 className="text-sm font-semibold mb-2">Model Hyperparameters</h4>
                        <div className="bg-gray-50 border border-gray-200 rounded p-3 text-sm text-gray-700">
                          <ul className="space-y-1">
                            {Object.entries(result.hyperparameters).map(([key, value]) => (
                              <li key={key} className="flex justify-between">
                                <span className="font-medium">{key}</span>
                                <span className="font-mono">{value ?? 'None'}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                    {result.featureImportances && result.featureImportances.length > 0 && (
                      <div className="lg:col-span-4 mt-4">
                        <h4 className="text-sm font-semibold mb-2">Top Features</h4>
                        <ul className="text-sm text-gray-700 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                          {result.featureImportances.slice(0, 6).map((feature) => (
                            <li key={feature.feature} className="flex justify-between bg-gray-50 px-3 py-2 rounded">
                              <span>{feature.feature}</span>
                              <span className="font-mono">{feature.importance.toFixed(3)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                 </div>
            </Card>
        ))}
    </div>
  );

  const renderRocCurve = () => (
    <Card>
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-800">ROC Curves</h3>
            <div className="flex space-x-2 mt-3 md:mt-0">
                <Button onClick={handleDownloadRocSvg} text="Download SVG" icon={Download} variant="secondary" />
                <Button onClick={handleDownloadRocPng} text="Download PNG" icon={ImageDown} variant="secondary" />
            </div>
        </div>
        <div ref={rocContainerRef} className="h-96 w-full">
            <ResponsiveContainer>
                <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="fpr" name="False Positive Rate" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }} />
                    <YAxis type="number" dataKey="tpr" name="True Positive Rate" domain={[0, 1]} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="tpr" stroke="#ccc" name="Random" dot={false} strokeDasharray="5 5" />
                    {supervisedResults.filter(result => result.rocCurve).map((result, i) => (
                        <Line key={result.name} type="monotone" data={result.rocCurve!} dataKey="tpr" name={`${result.name} (AUC=${result.metrics?.auc.toFixed(3)})`} stroke={colors[i % colors.length]} dot={false} strokeWidth={2}/>
                    ))}
                </LineChart>
            </ResponsiveContainer>
        </div>
    </Card>
  );

  const handleDownloadClusters = () => {
    if (!results) return;
    const kMeans = results.find(r => r.name === 'K-Means Clustering')?.kMeansResult;
    if (!kMeans) return;
    const rows: string[][] = [['Cluster', 'ID']];
    Object.entries(kMeans.clusters).forEach(([cluster, ids]) => {
      ids.forEach(id => {
        rows.push([cluster, String(id)]);
      });
    });
    const csvContent = rows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'kmeans_clusters.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const renderKMeans = () => {
    if (!kMeansResult) return <Card><p>K-Means was not selected.</p></Card>;
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
                <h3 className="text-xl font-bold text-gray-800 mb-4">K-Means: Elbow Plot</h3>
                <div className="h-80 w-full">
                    <ResponsiveContainer>
                        <LineChart data={kMeansResult.elbowPlot} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="k" label={{ value: 'Number of Clusters (k)', position: 'insideBottom', offset: -5 }}/>
                            <YAxis label={{ value: 'Inertia', angle: -90, position: 'insideLeft' }} />
                            <Tooltip />
                            <Line type="monotone" dataKey="inertia" stroke="#8884d8" strokeWidth={2} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </Card>
            <Card>
                <h3 className="text-xl font-bold text-gray-800 mb-4">Cluster Analysis (k={kMeansResult.clusterAnalysis.length})</h3>
                <table className="w-full text-sm text-left text-gray-500">
                    <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                        <tr>
                            <th scope="col" className="px-4 py-2">Cluster</th>
                            <th scope="col" className="px-4 py-2">Sample Count</th>
                            <th scope="col" className="px-4 py-2">% Positive Class</th>
                        </tr>
                    </thead>
                    <tbody>
                        {kMeansResult.clusterAnalysis.map(c => (
                            <tr key={c.cluster} className="bg-white border-b">
                                <td className="px-4 py-2 font-medium text-gray-900">{c.cluster}</td>
                                <td className="px-4 py-2">{c.count}</td>
                                <td className="px-4 py-2">{(c.positiveRate * 100).toFixed(1)}%</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                <div className="mt-4 text-right">
                    <Button onClick={handleDownloadClusters} text="Export Cluster Assignments" icon={Download} />
                </div>
            </Card>
        </div>
    )
  };

  const bestModel = getBestModel();

  const handleDownloadReport = () => {
    if (!results) return;
    const header = ['Model', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'VPP', 'VPN', 'F1', 'TP', 'FP', 'TN', 'FN', 'Hyperparameters'];
    const rows = [header];
    results.filter(r => r.metrics).forEach(result => {
      const metrics = result.metrics!;
      const hyperparams = formatHyperparameters(result.hyperparameters);
      rows.push([
        result.name,
        metrics.auc.toFixed(4),
        metrics.accuracy.toFixed(4),
        metrics.sensitivity.toFixed(4),
        metrics.specificity.toFixed(4),
        metrics.vpp.toFixed(4),
        metrics.vpn.toFixed(4),
        metrics.f1Score.toFixed(4),
        metrics.confusionMatrix.tp.toString(),
        metrics.confusionMatrix.fp.toString(),
        metrics.confusionMatrix.tn.toString(),
        metrics.confusionMatrix.fn.toString(),
        hyperparams,
      ]);
    });
    const csv = rows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'model_metrics.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
        <Card>
             <div className="flex flex-col md:flex-row justify-between items-start md:items-center">
                <div>
                    <h2 className="text-3xl font-bold text-gray-800 mb-2">Modeling Results</h2>
                    <p className="text-gray-600">The analysis is complete. Explore the results below.</p>
                </div>
                 <div className="flex items-center space-x-2 mt-4 md:mt-0">
                    <Button onClick={onRestart} text="Start New Analysis" icon={Redo} variant="secondary" />
                    <Button onClick={handleDownloadReport} text="Download Report" icon={FileText} />
                </div>
             </div>
             {bestModel && (
                 <div className="mt-6 bg-blue-50 border border-blue-200 p-4 rounded-lg flex items-center space-x-4">
                    <Award className="w-10 h-10 text-blue-600 flex-shrink-0" />
                    <div>
                        <h4 className="font-bold text-blue-800">Best Performing Model</h4>
                        <p className="text-gray-700">Based on AUC, <strong>{bestModel.name}</strong> performed the best with an AUC of <strong>{bestModel.metrics?.auc.toFixed(3)}</strong>.</p>
                        {Object.keys(bestModel.hyperparameters || {}).length > 0 && (
                          <p className="text-sm text-blue-900 mt-2">Key hyperparameters: {formatHyperparameters(bestModel.hyperparameters)}</p>
                        )}
                    </div>
                 </div>
             )}
        </Card>

        <div>
            <div className="border-b border-gray-200">
                <nav className="-mb-px flex space-x-6" aria-label="Tabs">
                    <button onClick={() => setActiveTab('comparison')} className={`${activeTab === 'comparison' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}><BarChartHorizontal className="w-4 h-4 mr-2"/>Comparison</button>
                    <button onClick={() => setActiveTab('roc')} className={`${activeTab === 'roc' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}><BrainCircuit className="w-4 h-4 mr-2"/>ROC Curve</button>
                    {kMeansResult && <button onClick={() => setActiveTab('kmeans')} className={`${activeTab === 'kmeans' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}><BarChartHorizontal className="w-4 h-4 mr-2"/>K-Means</button>}
                    <button onClick={() => setActiveTab('summary')} className={`${activeTab === 'summary' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}><Sparkles className="w-4 h-4 mr-2"/>AI Summary</button>
                </nav>
            </div>
            <div className="mt-6">
                {activeTab === 'comparison' && renderComparisonTable()}
                {activeTab === 'roc' && renderRocCurve()}
                {activeTab === 'kmeans' && renderKMeans()}
                {activeTab === 'summary' && (
                    <Card>
                        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center"><Bot className="mr-2"/>Gemini-Powered AI Summary</h3>
                        {!geminiSummary && !geminiLoading && (
                            <div className="text-center">
                                <p className="text-gray-600 mb-4">Get a natural language summary and interpretation of your model results.</p>
                                <Button onClick={handleGenerateSummary} text="Generate Summary" icon={Sparkles} />
                            </div>
                        )}
                        {geminiLoading && <div className="flex items-center space-x-2"><Spinner/><p>Generating insights...</p></div>}
                        {geminiSummary && <div className="prose prose-blue max-w-none" dangerouslySetInnerHTML={{ __html: geminiSummary }} />}
                    </Card>
                )}
            </div>
        </div>
    </div>
  );
};

export default ResultsStep;
