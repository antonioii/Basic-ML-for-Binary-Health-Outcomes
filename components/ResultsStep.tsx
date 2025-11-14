
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { DataSet, TrainingConfig, ModelResult, ModelMetrics } from '../types';
import { trainModels } from '../services/mlService';
import { generateSummary } from '../services/geminiService';
import { buildTrainingEstimate, formatTrainingTime } from '../services/trainingEstimator';
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

type MetricColumnKey = 'auc' | 'accuracy' | 'sensitivity' | 'specificity' | 'vpp' | 'vpn' | 'f1Score';
type MetricStdColumnKey = 'aucStd' | 'accuracyStd' | 'sensitivityStd' | 'specificityStd' | 'vppStd' | 'vpnStd' | 'f1ScoreStd';

const hasValidAuc = (metrics?: ModelMetrics | null): metrics is ModelMetrics & { auc: number } =>
  typeof metrics?.auc === 'number' && !Number.isNaN(metrics.auc);

const isRocEligible = (result: ModelResult): result is ModelResult & {
  metrics: ModelMetrics & { auc: number };
  rocCurve: NonNullable<ModelResult['rocCurve']>;
} => Array.isArray(result.rocCurve) && hasValidAuc(result.metrics);

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

// helper function to find the correct <svg> graphic
function getMainChartSvg(root: HTMLElement): SVGSVGElement | null {
  const svgs = Array.from(root.querySelectorAll('svg')) as SVGSVGElement[];
  if (svgs.length === 0) return null;

  // 1) Choose svg with role="application"
  const withRole = svgs.find(s => s.getAttribute('role') === 'application');
  if (withRole) return withRole;

  // 2) fallback: larger SVG in visible area
  let best: { el: SVGSVGElement; area: number } | null = null;
  for (const s of svgs) {
    const rect = s.getBoundingClientRect();
    const area = rect.width * rect.height;
    if (!best || area > best.area) best = { el: s, area };
  }
  return best?.el ?? null;
}

const ResultsStep: React.FC<ResultsStepProps> = ({ dataSet, trainingConfig, results, onResultsComplete, onRestart }) => {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('comparison');
  const [geminiSummary, setGeminiSummary] = useState('');
  const [geminiLoading, setGeminiLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeStageIndex, setActiveStageIndex] = useState(0);
  const rocContainerRef = useRef<HTMLDivElement | null>(null);
  const trainingEstimate = useMemo(() => buildTrainingEstimate(dataSet, trainingConfig), [dataSet, trainingConfig]);
  const formattedEstimate = useMemo(() => formatTrainingTime(trainingEstimate.totalSeconds), [trainingEstimate.totalSeconds]);
  const formatMetricWithStd = useCallback((
    value: number | null | undefined,
    std?: number | null,
    formatValue?: (val: number) => string,
    formatStd?: (val: number) => string,
  ) => {
    if (value == null || Number.isNaN(value)) {
      return 'N/A';
    }
    const formattedValue = formatValue ? formatValue(value) : value.toFixed(3);
    if (std == null || Number.isNaN(std)) {
      return formattedValue;
    }
    const formattedStd = formatStd ? formatStd(std) : std.toFixed(3);
    return `${formattedValue} ± ${formattedStd}`;
  }, []);

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

  useEffect(() => {
    if (!loading || trainingEstimate.stages.length === 0) {
      return;
    }
    setActiveStageIndex(0);
    const start = performance.now();
    const interval = window.setInterval(() => {
      const elapsedSeconds = (performance.now() - start) / 1000;
      let cumulative = 0;
      let currentIndex = trainingEstimate.stages.length - 1;
      for (let i = 0; i < trainingEstimate.stages.length; i++) {
        cumulative += trainingEstimate.stages[i].seconds;
        if (elapsedSeconds < cumulative) {
          currentIndex = i;
          break;
        }
      }
      setActiveStageIndex(currentIndex);
    }, 400);
    return () => window.clearInterval(interval);
  }, [loading, trainingEstimate]);

  const stageCount = trainingEstimate.stages.length;
  const currentStage = stageCount > 0
    ? trainingEstimate.stages[Math.min(activeStageIndex, stageCount - 1)]
    : null;

  useEffect(() => {
    if (!loading && stageCount > 0) {
      setActiveStageIndex(stageCount - 1);
    }
  }, [loading, stageCount]);

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
    const candidates = results.filter((r): r is ModelResult & { metrics: ModelMetrics & { auc: number } } => hasValidAuc(r.metrics));
    if (candidates.length === 0) return null;
    return candidates.slice(1).reduce(
      (best, current) => (current.metrics.auc > best.metrics.auc ? current : best),
      candidates[0],
    );
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
  const svg = getMainChartSvg(rocContainerRef.current);
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
  const svg = getMainChartSvg(rocContainerRef.current);
  if (!svg) return;

  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svg);

  if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {
    source = source.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
  }

  const svgBlob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  const image = new Image();
  image.crossOrigin = 'anonymous';
  image.onload = () => {
    const rect = svg.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;

    const canvas = document.createElement('canvas');
    canvas.width  = Math.max(1, Math.round(rect.width  * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      URL.revokeObjectURL(url);
      return;
    }

    // fundo branco para não ficar transparente
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // desenha escalado
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.drawImage(image, 0, 0, rect.width, rect.height);

    const pngUrl = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = pngUrl;
    link.download = 'roc_curve.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };
  image.onerror = () => URL.revokeObjectURL(url);
  image.src = url;
}, []);


  if (loading) {
    return (
      <Card>
        <div className="flex flex-col items-center justify-center h-80">
          <Spinner />
          <p className="mt-4 text-lg font-medium text-gray-700">Training Models...</p>
          <p className="text-gray-500 text-center px-4">This may take a moment. We're running cross-validation for robust results.</p>
          <p className="text-sm text-gray-500 mt-2">
            Estimated completion: <span className="font-semibold text-gray-700">~{formattedEstimate}</span>
          </p>
          {currentStage && (
            <p className="text-sm text-blue-600 mt-1 text-center">
              Stage {Math.min(activeStageIndex + 1, stageCount)} of {stageCount}: <span className="font-semibold text-blue-700">{currentStage.label}</span>
            </p>
          )}
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
  
  const supervisedResults = results.filter(r => r.name !== 'K-Means Clustering');
  const rocEligibleResults = supervisedResults.filter(isRocEligible);
  const kMeansResult = results.find(r => r.name === 'K-Means Clustering')?.kMeansResult;

  const renderComparisonTable = () => (
    <div className="space-y-6">
        {supervisedResults.map((result) => (
            <Card key={result.name}>
                 <h3 className="text-xl font-bold text-gray-800 mb-4">{result.name}</h3>
                 {result.metrics ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="lg:col-span-1">
                        <h4 className="text-sm font-semibold mb-2 text-center">Confusion Matrix</h4>
                        <ConfusionMatrixDisplay cm={result.metrics.confusionMatrix} />
                    </div>
                    <div className="lg:col-span-3 grid grid-cols-2 sm:grid-cols-3 gap-3">
                        <MetricCard title="AUC" value={formatMetricWithStd(result.metrics.auc, result.metrics.aucStd)} />
                        <MetricCard
                          title="Accuracy"
                          value={formatMetricWithStd(
                            result.metrics.accuracy,
                            result.metrics.accuracyStd,
                            (v) => `${(v * 100).toFixed(1)}%`,
                            (std) => `${(std * 100).toFixed(1)}%`,
                          )}
                        />
                        <MetricCard title="F1-Score" value={formatMetricWithStd(result.metrics.f1Score, result.metrics.f1ScoreStd)} />
                        <MetricCard title="Sensitivity" value={formatMetricWithStd(result.metrics.sensitivity, result.metrics.sensitivityStd)} />
                        <MetricCard title="Specificity" value={formatMetricWithStd(result.metrics.specificity, result.metrics.specificityStd)} />
                        <MetricCard title="VPP / Precision" value={formatMetricWithStd(result.metrics.vpp, result.metrics.vppStd)} />
                        <MetricCard title="VPN" value={formatMetricWithStd(result.metrics.vpn, result.metrics.vpnStd)} />
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
                 ) : (
                  <p className="text-sm text-gray-600 bg-gray-50 border border-dashed border-gray-300 rounded p-4">
                    {result.statusMessage ?? 'This model did not return evaluation metrics.'}
                  </p>
                 )}
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
            {rocEligibleResults.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm text-gray-500 text-center px-4">
                ROC curves will be displayed once at least one supervised model finishes successfully.
              </div>
            ) : (
              <ResponsiveContainer>
                  <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="fpr" name="False Positive Rate" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }} />
                      <YAxis type="number" dataKey="tpr" name="True Positive Rate" domain={[0, 1]} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="tpr" stroke="#ccc" name="Random" dot={false} strokeDasharray="5 5" data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} />
                      {rocEligibleResults.map((result, i) => (
                          <Line key={result.name} type="monotone" data={result.rocCurve!} dataKey="tpr" name={`${result.name} (AUC=${result.metrics!.auc.toFixed(3)})`} stroke={colors[i % colors.length]} dot={false} strokeWidth={2}/>
                      ))}
                  </LineChart>
              </ResponsiveContainer>
            )}
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
    const metricColumns: { label: string; key: MetricColumnKey; stdKey: MetricStdColumnKey }[] = [
      { label: 'AUC', key: 'auc', stdKey: 'aucStd' },
      { label: 'Accuracy', key: 'accuracy', stdKey: 'accuracyStd' },
      { label: 'Sensitivity', key: 'sensitivity', stdKey: 'sensitivityStd' },
      { label: 'Specificity', key: 'specificity', stdKey: 'specificityStd' },
      { label: 'VPP', key: 'vpp', stdKey: 'vppStd' },
      { label: 'VPN', key: 'vpn', stdKey: 'vpnStd' },
      { label: 'F1', key: 'f1Score', stdKey: 'f1ScoreStd' },
    ];
    const formatCsvMetric = (value: number | null | undefined, decimals = 4) => (
      typeof value === 'number' && !Number.isNaN(value) ? value.toFixed(decimals) : 'N/A'
    );
    const header = [
      'Model',
      ...metricColumns.flatMap(({ label }) => [label, `${label}_SD`]),
      'TP',
      'FP',
      'TN',
      'FN',
      'Hyperparameters',
    ];
    const rows = [header];
    results.filter(r => r.metrics).forEach(result => {
      const metrics = result.metrics!;
      const hyperparams = formatHyperparameters(result.hyperparameters);
      const row = [result.name];
      metricColumns.forEach(({ key, stdKey }) => {
        row.push(formatCsvMetric(metrics[key]));
        row.push(formatCsvMetric(metrics[stdKey]));
      });
      row.push(
        metrics.confusionMatrix.tp.toString(),
        metrics.confusionMatrix.fp.toString(),
        metrics.confusionMatrix.tn.toString(),
        metrics.confusionMatrix.fn.toString(),
        hyperparams,
      );
      rows.push(row);
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
                        <p className="text-gray-700">
                          Based on AUC, <strong>{bestModel.name}</strong> performed the best with an AUC of{' '}
                          <strong>{typeof bestModel.metrics?.auc === 'number' ? bestModel.metrics.auc.toFixed(3) : 'N/A'}</strong>.
                        </p>
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
