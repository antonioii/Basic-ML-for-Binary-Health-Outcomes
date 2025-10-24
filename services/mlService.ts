
import { DataSet, TrainingConfig, ModelResult, ModelMetrics, RocPoint, KMeansResult } from '../types';

// Helper to simulate async operations
const simulateProcessing = (duration: number) => new Promise(resolve => setTimeout(resolve, duration));

// Helper to generate a plausible but fake ROC curve
const generateFakeRocCurve = (auc: number): RocPoint[] => {
    const points: RocPoint[] = [{ fpr: 0, tpr: 0 }];
    let x = 0;
    let y = 0;
    while(x < 1 && y < 1) {
        x += Math.random() * 0.1;
        y += Math.random() * 0.2 * (1-x) + (auc - 0.5) * 0.1;
        points.push({ fpr: Math.min(1, x), tpr: Math.min(1, y) });
    }
    points.push({ fpr: 1, tpr: 1 });
    return points.sort((a,b) => a.fpr - b.fpr);
};

// Helper to create fake but consistent metrics
const generateFakeMetrics = (baseAuc: number): ModelMetrics => {
    const accuracy = baseAuc + (Math.random() - 0.5) * 0.1;
    const sensitivity = baseAuc + (Math.random() - 0.5) * 0.15;
    const specificity = baseAuc + (Math.random() - 0.5) * 0.15;
    const vpp = baseAuc + (Math.random() - 0.5) * 0.1;
    const tp = Math.round(100 * accuracy * sensitivity);
    const tn = Math.round(100 * accuracy * specificity);
    const fp = Math.round(100 * (1 - specificity));
    const fn = Math.round(100 * (1 - sensitivity));

    return {
        auc: baseAuc,
        accuracy: Math.min(0.99, Math.max(0.5, accuracy)),
        f1Score: 2 * (vpp * sensitivity) / (vpp + sensitivity) || 0,
        sensitivity: Math.min(0.99, Math.max(0.1, sensitivity)),
        specificity: Math.min(0.99, Math.max(0.1, specificity)),
        vpp: Math.min(0.99, Math.max(0.1, vpp)),
        vpn: 1 - vpp,
        confusionMatrix: { tp, tn, fp, fn }
    };
};

// Mock Training Functions
const trainLogisticRegression = async (dataSet: DataSet): Promise<ModelResult> => {
    await simulateProcessing(1000);
    const auc = 0.75 + Math.random() * 0.1;
    return {
        name: 'Logistic Regression',
        metrics: generateFakeMetrics(auc),
        rocCurve: generateFakeRocCurve(auc)
    };
};

const trainKNN = async (dataSet: DataSet): Promise<ModelResult> => {
    await simulateProcessing(1500);
    const auc = 0.80 + Math.random() * 0.1;
    return {
        name: 'K-Nearest Neighbors (KNN)',
        metrics: generateFakeMetrics(auc),
        rocCurve: generateFakeRocCurve(auc)
    };
};

const trainSVM = async (dataSet: DataSet, config: TrainingConfig): Promise<ModelResult> => {
    await simulateProcessing(2000);
    let auc = 0.82 + Math.random() * 0.1;
    if(config.svmFlexibility === 'High (Flexible)') auc += 0.03;
    if(config.svmFlexibility === 'Low (Rigid)') auc -= 0.03;

    return {
        name: 'Support Vector Machine (SVM)',
        metrics: generateFakeMetrics(auc),
        rocCurve: generateFakeRocCurve(auc)
    };
};

const trainRandomForest = async (dataSet: DataSet): Promise<ModelResult> => {
    await simulateProcessing(2500);
    const auc = 0.88 + Math.random() * 0.08;
    return {
        name: 'Random Forest',
        metrics: generateFakeMetrics(auc),
        rocCurve: generateFakeRocCurve(auc),
        featureImportances: dataSet.featureCols
            .map(f => ({ feature: f, importance: Math.random() }))
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 10)
    };
};

const trainGradientBoosting = async (dataSet: DataSet): Promise<ModelResult> => {
    await simulateProcessing(2800);
    const auc = 0.90 + Math.random() * 0.07;
    return {
        name: 'Gradient Boosting',
        metrics: generateFakeMetrics(auc),
        rocCurve: generateFakeRocCurve(auc),
        featureImportances: dataSet.featureCols
            .map(f => ({ feature: f, importance: Math.random() }))
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 10)
    };
};


export const runKMeansElbow = (dataSet: DataSet): { k: number; inertia: number }[] => {
    const elbowPlot: { k: number; inertia: number }[] = [];
    let lastInertia = 1000 + Math.random() * 200;
    for (let k = 1; k <= 10; k++) {
        const inertiaDrop = (lastInertia / (k * k)) * (0.8 + Math.random() * 0.4);
        lastInertia -= inertiaDrop;
        elbowPlot.push({ k, inertia: Math.max(0, lastInertia) });
    }
    return elbowPlot;
};

const trainKMeans = async (dataSet: DataSet, config: TrainingConfig): Promise<ModelResult> => {
    await simulateProcessing(1200);
    const { data, idCol, targetCol } = dataSet;
    const k = config.kMeansClusters;
    
    const clusters: Record<string, (string|number)[]> = {};
    const clusterAnalysis: { cluster: number; count: number; positiveRate: number }[] = [];

    for (let i = 0; i < k; i++) {
        clusters[`Cluster ${i}`] = [];
    }
    data.forEach(row => {
        const clusterIndex = Math.floor(Math.random() * k);
        clusters[`Cluster ${clusterIndex}`].push(row[idCol]);
    });

    for (let i = 0; i < k; i++) {
        const idsInCluster = new Set(clusters[`Cluster ${i}`]);
        const samplesInCluster = data.filter(row => idsInCluster.has(row[idCol]));
        const positiveCount = samplesInCluster.filter(row => row[targetCol] === 1).length;
        
        clusterAnalysis.push({
            cluster: i,
            count: samplesInCluster.length,
            positiveRate: samplesInCluster.length > 0 ? positiveCount / samplesInCluster.length : 0
        });
    }

    const kMeansResult: KMeansResult = {
        elbowPlot: runKMeansElbow(dataSet),
        clusters: clusters,
        clusterAnalysis: clusterAnalysis
    };

    return {
        name: 'K-Means Clustering',
        metrics: null,
        rocCurve: null,
        kMeansResult,
    };
};


// Main orchestrator function
export const trainModels = async (dataSet: DataSet, config: TrainingConfig): Promise<ModelResult[]> => {
    const promises: Promise<ModelResult>[] = [];
    
    const modelMap: Record<string, (ds: DataSet, cfg: TrainingConfig) => Promise<ModelResult>> = {
        "Logistic Regression": trainLogisticRegression,
        "K-Nearest Neighbors (KNN)": trainKNN,
        "Support Vector Machine (SVM)": trainSVM,
        "Random Forest": trainRandomForest,
        "Gradient Boosting": trainGradientBoosting,
        "K-Means Clustering": trainKMeans
    };
    
    config.models.forEach(modelName => {
        if(modelMap[modelName]){
            promises.push(modelMap[modelName](dataSet, config));
        }
    });

    const results = await Promise.all(promises);
    return results.sort((a,b) => (b.metrics?.auc ?? -1) - (a.metrics?.auc ?? -1));
};
