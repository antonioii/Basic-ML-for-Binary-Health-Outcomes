import { DataSet, EdaResult, VariableType, VariableInfo, OutlierInfo, CorrelationInfo, CleaningSuggestion, DataRow } from '../types';

const CORRELATION_THRESHOLD = 0.8;
const IMBALANCE_THRESHOLD = 0.2;

const identifyVariableType = (col: string, data: DataRow[]): VariableType => {
  const values = new Set(data.map(row => row[col]).filter(v => v !== null && v !== undefined));
  const hasOnly01 = Array.from(values).every(v => v === 0 || v === 1);
  if (hasOnly01 && values.size <= 2) {
    return VariableType.BINARY;
  }
  if (values.size < 15) { // Heuristic for low cardinality
    return VariableType.CATEGORICAL;
  }
  return VariableType.CONTINUOUS;
};

const calculateMissing = (data: DataRow[], cols: string[]): { variable: string; percentage: number }[] => {
  const missingInfo = cols.map(col => {
    const missingCount = data.filter(row => row[col] === null || row[col] === undefined).length;
    return { variable: col, percentage: (missingCount / data.length) * 100 };
  });
  return missingInfo.filter(info => info.percentage > 0);
};

const detectOutliers = (dataSet: DataSet): OutlierInfo[] => {
    const { data, idCol, featureCols } = dataSet;
    const outlierInfo: OutlierInfo[] = [];

    const continuousCols = featureCols.filter(col => {
        const values = new Set(data.map(r => r[col]));
        return values.size > 10; // Simple heuristic for continuous
    });

    for (const col of continuousCols) {
        const values = data.map(row => row[col] as number).filter(v => v !== null && v !== undefined).sort((a, b) => a - b);
        if (values.length < 5) continue;

        const q1 = values[Math.floor(values.length / 4)];
        const q3 = values[Math.floor(values.length * 3 / 4)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;

        const outliers = data.filter(row => {
            const val = row[col] as number;
            return val < lowerBound || val > upperBound;
        });

        if (outliers.length > 0) {
            outlierInfo.push({
                variable: col,
                count: outliers.length,
                ids: outliers.map(o => o[idCol])
            });
        }
    }
    return outlierInfo;
};

const calculateCorrelations = (dataSet: DataSet): CorrelationInfo[] => {
    const { data, featureCols } = dataSet;
    const correlationInfo: CorrelationInfo[] = [];
    const numericCols = featureCols.filter(col => typeof data[0][col] === 'number');

    for (let i = 0; i < numericCols.length; i++) {
        for (let j = i + 1; j < numericCols.length; j++) {
            const colA = numericCols[i];
            const colB = numericCols[j];
            
            const valuesA = data.map(r => r[colA] as number);
            const valuesB = data.map(r => r[colB] as number);
            
            const meanA = valuesA.reduce((a, b) => a + b, 0) / valuesA.length;
            const meanB = valuesB.reduce((a, b) => a + b, 0) / valuesB.length;
            
            let numerator = 0;
            let denA = 0;
            let denB = 0;

            for (let k = 0; k < data.length; k++) {
                const diffA = valuesA[k] - meanA;
                const diffB = valuesB[k] - meanB;
                numerator += diffA * diffB;
                denA += diffA * diffA;
                denB += diffB * diffB;
            }
            
            const correlation = numerator / (Math.sqrt(denA) * Math.sqrt(denB));

            if (Math.abs(correlation) >= CORRELATION_THRESHOLD) {
                correlationInfo.push({
                    pair: [colA, colB],
                    value: correlation
                });
            }
        }
    }
    return correlationInfo;
};

const generateSuggestions = (edaResult: EdaResult): CleaningSuggestion[] => {
    const suggestions: CleaningSuggestion[] = [];
    
    edaResult.missingInfo.forEach((info, index) => {
        suggestions.push({
            id: `missing-${index}`,
            type: 'missing',
            description: `Remove rows with missing values in '${info.variable}' (${info.percentage.toFixed(1)}% missing).`,
            details: { variable: info.variable },
            apply: true
        });
    });

    edaResult.outlierInfo.forEach((info, index) => {
        suggestions.push({
            id: `outlier-${index}`,
            type: 'outlier',
            description: `Remove ${info.count} outlier(s) detected in '${info.variable}'.`,
            details: { variable: info.variable, ids: info.ids },
            apply: true
        });
    });

    edaResult.correlationInfo.forEach((info, index) => {
        suggestions.push({
            id: `corr-${index}`,
            type: 'correlation',
            description: `Remove column '${info.pair[1]}' due to high correlation with '${info.pair[0]}' (r=${info.value.toFixed(2)}).`,
            details: { variables: info.pair, value: info.value },
            apply: true
        });
    });

    return suggestions;
};

export const performEDA = (dataSet: DataSet): EdaResult => {
  const { data, idCol, targetCol, featureCols } = dataSet;
  
  const variableInfo: VariableInfo[] = featureCols.map(col => ({
    name: col,
    type: identifyVariableType(col, data),
    missing: data.filter(row => row[col] === null || row[col] === undefined).length,
    distinctValues: new Set(data.map(row => row[col])).size,
  }));
  
  // FIX: Explicitly typing the accumulator for `reduce` fixes type inference issues
  // that could lead to downstream errors when calculating `totalTarget` and `imbalance`.
  const targetCounts = data.reduce((acc: { '0': number; '1': number }, row) => {
    const val = row[targetCol];
    if (val === 0) acc['0']++;
    if (val === 1) acc['1']++;
    return acc;
  }, { '0': 0, '1': 0 });

  const totalTarget = targetCounts['0'] + targetCounts['1'];
  const imbalance = (Math.min(targetCounts['0'], targetCounts['1']) / totalTarget) < IMBALANCE_THRESHOLD;

  const result: Omit<EdaResult, 'cleaningSuggestions'> = {
    totalRows: data.length,
    totalCols: featureCols.length + 2,
    variableInfo,
    targetDistribution: { ...targetCounts, imbalance },
    missingInfo: calculateMissing(data, featureCols),
    outlierInfo: detectOutliers(dataSet),
    correlationInfo: calculateCorrelations(dataSet),
  };

  const cleaningSuggestions = generateSuggestions(result as EdaResult);

  return { ...result, cleaningSuggestions };
};
