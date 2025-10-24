
import { GoogleGenAI } from "@google/genai";
import { ModelResult } from "../types";
import { marked } from 'marked';

// IMPORTANT: This service assumes the API_KEY is available as an environment variable.
// In a real browser environment, this should be handled via a secure backend proxy
// or a build process that injects the key. For this example, we assume it's present.
const API_KEY = process.env.API_KEY;

let ai: GoogleGenAI | null = null;
if(API_KEY) {
    ai = new GoogleGenAI({ apiKey: API_KEY });
}

export const generateSummary = async (results: ModelResult[]): Promise<string> => {
    if (!ai) {
        return Promise.resolve("<h2>Gemini API not configured</h2><p>Please provide an API key to enable this feature.</p>");
    }

    const supervisedResults = results.filter(r => r.metrics);
    if(supervisedResults.length === 0) {
        return Promise.resolve("<p>No supervised models were trained, so no summary can be generated.</p>");
    }

    const bestModel = supervisedResults.reduce((best, current) => (current.metrics!.auc > best.metrics!.auc ? current : best));

    const prompt = `
        As an expert data scientist specializing in clinical epidemiology, analyze the following machine learning model results. 
        The goal is to predict a binary health outcome. Provide a concise, insightful summary in Markdown format.

        **Analysis Context:**
        - **Target Variable:** A binary health outcome (e.g., presence/absence of a condition).
        - **Metrics:** Standard classification metrics are provided (AUC, Accuracy, Sensitivity, Specificity, etc.).

        **Model Performance Data:**
        ${supervisedResults.map(r => `
        - **Model: ${r.name}**
          - AUC: ${r.metrics!.auc.toFixed(3)}
          - Accuracy: ${(r.metrics!.accuracy * 100).toFixed(1)}%
          - Sensitivity (Recall): ${r.metrics!.sensitivity.toFixed(3)}
          - Specificity: ${r.metrics!.specificity.toFixed(3)}
          - F1-Score: ${r.metrics!.f1Score.toFixed(3)}
        `).join('')}

        **Your Task:**
        1.  **Overall Performance:** Briefly state which model performed best overall, using AUC as the primary indicator. Mention its key performance metrics.
        2.  **Model Comparison:** Compare the top-performing models. Are they significantly different? Does one offer a better trade-off (e.g., higher sensitivity at the cost of specificity)?
        3.  **Clinical Interpretation:** Based on the metrics, discuss the potential clinical utility. For example, is the best model better at correctly identifying patients with the condition (high sensitivity) or those without it (high specificity)? What are the implications of false positives vs. false negatives in this context?
        4.  **Recommendations:** Suggest next steps. This could include feature importance analysis (if available for tree-based models), further hyperparameter tuning, or considering the clinical context for model selection beyond just AUC.

        Structure your response with clear headings (e.g., **Overall Performance**, **Model Comparison**, etc.). Be professional and clear.
    `;

    try {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: [{ parts: [{ text: prompt }] }],
        });
        const text = response.text;
        return marked(text) as string;
    } catch (error) {
        console.error("Gemini API call failed:", error);
        throw new Error("Failed to generate summary from Gemini API.");
    }
};
