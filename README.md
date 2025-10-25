<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Health Data Science ML Pipeline

This project provides an end-to-end workflow for binary clinical outcome modelling. It includes a React front-end for orchestrating the analysis workflow and a FastAPI backend that performs data validation, exploratory data analysis (EDA), cleaning, and machine-learning model training using scikit-learn.

## Architecture Overview

- **Frontend** (`npm run dev`): Guides the user through uploading the dataset, inspecting the automated EDA report, applying suggested cleaning steps, configuring model training, and reviewing the results.
- **Backend** (`uvicorn backend.main:app`): Implements all analytics logic in Python. The API loads Excel/CSV files, validates the schema, generates EDA artefacts (variable typing, missingness, outliers, correlations, histogram/box plot summaries), applies configurable cleaning actions, runs cross-validated ML pipelines, and returns metrics/visualisation data to the UI.

The services communicate over HTTP; set `VITE_API_URL` in the frontend to point to the backend (defaults to `http://localhost:8000/api`).

## Running Locally

### 1. Backend

Requirements: Python 3.10+

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

Requirements: Node.js 18+

```bash
npm install
# Optional: create .env.local with VITE_API_URL and GEMINI_API_KEY variables
npm run dev
```

Open the application at the URL printed by Vite (typically `http://localhost:5173`). The frontend expects the backend to be running on `http://localhost:8000`; adjust `VITE_API_URL` if you bind to a different host/port.

## Environment Variables

Create a `.env.local` file in the project root to override defaults:

```
VITE_API_URL=http://localhost:8000/api
GEMINI_API_KEY=your-optional-api-key
```

The Gemini key is only required if you want to generate the optional AI-written result summary.

## Features Implemented

- Strict input validation (ID uniqueness, numeric features, binary target).
- Automated EDA with variable typing, missingness, outlier detection (IQR), correlation screening, histograms, and box-plot summaries.
- Cleaning workflow with auditable suggestions (missing values, outliers, multicollinearity) applied server-side.
- Cross-validated (Stratified 10-fold) pipelines for Logistic Regression, KNN (odd k up to n-1), SVM (flexibility presets), Random Forest, and Gradient Boosting with proper preprocessing.
- ROC curves, confusion matrices, clinical metrics (sensitivity, specificity, VPP, VPN, F1, AUC, accuracy), and feature importances for tree models.
- K-Means elbow computation, cluster assignment analysis, and CSV export of cluster membership with positive-rate summaries.

Refer to the in-app workflow for additional guidance on dataset format and interpretation of outputs.
