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

Follow the steps below to install dependencies, configure environment variables, and run both services.

### 1. Backend (FastAPI)

**Prerequisites:** Python 3.10 or newer.

1. Create and activate an isolated environment:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install the backend dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > The requirements file includes `python-multipart`, which enables file uploads for dataset ingestion.

3. (Optional) Verify that the service starts successfully:

   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   You should see Uvicorn log output indicating the API is serving requests on `http://localhost:8000`.

### 2. Frontend (Vite + React)

**Prerequisites:** Node.js 18 or newer and npm (comes with Node.js).

1. Install the JavaScript dependencies in the project root:

   ```bash
   npm install
   ```

2. Create a `.env.local` file (explained below) so the frontend can reach the backend and optional AI service.

3. Start the development server:

   ```bash
   npm run dev
   ```

   Vite will print a local URL (typically `http://localhost:5173`). Ensure the backend is running on `http://localhost:8000` or update `VITE_API_URL` to match your backend host/port.

## Environment Variables & API Keys

Create a `.env.local` file in the repository root to configure runtime values consumed by the frontend build:

```
VITE_API_URL=http://localhost:8000/api
GEMINI_API_KEY=your-gemini-api-key
```

- `VITE_API_URL` points the React app to your FastAPI instance. Adjust it if the backend runs on a different host/port.
- `GEMINI_API_KEY` enables optional AI-generated summaries of the model results via Google Gemini. If you omit the key, the UI will fall back to a helpful placeholder message.

> **Where do I get a Gemini API key?** Create a free key in the [Google AI Studio](https://aistudio.google.com/app/apikey). Store the value securely and never commit it to source control. The frontend build system injects the key into `process.env` at runtime; no additional backend configuration is required.

After setting the variables, restart `npm run dev` so Vite picks up the changes. If you are running the backend in a new terminal session, remember to activate the virtual environment again before launching Uvicorn.

## Features Implemented

- Strict input validation (ID uniqueness, numeric features, binary target).
- Automated EDA with variable typing, missingness, outlier detection (IQR), correlation screening, histograms, and box-plot summaries.
- Cleaning workflow with auditable suggestions (missing values, outliers, multicollinearity) applied server-side.
- Cross-validated (Stratified 10-fold) pipelines for Logistic Regression, KNN (odd k up to n-1), SVM (flexibility presets), Random Forest, and Gradient Boosting with proper preprocessing.
- ROC curves, confusion matrices, clinical metrics (sensitivity, specificity, VPP, VPN, F1, AUC, accuracy), and feature importances for tree models.
- K-Means elbow computation, cluster assignment analysis, and CSV export of cluster membership with positive-rate summaries.

Refer to the in-app workflow for additional guidance on dataset format and interpretation of outputs.
