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

## Running the Backend in Google Colab (with LocalTunnel)

The repository ships with a helper script (`backend/colab_runner.py`) that starts the FastAPI backend and exposes it publicly
through [LocalTunnel](https://theboroer.github.io/localtunnel-www/). This workflow is ideal when you want to experiment directly
in a Google Colab notebook and share the API URL with the React frontend or other collaborators.

### 1. Clone the repository inside Colab

```python
!git clone https://github.com/<your-account>/Basic-ML-for-Binary-Health-Outcomes.git
%cd Basic-ML-for-Binary-Health-Outcomes
```

### 2. Install dependencies

Install the Python requirements as well as the Node.js LocalTunnel CLI:

```python
!pip install -r backend/requirements.txt
!npm install -g localtunnel
```

> Colab already bundles Node.js/npm. If the `npm` command is missing, run `!apt-get install nodejs npm` first.

### 3. Launch the backend and tunnel

Run the Colab helper to start Uvicorn and automatically open a LocalTunnel:

```python
!python backend/colab_runner.py --port 8000
```

The cell prints both the local address (`http://0.0.0.0:8000`) and the public URL generated by LocalTunnel. Keep the cell
running so the API stays online. Interrupt the execution (stop button / `Ctrl+C`) to close the tunnel and terminate the server.

> **Customisation tips:** pass `--no-localtunnel` if you only need the private server, `--subdomain your-name` to request a
> specific LocalTunnel hostname, or `--log-level debug` to increase backend verbosity.

### 4. Connect the frontend (optional)

If you want to exercise the UI from your local machine, update `VITE_API_URL` in `.env.local` to point at the LocalTunnel URL
displayed in the notebook. Afterwards run the frontend locally (`npm run dev`) and interact with the Colab-hosted backend.

## Quick Start Recap (Colab or Local Backend + Frontend)

The checklist below summarises the workflow shared by the project maintainer for quickly spinning up the full stack, either with
the backend in Google Colab or on your local machine:

1. **Run the FastAPI backend**
   - In Colab, execute the notebook cell that invokes `%%bash` to install dependencies and start `backend/main.py`. The helper
     script prints a public `loca.lt` URL and the peer IP that acts as an access token.
   - Locally, create a virtual environment, install the pinned dependencies (see the `pip install` block above), and run
     `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`.
2. **Verify the API** by opening `https://<your-tunnel>.loca.lt/docs` (or `http://localhost:8000/docs`) to confirm the Swagger UI
   loads.
3. **Set up the frontend**
   - In the repository root, run `npm install` and create a `.env.local` file containing:
     ```env
     VITE_API_URL=https://<your-tunnel>.loca.lt/api  # or http://localhost:8000/api when running locally
     GEMINI_API_KEY=<optional Gemini token>
     ```
   - Start the UI with `npm run dev` and open the printed URL (usually `http://localhost:5173`). The interface guides you through
     the pipeline: **Upload → EDA → Clean → Configure → Results**.
4. **Upload your dataset** via the frontend. Files are relayed to `/api/datasets` on the backend (tunnel URL when using Colab).
5. **Keep services running** until you finish exploring the results. Stop `npm run dev` with `Ctrl+C` and interrupt the Colab cell
   (or terminate Uvicorn) when done.

> If the LocalTunnel hostname changes during a Colab session, update `VITE_API_URL` in `.env.local` and restart `npm run dev` so
> Vite picks up the new value.
