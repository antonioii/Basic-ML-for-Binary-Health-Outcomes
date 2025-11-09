# Repository Guidelines

## Project Structure & Module Organization
The React/Vite front end lives at the repo root: `App.tsx` wires the upload → EDA → cleaning → configuration → results flow, UI blocks live in `components/` (atoms in `components/common`), and API/Gemini helpers plus shared types live in `services/` and `types.ts`. The FastAPI backend is contained in `backend/`, with routes in `main.py`, DTOs in `schemas.py`, cached datasets in `dataset_store.py`, analytics in `services/{eda,cleaning,training}.py`, and `launcher.py` gluing both stacks for Windows users. `backend/services/training.py` orchestrates every supervised learner plus K-Means; stick to that module when adding models or tweaking hyperparameter grids so the UI stays in sync.

### Model Training Features
- Supervised options: Logistic Regression, Elastic Net (LogReg), KNN, SVM (with Low/Medium/High flexibility presets), Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Gaussian Naive Bayes, soft/hard VotingClassifier, and StackingClassifier (LogReg meta-estimator). K-Means keeps its dedicated elbow + clustering output.  
- All supervised models run inside sklearn Pipelines with the shared preprocessing builder, `StratifiedKFold` splits, and `GridSearchCV` (use sensible search grids and keep randomness at `random_state=42`).  
- Ensemble builders only activate when at least two distinct base estimators were successfully trained; otherwise return a `status_message` so the UI can explain the skip. Optional deps (xgboost/lightgbm/catboost) are pinned in `backend/requirements.txt` and installed by the launcher, but keep the graceful error handling so manual setups still get actionable feedback.  
- Dataset payloads now include `classDistribution` (populated via `backend/utils/datasets.py`) so the front end can infer realistic fold counts without recalculating from scratch—update this helper whenever dataset metadata evolves.
- The training ETA lives in `services/trainingEstimator.ts`, which weighs dataset size, fold counts, model complexity, and search grids to produce both a total duration and per-stage breakdown (e.g., “Creating 8-fold splits…”, “Training Random Forest…”). `ResultsStep` consumes that estimate to display a live stage indicator alongside the ETA; keep this contract in sync if backend orchestration changes.

## Build, Test, and Development Commands
Run `npm install` once, then `npm run dev` for hot reload on `http://localhost:3000`. Verify release output with `npm run build && npm run preview`. Backend work starts with `python -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt`, and `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload` serves the API plus `/docs`. Use `python launcher.py` whenever you need to mimic the turnkey workflow.

## Coding Style & Naming Conventions
Maintain two-space indentation and single quotes in TypeScript, keep components PascalCase (`ResultsStep.tsx`), hooks/utilities camelCase, and register new types in `types.ts` rather than repeating interfaces. Favor functional components, React hooks, and colocated state; remote calls should pass through `services/apiClient.ts`. Python code should stay fully typed, `snake_case`, and organized as pure helpers inside `backend/services/`, while FastAPI routes remain thin orchestrators with shared validation helpers.

## Testing Guidelines
There is no committed suite yet, so add coverage alongside every feature. Place backend specs under `backend/tests/`, run them with `pytest -q backend`, and use anonymized `.xlsx` fixtures in `tests/data/` to pin EDA/training outputs. For the UI, add Vitest + React Testing Library cases named `ComponentName.test.tsx`, mock HTTP calls with MSW, and wire `npm test` into your workflow. Target ≥80 % coverage on touched modules and mention skipped scenarios in the PR.

## Commit & Pull Request Guidelines
Follow the existing imperative, single-line subjects (`Implement drag-and-drop`, `bugFix: ...`). Keep them under ~50 characters, expanding on motivation or rollout steps in the body. Before opening a PR, squash WIP commits, describe the change, list commands you ran (`npm run dev`, `pytest`), attach screenshots for UI updates, and call out launcher or schema edits so reviewers can retest integrations.

## Security & Configuration Tips
Never commit `.env.local`, API keys, or clinical datasets. Document new environment variables in the README, scrub notebooks, and anonymize IDs in sample files. Align both `package.json` and `backend/requirements.txt` when adding dependencies so the launcher stays reproducible.
