# Customer Churn Prediction

[![CI](https://github.com/devdalal2002/Customer-Churn-Prediction/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/devdalal2002/Customer-Churn-Prediction/actions/workflows/ci.yml)

This repository provides a reproducible end-to-end pipeline to preprocess the Telco customer churn dataset, train and evaluate baseline models, and export scored data for dashboards (Streamlit) and Power BI. It also includes explainability via SHAP and basic unit tests.

## Quick Summary
- End-to-end steps: raw CSV → preprocess → feature engineering → train + select models → evaluate → explain (SHAP) → export scored CSV for Power BI
- Supported models: LogisticRegression (baseline), RandomForest (grid search), XGBoost (optional)
- Artifacts: `models/pipeline.pkl` (preferred), `models/tuned_model.pkl`, `models/scaler_encoder.pkl`.

## Quickstart
1. Create a Python environment and install dependencies:

```pwsh
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

2. Run the full pipeline (preprocessing, features, training, evaluation, explainability):

```pwsh
python churn_pipeline.py
```

3. Export a scored CSV for Power BI (or regenerate):

```pwsh
python scripts/export_for_powerbi.py
```

4. Launch the Streamlit dashboard locally to do interactive scoring:

```pwsh
streamlit run dashboard/streamlit_app.py
```

5. Run the test suite (CI uses the same):

```pwsh
python -m pytest -q
```

## Project Structure
- `churn_pipeline.py` — Master pipeline that orchestrates preprocess → features → train → evaluate → explain.
- `config.yaml` — File of checkpoints for paths and training settings (random seed, test size, model paths).
- `src/` — Code modules.
  - `data_preprocessing.py` — functions to load data, idempotent `TotalCharges` imputation, save processed CSV.
  - `feature_engineering.py` — derived features and a `build_preprocessor` function that returns a ColumnTransformer pipeline.
  - `model_training.py` — training helpers: trains LogisticRegression, RandomForest (grid search), and optional XGBoost then selects the best model by F1 score (tie-breaker ROC-AUC).
  - `model_evaluation.py` — evaluation utilities that produce metrics and plots in `reports/figures/`.
  - `explainability.py` — SHAP explainability exports `shap_summary.png` and `shap_top10.png` if SHAP is installed.
  - `utils.py` — helpful utilities like YAML loading.
- `dashboard/streamlit_app.py` — Streamlit application for quick interactive scoring; attempts to load `models/pipeline.pkl` or fallback to `models/scaler_encoder.pkl` + `models/tuned_model.pkl`.
- `scripts/export_for_powerbi.py` — produces `data/processed/telco_powerbi.csv` with `churn_proba` and `churn_pred` values plus metrics for Power BI import.
- `data/raw/` — original `Telco-Customer-Churn.csv`.
- `data/processed/` — canonical processed dataset `telco_clean.csv` and `telco_powerbi.csv`.
- `models/` — artifacts produced by training. The README here describes expected files.
- `reports/figures/` — metrics and plots: ROC/PR, confusion matrix, SHAP plots.
- `notebooks/` — notebooks (01–07) scoped by purpose: EDA, cleaning, features, training, evaluation, explainability, dashboard development.
- `tests/` — pytest unit and integration tests.

## What the models do
- `LogisticRegression` (baseline): simple interpretable model, trained on preprocessed features with `class_weight='balanced'`.
- `RandomForest` (grid searched): tuned with `n_estimators` and `max_depth` hyperparameter grid via GridSearchCV.
- `XGBoost` (optional): trained if `xgboost` is installed; grid searched similarly.

Train & selection algorithm (in `src/model_training.py`):
- Build candidate models (LogReg, RF, optional XGB); if a preprocessor is given, it composes a scikit-learn Pipeline (`preprocessor + clf`).
- Evaluate candidates on a test set by F1 score (primary) and ROC-AUC (tie-breaker). The best model is persisted to `models/tuned_model.pkl` and the combined pipeline is saved to `models/pipeline.pkl`.

## Models and artifacts
- `models/pipeline.pkl` — Full pipeline (preprocessor + classifier); this is the preferred artifact for scoring and explainability.
- `models/tuned_model.pkl` — The fitted classifier only (no preprocessor). Useful for retraining or composing with a preprocessor.
- `models/scaler_encoder.pkl` — The fitted preprocessor (ColumnTransformer/encoders/scalers).

Notes:
- All artifacts are saved with `joblib.dump`. Re-running the pipeline overwrites them.
- These artifacts are listed in `.gitignore` to prevent committing large files.

## Explainability
- `src/explainability.py` uses SHAP (if installed) to compute per-feature attributions. If SHAP is not installed, the script logs a message and continues without SHAP outputs.

## Power BI integration
- Use `scripts/export_for_powerbi.py` to produce `data/processed/telco_powerbi.csv` containing `churn_proba` and `churn_pred` columns ready for Power BI import.
- The script also saves `reports/figures/powerbi_metrics.csv` and ROC/PR plots for use in Power BI or reporting.
- Or, from the Streamlit dashboard you can click **Generate & download Power BI CSV** after uploading/processing a file to generate and download `telco_powerbi.csv` directly.

## Streamlit Dashboard
- The dashboard in `dashboard/streamlit_app.py` loads `models/pipeline.pkl` if it exists. If not, it will attempt to compose a pipeline using `models/scaler_encoder.pkl` and `models/tuned_model.pkl`.
- Note: When you run the app with `streamlit run dashboard/streamlit_app.py`, the script will ensure the repository root is added to `sys.path` so `src.*` imports work correctly. Still, it's recommended to run Streamlit from the repository root:

```pwsh
cd "D:\VS code projects\customer-churn-prediction"
streamlit run dashboard/streamlit_app.py
```
- Upload a CSV with similar features to the training dataset and the dashboard will process (preprocess → feature engineering) and score the upload automatically. Mapping UI has been removed — the app automatically maps commonly-named id/target columns (case-insensitive substring matching) and infers features (all non-id/target columns). If auto-train is enabled it will retry training with inferred features when necessary. Uploads are configured by default to auto-run and (optionally) train and overwrite the existing pipeline (controlled by `train_on_upload_overwrite` in `config.yaml`).

## Testing & CI
- Unit tests are in `tests/`. CI configuration in `.github/workflows/ci.yml` runs the tests on push/PR to `main`.

## Recommended next steps (optional)
- Add `requirements-dev.txt` with optional dependencies (xgboost, shap, ydata_profiling). Keep `requirements.txt` minimal for a light runtime.
- Add automated model artifact versioning (timestamped filenames) and add metadata (metrics, model name) to `reports/` or a small `models/metadata.json` for traceability.
- Add a scheduled GitHub Action to run the pipeline nightly and upload `telco_powerbi.csv` to cloud storage (S3, Azure Blob) for Power BI scheduled refresh.
- Add a minimal `FastAPI` scoring endpoint and a `Dockerfile` for production deployment.

## Helpful Commands
- Run full pipeline:

```pwsh
python churn_pipeline.py
```

- Export Power BI CSV:

```pwsh
python scripts/export_for_powerbi.py
```

- Run Streamlit dashboard:

```pwsh
streamlit run dashboard/streamlit_app.py
```

- Run tests:

```pwsh
python -m pytest -q
```

## Where to look next
- To change model hyperparameters, edit `src/model_training.py`.
- To update preprocessing or derived features, edit `src/data_preprocessing.py` or `src/feature_engineering.py`.
- If you want a hosted prediction service, I can add a `src/api.py` using FastAPI and a `Dockerfile` to run the pipeline artifact in a container.

---
If you'd like, I can now add one of the recommended enhancements: `requirements-dev.txt`, scheduled nightly export CI, or a minimal FastAPI service + Dockerfile — tell me which one to implement next.
