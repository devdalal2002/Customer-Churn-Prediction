# Customer Churn Prediction Project - TODO List

## Phase 1: Extend the Pipeline
- [x] Update `churn_pipeline.py` to orchestrate preprocessing, feature engineering, model training, evaluation, and explainability.
- [x] Ensure data directories (`data/interim`, `data/processed`) exist and are created by the pipeline where needed.
- [x] Add missing imports and small helper functions to integrate all modules.

## Phase 2: Create Flask API (Optional, if needed)
- [ ] Create src/app.py for a Flask API to serve predictions (as mentioned in README).
- [ ] Implement endpoints for prediction and health check.

## Phase 3: Update Dependencies
- [ ] Add shap to requirements.txt for explainability.
- [ ] Add pyyaml for config loading.

## Phase 4: Testing and Validation
- [x] Run the full pipeline and verified outputs are created to `reports/figures` and `models/`.
- [x] Test the dashboard (Streamlit) and fallback pipeline composition.
- [x] Provided code snippets and organized notebooks 01–07 as per titles.

## Phase 5: Documentation
- [x] README updated with quick start and Power BI export instructions.
