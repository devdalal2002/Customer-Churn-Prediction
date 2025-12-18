This folder will contain serialized model artifacts (pickles) produced by training.

- `pipeline.pkl`             # Full pipeline (preprocessor + estimator); preferred for deployment
- `tuned_model.pkl`          # Best-only estimator (not including preprocessor)
- `scaler_encoder.pkl`       # preprocessor (scalers/encoders); used when composition is preferred

Note: For a minimal repository we keep only the model scaffolding and avoid committing large binary model files; use `churn_pipeline.py` to train and reproduce these artifacts locally.
