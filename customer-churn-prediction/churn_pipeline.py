"""
Master pipeline runner that calls preprocessing, feature engineering, training, evaluation, and explainability.
Usage: python churn_pipeline.py
"""
import os
from src.data_preprocessing import load_raw_data, clean_total_charges, save_processed
from src.feature_engineering import create_features, get_numeric_features, get_categorical_features, build_preprocessor, split_data
from src.model_training import train_and_select
from src.model_evaluation import evaluate_model, save_metrics
from sklearn.pipeline import Pipeline
from src.explainability import explain_model_shap
from src.utils import load_yaml
import joblib

CONFIG = load_yaml('config.yaml')
RAW_PATH = CONFIG['defaults']['raw']
PROCESSED_PATH = CONFIG['defaults']['processed']
MODEL_PATH = CONFIG['models']['tuned']
PREPROCESSOR_PATH = CONFIG['models']['preprocessor']

def run_all():
    print('Running full churn pipeline')

    # Step 1: Load and preprocess data
    df = load_raw_data(RAW_PATH)
    df = clean_total_charges(df)
    save_processed(df, PROCESSED_PATH)
    print('Data preprocessing complete.')

    # Step 2: Feature engineering
    df = create_features(df)
    numeric_features = get_numeric_features(df)
    categorical_features = get_categorical_features(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target='Churn', test_size=CONFIG['training']['test_size'], random_state=CONFIG['training']['random_state'])

    # Fit preprocessor and optionally save
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print('Feature engineering complete.')

    # Step 3: Train model
    # Train using raw X and preprocessor where the train_and_select will handle pipeline composition
    model_path = train_and_select(X_train, y_train, X_test, y_test, model_dir=os.path.dirname(MODEL_PATH), random_state=CONFIG['training']['random_state'], preprocessor=preprocessor, pipeline_path=CONFIG['models'].get('pipeline'))
    model = joblib.load(model_path)
    print('Model training complete.')

    # Step 4: Evaluate model
    # If model is a pipeline (contains a preprocessor), evaluate against raw X_test, else use processed
    if isinstance(model, Pipeline):
        metrics = evaluate_model(model, X_test, y_test, out_dir='reports/figures')
    else:
        metrics = evaluate_model(model, X_test_processed, y_test, out_dir='reports/figures')
    save_metrics(metrics, 'reports/figures/metrics.txt')
    print('Model evaluation complete.')

    # Step 5: Explainability
    # For explainability, pass raw X (the pipeline will include preprocessing) and raw feature names
    explain_model_shap(model, X_test, feature_names=X_test.columns.tolist(), out_dir='reports/figures')
    print('Explainability analysis complete.')

    print('Full pipeline finished. Check reports/figures for outputs.')

if __name__ == '__main__':
    run_all()
