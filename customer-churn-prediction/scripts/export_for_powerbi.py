"""
Export processed dataset and model predictions for Power BI consumption.

This script loads the processed dataset (`data/processed/telco_clean.csv`),
attempts to load a full pipeline at `models/pipeline.pkl`, if not present try a fallback
by composing a preprocessor (`models/scaler_encoder.pkl`) and a tuned model (`models/tuned_model.pkl`).
It adds `churn_proba` and `churn_pred` fields and saves `data/processed/telco_powerbi.csv`.
It also writes summary metrics and ROC/PR plots to `reports/figures/` for importing into Power BI.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_yaml
from src.feature_engineering import create_features


def load_pipeline(config):
    pipeline_path = config['models'].get('pipeline')
    preproc_path = config['models'].get('preprocessor')
    tuned_path = config['models'].get('tuned')
    if pipeline_path and os.path.exists(pipeline_path):
        return joblib.load(pipeline_path)
    if preproc_path and tuned_path and os.path.exists(preproc_path) and os.path.exists(tuned_path):
        pre = joblib.load(preproc_path)
        tune = joblib.load(tuned_path)
        return Pipeline([('preprocessor', pre), ('clf', tune)])
    return None


def compute_metrics(y_true, y_proba, y_pred):
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred))
    metrics['recall'] = float(recall_score(y_true, y_pred))
    metrics['f1'] = float(f1_score(y_true, y_pred))
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics['roc_auc'] = float(auc(fpr, tpr))
    else:
        metrics['roc_auc'] = None
    return metrics


def save_plots(y_true, y_proba, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(out_dir, 'powerbi_roc_curve.png'))
        plt.close()

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'powerbi_pr_curve.png'))
        plt.close()


def main():
    config = load_yaml('config.yaml')
    processed_path = config['defaults'].get('processed', 'data/processed/telco_clean.csv')
    if not os.path.exists(processed_path):
        # try to find a matching processed file in the workspace
        candidates = []
        for root, dirs, files in os.walk(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))):
            for f in files:
                if f.lower() in ('telco_clean.csv', 'telco_clean_preview.csv', 'telco_interim.csv'):
                    candidates.append(os.path.join(root, f))
        if candidates:
            processed_path = candidates[0]
            print('Using found processed file:', processed_path)
        else:
            raise FileNotFoundError(f"Processed file not found: {processed_path}")
    df = pd.read_csv(processed_path)
    pipeline = load_pipeline(config)
    if pipeline is None:
        print('No pipeline found; cannot score. Ensure models/pipeline.pkl or models/preprocessor & tuned_model exist.')
        # Still write the processed csv for Power BI
        out_path = os.path.join('data', 'processed', 'telco_powerbi.csv')
        df.to_csv(out_path, index=False)
        print('Wrote processed csv to', out_path)
        return

    # Create derived features that the pipeline expects
    df = create_features(df)
    # Prepare features for prediction, drop identifiers and the target
    X = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    y = None
    if 'Churn' in df.columns:
        y = df['Churn'].map({'No': 0, 'Yes': 1})

    proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    pred = pipeline.predict(X)
    df['churn_proba'] = proba
    df['churn_pred'] = pred

    out_path = os.path.join('data', 'processed', 'telco_powerbi.csv')
    df.to_csv(out_path, index=False)
    print('Wrote scored dataset to', out_path)

    # Save metrics
    if y is not None:
        y_true = y.values
        metrics = compute_metrics(y_true, proba, pred)
        metrics_path = os.path.join('reports', 'figures', 'powerbi_metrics.csv')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        pd.Series(metrics).to_csv(metrics_path, header=False)
        print('Saved metrics to', metrics_path)
        save_plots(y_true, proba, out_dir='reports/figures')


if __name__ == '__main__':
    main()
