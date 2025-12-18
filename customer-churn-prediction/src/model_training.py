"""
Model training utilities: build pipelines, run CV and save models.

This module exposes `train_and_select` which trains LogisticRegression, RandomForest,
and XGBoost (if available), performs simple grid search / CV and saves the best model.
"""
from typing import Optional
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier  # optional
except Exception:
    XGBClassifier = None


def train_and_select(X_train, y_train, X_test, y_test, model_dir: str = 'models', random_state: int = 42, preprocessor=None, pipeline_path: str = None) -> str:
    """Train baseline models and grid-search where applicable, return path to best model.

    Outputs:
    - saves best model as models/tuned_model.pkl and returns that path
    """
    os.makedirs(model_dir, exist_ok=True)

    # Logistic Regression baseline (no hyperparam search)
    # If a preprocessor is provided, we construct a pipeline
    if preprocessor is not None:
        lr = Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state))])
        lr_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1')
        print('LR CV f1 mean:', float(np.mean(lr_scores)))
        lr.fit(X_train, y_train)
    else:
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
        lr_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1')
        print('LR CV f1 mean:', float(np.mean(lr_scores)))
        lr.fit(X_train, y_train)

    # Random Forest with a small grid
    # Random Forest with grid - integrate preprocessor into a pipeline if provided
    rf = RandomForestClassifier(random_state=random_state)
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
    if preprocessor is not None:
        rf_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', rf)])
        rf_grid_prefixed = {f'clf__{k}': v for k, v in rf_grid.items()}
        rf_search = GridSearchCV(rf_pipeline, rf_grid_prefixed, cv=3, scoring='f1', n_jobs=-1)
        rf_search.fit(X_train, y_train)
    else:
        rf_search = GridSearchCV(rf, rf_grid, cv=3, scoring='f1', n_jobs=-1)
        rf_search.fit(X_train, y_train)
    print('RF best params:', rf_search.best_params_)

    candidates = {'logistic': lr, 'random_forest': rf_search.best_estimator_}

    # XGBoost (optional)
    if XGBClassifier is not None:
        xgb = XGBClassifier(eval_metric='logloss', random_state=random_state)
        xgb_grid = {'n_estimators': [100, 200], 'max_depth': [3, 6]}
        if preprocessor is not None:
            xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', xgb)])
            xgb_grid_prefixed = {f'clf__{k}': v for k, v in xgb_grid.items()}
            xgb_search = GridSearchCV(xgb_pipeline, xgb_grid_prefixed, cv=3, scoring='f1', n_jobs=-1)
            xgb_search.fit(X_train, y_train)
        else:
            xgb_search = GridSearchCV(xgb, xgb_grid, cv=3, scoring='f1', n_jobs=-1)
            xgb_search.fit(X_train, y_train)
        print('XGB best params:', xgb_search.best_params_)
        candidates['xgboost'] = xgb_search.best_estimator_

    # Evaluate candidates on test set and pick best by F1, tie-breaker ROC-AUC
    best_name = None
    best_score = -1.0
    best_auc = -1.0
    for name, model in candidates.items():
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas) if probas is not None else 0.0
        print(f"{name} -> test f1: {f1:.4f}, roc_auc: {auc:.4f}")
        if (f1 > best_score) or (f1 == best_score and auc > best_auc):
            best_name = name
            best_score = f1
            best_auc = auc
            best_model = model

    out_path = os.path.join(model_dir, 'tuned_model.pkl')
    joblib.dump(best_model, out_path)
    # If pipeline_path provided and best_model is not a pipeline already, combine preprocessor + estimator
    if pipeline_path is not None:
        if preprocessor is not None and not isinstance(best_model, Pipeline):
            full_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', best_model)])
        else:
            full_pipeline = best_model
        joblib.dump(full_pipeline, pipeline_path)
        print('Saved full pipeline to', pipeline_path)
    print('Saved best model (%s) to %s' % (best_name, out_path))
    return out_path


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)

