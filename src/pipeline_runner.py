from typing import Tuple, Dict, Any
import os
import joblib
import pandas as pd
from src.data_preprocessing import preprocess_dataframe
from src.feature_engineering import build_preprocessor_from_config
from src.model_training import train_and_select


def process_and_score(df: pd.DataFrame, cfg: dict, try_auto_train: bool = True, force_train: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process a dataframe and attempt to score it with an existing pipeline.

    If no pipeline exists and try_auto_train is True (or force_train True), will
    attempt to train a model on the uploaded data and then score.

    If force_train is True, the function will attempt to train regardless of an
    existing pipeline (overwriting it when saved).

    Returns (processed_df, status_dict) where status_dict contains keys:
    - 'scored' (bool): whether scoring succeeded
    - 'trained' (bool): whether an auto-train occurred
    - 'message' (str): user-facing status message
    """
    status = {'scored': False, 'trained': False, 'message': ''}

    # Preprocess
    df_proc = preprocess_dataframe(df, cfg=cfg)

    # Try to load existing pipeline
    pipeline_path = cfg.get('models', {}).get('pipeline')
    model = None
    if pipeline_path and os.path.exists(pipeline_path):
        try:
            model = joblib.load(pipeline_path)
        except Exception as e:
            status['message'] += f"Failed loading pipeline: {e}. "
            model = None

    # Prepare features
    from src.utils import map_requested_to_actual
    features_cfg = cfg.get('features', {})
    if features_cfg.get('numeric') or features_cfg.get('categorical'):
        numeric = features_cfg.get('numeric', [])
        categorical = features_cfg.get('categorical', [])
        requested_feature_cols = numeric + categorical
    else:
        # fallback: use all non-id/target cols
        id_col = cfg.get('id_column', 'customerID')
        target_col = cfg.get('target_column', 'Churn')
        requested_feature_cols = [c for c in df_proc.columns if c not in (id_col, target_col)]

    # Map requested features to actual DataFrame columns using case-insensitive/substring matching
    mapped_cols, missing_cols, mapping = map_requested_to_actual(requested_feature_cols, list(df_proc.columns))

    # If many requested features are missing, fall back to using inferred features (all non-id/target)
    fallback_ratio = cfg.get('feature_mapping', {}).get('fallback_ratio', 0.5)
    if requested_feature_cols and (len(mapped_cols) / max(1, len(requested_feature_cols)) < fallback_ratio):
        id_col = cfg.get('id_column', 'customerID')
        target_col = cfg.get('target_column', 'Churn')
        feature_cols_present = [c for c in df_proc.columns if c not in (id_col, target_col)]
        status['message'] += f"Many requested features were missing ({missing_cols}), falling back to inferred features: {feature_cols_present}. "
    else:
        feature_cols_present = mapped_cols
        if missing_cols:
            status['message'] += f"Some requested features were not found and were mapped/skipped: {missing_cols}. "

    if not feature_cols_present:
        status['message'] += f"No usable feature columns were found. Requested: {requested_feature_cols}. "

    X = df_proc[feature_cols_present] if feature_cols_present else df_proc.drop(columns=[cfg.get('id_column', 'customerID'), cfg.get('target_column', 'Churn')], errors='ignore')

    # Try scoring with existing model (if available) unless we are forcing training
    if model is not None and not force_train:
        try:
            preds = model.predict_proba(X)[:, 1]
            df_proc['churn_proba'] = preds
            status['scored'] = True
            status['message'] += 'Scored with existing pipeline. '
        except Exception as e:
            status['message'] += f"Existing pipeline couldn't score data: {e}. "
            status['scored'] = False

    # If not scored and auto-train allowed, or if force_train is True, attempt training
    if (not status['scored'] and try_auto_train) or force_train:
        # check training eligibility
        target_col = cfg.get('target_column', 'Churn')
        from src.utils import can_train_on_dataframe
        ok, msg = can_train_on_dataframe(df_proc, target_col, min_rows=cfg.get('training', {}).get('min_rows', 200), min_class_count=cfg.get('training', {}).get('min_class_count', 10))
        if not ok:
            status['message'] += f"Auto-train skipped: {msg}"
        else:
            # split and train
            try:
                from src.feature_engineering import split_data
                X_train, X_test, y_train, y_test = split_data(df_proc, target=target_col, test_size=cfg.get('training', {}).get('test_size', 0.2), random_state=cfg.get('training', {}).get('random_state', 42))

                # Build a config copy that restricts features to the ones actually present
                cfg_copy = dict(cfg)
                cfg_copy.setdefault('features', {})
                # put all present feature columns as 'numeric' by default if original lists were empty
                original_numeric = cfg.get('features', {}).get('numeric', [])
                original_categorical = cfg.get('features', {}).get('categorical', [])
                if original_numeric or original_categorical:
                    # preserve type split if possible by intersecting
                    cfg_copy['features']['numeric'] = [c for c in cfg.get('features', {}).get('numeric', []) if c in feature_cols_present]
                    cfg_copy['features']['categorical'] = [c for c in cfg.get('features', {}).get('categorical', []) if c in feature_cols_present]
                else:
                    # no original split -> treat everything as numeric/categorical inference will happen in builder
                    cfg_copy['features']['numeric'] = [c for c in feature_cols_present]
                    cfg_copy['features']['categorical'] = []

                pre = build_preprocessor_from_config(cfg_copy, df_proc)

                out_path = train_and_select(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    model_dir='models',
                    random_state=cfg.get('training', {}).get('random_state', 42),
                    preprocessor=pre,
                    pipeline_path=pipeline_path,
                    cv_lr=cfg.get('training', {}).get('cv_lr', 5),
                    cv_rf=cfg.get('training', {}).get('cv_rf', 3)
                )
                status['trained'] = True
                status['message'] += f"Auto-trained and saved model to {out_path}. "
                if force_train:
                    status['message'] += 'Forced training requested — existing pipeline was overwritten (if present). '
                # attempt to load pipeline and score
                if pipeline_path and os.path.exists(pipeline_path):
                    try:
                        model2 = joblib.load(pipeline_path)
                        preds = model2.predict_proba(X)[:, 1]
                        df_proc['churn_proba'] = preds
                        status['scored'] = True
                        status['message'] += 'Scored with newly trained pipeline.'
                    except Exception as e:
                        status['message'] += f"Trained model couldn't score data: {e}. "
                else:
                    status['message'] += 'Pipeline path not found after training.'
            except Exception as e:
                # If training failed due to column mismatch in preprocessor, try a second attempt with fully inferred features
                err_msg = str(e)
                if 'not a column' in err_msg or 'not a column of the dataframe' in err_msg or 'A given column is not a column' in err_msg:
                    try:
                        # Retry with fully inferred feature set
                        id_col = cfg.get('id_column', 'customerID')
                        target_col = cfg.get('target_column', 'Churn')
                        inferred_features = [c for c in df_proc.columns if c not in (id_col, target_col)]
                        cfg_retry = dict(cfg)
                        cfg_retry.setdefault('features', {})
                        cfg_retry['features']['numeric'] = inferred_features
                        cfg_retry['features']['categorical'] = []
                        pre_retry = build_preprocessor_from_config(cfg_retry, df_proc)
                        out_path = train_and_select(X_train, y_train, X_test, y_test, model_dir='models', random_state=cfg.get('training', {}).get('random_state', 42), preprocessor=pre_retry, pipeline_path=pipeline_path)
                        status['trained'] = True
                        status['message'] += f"Auto-trained (retry inferred features) and saved model to {out_path}. "
                        if pipeline_path and os.path.exists(pipeline_path):
                            model2 = joblib.load(pipeline_path)
                            preds = model2.predict_proba(X)[:, 1]
                            df_proc['churn_proba'] = preds
                            status['scored'] = True
                            status['message'] += 'Scored with newly trained pipeline (retry).' 
                    except Exception as e2:
                        status['message'] += f"Auto-train failed on retry: {e2}. Original error: {err_msg}"
                else:
                    status['message'] += f"Auto-train failed: {e}"
        # check training eligibility
        target_col = cfg.get('target_column', 'Churn')
        from src.utils import can_train_on_dataframe
        ok, msg = can_train_on_dataframe(df_proc, target_col, min_rows=cfg.get('training', {}).get('min_rows', 200), min_class_count=cfg.get('training', {}).get('min_class_count', 10))
        if not ok:
            status['message'] += f"Auto-train skipped: {msg}"
        else:
            # split and train
            try:
                from src.feature_engineering import split_data
                X_train, X_test, y_train, y_test = split_data(df_proc, target=target_col, test_size=cfg.get('training', {}).get('test_size', 0.2), random_state=cfg.get('training', {}).get('random_state', 42))

                # Build a config copy that restricts features to the ones actually present
                cfg_copy = dict(cfg)
                cfg_copy.setdefault('features', {})
                # put all present feature columns as 'numeric' by default if original lists were empty
                original_numeric = cfg.get('features', {}).get('numeric', [])
                original_categorical = cfg.get('features', {}).get('categorical', [])
                if original_numeric or original_categorical:
                    # preserve type split if possible by intersecting
                    cfg_copy['features']['numeric'] = [c for c in cfg.get('features', {}).get('numeric', []) if c in feature_cols_present]
                    cfg_copy['features']['categorical'] = [c for c in cfg.get('features', {}).get('categorical', []) if c in feature_cols_present]
                else:
                    # no original split -> treat everything as numeric/categorical inference will happen in builder
                    cfg_copy['features']['numeric'] = [c for c in feature_cols_present]
                    cfg_copy['features']['categorical'] = []

                pre = build_preprocessor_from_config(cfg_copy, df_proc)

                out_path = train_and_select(X_train, y_train, X_test, y_test, model_dir='models', random_state=cfg.get('training', {}).get('random_state', 42), preprocessor=pre, pipeline_path=pipeline_path)
                status['trained'] = True
                status['message'] += f"Auto-trained and saved model to {out_path}. "

                # attempt to load pipeline and score
                if pipeline_path and os.path.exists(pipeline_path):
                    try:
                        model2 = joblib.load(pipeline_path)
                        preds = model2.predict_proba(X)[:, 1]
                        df_proc['churn_proba'] = preds
                        status['scored'] = True
                        status['message'] += 'Scored with newly trained pipeline.'
                    except Exception as e:
                        status['message'] += f"Trained model couldn't score data: {e}. "
                else:
                    status['message'] += 'Pipeline path not found after training.'
            except Exception as e:
                # If training failed due to column mismatch in preprocessor, try a second attempt with fully inferred features
                err_msg = str(e)
                if 'not a column' in err_msg or 'not a column of the dataframe' in err_msg or 'A given column is not a column' in err_msg:
                    try:
                        # Retry with fully inferred feature set
                        id_col = cfg.get('id_column', 'customerID')
                        target_col = cfg.get('target_column', 'Churn')
                        inferred_features = [c for c in df_proc.columns if c not in (id_col, target_col)]
                        cfg_retry = dict(cfg)
                        cfg_retry.setdefault('features', {})
                        cfg_retry['features']['numeric'] = inferred_features
                        cfg_retry['features']['categorical'] = []
                        pre_retry = build_preprocessor_from_config(cfg_retry, df_proc)
                        out_path = train_and_select(X_train, y_train, X_test, y_test, model_dir='models', random_state=cfg.get('training', {}).get('random_state', 42), preprocessor=pre_retry, pipeline_path=pipeline_path)
                        status['trained'] = True
                        status['message'] += f"Auto-trained (retry inferred features) and saved model to {out_path}. "
                        if pipeline_path and os.path.exists(pipeline_path):
                            model2 = joblib.load(pipeline_path)
                            preds = model2.predict_proba(X)[:, 1]
                            df_proc['churn_proba'] = preds
                            status['scored'] = True
                            status['message'] += 'Scored with newly trained pipeline (retry).' 
                    except Exception as e2:
                        status['message'] += f"Auto-train failed on retry: {e2}. Original error: {err_msg}"
                else:
                    status['message'] += f"Auto-train failed: {e}"

    # If still not scored, provide hint
    if not status['scored'] and not status['trained']:
        status['message'] += 'No scoring available. Consider training a model or adjusting features.'

    # Create binary predictions if probabilities present
    if 'churn_proba' in df_proc.columns:
        thresh = cfg.get('prediction_threshold', 0.5)
        df_proc['churn_pred'] = (df_proc['churn_proba'] >= thresh).astype(int)

    return df_proc, status