import os
import sys
# Ensure parent repo root is on sys.path so `import src.*` works when Streamlit runs from the `dashboard/` path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import joblib
from src.utils import load_yaml
from sklearn.pipeline import Pipeline

CONFIG = load_yaml('config.yaml')
PIPELINE_PATH = CONFIG['models'].get('pipeline')
MODEL_PATH = CONFIG['models'].get('tuned')
PREPROCESSOR_PATH = CONFIG['models'].get('preprocessor')

st.title("Customer Churn Prediction Dashboard")
st.write("Upload a CSV to predict churn using the trained pipeline.")

# Sidebar controls
st.sidebar.header("Display options")
threshold = st.sidebar.slider("Prediction threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
hide_id = st.sidebar.checkbox("Hide ID column", value=False)

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Column mapping UI: suggest id + target candidates, allow user to pick features
    from src.utils import infer_column_mapping
    uploaded_cols = df.columns.tolist()
    suggestions = infer_column_mapping(uploaded_cols)

    # define default mapping choices so they're available regardless of automation mode
    id_choice = suggestions['id'] if suggestions.get('id') in uploaded_cols else None
    target_choice = suggestions['target'] if suggestions.get('target') in uploaded_cols else None

    # Fully automated upload flow (always enabled)
    fully_auto = True
    auto_train_cfg = st.sidebar.checkbox("Auto-train on upload if pipeline is missing", value=CONFIG.get('auto_train_on_upload', True))
    force_train_cfg = st.sidebar.checkbox("Train on upload and overwrite existing pipeline (default)", value=CONFIG.get('train_on_upload_overwrite', True))

    # The app runs preprocessing -> (optional train) -> scoring automatically when a file is uploaded.
    st.info("Uploads are processed automatically: preprocess → (optional train) → score. Mapping UI has been removed; to change default id/target names or features, edit `config.yaml`.")

    cfg_local = CONFIG.copy()
    id_col = cfg_local.get('id_column', 'customerID')
    target_col = cfg_local.get('target_column', 'Churn')

    df_proc = df.copy()
    # rename id/target if detected
    if suggestions['id'] in uploaded_cols and suggestions['id'] != id_col:
        df_proc = df_proc.rename(columns={suggestions['id']: id_col})
    if suggestions['target'] in uploaded_cols and suggestions['target'] != target_col:
        df_proc = df_proc.rename(columns={suggestions['target']: target_col})

    # infer features automatically
    numeric = [c for c in df_proc.select_dtypes(include=['number']).columns.tolist() if c not in (id_col, target_col)]
    categorical = [c for c in df_proc.select_dtypes(include=['object', 'category']).columns.tolist() if c not in (id_col, target_col)]
    cfg_local['features'] = {'numeric': numeric, 'categorical': categorical}

    # If target isn't present, try to detect likely target candidates and ask the user to confirm
    auto_train_to_use = auto_train_cfg
    if target_col not in df_proc.columns:
        from src.utils import detect_target_candidates
        cands = detect_target_candidates(df_proc, id_column=id_col)
        if cands:
            st.warning("No explicit target column found — here are candidate columns I detected (confirm to enable training):")
            # Show top 3 candidates with score and reason
            for name, score, reasons in cands[:3]:
                st.write(f"- **{name}** (score={score:.2f}) — reasons: {', '.join(reasons)}")
            selected = st.selectbox("Select the column to use as target (or choose None)", options=[None] + [c[0] for c in cands[:5]], index=0)
            confirm = st.checkbox("Confirm selected column as target and enable training", value=False)
            if selected and confirm:
                df_proc = df_proc.rename(columns={selected: target_col})
                # refresh inferred features using renamed df_proc
                numeric = [c for c in df_proc.select_dtypes(include=['number']).columns.tolist() if c not in (id_col, target_col)]
                categorical = [c for c in df_proc.select_dtypes(include=['object', 'category']).columns.tolist() if c not in (id_col, target_col)]
                cfg_local['features'] = {'numeric': numeric, 'categorical': categorical}
                auto_train_to_use = True
            else:
                st.info('Training will remain disabled for this upload until a target column is confirmed.')
                auto_train_to_use = False
        else:
            st.warning("No candidate target columns detected — training will be disabled for this upload.")
            auto_train_to_use = False

    with st.spinner('Processing and scoring uploaded file...'):
        try:
            from src.pipeline_runner import process_and_score
            df_proc, status = process_and_score(df_proc, cfg_local, try_auto_train=auto_train_to_use, force_train=(force_train_cfg and auto_train_to_use))
            # show only final results: messages + scored ids (if any)
            st.success(status.get('message', 'Processing complete'))
            if 'churn_proba' in df_proc.columns:
                thresh = cfg_local.get('prediction_threshold', 0.5)
                df_proc['churn_pred'] = (df_proc['churn_proba'] >= thresh).astype(int)
                # show only predicted churners
                out = df_proc[df_proc['churn_pred'] == 1]
                if out.empty:
                    st.warning('No customers predicted to churn at the configured threshold.')
                else:
                    display_cols = [id_col] if id_col in df_proc.columns else []
                    display_cols += ['churn_proba', 'churn_pred']
                    st.dataframe(out[display_cols].head(200))
            else:
                st.warning('No scoring available for this upload. See the message above for details.')

            # allow download of processed/scored CSV
            st.download_button('Download processed & scored CSV', df_proc.to_csv(index=False), file_name='processed_scored.csv')

            # Export Power BI CSV: run the export script which will attempt to score and write
            if st.button('Generate & download Power BI CSV'):
                import subprocess, sys, os
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'export_for_powerbi.py'))
                with st.spinner('Generating Power BI CSV...'):
                    try:
                        res = subprocess.run([sys.executable, script_path], cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) , capture_output=True)
                        if res.returncode != 0:
                            st.error(f'Export script failed: {res.stderr.decode("utf-8")}')
                        else:
                            out_path = os.path.join('data', 'processed', 'telco_powerbi.csv')
                            if os.path.exists(out_path):
                                with open(out_path, 'rb') as fh:
                                    data = fh.read()
                                    st.download_button('Download Power BI CSV', data, file_name='telco_powerbi.csv')
                                st.success('Power BI CSV generated successfully')
                            else:
                                st.error('Export script finished but output file was not found.')
                    except Exception as e:
                        st.error(f'Could not generate Power BI CSV: {e}')

            st.session_state['last_processed'] = df_proc
        except Exception as e:
            st.error(f'Automated processing failed: {e}')


# Auto-train section (opt-in)
if st.session_state.get('last_processed') is not None:
    st.header("Optional: Train a model on this uploaded dataset")
    st.write("This is an **opt-in** action — training will run locally in your session. Use only if you trust the uploaded data.")

    df_last = st.session_state['last_processed']
    id_col = CONFIG.get('id_column', 'customerID')
    target_col = CONFIG.get('target_column', 'Churn')

    enable_train = st.checkbox("Enable Auto-train on uploaded data (opt-in)", value=False)
    if enable_train:
        st.write(f"Dataset rows: {len(df_last)}")
        if target_col in df_last.columns:
            try:
                from src.utils import safe_value_counts
                vc = safe_value_counts(df_last, target_col, dropna=False)
                st.write(vc)
                # warn user if duplicates were present
                ser = df_last[target_col]
                if isinstance(ser, pd.DataFrame) and ser.shape[1] > 1:
                    st.warning("Multiple columns named '{target_col}' found; using the first matching column for counts and training. Consider renaming duplicates in your upload.")
            except KeyError:
                st.warning(f"Target column '{target_col}' not found in the dataset")
            except Exception as e:
                st.warning(f"Could not compute value counts for '{target_col}': {e}")
        min_rows = st.number_input("Minimum rows required to train", min_value=50, value=200)
        min_class_count = st.number_input("Minimum samples per class", min_value=2, value=10)

        from src.utils import can_train_on_dataframe
        ok, msg = can_train_on_dataframe(df_last, target_col, min_rows=min_rows, min_class_count=min_class_count)
        if not ok:
            st.warning(msg)
        else:
            if st.button("Start training on uploaded dataset"):
                st.info("Starting training — this may take a few minutes. Progress will appear in the logs.")
                try:
                    # prepare splits using split_data helper
                    from src.feature_engineering import split_data, build_preprocessor_from_config
                    X_train, X_test, y_train, y_test = split_data(df_last, target=target_col, test_size=CONFIG['training'].get('test_size', 0.2), random_state=CONFIG['training'].get('random_state', 42))

                    # build preprocessor from the current cfg (features previously set)
                    pre = build_preprocessor_from_config(CONFIG, df_last)

                    # call train_and_select (saves tuned_model and optionally pipeline)
                    from src.model_training import train_and_select
                    out_path = train_and_select(X_train, y_train, X_test, y_test, model_dir='models', random_state=CONFIG['training'].get('random_state', 42), preprocessor=pre, pipeline_path=PIPELINE_PATH)
                    st.success(f"Training complete. Best model saved to {out_path}")

                    # Offer to run scoring immediately with the new pipeline
                    if os.path.exists(PIPELINE_PATH):
                        full_pipe = joblib.load(PIPELINE_PATH)
                        try:
                            X = df_last.drop(columns=[id_col, target_col], errors='ignore')
                            preds = full_pipe.predict_proba(X)[:, 1]
                            df_last['churn_proba'] = preds
                            df_last['churn_pred'] = (df_last['churn_proba'] >= threshold).astype(int)
                            st.success('Scored uploaded dataset with newly trained pipeline')
                            st.dataframe(df_last[[id_col, 'churn_proba', 'churn_pred']].head(200))
                            st.session_state['last_processed'] = df_last
                        except Exception as e:
                            st.warning(f"Could not score the uploaded dataset with the new pipeline: {e}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

