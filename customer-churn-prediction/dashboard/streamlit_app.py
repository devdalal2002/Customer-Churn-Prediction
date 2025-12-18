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

    # Configurable defaults from config.yaml (with sane fallbacks)
    id_col = CONFIG.get('id_column', 'customerID')
    cfg_display_cols = CONFIG.get('display_columns', []) or []

    # Load pipeline or fallback
    model = None
    if PIPELINE_PATH and os.path.exists(PIPELINE_PATH):
        model = joblib.load(PIPELINE_PATH)
    else:
        # fallback: load preprocessor and tuned model
        if PREPROCESSOR_PATH and os.path.exists(PREPROCESSOR_PATH) and MODEL_PATH and os.path.exists(MODEL_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            base_model = joblib.load(MODEL_PATH)
            if preprocessor is not None and base_model is not None:
                model = Pipeline([('preprocessor', preprocessor), ('clf', base_model)])

    if model is None:
        st.error('No trained pipeline available. Please run the churn pipeline to create models/pipeline.pkl')
    else:
        # Make predictions
        try:
            X = df.drop(columns=[id_col], errors='ignore')
            # Ensure we pass only features expected by the model/pipeline
            preds = model.predict_proba(X)[:, 1]
            df['churn_proba'] = preds
            df['churn_pred'] = (df['churn_proba'] >= threshold).astype(int)

            # Allow user to select columns to display
            # Default: id_col + churn_proba + churn_pred + cfg_display_cols (if present in df)
            default_cols = []
            if id_col in df.columns:
                default_cols.append(id_col)
            default_cols += ['churn_proba', 'churn_pred']
            default_cols += [c for c in cfg_display_cols if c in df.columns]

            # Multiselect populated with sensible ordering
            all_options = list(df.columns)
            display_cols = st.multiselect("Columns to display (select to include)", options=all_options, default=default_cols)

            # Respect hide_id toggle
            if hide_id and id_col in display_cols:
                display_cols = [c for c in display_cols if c != id_col]

            # Small metrics and filtering
            st.write(f"Predicted churn (threshold={threshold}): {int(df['churn_pred'].sum())} customers")

            show_only = st.sidebar.checkbox("Show only predicted churners", value=True)

            if show_only:
                filtered = df[df['churn_pred'] == 1]
            else:
                filtered = df

            if filtered.empty:
                st.warning("No customers meet the current threshold — try lowering the threshold or toggling 'Show only predicted churners'.")
            else:
                st.dataframe(filtered[display_cols].head(200))

            # Option to download the scored CSV (full scored df)
            if st.button("Download scored CSV"):
                st.download_button("Download CSV", df.to_csv(index=False), file_name='scored.csv')

        except Exception as e:
            st.error(f"Failed to predict: {e}")
