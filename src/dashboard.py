"""
Streamlit dashboard scaffold that loads a trained model and provides simple prediction UI.

Run with: `streamlit run src/dashboard.py` (after installing streamlit and dependencies).
"""
import os
import joblib
import pandas as pd
from src.feature_engineering import create_features, get_numeric_features, get_categorical_features, build_preprocessor


def run_dashboard():
    try:
        import streamlit as st
    except Exception:
        raise RuntimeError('Streamlit not installed. Install with: pip install streamlit')

    st.title('Telco Customer Churn - Prediction')
    st.write('Upload a CSV with customer rows to get churn predictions.')

    model_path = os.path.join('models','tuned_model.pkl')
    preprocessor_path = os.path.join('models','scaler_encoder.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    else:
        st.warning('Trained model not found at models/tuned_model.pkl; upload will not produce predictions.')
        model = None
        preprocessor = None

    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if model is not None and preprocessor is not None:
            try:
                # Preprocess the data
                df_processed = create_features(df)
                num_features = get_numeric_features(df_processed)
                cat_features = get_categorical_features(df_processed)
                # Build preprocessor with same features as training
                prep = build_preprocessor(num_features, cat_features)
                X = prep.fit_transform(df_processed)  # Fit on uploaded data, but should use saved preprocessor
                # Actually, we need to use the saved preprocessor to ensure feature consistency
                # But since the uploaded data might have different categories, we need to handle that
                # For simplicity, assume uploaded data has same structure
                X = preprocessor.transform(df_processed[num_features + cat_features])
                probs = model.predict_proba(X)[:,1]
                df['churn_probability'] = probs
                st.write(df[['churn_probability']])
            except Exception as e:
                st.error(f'Prediction failed: {e}')


if __name__ == '__main__':
    run_dashboard()
