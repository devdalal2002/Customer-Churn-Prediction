import os
import tempfile
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.model_training import train_and_select
from src.feature_engineering import build_preprocessor, create_features


def make_small_df():
    # Create a dataset with several rows to satisfy CV requirements
    df = pd.DataFrame({
        'customerID': [str(i) for i in range(1, 13)],
        'tenure': [i % 12 + 1 for i in range(12)],
        'MonthlyCharges': [10 + (i % 3) * 10 for i in range(12)],
        'TotalCharges': [20 + (i % 5) * 10 for i in range(12)],
        'gender': ['Male' if i % 2 == 0 else 'Female' for i in range(12)],
        'Churn': ['No' if i % 2 == 0 else 'Yes' for i in range(12)],
    })
    df = create_features(df)
    return df


def test_train_and_save_pipeline(tmp_path):
    df = make_small_df()
    from src.feature_engineering import get_numeric_features, get_categorical_features
    numerics = get_numeric_features(df)
    cats = get_categorical_features(df)
    preprocessor = build_preprocessor(numerics, cats)

    X_train = df.drop(columns=['Churn', 'customerID'])
    y_train = df['Churn'].map({'No': 0, 'Yes': 1})
    # For simplicity reuse both train and test
    pipeline_path = tmp_path / 'pipeline.pkl'
    model_dir = tmp_path / 'models'
    os.makedirs(model_dir, exist_ok=True)

    # Should complete quickly on tiny data
    model_path = train_and_select(X_train, y_train, X_train, y_train, model_dir=str(model_dir), preprocessor=preprocessor, pipeline_path=str(pipeline_path))
    assert os.path.exists(model_path)
    assert os.path.exists(str(pipeline_path))
