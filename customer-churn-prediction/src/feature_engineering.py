"""
Feature engineering scaffolds: transformers and helper functions.
"""
from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def get_numeric_features(df: pd.DataFrame) -> List[str]:
    numerics = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # remove possible identifiers and the target if present
    return [c for c in numerics if c.lower() not in ('customerid', 'churn')]


def get_categorical_features(df: pd.DataFrame) -> List[str]:
    cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # exclude customer identifier and target
    return [c for c in cats if c.lower() not in ('customerid', 'churn')]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: TenureGroup and TotalChargesPerMonth."""
    df = df.copy()
    # TenureGroup buckets
    bins = [-1, 0, 12, 24, 48, 72]
    labels = ['<1', '1-12', '13-24', '25-48', '49+']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    # TotalChargesPerMonth: guard divide by zero
    df['TotalChargesPerMonth'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    return df


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]):
    """Return a ColumnTransformer that scales numeric and one-hot encodes categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'
    )
    return preprocessor


def build_preprocessor_from_config(cfg: dict, df: pd.DataFrame):
    """Build a preprocessor using config-driven feature lists when present.

    Config keys:
    features:
      numeric: [col1, col2]
      categorical: [col3, col4]
    If a list is not provided, features are inferred from the dataframe.
    """
    features_cfg = cfg.get('features', {}) if isinstance(cfg, dict) else {}
    numeric = features_cfg.get('numeric') or get_numeric_features(df)
    categorical = features_cfg.get('categorical') or get_categorical_features(df)
    return build_preprocessor(numeric, categorical)


def split_data(df: pd.DataFrame, target: str = 'Churn', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into X_train, X_test, y_train, y_test. Converts target to binary (0/1)."""
    df = df.copy()
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe")

    y = df[target].map({'No': 0, 'Yes': 1}) if df[target].dtype == object else df[target]
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# End of feature_engineering.py
