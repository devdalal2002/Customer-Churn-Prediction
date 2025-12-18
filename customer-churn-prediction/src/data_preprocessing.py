"""
Cleaning and preprocessing helpers for the Telco churn project.
Functions:
- load_raw_data(path)
- clean_total_charges(df)
- save_processed(df, path)
"""
from typing import Tuple, Dict
import os
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Create a TotalCharges_imputed flag and fill TotalCharges idempotently.

    Rules implemented:
    - Convert TotalCharges to numeric (coerce errors).
    - If conversion produced NaN and tenure == 0 -> set TotalCharges = 0.0 and mark imputed.
    - Else fill NaNs with MonthlyCharges * tenure.
    - Adds column `TotalCharges_imputed` (bool).
    """
    tc = pd.to_numeric(df['TotalCharges'], errors='coerce')
    imputed_flag = tc.isna()
    df = df.copy()
    # ensure flag exists but don't overwrite if present
    if 'TotalCharges_imputed' not in df.columns:
        df['TotalCharges_imputed'] = imputed_flag
    else:
        df['TotalCharges_imputed'] = df['TotalCharges_imputed'] | imputed_flag

    # fill where tenure == 0 -> 0.0
    mask_zero_tenure = df['tenure'] == 0
    df.loc[mask_zero_tenure & df['TotalCharges_imputed'], 'TotalCharges'] = 0.0

    # remaining NaNs -> MonthlyCharges * tenure
    tc2 = pd.to_numeric(df['TotalCharges'], errors='coerce')
    remaining_na = tc2.isna()
    df.loc[remaining_na, 'TotalCharges'] = (df.loc[remaining_na, 'MonthlyCharges'] * df.loc[remaining_na, 'tenure']).fillna(0.0)

    # ensure numeric dtype
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    return df


def summary_missing_duplicates(df: pd.DataFrame) -> Dict[str, int]:
    """Return a small report dict with missing counts and duplicate count."""
    return {
        'total_rows': len(df),
        'total_columns': df.shape[1],
        'missing_values': int(df.isna().sum().sum()),
        'duplicates': int(df.duplicated().sum())
    }


def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """Encode categorical variables using one-hot encoding (pandas.get_dummies).

    Parameters
    - df: input DataFrame
    - drop_first: whether to drop the first level to avoid multicollinearity

    Returns a new DataFrame with encoded categorical features.
    """
    df = df.copy()
    # identify object/category dtype columns (exclude customerID)
    cats = [c for c in df.select_dtypes(include=['object', 'category']).columns if c.lower() not in ('customerid',)]
    if not cats:
        return df
    df = pd.get_dummies(df, columns=cats, drop_first=drop_first)
    return df


def save_interim(df: pd.DataFrame, path: str) -> None:
    """Save the cleaned DataFrame to an interim CSV path, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def generic_impute(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """Generic imputation driven by config.

    Config structure (partial):
    imputation:
      default_numeric: median|mean|zero
      default_categorical: mode|missing
      column_imputations:
        TotalCharges: telco_total_charges
    """
    df = df.copy()
    cfg = cfg or {}
    impute_cfg = cfg.get('imputation', {})
    col_imps = impute_cfg.get('column_imputations', {})

    # Special-case Telco TotalCharges behaviour if configured
    if col_imps.get('TotalCharges') == 'telco_total_charges':
        # delegate to existing Telco-specific function
        df = clean_total_charges(df)

    # Numeric defaults
    default_num = impute_cfg.get('default_numeric', 'median')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            if default_num == 'median':
                df[c] = df[c].fillna(df[c].median())
            elif default_num == 'mean':
                df[c] = df[c].fillna(df[c].mean())
            elif default_num == 'zero':
                df[c] = df[c].fillna(0.0)

    # Categorical defaults
    default_cat = impute_cfg.get('default_categorical', 'mode')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        if df[c].isna().any():
            if default_cat == 'mode':
                mode_val = df[c].mode()
                if not mode_val.empty:
                    df[c] = df[c].fillna(mode_val.iloc[0])
                else:
                    df[c] = df[c].fillna('missing')
            else:
                df[c] = df[c].fillna('missing')

    return df


def build_and_save_cleaned(raw_path: str, interim_path: str, cfg: dict = None) -> pd.DataFrame:
    """Full convenience: load raw, clean, encode, save to interim_path and return df.

    If a config dict is provided, use `generic_impute` with it and respect any column-specific rules.
    """
    df = load_raw_data(raw_path)
    print('Loaded raw:', raw_path)
    report = summary_missing_duplicates(df)
    print('Initial data summary:', report)

    if cfg and isinstance(cfg, dict):
        df = generic_impute(df, cfg=cfg)
    else:
        # default behavior: preserve old Telco-specific logic
        df = clean_total_charges(df)

    df = encode_categoricals(df)
    save_interim(df, interim_path)
    print('Saved cleaned interim to:', interim_path)
    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save the processed dataframe to a path (processed CSV)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
