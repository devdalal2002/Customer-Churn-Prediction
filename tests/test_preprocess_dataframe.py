import pandas as pd
from src.data_preprocessing import preprocess_dataframe


def test_preprocess_basic():
    df = pd.DataFrame({'tenure': [0, 10], 'MonthlyCharges': [50.0, 20.0], 'TotalCharges': [None, '200']})
    out = preprocess_dataframe(df, cfg={'imputation': {'column_imputations': {'TotalCharges': 'telco_total_charges'}}})
    assert 'TotalCharges_imputed' in out.columns
    assert 'TenureGroup' in out.columns or 'TotalChargesPerMonth' in out.columns


def test_preprocess_non_telco():
    df = pd.DataFrame({'a': [1, None], 'b': ['x', None]})
    out = preprocess_dataframe(df, cfg={})
    assert 'a' in out.columns and out['a'].isna().sum() == 0